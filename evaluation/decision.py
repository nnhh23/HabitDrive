"""Local decision model for the smoke run (substitute backend).

Uses Llama-3.1-8B-Instruct via transformers. Paper-specified decision models
(DeepSeek V4 Flash / Gemini 3.5 Flash-Lite) are NOT reachable from this server
(the OpenAI-compatible gateway is unreachable), so this is a *substitute
backend* and must not be reported as a paper-model reproduction.
"""
from __future__ import annotations

import json
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/path/to/weights/Llama-3.1-8B-Instruct"

SYS = ("You are an in-vehicle assistant. Answer with ONLY a JSON object, no "
       "extra text.")


class LocalLLM:
    def __init__(self, model_path=MODEL_PATH, gpu=0, max_new_tokens=256,
                 temperature=0.0, max_input_tokens=3500):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_input_tokens = max_input_tokens
        torch.cuda.set_device(gpu)
        self.tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map={"": f"cuda:{gpu}"},
            torch_dtype=torch.bfloat16, local_files_only=True)
        self.model.eval()

    def generate(self, prompts):
        msgs = [[{"role": "system", "content": SYS},
                 {"role": "user", "content": p}] for p in prompts]
        texts = [self.tok.apply_chat_template(m, tokenize=False,
                                              add_generation_prompt=True)
                 for m in msgs]
        enc = self.tok(texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=self.max_input_tokens)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        inp_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0, temperature=self.temperature,
                top_p=0.9, pad_token_id=self.tok.eos_token_id)
        res = []
        for i in range(len(prompts)):
            res.append(self.tok.decode(out[i][inp_len:],
                                       skip_special_tokens=True).strip())
        return res


def truncate_to_tokens(tok, text, budget, from_end=False):
    """Truncate text to at most `budget` tokens; keep the beginning unless
    from_end=True (keep the tail)."""
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= budget:
        return text
    kept = ids[-budget:] if from_end else ids[:budget]
    return tok.decode(kept, skip_special_tokens=True).strip()


def extract_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# ---- prompt builders ----

def fmt_history(history_messages, limit=None):
    lines = []
    for m in history_messages[:limit]:
        who = m.get("speaker_name") or m.get("speaker_id") or "assistant"
        if m["role"] == "assistant":
            who = "assistant"
        lines.append(f"[{m['timestamp'][:16]}] {who}: {m['text']}")
    return "\n".join(lines)


def route_destination_prompt(task, memory_text):
    pub = task["public"]
    req = pub["request"]
    ctx = pub["context"]
    occupants = ", ".join(ctx.get("occupant_ids", []))
    return (
        f"Trip request from {req['speaker_name']} ({req['speaker_id']}): "
        f"\"{req['text']}\"\n"
        f"Context: weather={ctx['weather']}, time={ctx['time_period']}, "
        f"trip={ctx['trip_context']}, occupants={occupants}.\n"
        f"\nRemembered information:\n{memory_text}\n"
        f"\nResolve the DESTINATION for this request. "
        f"Output ONLY JSON: {{\"destination\": \"<place name>\"}}."
    )


def route_selection_prompt(task, memory_text):
    pub = task["public"]
    req = pub["request"]
    ctx = pub["context"]
    routes = pub["stages"]["route_selection"]["candidates"]
    cand_txt = "\n".join(
        f"{r['route_id']}: highway={r['uses_highway']}, "
        f"toll={r['uses_toll_road']}, congestion={r['congestion_level']}, "
        f"traffic_lights={r['traffic_lights']}, lane_changes={r['lane_changes']}, "
        f"smoothness={r['smoothness']}, scenery={r['scenery']}, "
        f"time={r['travel_time_minutes']}min, distance={r['distance_km']}km, "
        f"rest_stop={r['rest_stop_available']}" for r in routes)
    return (
        f"Trip request: \"{req['text']}\" (speaker {req['speaker_name']}).\n"
        f"Context: weather={ctx['weather']}, time={ctx['time_period']}, "
        f"trip={ctx['trip_context']}.\n"
        f"\nRemembered information:\n{memory_text}\n"
        f"\nCandidate routes:\n{cand_txt}\n"
        f"\nSelect the route that satisfies the user's preferences. "
        f"Output ONLY JSON: {{\"route_id\": \"<id>\"}}."
    )


def parking_prompt(task, memory_text):
    pub = task["public"]
    lots = pub["stages"]["parking_selection"]["candidates"]
    cand_txt = "\n".join(
        f"{p['parking_id']}: walk={p['walking_distance_m']}m, "
        f"cost={p['cost_level']}, indoor={p['indoor']}, "
        f"easy={p['easy_to_park']}, accessible={p['accessible_pickup']}, "
        f"lot={p.get('lot_name','')}" for p in lots)
    return (
        f"Task: \"{pub['request']['text']}\".\n"
        f"\nRemembered information:\n{memory_text}\n"
        f"\nCandidate parking:\n{cand_txt}\n"
        f"\nSelect the parking option that satisfies the user's preferences. "
        f"Output ONLY JSON: {{\"parking_id\": \"<id>\"}}."
    )


def cabin_prompt(task, memory_text):
    pub = task["public"]
    env = pub["environment"]
    ctx = env["context"]
    init = env["initial_device_state"]
    fields = env["adjustable_fields"]
    occupants = ", ".join(o["seat"] + "=" + o["member_id"]
                          for o in ctx.get("occupants", []))
    init_txt = json.dumps(init, ensure_ascii=False)
    fields_txt = json.dumps(fields, ensure_ascii=False)
    return (
        f"Request: \"{pub['request']['text']}\" (speaker "
        f"{pub['request']['speaker_name']}).\n"
        f"Context: weather={ctx['weather']}, outside={ctx['outside_temperature_band']}, "
        f"time={ctx['time_period']}, trip={ctx['trip_context']}, "
        f"occupants={occupants}.\n"
        f"Initial device state: {init_txt}\n"
        f"Adjustable fields (allowed values): {fields_txt}\n"
        f"\nRemembered information:\n{memory_text}\n"
        f"\nProduce a setting plan applying the user's usual preferences. "
        f"Leave unrelated/protected settings unchanged. "
        f"Output ONLY JSON: {{\"settings\": {{\"<field>\": \"<value>\"}}}}."
    )
