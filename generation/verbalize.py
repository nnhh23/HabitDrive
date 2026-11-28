"""GPU verbalizer: convert structured history events into natural two-turn
dialogue using a local Llama model, grounded in the structured facts.

Gold is decided before language: the model only paraphrases; template
dialogue is the deterministic fallback and never changes the structured fact.

Usage (one worker per GPU):
  python verbalize.py --gpu 0 --shard 0 --num-shards 4 \
    --input ../data/structured_history.json --output-dir ../data/history
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SYS = ("You are verbalizing a structured in-vehicle interaction into a natural "
       "two-turn dialogue. Produce ONLY a JSON object with keys "
       "\"user_utterance\" and \"assistant_response\". Preserve the speaker, all "
       "factual values (names, numbers, conditions, destinations), and any "
       "one-trip-only scope exactly as given. Do not invent new preferences or "
       "change any fact.")

UNRELATED_EXCHANGES = [
    ("What time is it right now?", "It's currently 3:24 PM."),
    ("How's the traffic on the ring road?", "Traffic on the ring road is light right now."),
    ("Can you check the weather for today?", "It will be partly cloudy with a high of 22 degrees."),
    ("Remind me to get gas this week.", "Sure, I've noted a reminder to refuel this week."),
    ("How far is the nearest charging station?", "The nearest charging station is about 2 km away."),
    ("Play something relaxing.", "Playing a calm playlist now."),
    ("Is there any news about the road closure?", "No major road closures reported on your usual routes."),
    ("Thanks, that's all for now.", "You're welcome. Drive safely."),
]


def _disp(attr):
    return "temperature" if attr == "temperature_c" else attr.replace("_", " ")


def describe_fact(ev):
    f = ev.get("fact", {})
    op = f.get("operation", ev.get("operation", ""))
    cat = f.get("category", "")
    st = f.get("state", {})
    cond = f.get("condition", {})
    if cat == "location":
        role = st.get("semantic_role", "")
        place = st.get("place_name", "")
        if op == "establish":
            return (f"Speaker establishes a {role} location: {place}.")
        if op == "content_update":
            return f"Speaker updates their {role} location to: {place}."
        if op == "cancellation":
            return f"Speaker asks to forget the location {place}."
        return f"Speaker uses the {role} location {place}."
    if cat == "route_preference":
        attr = _disp(st.get("attribute", ""))
        val = st.get("preference", "")
        cond_s = describe_condition(cond)
        return (f"Speaker expresses a route preference: {val} {attr}"
                f"{cond_s}.")
    if cat == "parking_preference":
        pt = st.get("preference_type", "")
        lot = st.get("lot_name", "")
        cond_s = describe_condition(cond)
        if pt == "preferred_parking_lot" and lot:
            return f"Speaker's parking preference: park at {lot}{cond_s}."
        return f"Speaker's parking preference: {pt.replace('_',' ')}{cond_s}."
    if cat == "passenger_constraint":
        ct = st.get("constraint_type", "").replace("_", " ")
        cond_s = describe_condition(cond)
        return f"Speaker states a passenger constraint: {ct}{cond_s}."
    if cat in ("climate", "seat", "media", "window", "ambient_light"):
        zone = st.get("target_zone", "")
        attr = _disp(st.get("attribute", ""))
        val = st.get("value", "")
        cond_s = describe_condition(cond)
        return (f"Speaker expresses a cabin preference: set {zone} {attr} "
                f"to {val}{cond_s}.")
    if op == "temporary":
        instr = f.get("instruction", {})
        if f.get("domain") == "route":
            attr = _disp(instr.get("attribute", "highway"))
            pref = instr.get("preference", "prefer")
            return (f"Speaker makes a ONE-TRIP-ONLY request: {pref} {attr} "
                    "for this trip only.")
        attr = _disp(instr.get("attribute", "temperature_c"))
        val = instr.get("value", 26)
        return (f"Speaker makes a ONE-TRIP-ONLY request: set {attr} to {val} "
                "for this trip only.")
    return "Unrelated exchange."


def describe_condition(cond):
    if not cond:
        return ""
    k, v = next(iter(cond.items()))
    if k == "destination":
        return f" when going to the {v}"
    if k == "weather":
        return f" when it is {v}"
    if k == "time_period":
        return f" during {v}"
    if k == "trip_context":
        return f" for {v}"
    if k == "outside_temperature_band":
        return f" when it is {v} outside"
    return f" when {k} is {v}"


def prompt_for(ev):
    tpl = ev.get("user_utterance", ""), ev.get("agent_response", "")
    return (
        f"Speaker name: {ev.get('speaker_name','')}\n"
        f"Event type: {ev.get('event_type','')} "
        f"({ev.get('operation','')})\n"
        f"Fact: {describe_fact(ev)}\n"
        f"Template to paraphrase naturally (keep meaning and facts):\n"
        f"  user: \"{tpl[0]}\"\n"
        f"  assistant: \"{tpl[1]}\"\n"
        f"Return JSON: {{\"user_utterance\": \"...\", "
        f"\"assistant_response\": \"...\"}}"
    )


def extract_json(text):
    text = text.strip()
    # Strip markdown fences if present.
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict) and "user_utterance" in obj and \
                "assistant_response" in obj:
            return obj
    except Exception:
        pass
    return None


def fact_ok(ev, obj):
    u = (obj.get("user_utterance") or "").strip()
    a = (obj.get("assistant_response") or "").strip()
    if not u or not a:
        return False
    if len(u) > 120 or len(a) > 220:
        return False
    f = ev.get("fact", {})
    cat = f.get("category", "")
    st = f.get("state", {})
    # Key factual tokens that must survive paraphrasing.
    checks = []
    if cat == "location" and st.get("place_name"):
        checks.append(st["place_name"])
    if cat == "parking_preference" and st.get("lot_name"):
        checks.append(st["lot_name"])
    if cat in ("climate", "seat", "media", "window", "ambient_light") \
            and st.get("value") is not None:
        checks.append(str(st["value"]))
    blob = (u + " " + a).lower()
    for c in checks:
        if str(c).lower() not in blob:
            return False
    return True


def generate_batch(model, tok, prompts, max_new_tokens):
    msgs = [[{"role": "system", "content": SYS},
             {"role": "user", "content": p}] for p in prompts]
    texts = [tok.apply_chat_template(m, tokenize=False,
                                     add_generation_prompt=True) for m in msgs]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=900)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    inp_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=0.25, top_p=0.9, repetition_penalty=1.05,
            pad_token_id=tok.eos_token_id)
    results = []
    for i in range(len(prompts)):
        gen = out[i][inp_len:]
        results.append(tok.decode(gen, skip_special_tokens=True).strip())
    return results


def process_household(model, tok, hh, batch_size, max_new_tokens):
    events = hh["events"]
    out_events = []
    # Only verbalize preference-carrying events; unrelated stay template.
    idxs = [i for i, e in enumerate(events)
            if e["event_type"] != "unrelated"]
    results = {}
    for start in range(0, len(idxs), batch_size):
        chunk_idx = idxs[start:start + batch_size]
        prompts = [prompt_for(events[i]) for i in chunk_idx]
        texts = generate_batch(model, tok, prompts, max_new_tokens)
        for i, text in zip(chunk_idx, texts):
            obj = extract_json(text)
            if obj is not None and fact_ok(events[i], obj):
                results[i] = (obj["user_utterance"], obj["assistant_response"])
            else:
                results[i] = (events[i].get("user_utterance", ""),
                              events[i].get("agent_response", ""))
    for i, e in enumerate(events):
        e2 = dict(e)
        if i in results:
            e2["user_utterance"], e2["agent_response"] = results[i]
        out_events.append(e2)
    return {"household_id": hh["household_id"],
            "namespace_id": hh.get("namespace_id", hh["household_id"]),
            "events": out_events}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model", default="/path/to/weights/Llama-3.1-8B-Instruct")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=180)
    ap.add_argument("--limit", type=int, default=0, help="max households (debug)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.cuda.set_device(args.gpu)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map={"": f"cuda:{args.gpu}"},
        torch_dtype=torch.bfloat16, local_files_only=True)
    model.eval()
    print(f"[gpu{args.gpu}] model loaded in {time.time()-t0:.1f}s", flush=True)

    data = json.load(open(args.input, encoding="utf-8"))
    histories = data["histories"]
    if args.limit:
        histories = histories[:args.limit]
    my = [h for i, h in enumerate(histories) if i % args.num_shards == args.shard]
    print(f"[gpu{args.gpu}] shard {args.shard}/{args.num_shards}: "
          f"{len(my)} households", flush=True)

    done = 0
    for h in my:
        out_path = os.path.join(args.output_dir, f"{h['household_id']}.json")
        if os.path.exists(out_path):
            done += 1
            continue
        t1 = time.time()
        out = process_household(model, tok, h, args.batch_size,
                                args.max_new_tokens)
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        os.replace(tmp, out_path)
        done += 1
        if done % 5 == 0 or done == len(my):
            print(f"[gpu{args.gpu}] {done}/{len(my)} households "
                  f"({time.time()-t1:.1f}s last)", flush=True)
    print(f"[gpu{args.gpu}] finished {done} households", flush=True)


if __name__ == "__main__":
    main()
