"""GPU smoke test: actually generate a short text with a local model on GPU."""
import sys, json, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "/path/to/weights/Llama-3.1-8B-Instruct"

def main():
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          "count", torch.cuda.device_count(), flush=True)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} "
                  f"{torch.cuda.get_device_properties(i).total_memory/1e9:.1f}GB", flush=True)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, device_map="auto", torch_dtype=torch.bfloat16, local_files_only=True)
    model.eval()
    print(f"model loaded in {time.time()-t0:.1f}s on "
          f"{next(model.parameters()).device}", flush=True)

    messages = [
        {"role": "system", "content": "You are a helpful in-vehicle assistant."},
        {"role": "user", "content": "Please confirm in one sentence that you can hear me."},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=60, do_sample=False,
                             temperature=1.0, top_p=1.0)
    text = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    print("=== GENERATED ===", flush=True)
    print(text, flush=True)
    print("=== OK ===", flush=True)
    print(json.dumps({"status": "ok", "output": text,
                      "device": str(next(model.parameters()).device),
                      "elapsed_s": round(time.time()-t0, 2)}))

if __name__ == "__main__":
    main()
