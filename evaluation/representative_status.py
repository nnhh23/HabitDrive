"""Record the honest status of the five representative memory systems:
installed / not_installed / interface_available / e2e_run / blocked.

Separates "import works" from "can actually build+run", because the latter
needs a remote extraction LLM + dense embedding backend (unreachable) or a
local embedding model (not installed).
"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys

ROOT = "/path/to/habitdrive_eval"
THIRD = "/path/to/habitdrive/third_party"

SPECS = {
    "mem0": {"module": "mem0", "version_attr": "__version__",
             "blocked_reason": "extraction LLM + embedding via OpenAI-compatible "
             "gateway (unreachable); no local embedding model"},
    "memobase": {"module": "memobase", "version_attr": "__version__",
                 "blocked_reason": "client only; self-hosted backend "
                 "(PostgreSQL+Redis FastAPI) not running"},
    "lightmem": {"module": "lightmem",
                 "blocked_reason": "requires remote LLM/embedding or local "
                 "LLMLingua-2 + embedding model (unavailable)"},
    "memoryos": {"module": "memoryos", "path": "MemoryOS/memoryos-pypi",
                 "blocked_reason": "requires remote LLM/embedding or local HF "
                 "embeddings + faiss-gpu (unavailable)"},
    "a_mem": {"module": "memory_layer", "path": "AgenticMemory",
              "blocked_reason": "missing dependency rank_bm25 and remote "
              "embeddings (unreachable)"},
}


def check():
    results = {}
    for name, spec in SPECS.items():
        r = {"name": name}
        if spec.get("path"):
            sys.path.insert(0, f"{THIRD}/{spec['path']}")
        try:
            mod = importlib.import_module(spec["module"])
            r["import"] = "ok"
            r["version"] = getattr(mod, spec.get("version_attr", ""), "unknown")
            r["status"] = "installed"
        except Exception as e:
            r["import"] = f"failed: {type(e).__name__}: {e}"
            r["status"] = "not_installed"
        r["interface_available"] = False
        r["e2e_run"] = False
        r["blocked_reason"] = spec["blocked_reason"]
        results[name] = r
    return results


if __name__ == "__main__":
    res = check()
    with open(f"{ROOT}/reports/representative_status.json", "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1))
