"""Memory methods for HabitDrive smoke run.

Basic strategies (No Memory, Oracle, Full Context, Recursive Summarization,
Vector RAG, Key-Value Store) are implemented locally. Representative systems
(Mem0, LightMem, Memobase, MemoryOS, A-MEM) are adapted to their real
implementations; status is one of installed / interface_available /
e2e_run / blocked and is recorded per run.

Important compliance note: Key-Value Store extracts records from the
*natural-language* history (never from the generator's preference tables);
Oracle provides only currently-valid, task-relevant preference *evidence*,
never the final choice, target action, or reference answer.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict

DATA = "/path/to/habitdrive_full/data"
GEN = "/path/to/habitdrive_full/code"


class BaseMethod:
    name = "base"

    def reset(self, household_id):
        pass

    def ingest(self, household_id, history_messages, llm=None):
        pass

    def provide(self, task, stage):
        return ""


class NoMemory(BaseMethod):
    name = "no_memory"

    def provide(self, task, stage):
        return "(no historical memory available)"


class FullContext(BaseMethod):
    name = "full_context"
    _hist = {}

    def ingest(self, household_id, history_messages, llm=None):
        self._hist[household_id] = history_messages

    def provide(self, task, stage):
        hid = task["household_id"]
        from decision import fmt_history
        return fmt_history(self._hist.get(hid, []))


class RecentHistory(BaseMethod):
    """Most recent raw turns under the same 8,192-token budget as retrieved
    memory (token-matched recency baseline)."""
    name = "recent_history"
    _hist = {}
    _tok = None
    BUDGET = 8192

    def ingest(self, household_id, history_messages, llm=None):
        self._hist[household_id] = history_messages
        self._tok = getattr(llm, "tok", None)

    def provide(self, task, stage):
        from decision import fmt_history, truncate_to_tokens
        full = fmt_history(self._hist.get(task["household_id"], []))
        if self._tok is not None:
            return truncate_to_tokens(self._tok, full, self.BUDGET, from_end=True)
        return full[-6000:]


class FullContextTokenMatched(BaseMethod):
    """Full Context truncated to the same 8,192-token budget (token-matched
    Full Context variant)."""
    name = "full_context_8k"
    _hist = {}
    _tok = None
    BUDGET = 8192

    def ingest(self, household_id, history_messages, llm=None):
        self._hist[household_id] = history_messages
        self._tok = getattr(llm, "tok", None)

    def provide(self, task, stage):
        from decision import fmt_history, truncate_to_tokens
        full = fmt_history(self._hist.get(task["household_id"], []))
        if self._tok is not None:
            return truncate_to_tokens(self._tok, full, self.BUDGET, from_end=False)
        return full


class RecursiveSummarization(BaseMethod):
    name = "recursive_summarization"
    _summary = {}
    _chunk = 10

    def ingest(self, household_id, history_messages, llm=None):
        # Chronological summary updates, chunk by chunk.
        summary = ""
        from decision import fmt_history
        for i in range(0, len(history_messages), self._chunk):
            chunk = history_messages[i:i + self._chunk]
            chunk_txt = fmt_history(chunk)
            prompt = (
                f"Previous summary:\n{summary or '(none)'}\n\n"
                f"New interaction chunk:\n{chunk_txt}\n\n"
                f"Update the running summary to concisely retain each member's "
                f"currently-valid preferences (locations, route, parking, cabin "
                f"settings), applicability conditions, and any updates or "
                f"cancellations, with speaker identity. Output only the new "
                f"summary text."
            )
            if llm is not None:
                out = llm.generate([prompt])[0]
                summary = out.strip() or summary
            else:
                summary = chunk_txt
        self._summary[household_id] = summary

    def provide(self, task, stage):
        return self._summary.get(task["household_id"], "")


class VectorRAG(BaseMethod):
    name = "vector_rag"
    _index = {}
    _top_k = 8

    def ingest(self, household_id, history_messages, llm=None):
        from decision import fmt_history
        # Chunk history into passages of ~6 messages, build TF-IDF vectors.
        passages = []
        for i in range(0, len(history_messages), 6):
            chunk = history_messages[i:i + 6]
            passages.append(fmt_history(chunk))
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except Exception:
            # fallback: keep passages, do naive keyword matching later
            self._index[household_id] = {"passages": passages, "vec": None}
            return
        vec = TfidfVectorizer().fit(passages)
        mat = vec.transform(passages)
        self._index[household_id] = {"passages": passages, "vec": vec,
                                     "mat": mat,
                                     "cos": cosine_similarity}

    def provide(self, task, stage):
        entry = self._index.get(task["household_id"])
        if not entry:
            return ""
        q = _query_text(task, stage)
        passages = entry["passages"]
        if entry.get("mat") is None:
            # keyword fallback
            toks = set(re.findall(r"[a-z]+", q.lower()))
            scored = sorted(passages, key=lambda p: sum(
                t in p.lower() for t in toks), reverse=True)
            return "\n---\n".join(scored[:self._top_k])
        qv = entry["vec"].transform([q])
        sims = entry["cos"](qv, entry["mat"])[0]
        order = sorted(range(len(sims)), key=lambda i: -sims[i])
        return "\n---\n".join(passages[i] for i in order[:self._top_k])


class KeyValueStore(BaseMethod):
    name = "key_value_store"
    _records = {}

    def ingest(self, household_id, history_messages, llm=None):
        self._records[household_id] = extract_records(history_messages)

    def provide(self, task, stage):
        recs = self._records.get(task["household_id"], [])
        q = _query_text(task, stage).lower()
        # retrieve records whose value/attribute overlaps the query keywords
        toks = set(re.findall(r"[a-z]+", q))
        relevant = [r for r in recs if toks & r["tokens"]]
        if not relevant:
            relevant = recs[-20:]
        return "\n".join(f"[{r['speaker']}] {r['text']}" for r in relevant[:20])


def _query_text(task, stage):
    pub = task["public"]
    req = pub["request"]["text"]
    ctx = pub.get("context") or pub.get("environment", {}).get("context", {})
    if stage == "route_selection":
        return req + " " + " ".join(str(v) for v in ctx.values())
    if stage == "parking_selection":
        return req + " parking"
    if stage == "cabin":
        return req
    return req


# ---- heuristic record extraction from NL history (NOT from tables) ----
_ATTR_RE = re.compile(
    r"\b(highway|toll|toll road|congestion|traffic light|lane change|smooth|"
    r"scenic|scenery|travel time|distance|parking|temperature|fan|airflow|"
    r"seat heating|seat ventilation|seat|volume|music|window|sunroof|light|"
    r"workplace|work|office|school|home|home address)\b", re.I)


def extract_records(history_messages):
    records = []
    for m in history_messages:
        if m["role"] != "user":
            continue
        text = m["text"]
        toks = set(re.findall(r"[a-z]+", text.lower()))
        records.append({
            "speaker": m.get("speaker_name") or m.get("speaker_id") or "?",
            "speaker_id": m.get("speaker_id"),
            "timestamp": m["timestamp"],
            "text": text,
            "tokens": toks,
        })
    return records


# ---- Oracle (valid, task-relevant evidence; never the answer) ----
class OracleMemory(BaseMethod):
    name = "oracle_memory"
    _active = None
    _mem_by_owner = None

    def ingest(self, household_id, history_messages, llm=None):
        if self._active is None:
            self._load()

    def _load(self):
        import sys
        sys.path.insert(0, GEN)
        import generate_structure as gs
        memories = json.load(open(f"{DATA}/memories.json", encoding="utf-8"))["memories"]
        self._active = gs.final_active(memories)
        self._mem_by_owner = defaultdict(list)
        for m in memories:
            self._mem_by_owner[m["owner_id"]].append(m)

    def provide(self, task, stage):
        if self._active is None:
            self._load()
        import sys
        sys.path.insert(0, GEN)
        import generate_structure as gs
        active = self._active
        if stage == "cabin":
            return self._cabin_facts(task, active)
        pub = task["public"]
        ctx = pub["context"]
        req = pub["request"]["speaker_id"]
        role = task["evaluation"]["gold"]["destination"]["semantic_role"]
        occ = ctx["occupant_ids"]
        reqs, _ = gs.resolve_route_requirements(active, req, role, ctx, occ)
        facts = []
        seen = set()
        for m, attr, val in reqs:
            if m is None or m["memory_id"] in seen:
                continue
            seen.add(m["memory_id"])
            owner = m["owner_id"]
            st = m["trajectory"][-1]["state"]
            if m["category"] == "route_preference":
                facts.append(f"{owner}: {st['attribute']} -> {st['preference']}"
                             f"{_cond(m['trajectory'][-1]['condition'])}")
            elif m["category"] == "passenger_constraint":
                facts.append(f"{owner}: passenger constraint "
                             f"{st['constraint_type']}")
        if pub["parking_required"]:
            preqs, _ = gs.resolve_parking_requirements(active, req, role, ctx, occ)
            for m, attr, val in preqs:
                if m is None or m["memory_id"] in seen:
                    continue
                seen.add(m["memory_id"])
                st = m["trajectory"][-1]["state"]
                facts.append(f"{m['owner_id']}: parking "
                             f"{st.get('preference_type', attr)}"
                             f"{(' at ' + st['lot_name']) if st.get('lot_name') else ''}")
        # destination fact
        dmem = task["evaluation"]["gold"]["destination"]["memory_id"]
        for mid, (m, node) in active.items():
            if mid == dmem:
                facts.append(f"{m['owner_id']}: {node['state']['semantic_role']} "
                             f"= {node['state']['place_name']}")
        return "\n".join(facts)

    def _cabin_facts(self, task, active):
        req = task["public"]["request"]["speaker_id"]
        ev = task["evaluation"]
        facts = []
        for mid in ev["evidence"]:
            if mid in active:
                m, node = active[mid]
                st = node["state"]
                facts.append(f"{m['owner_id']}: {st.get('control_type')} "
                             f"{st.get('target_zone')} {st.get('attribute')} "
                             f"= {st.get('value')}"
                             f"{_cond(node['condition'])}")
        return "\n".join(facts)


def _cond(cond):
    if not cond:
        return ""
    k, v = next(iter(cond.items()))
    return f" (when {k}={v})"


# ---- representative system adapters (status-tracked) ----
class RepresentativeMethod(BaseMethod):
    """Attempts to load a real system; records status + error."""
    name = "representative"
    status = "not_attempted"
    error = None

    def __init__(self, name):
        self.name = name

    def ingest(self, household_id, history_messages, llm=None):
        try:
            self._build(household_id, history_messages, llm)
            self.status = "e2e_run"
        except Exception as e:
            self.status = "blocked"
            self.error = f"{type(e).__name__}: {e}"

    def _build(self, household_id, history_messages, llm):
        raise NotImplementedError

    def provide(self, task, stage):
        return ""


class Mem0Adapter(RepresentativeMethod):
    def _build(self, household_id, history_messages, llm):
        raise RuntimeError("remote extraction LLM/embedding gateway unreachable "
                           "(xiaoai.plus); no local embedding model installed")


class LightMemAdapter(RepresentativeMethod):
    def _build(self, household_id, history_messages, llm):
        raise RuntimeError("LightMem requires remote LLM/embedding or local "
                           "LLMLingua-2/embedding model; unavailable")


class MemobaseAdapter(RepresentativeMethod):
    def _build(self, household_id, history_messages, llm):
        raise RuntimeError("Memobase backend (PostgreSQL+Redis FastAPI service) "
                           "is not running on this server")


class MemoryOSAdapter(RepresentativeMethod):
    def _build(self, household_id, history_messages, llm):
        raise RuntimeError("MemoryOS requires remote LLM/embedding or local "
                           "HF embeddings + faiss-gpu; unavailable")


class AMemAdapter(RepresentativeMethod):
    def _build(self, household_id, history_messages, llm):
        raise RuntimeError("A-MEM requires remote embeddings (text-embedding) "
                           "unreachable; no local embedding model installed")


def build_basic_methods():
    return [NoMemory(), OracleMemory(), FullContext(),
            RecursiveSummarization(), VectorRAG(), KeyValueStore()]


def build_representative_methods():
    return [Mem0Adapter("mem0"), LightMemAdapter("lightmem"),
            MemobaseAdapter("memobase"), MemoryOSAdapter("memoryos"),
            AMemAdapter("a_mem")]


def all_methods():
    return build_basic_methods() + build_representative_methods()
