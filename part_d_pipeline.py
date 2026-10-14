from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm

ROOT = Path(__file__).resolve().parent
POLICY_DIR = ROOT / "data" / "policies"
RAW = ROOT / "data" / "raw"
DB = ROOT / "data" / "instamart.db"
OUT = ROOT / "outputs"
STATE_DIR = ROOT / "state"
MAX_WORDS = 40
OVERLAP_WORDS = 10


class SimulatedCrash(RuntimeError):
    pass


def chunk_documents(policy_dir: Path = POLICY_DIR, overlap: int = OVERLAP_WORDS) -> list[dict[str, object]]:
    chunks = []
    step = MAX_WORDS - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than chunk size")
    for path in sorted(policy_dir.glob("*.txt")):
        words = path.read_text(encoding="utf-8").split()
        start = 0
        while start < len(words):
            end = min(start + MAX_WORDS, len(words))
            chunks.append({"source": path.name, "text": " ".join(words[start:end])})
            if end >= len(words):
                break
            start += step
    return chunks


class PolicyRetriever:
    def __init__(self, policy_dir: Path = POLICY_DIR, overlap: int = OVERLAP_WORDS):
        self.chunks = chunk_documents(policy_dir, overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([chunk["text"] for chunk in self.chunks])

    def retrieve(self, query: str, k: int = 3) -> list[dict[str, object]]:
        vector = self.vectorizer.transform([query])
        scores = cosine_similarity(vector, self.matrix).ravel()
        indexes = scores.argsort()[::-1][:k]
        return [
            {"source": self.chunks[int(index)]["source"], "score": float(scores[index]), "text": self.chunks[int(index)]["text"]}
            for index in indexes
        ]

    def answer(self, question: str) -> dict[str, object]:
        results = self.retrieve(question, k=3)
        context = "\n".join(f"[{item['source']}]\n{item['text']}" for item in results)
        user_content = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
        response = mock_llm.chat([
            {"role": "system", "content": "POLICY ASSISTANT. Answer only from the supplied context. If the information is not in the context, say: I could not find that information in the policy documents."},
            {"role": "user", "content": user_content},
        ])
        return {
            "answer": response.choices[0].message.content or "I could not find that information in the policy documents.",
            "sources": [item["source"] for item in results],
        }


_ACTIVE_RETRIEVER: PolicyRetriever | None = None
_ACTIVE_SCENARIO_ID = ""
LEDGER = OUT / "refund_ledger.jsonl"


def retrieve(query: str, k: int = 3) -> list[dict[str, object]]:
    global _ACTIVE_RETRIEVER
    if _ACTIVE_RETRIEVER is None:
        _ACTIVE_RETRIEVER = PolicyRetriever()
    return _ACTIVE_RETRIEVER.retrieve(query, k=k)


def get_order(order_id: str) -> dict[str, object]:
    import sqlite3
    with sqlite3.connect(DB) as connection:
        row = connection.execute(
            "SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ? LIMIT 1",
            (order_id,),
        ).fetchone()
    if row is None:
        return {"error": "order not found"}
    delivery_minutes = None if row[3] in (None, "") else int(row[3])
    return {
        "order_id": row[0], "status": row[1], "order_amount": float(row[2]),
        "delivery_minutes": delivery_minutes,
    }


def search_policy(query: str) -> dict[str, object]:
    return {"results": retrieve(query, k=3)}


def calculate_refund(order_amount: float, delivery_minutes: int | None) -> dict[str, float]:
    amount = float(order_amount) if delivery_minutes is not None and int(delivery_minutes) > 45 else 0.0
    return {"refund_inr": round(amount, 2)}


def _record_ledger(tool_name: str, order_id: str, refund_inr: float) -> dict[str, float]:
    OUT.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenario_id": _ACTIVE_SCENARIO_ID,
        "tool_name": tool_name,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2),
    }
    with LEDGER.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"refund_inr": row["refund_inr"]}


def issue_refund(order_id: str, refund_inr: float) -> dict[str, float]:
    return _record_ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id: str, refund_inr: float) -> dict[str, float]:
    return _record_ledger("escalate_to_human", order_id, refund_inr)


TOOLS_SPEC = [
    {"name": "get_order", "description": "Read order status and delivery facts from the analytical database.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "search_policy", "description": "Retrieve the top policy chunks relevant to a query.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "calculate_refund", "description": "Calculate the full order refund only when delivery exceeded 45 minutes.", "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"}, "delivery_minutes": {"type": ["integer", "null"]}}, "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund", "description": "Append an eligible simulated refund action to the JSONL ledger.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human", "description": "Append an above-INR-1000 refund request for human approval to the JSONL ledger.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]}},
]
TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


def _json_tool_value(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _checkpoint(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _status(final_answer: str) -> str:
    return final_answer.split(":", 1)[0].strip() if ":" in final_answer else "FALLBACK"


def _agent_loop(state: dict[str, object], token_limit: int, time_limit_seconds: float,
                crash_after_tools: int | None = None) -> dict[str, object]:
    checkpoint_path = STATE_DIR / f"{state['run_id']}.json"
    started = time.monotonic()
    while int(state["llm_calls"]) < int(state["max_iterations"]):
        if time.monotonic() - started >= time_limit_seconds:
            final = "FALLBACK_TIMEOUT: hand this request to a human agent; the time limit was reached."
            state.update({"status": "FALLBACK_TIMEOUT", "final_answer": final})
            _checkpoint(checkpoint_path, state)
            return state
        response = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] = int(state["llm_calls"]) + 1
        state["tokens_used"] = int(state["tokens_used"]) + int(response.usage.total_tokens)
        assistant = response.choices[0].message
        calls = assistant.tool_calls or []
        if int(state["tokens_used"]) > token_limit:
            final = "FALLBACK_TOKEN_LIMIT: hand this request to a human agent; the token limit was exceeded."
            state.update({"status": "FALLBACK_TOKEN_LIMIT", "final_answer": final})
            _checkpoint(checkpoint_path, state)
            return state
        if not calls:
            final = assistant.content or "FALLBACK: hand this request to a human agent; the model returned no answer."
            state.update({"status": _status(final), "final_answer": final})
            _checkpoint(checkpoint_path, state)
            return state

        call = calls[0]
        call_record = {"id": str(call.id), "name": str(call.name), "arguments": call.arguments}
        state["messages"].append({"role": "assistant", "content": None, "tool_calls": [call_record]})
        name = call_record["name"]
        arguments = call_record["arguments"]
        if not isinstance(arguments, dict):
            arguments = json.loads(arguments)
        if name not in TOOL_MAP:
            observation = {"error": "tool is not allowlisted"}
        elif name in {"issue_refund", "escalate_to_human"}:
            order = next((step["observation"] for step in reversed(state["steps"]) if step["tool_name"] == "get_order"), {})
            calculated = next((step["observation"] for step in reversed(state["steps"]) if step["tool_name"] == "calculate_refund"), {})
            requested = float(arguments.get("refund_inr", -1))
            order_id = str(arguments.get("order_id", ""))
            matches_order = order.get("order_id") == order_id and order.get("status") == "completed"
            matches_amount = calculated.get("refund_inr") == requested and requested > 0
            correct_action = (name == "issue_refund" and requested <= 1000) or (name == "escalate_to_human" and requested > 1000)
            observation = TOOL_MAP[name](**arguments) if matches_order and matches_amount and correct_action else {"error": "refund tool guard rejected this action"}
        else:
            observation = TOOL_MAP[name](**arguments)
        state["steps"].append({"tool_name": name, "arguments": arguments, "observation": observation})
        state["messages"].append({
            "role": "tool", "tool_call_id": call_record["id"], "name": name,
            "content": _json_tool_value(observation),
        })
        state["step_number"] = int(state["step_number"]) + 1
        state["tools_executed"] = state["tools_executed"] + [name]
        _checkpoint(checkpoint_path, state)
        if crash_after_tools is not None and len(state["tools_executed"]) >= crash_after_tools:
            raise SimulatedCrash(f"simulated crash after tool {name}")

    final = "FALLBACK_MAX_ITERATIONS: hand this request to a human agent; the iteration limit was reached."
    state.update({"status": "FALLBACK_MAX_ITERATIONS", "final_answer": final})
    _checkpoint(checkpoint_path, state)
    return state


def run_agent(user_message: str, scenario_id: str, max_iterations: int = 6,
              token_limit: int = 6000, time_limit_seconds: float = 10,
              run_id: str | None = None, crash_after_tools: int | None = None) -> dict[str, object]:
    global _ACTIVE_SCENARIO_ID, _ACTIVE_RETRIEVER
    _ACTIVE_SCENARIO_ID = scenario_id
    if _ACTIVE_RETRIEVER is None:
        _ACTIVE_RETRIEVER = PolicyRetriever()
    run_id = run_id or scenario_id
    state = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": [
            {"role": "system", "content": "AGENT. Use the order and policy tools as evidence. Follow the refund rules. Never invent order facts or issue an unverified refund."},
            {"role": "user", "content": user_message},
        ],
        "tokens_used": 0,
        "step_number": 0,
        "tools_executed": [],
        "steps": [],
        "llm_calls": 0,
        "max_iterations": max_iterations,
        "status": "running",
        "final_answer": "",
    }
    _checkpoint(STATE_DIR / f"{run_id}.json", state)
    return _agent_loop(state, token_limit, time_limit_seconds, crash_after_tools)


def resume_agent(run_id: str) -> dict[str, object]:
    global _ACTIVE_SCENARIO_ID, _ACTIVE_RETRIEVER
    state = json.loads((STATE_DIR / f"{run_id}.json").read_text(encoding="utf-8"))
    if state.get("status") != "running":
        return state
    _ACTIVE_SCENARIO_ID = str(state["scenario_id"])
    if _ACTIVE_RETRIEVER is None:
        _ACTIVE_RETRIEVER = PolicyRetriever()
    return _agent_loop(state, token_limit=6000, time_limit_seconds=10)


def _evaluate_policy(retriever: PolicyRetriever, output_name: str) -> tuple[pd.DataFrame, dict[str, object]]:
    questions = pd.read_csv(RAW / "policy_eval.csv", keep_default_na=False)
    rows = []
    for item in questions.itertuples(index=False):
        results = retriever.retrieve(item.question, k=3)
        result = retriever.answer(item.question)
        answerable = item.expected_doc != "NONE"
        rows.append({
            "question": item.question,
            "expected_doc": item.expected_doc,
            "expected_fact": item.expected_fact,
            "retrieved_sources": "|".join(chunk["source"] for chunk in results),
            "answer": result["answer"],
            "doc_hit": item.expected_doc in [chunk["source"] for chunk in results] if answerable else "",
            "fact_hit": item.expected_fact in result["answer"] if answerable else "",
            "fallback": result["answer"].startswith("I could not find"),
        })
    frame = pd.DataFrame(rows)
    answerable = frame[frame["expected_doc"].ne("NONE")]
    unanswerable = frame[frame["expected_doc"].eq("NONE")]
    metrics = {
        "chunks": len(retriever.chunks),
        "answerable_questions": len(answerable),
        "doc_hits": int(answerable["doc_hit"].astype(bool).sum()),
        "fact_hits": int(answerable["fact_hit"].astype(bool).sum()),
        "unanswerable_questions": len(unanswerable),
        "fallbacks": int(unanswerable["fallback"].sum()),
        "failed_questions": frame.loc[
            frame["expected_doc"].ne("NONE") & ~frame["doc_hit"].astype(bool), "question"
        ].tolist(),
        "fact_failed_questions": frame.loc[
            frame["expected_doc"].ne("NONE") & ~frame["fact_hit"].astype(bool), "question"
        ].tolist(),
    }
    frame.to_csv(OUT / output_name, index=False)
    return frame, metrics


def run() -> dict[str, object]:
    global _ACTIVE_RETRIEVER
    OUT.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _ACTIVE_RETRIEVER = PolicyRetriever(overlap=10)
    chunk_counts = {
        str(source): int(count)
        for source, count in pd.Series([chunk["source"] for chunk in _ACTIVE_RETRIEVER.chunks]).value_counts().items()
    }
    _, overlap_metrics = _evaluate_policy(_ACTIVE_RETRIEVER, "policy_eval_results.csv")
    no_overlap = PolicyRetriever(overlap=0)
    _, no_overlap_metrics = _evaluate_policy(no_overlap, "policy_eval_zero_overlap.csv")

    for filename in ("agent_runs.jsonl", "refund_ledger.jsonl", "agent_demos.json"):
        path = OUT / filename
        if path.exists():
            path.unlink()
    for path in STATE_DIR.glob("*.json"):
        path.unlink()
    scenarios = pd.read_csv(RAW / "agent_scenarios.csv", keep_default_na=False)
    scenario_rows = []
    with (OUT / "agent_runs.jsonl").open("w", encoding="utf-8") as file:
        for item in scenarios.to_dict("records"):
            result = run_agent(item["message"], item["scenario_id"], run_id=f"main-{item['scenario_id']}")
            row = {
                "scenario_id": item["scenario_id"],
                "status": result["status"],
                "final_answer": result["final_answer"],
                "llm_calls": result["llm_calls"],
                "tokens_used": result["tokens_used"],
                "steps": result["steps"],
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            scenario_rows.append(row)

    first = scenarios.iloc[0]
    token_demo = run_agent(first["message"], "S1", token_limit=200, run_id="demo-token-limit")
    timeout_demo = run_agent(first["message"], "S1", time_limit_seconds=0, run_id="demo-timeout")
    crash_run_id = "demo-crash-resume-S1"
    crashed_tools = []
    try:
        run_agent(first["message"], "S1", run_id=crash_run_id, crash_after_tools=2)
    except SimulatedCrash:
        crashed_tools = json.loads((STATE_DIR / f"{crash_run_id}.json").read_text(encoding="utf-8"))["tools_executed"]
    resumed = resume_agent(crash_run_id)
    tools_across_resume = crashed_tools + resumed["tools_executed"][len(crashed_tools):]
    demos = {
        "token_limit_demo_status": token_demo["status"],
        "timeout_demo_status": timeout_demo["status"],
        "crash_resume_status": resumed["status"],
        "crash_resume_final_answer": resumed["final_answer"],
        "crash_resume_tools": tools_across_resume,
        "crash_resume_llm_calls": resumed["llm_calls"],
        "crash_resume_tokens": resumed["tokens_used"],
    }
    (OUT / "agent_demos.json").write_text(json.dumps(demos, indent=2), encoding="utf-8")
    report = {
        "chunk_counts": chunk_counts,
        "overlap_10": overlap_metrics,
        "overlap_0": no_overlap_metrics,
        "agent_scenarios": scenario_rows,
        "agent_demo": demos,
        "tools_spec": TOOLS_SPEC,
    }
    (OUT / "part_d_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
