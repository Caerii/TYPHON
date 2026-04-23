from __future__ import annotations

from typhon.benchmarks.base import BenchmarkSample, BenchmarkSpec


def _answer_style_instruction(expected_answer_type: str) -> str:
    if expected_answer_type == "classification":
        return "Return only the label or option text."
    if expected_answer_type in {"long_text", "summary"}:
        return "Return only the answer text grounded in the context, with no extra explanation."
    if expected_answer_type == "code":
        return "Return only the code answer with no explanation."
    return "Return only the shortest factual answer phrase with no explanation."


def build_benchmark_prompt(spec: BenchmarkSpec, sample: BenchmarkSample) -> tuple[str, str]:
    answer_style = _answer_style_instruction(sample.expected_answer_type)
    system_prompt = (
        "You are evaluating long-context memory. "
        "Answer the user's question using only the provided context. "
        "Be concise and factual. "
        "If the answer is not supported by the context, say that explicitly. "
        f"{answer_style}"
    )
    user_prompt = (
        f"Benchmark: {spec.name}\n"
        f"Task type: {sample.task_type}\n"
        f"Question:\n{sample.question}\n\n"
        f"Context:\n{sample.context}\n\n"
        f"Output style: {answer_style}"
    )
    return system_prompt, user_prompt


def build_selected_context_prompt(
    spec: BenchmarkSpec,
    sample: BenchmarkSample,
    *,
    strategy_id: str,
    context_segments: list[str],
) -> tuple[str, str]:
    answer_style = _answer_style_instruction(sample.expected_answer_type)
    system_prompt = (
        "You are evaluating long-context memory. "
        "Answer the user's question using only the provided context excerpts. "
        "The excerpts may be a compressed or selected view of a larger context. "
        "Be concise and factual. "
        "If the answer is not supported by the excerpts, say that explicitly. "
        f"{answer_style}"
    )
    if not context_segments:
        context_block = "[no context excerpts selected]"
    else:
        lines: list[str] = []
        for index, segment in enumerate(context_segments, start=1):
            lines.append(f"[Excerpt {index}]\n{segment}")
        context_block = "\n\n".join(lines)

    user_prompt = (
        f"Benchmark: {spec.name}\n"
        f"Task type: {sample.task_type}\n"
        f"Context strategy: {strategy_id}\n"
        f"Question:\n{sample.question}\n\n"
        f"Context excerpts:\n{context_block}\n\n"
        f"Output style: {answer_style}"
    )
    return system_prompt, user_prompt
