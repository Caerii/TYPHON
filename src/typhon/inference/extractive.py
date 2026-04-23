from __future__ import annotations

from typhon.eval.heuristics import predict_answer
from typhon.inference.base import BackendStatus, GenerationRequest, GenerationResult


class ExtractiveHeuristicBackend:
    backend_id = "extractive_heuristic"

    def status(self) -> BackendStatus:
        return BackendStatus(
            backend_id=self.backend_id,
            available=True,
            details={"kind": "local_python"},
            message="Always available heuristic extractive backend.",
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        question = request.question or request.user_prompt
        retrieval_text = request.context or request.user_prompt
        content = predict_answer(
            question=question,
            retrieval_texts=[retrieval_text],
            expected_answer_type=request.expected_answer_type or "short_text",
        )
        return GenerationResult(
            content=content,
            raw_response={"backend": self.backend_id},
            usage={},
        )
