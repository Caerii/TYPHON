from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from typhon.policies.interfaces import WriteDecision, WriteSignal


@dataclass(frozen=True)
class LayeredWritePlan:
    fast_weight: bool
    episodic: bool
    cross_episode: bool
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fast_weight": self.fast_weight,
            "episodic": self.episodic,
            "cross_episode": self.cross_episode,
            "reasons": self.reasons,
        }


class HeuristicUtilityWritePolicy:
    def __init__(self, family: str) -> None:
        self.family = family

    def decide(self, signal: WriteSignal) -> WriteDecision:
        score = (
            0.4 * signal.predicted_utility
            + 0.25 * signal.surprise
            + 0.2 * signal.gradient_norm
            + 0.15 * signal.novelty
        )
        if score >= 0.7:
            return WriteDecision(
                action="write",
                target_layer="fast_weight",
                score=score,
                reason="High combined utility and adaptation signal.",
            )
        if score >= 0.4:
            return WriteDecision(
                action="consider",
                target_layer="episodic",
                score=score,
                reason="Moderate utility suggests selective persistence.",
            )
        return WriteDecision(
            action="skip",
            target_layer=None,
            score=score,
            reason="Signal too weak for explicit write.",
        )

    def layered_plan(self, signal: WriteSignal) -> LayeredWritePlan:
        reasons: list[str] = []
        fast_weight = signal.predicted_utility >= 0.5 or signal.gradient_norm >= 0.55
        episodic = signal.surprise >= 0.55 or signal.novelty >= 0.65
        cross_episode = False

        if fast_weight:
            reasons.append("fast_weight: utility or adaptation signal is high")
        if episodic:
            reasons.append("episodic: novelty or surprise is high")
        if self.family in {"conversational_memory", "streaming_agentic_memory", "continual_learning"}:
            if signal.predicted_utility >= 0.45 or signal.metadata.get("latent_constraint", False):
                cross_episode = True
                reasons.append("cross_episode: family requires persistence across interactions")

        if not reasons:
            reasons.append("no_explicit_write: local recall is sufficient for this chunk")

        return LayeredWritePlan(
            fast_weight=fast_weight,
            episodic=episodic,
            cross_episode=cross_episode,
            reasons=reasons,
        )
