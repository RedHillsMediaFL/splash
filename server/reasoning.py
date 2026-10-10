# Modified by Red Hills Media on 2026-10-10 for this M1 Ultra source release.
"""Request-local reasoning effort profiles and answer-space reservation."""

from dataclasses import dataclass

from .errors import APIError

REASONING_TOKEN_BUDGETS = {
    "none": 0,
    "low": 64,
    "medium": 256,
    "high": 1024,
    "xhigh": 4096,
    "max": 16384,
}
REASONING_INSTRUCTIONS = {
    "low": "Check the decisive prerequisite briefly, then answer or call the next eligible tool.",
    "medium": "Check prerequisites and the next step; repair recoverable errors and stop on task completion or unresolved permission/safety blocks.",
    "high": "Check dependencies, evidence, and failure branches before acting; verify the answer concisely.",
    "xhigh": "Examine alternatives, edge cases, and conflicting evidence; verify all action prerequisites before acting.",
    "max": "Use thorough analysis when needed, compare approaches, and verify constraints and results before acting.",
}


def effective_effort(effort, enable_thinking):
    if enable_thinking is False:
        return "none"
    if enable_thinking is True and effort in (None, "none"):
        return "medium"
    return "low" if effort == "minimal" else effort


def reasoning_instruction(effort):
    if effort in (None, "none"):
        return None
    return (
        f"Reasoning effort: {effort}. {REASONING_INSTRUCTIONS[effort]} "
        f"Your reasoning ceiling is {REASONING_TOKEN_BUDGETS[effort]} tokens, "
        "or less if the total output budget is small. Finish reasoning early when sufficient; "
        "close the thinking block and provide the final answer or tool call. "
        "More effort does not authorize additional actions. "
        "Repair recoverable tool errors within existing permissions. "
        "Never treat a failed prerequisite as satisfied or claim unverified success. "
        "For requested actions, use tools rather than stopping with a plan or an unexecuted command. "
        "Stop on unresolved permission or safety blocks, and verify uncertain external effects before retrying."
    )


def effort_messages(messages, effort):
    instruction = reasoning_instruction(effort)
    if instruction is None:
        return messages
    prepared = [dict(message) for message in messages]
    if prepared and prepared[0]["role"] == "system":
        prepared[0]["content"] = prepared[0]["content"] + "\n\n" + instruction
    else:
        prepared.insert(0, {"role": "system", "content": instruction})
    return prepared


@dataclass(frozen=True)
class ReasoningBudget:
    effort: str
    requested_tokens: int
    effective_tokens: int
    answer_reserve_tokens: int


def reasoning_budget(effort, thinking, total_tokens):
    if not thinking or effort is None:
        return None
    if total_tokens < 2:
        raise APIError(
            400, "thinking requires at least two output tokens: a close and an answer"
        )
    reserve = min(512, max(1, total_tokens // 4))
    requested = REASONING_TOKEN_BUDGETS[effort]
    return ReasoningBudget(
        effort, requested, min(requested, total_tokens - reserve - 1), reserve
    )


def reasoning_capabilities(default_effort):
    return {
        "supported_efforts": list(REASONING_TOKEN_BUDGETS),
        "aliases": {"minimal": "low"},
        "default_effort": default_effort,
        "token_budgets": dict(REASONING_TOKEN_BUDGETS),
        "budget_enforced": True,
        "early_exit": True,
    }
