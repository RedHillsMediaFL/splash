# Modified by Red Hills Media on 2026-10-10 for this M1 Ultra source release.
import json
import unittest
from unittest import mock

import numpy as np

from dev.tests import server_fixtures as fixtures
from server import protocol as wire
from server.constraints import ReasoningBudgetConstraint, TokenConstraint
from server.errors import APIError, ConstraintError
from server.metrics import reasoning_details
from server.reasoning import (
    REASONING_TOKEN_BUDGETS,
    effective_effort,
    effort_messages,
    reasoning_budget,
    reasoning_capabilities,
)
from server.tool_schema import THINK_END_TOKEN_ID, TOOL_CALL_OPEN_TOKEN_ID


def decode_masks(data):
    columns = (TokenConstraint.VOCABULARY + 31) // 32
    return np.frombuffer(data, dtype=np.uint32).reshape(-1, columns)


def permits(row, token):
    word, bit = divmod(token, 32)
    return bool(int(row[word]) & (1 << bit))


class ReasoningBudgetTests(unittest.TestCase):
    def test_canonical_levels_have_distinct_budgets(self):
        self.assertEqual(
            list(REASONING_TOKEN_BUDGETS),
            ["none", "low", "medium", "high", "xhigh", "max"],
        )
        budgets = list(REASONING_TOKEN_BUDGETS.values())
        self.assertEqual(budgets, sorted(set(budgets)))
        self.assertEqual(reasoning_capabilities("none")["default_effort"], "none")

    def test_none_preserves_exact_messages_and_legacy_overrides(self):
        messages = [{"role": "user", "content": "hello"}]
        self.assertIs(effort_messages(messages, "none"), messages)
        self.assertIs(effort_messages(messages, None), messages)
        self.assertEqual(effective_effort("max", False), "none")
        self.assertEqual(effective_effort("none", True), "medium")
        self.assertEqual(effective_effort("minimal", None), "low")
        self.assertIsNone(reasoning_budget(None, True, 100))
        self.assertIsNone(reasoning_budget("none", False, 1))

    def test_directive_preserves_user_instructions_without_mutating_history(self):
        messages = [
            {"role": "system", "content": "Keep secrets private."},
            {"role": "user", "content": "hello"},
        ]
        prepared = effort_messages(messages, "high")
        self.assertEqual(len(prepared), 2)
        self.assertIn("Keep secrets private.", prepared[0]["content"])
        self.assertIn("Reasoning effort: high.", prepared[0]["content"])
        self.assertEqual(messages[0]["content"], "Keep secrets private.")

    def test_output_budget_reserves_close_and_visible_answer(self):
        for total in (2, 4, 32, 512, 8192, 20000):
            for effort in list(REASONING_TOKEN_BUDGETS)[1:]:
                budget = reasoning_budget(effort, True, total)
                self.assertLessEqual(budget.effective_tokens, budget.requested_tokens)
                self.assertLessEqual(
                    budget.effective_tokens + budget.answer_reserve_tokens + 1, total
                )
                self.assertGreaterEqual(budget.answer_reserve_tokens, 1)
        with self.assertRaises(APIError):
            reasoning_budget("low", True, 1)

    def test_simulated_anchor_and_draft_crossing_budget_do_not_commit_state(self):
        constraint = ReasoningBudgetConstraint(3)
        constraint.commit([101, 102])
        rows = decode_masks(constraint.masks([103, THINK_END_TOKEN_ID, 104]))
        self.assertTrue(permits(rows[0], 103))
        self.assertFalse(permits(rows[0], TokenConstraint.EOS_TOKENS[0]))
        self.assertFalse(permits(rows[0], TOOL_CALL_OPEN_TOKEN_ID))
        self.assertTrue(permits(rows[1], THINK_END_TOKEN_ID))
        self.assertEqual(sum(int(word).bit_count() for word in rows[1]), 1)
        self.assertTrue(permits(rows[2], 104))
        self.assertTrue(permits(rows[2], TokenConstraint.EOS_TOKENS[0]))
        self.assertEqual(constraint.reasoning_tokens, 2)
        self.assertTrue(constraint.thinking)
        self.assertTrue(permits(decode_masks(constraint.masks([]))[0], 103))
        constraint.commit([103, THINK_END_TOKEN_ID, 104])
        constraint.finish()
        self.assertEqual(constraint.reasoning_tokens, 3)
        self.assertFalse(constraint.thinking)
        self.assertTrue(constraint.budget_exhausted)

    def test_rejected_draft_suffix_cannot_reopen_answer_masks(self):
        constraint = ReasoningBudgetConstraint(1)
        rows = decode_masks(constraint.masks([101, 102, THINK_END_TOKEN_ID, 103]))
        for row in rows[1:]:
            self.assertTrue(permits(row, THINK_END_TOKEN_ID))
            self.assertFalse(permits(row, 103))
            self.assertEqual(sum(int(word).bit_count() for word in row), 1)
        self.assertEqual(constraint.reasoning_tokens, 0)

    def test_early_exit_is_allowed_without_consuming_the_ceiling(self):
        constraint = ReasoningBudgetConstraint(64)
        constraint.commit([101, THINK_END_TOKEN_ID, 102])
        constraint.finish()
        self.assertEqual(constraint.reasoning_tokens, 1)
        self.assertFalse(constraint.budget_exhausted)
        self.assertTrue(permits(decode_masks(constraint.masks([]))[0], 102))

    def test_zero_budget_forces_immediate_close(self):
        constraint = ReasoningBudgetConstraint(0)
        row = decode_masks(constraint.masks([]))[0]
        self.assertTrue(permits(row, THINK_END_TOKEN_ID))
        self.assertEqual(sum(int(word).bit_count() for word in row), 1)
        constraint.commit([THINK_END_TOKEN_ID, 101])
        constraint.finish()
        self.assertEqual(constraint.reasoning_tokens, 0)
        self.assertTrue(constraint.budget_exhausted)

    def test_budget_intersects_and_never_bypasses_output_grammar(self):
        columns = (TokenConstraint.VOCABULARY + 31) // 32
        mask = np.zeros((2, columns), dtype=np.uint32)
        for token in (THINK_END_TOKEN_ID, 101):
            word, bit = divmod(token, 32)
            mask[0, word] |= np.uint32(1 << bit)
        word, bit = divmod(102, 32)
        mask[1, word] |= np.uint32(1 << bit)
        grammar = mock.Mock()
        grammar.masks.return_value = mask.tobytes()
        constraint = ReasoningBudgetConstraint(0, grammar)
        rows = decode_masks(constraint.masks([THINK_END_TOKEN_ID]))
        self.assertTrue(permits(rows[0], THINK_END_TOKEN_ID))
        self.assertFalse(permits(rows[0], 101))
        self.assertTrue(permits(rows[1], 102))
        self.assertFalse(permits(rows[1], 103))
        constraint.commit([THINK_END_TOKEN_ID, 102])
        constraint.finish()
        grammar.commit.assert_called_once_with((THINK_END_TOKEN_ID, 102))
        grammar.finish.assert_called_once_with()
        grammar.masks.return_value = mask[1:].tobytes()
        with self.assertRaisesRegex(ConstraintError, "conflicts"):
            ReasoningBudgetConstraint(0, grammar).masks([])

    def test_invalid_actual_tokens_and_simulation_count_fail_closed(self):
        for tokens in ([101, 102], [TOOL_CALL_OPEN_TOKEN_ID], [248046]):
            constraint = ReasoningBudgetConstraint(1)
            constraint.commit(tokens)
            with self.assertRaises(ConstraintError):
                constraint.finish()
        with self.assertRaises(ConstraintError):
            ReasoningBudgetConstraint(64).masks([101] * 9)
        with self.assertRaises(ConstraintError):
            ReasoningBudgetConstraint(64).commit([TokenConstraint.VOCABULARY])
        for value in (-1, True, 1.0):
            with self.assertRaises(ValueError):
                ReasoningBudgetConstraint(value)


class ReasoningFrontendTests(unittest.TestCase):
    def harness(self):
        harness = fixtures.Harness(
            fixtures.FakeRuntime(), max_context=24000, default_reasoning_effort="none"
        )
        self.addCleanup(harness.close)
        return harness

    def test_every_effort_prepares_a_real_native_constraint(self):
        harness = self.harness()
        for effort, expected in REASONING_TOKEN_BUDGETS.items():
            job = harness.app.prepare(
                fixtures.chat_body(reasoning_effort=effort, max_tokens=20000),
                deadline=fixtures.FOREVER,
            )
            self.assertEqual(job.reasoning_effort, effort)
            frame = harness.backend._generation_request(job).frame
            if effort == "none":
                self.assertIsNone(job.constraint)
                self.assertFalse(job.thinking)
                self.assertEqual(frame.constraint, wire.ConstraintMode.NONE)
            else:
                self.assertIsInstance(job.constraint, ReasoningBudgetConstraint)
                self.assertEqual(job.reasoning_budget_tokens, expected)
                self.assertEqual(frame.constraint, wire.ConstraintMode.TOKEN_MASK)

    def test_clamp_metadata_and_responses_use_the_same_budget(self):
        harness = self.harness()
        job = harness.app.prepare_responses(
            fixtures.responses_body(reasoning={"effort": "max"}, max_output_tokens=64),
            deadline=fixtures.FOREVER,
            reserve_input=fixtures.reserve_unbounded,
        )
        details = reasoning_details(job)
        self.assertEqual(details["reasoning_effort"], "max")
        self.assertEqual(details["reasoning_requested_budget_tokens"], 16384)
        self.assertEqual(details["reasoning_budget_tokens"], 47)
        self.assertTrue(details["reasoning_budget_clamped"])
        self.assertEqual(details["answer_reserve_tokens"], 16)

    def test_tiny_output_budget_is_rejected_only_when_thinking_is_enabled(self):
        harness = self.harness()
        with self.assertRaises(APIError):
            harness.app.prepare(
                fixtures.chat_body(reasoning_effort="low", max_tokens=1),
                deadline=fixtures.FOREVER,
            )
        job = harness.app.prepare(
            fixtures.chat_body(reasoning_effort="none", max_tokens=1),
            deadline=fixtures.FOREVER,
        )
        self.assertIsNone(job.constraint)

    def test_model_capabilities_are_machine_readable(self):
        harness = self.harness()
        status, _, response = harness.request("GET", "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(
            json.loads(response)["data"][0]["reasoning"], reasoning_capabilities("none")
        )
