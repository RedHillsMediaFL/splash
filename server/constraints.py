# Modified by Red Hills Media on 2026-10-10 for this M1 Ultra source release.
"""Tokenizer contract and cached token-level output constraints."""

import threading
from concurrent.futures import Future, wait

import numpy as np
from llguidance import LLExecutor, LLMatcher, LLTokenizer
from llguidance.hf import from_tokenizer as guidance_tokenizer
from llguidance.numpy import (
    allocate_token_bitmask,
    fill_next_token_bitmask_par,
    fill_next_token_bitmask_par_with_draft_tokens,
)

from . import runtime as engine_runtime
from .errors import APIError, ConstraintError
from .lru import LRUCache
from .tool_schema import (
    THINK_END,
    THINK_END_TOKEN_ID,
    TOOL_CALL_OPEN,
    TOOL_CALL_OPEN_TOKEN_ID,
)


class TokenConstraint:
    VOCABULARY = 248320
    # A mask request simulates at most the engine's target verify rows
    # (ExecutionLimits::targetVerifyRows, 8: the pending anchor and seven
    # draft proposals) and takes a mask before and after each.
    MAX_ROWS = 9
    EOS_TOKENS = (248044, 248046)

    def __init__(self, matcher, executor):
        self.matcher = matcher
        self.executor = executor
        self.bitmask = allocate_token_bitmask(self.MAX_ROWS, self.VOCABULARY)
        # Generated batches the grammar has not consumed yet. The reader
        # thread commits them; the mask thread consumes them before the next
        # mask, so reading the native stream never waits for the grammar.
        self._lock = threading.Lock()
        self._committed: list[tuple[int, ...]] = []

    def commit(self, token_ids):
        if any(not 0 <= token < self.VOCABULARY for token in token_ids):
            raise ConstraintError("generated token is out of range")
        with self._lock:
            self._committed.append(tuple(token_ids))

    def finish(self):
        """Checks the tokens generated after the last mask."""
        self._consume_committed()

    def masks(self, simulation_tokens):
        self._consume_committed()
        if len(simulation_tokens) >= self.MAX_ROWS:
            raise ConstraintError("too many simulation tokens")
        in_range = next(
            (
                index
                for index, token in enumerate(simulation_tokens)
                if not 0 <= token < self.VOCABULARY
            ),
            len(simulation_tokens),
        )
        valid_count = self.matcher.validate_tokens(list(simulation_tokens[:in_range]))
        valid_tokens = simulation_tokens[:valid_count]
        if valid_tokens:
            fill_next_token_bitmask_par_with_draft_tokens(
                self.executor,
                [(self.matcher, 0, list(valid_tokens))],
                self.bitmask,
            )
        else:
            fill_next_token_bitmask_par(
                self.executor, [(self.matcher, 0)], self.bitmask
            )
        rows = len(simulation_tokens) + 1
        valid_rows = valid_count + 1
        if valid_rows < rows:
            self.bitmask[valid_rows:rows] = self.bitmask[valid_rows - 1]
        if not self.bitmask[:valid_rows].any(axis=1).all():
            raise ConstraintError("output grammar has no valid token")
        return self.bitmask[:rows].tobytes()

    def _consume_committed(self):
        with self._lock:
            batches, self._committed = self._committed, []
        for token_ids in batches:
            # LLGuidance's bulk API rejects EOS after a NoExtension stop.
            stopped_eos = (
                len(token_ids) == 1
                and token_ids[0] in self.EOS_TOKENS
                and not self.matcher.is_error()
                and self.matcher.is_stopped()
                and self.matcher.is_accepting()
            )
            valid = (
                self.matcher.consume_token(token_ids[0])
                if stopped_eos
                else self.matcher.consume_tokens(token_ids)
            )
            if not valid:
                # The lines after the first dump the parser state, output
                # included.
                error = self.matcher.get_error()
                raise ConstraintError(
                    error.splitlines()[0] if error else "invalid token"
                )


class ReasoningBudgetConstraint:
    def __init__(self, budget, constraint=None):
        if type(budget) is not int or budget < 0:
            raise ValueError("reasoning budget must be a nonnegative integer")
        self.budget = budget
        self.constraint = constraint
        self.reasoning_tokens = 0
        self.thinking = True
        self.budget_exhausted = False
        self._lock = threading.Lock()
        self._committed = []

    def commit(self, token_ids):
        tokens = tuple(token_ids)
        if any(not 0 <= token < TokenConstraint.VOCABULARY for token in tokens):
            raise ConstraintError("generated token is out of range")
        with self._lock:
            self._committed.append(tokens)
            if self.constraint is not None:
                self.constraint.commit(tokens)

    def _consume_committed(self):
        with self._lock:
            batches, self._committed = self._committed, []
        for batch in batches:
            for token in batch:
                if not self.thinking:
                    continue
                if token == THINK_END_TOKEN_ID:
                    self.thinking = False
                    self.budget_exhausted = self.reasoning_tokens >= self.budget
                elif (
                    self.reasoning_tokens >= self.budget
                    or token in TokenConstraint.EOS_TOKENS
                    or token == TOOL_CALL_OPEN_TOKEN_ID
                ):
                    raise ConstraintError(
                        "generated token violated the reasoning budget"
                    )
                else:
                    self.reasoning_tokens += 1

    def masks(self, simulation_tokens):
        if len(simulation_tokens) >= TokenConstraint.MAX_ROWS:
            raise ConstraintError("too many simulation tokens")
        self._consume_committed()
        rows = len(simulation_tokens) + 1
        columns = (TokenConstraint.VOCABULARY + 31) // 32
        if self.constraint is None:
            masks = np.full((rows, columns), 0xFFFFFFFF, dtype=np.uint32)
        else:
            masks = (
                np.frombuffer(self.constraint.masks(simulation_tokens), dtype=np.uint32)
                .reshape(rows, columns)
                .copy()
            )
        thinking, used = self.thinking, self.reasoning_tokens
        valid = True
        for row_index in range(rows):
            row = masks[row_index]
            if not valid:
                row[:] = masks[row_index - 1]
                continue
            if thinking:
                if used >= self.budget:
                    close_word, close_bit = divmod(THINK_END_TOKEN_ID, 32)
                    permits_close = int(row[close_word]) & (1 << close_bit)
                    row[:] = 0
                    row[close_word] = permits_close
                else:
                    for token in (*TokenConstraint.EOS_TOKENS, TOOL_CALL_OPEN_TOKEN_ID):
                        word, bit = divmod(token, 32)
                        row[word] &= np.uint32(0xFFFFFFFF ^ (1 << bit))
            if not row.any():
                raise ConstraintError(
                    "reasoning budget conflicts with the output grammar"
                )
            if row_index < len(simulation_tokens):
                token = simulation_tokens[row_index]
                valid = 0 <= token < TokenConstraint.VOCABULARY
                if valid:
                    word, bit = divmod(token, 32)
                    valid = bool(int(row[word]) & (1 << bit))
                if valid and thinking:
                    if token == THINK_END_TOKEN_ID:
                        thinking = False
                    else:
                        used += 1
        return masks.tobytes()

    def finish(self):
        self._consume_committed()
        if self.constraint is not None:
            self.constraint.finish()


def validate_tokenizer(tokenizer):
    vocabulary = tokenizer.get_vocab()
    if not vocabulary or any(
        type(token_id) is not int or not 0 <= token_id < TokenConstraint.VOCABULARY
        for token_id in vocabulary.values()
    ):
        raise engine_runtime.EngineUnhealthy(
            "tokenizer vocabulary does not fit the native model"
        )
    expected = {
        "<|endoftext|>": TokenConstraint.EOS_TOKENS[0],
        "<|im_end|>": TokenConstraint.EOS_TOKENS[1],
        THINK_END: THINK_END_TOKEN_ID,
        TOOL_CALL_OPEN: TOOL_CALL_OPEN_TOKEN_ID,
    }
    for token, token_id in expected.items():
        if vocabulary.get(token) != token_id or tokenizer.encode(
            token, add_special_tokens=False
        ) != [token_id]:
            raise engine_runtime.EngineUnhealthy(
                f"tokenizer must encode {token!r} as native token {token_id}"
            )
    if tokenizer.eos_token_id not in TokenConstraint.EOS_TOKENS:
        raise engine_runtime.EngineUnhealthy(
            "tokenizer EOS token does not match the native model"
        )


def _grammar_error(error):
    # A compiler panic carries a backtrace rather than a reason, and the lines
    # after the first echo the grammar source with every schema it holds.
    if error.startswith("panic"):
        return APIError(400, "tool or output schema is too large to compile")
    return APIError(400, f"unsupported output schema: {error.splitlines()[0]}")


class ConstraintFactory:
    CACHE_SIZE = 32
    CACHE_SOURCE_BYTES = 8 * 1024 * 1024

    def __init__(self, tokenizer):
        self.tokenizer = guidance_tokenizer(
            tokenizer,
            n_vocab=TokenConstraint.VOCABULARY,
            eos_token=list(TokenConstraint.EOS_TOKENS),
            slices=LLTokenizer.json_slices(),
        )
        self.executor = LLExecutor()
        # Compiled matchers by grammar, within a budget of grammar bytes.
        self.cache = LRUCache(self.CACHE_SOURCE_BYTES, self.CACHE_SIZE)
        self.lock = threading.Lock()
        self.pending = {}
        self.hits = 0
        self.misses = 0

    def create(self, grammar, *, timeout=None, prefixes=None):
        """`prefixes`, called when the grammar is compiled, returns pairs of
        tokens the output must be able to begin with and the error for a
        grammar that cannot. A grammar can compile and still exceed the
        parser's limits where generation reaches a construct, after the whole
        prompt has been processed."""
        matcher = self._matcher(grammar, timeout, prefixes)
        return TokenConstraint(matcher.deep_copy(), self.executor)

    def _matcher(self, grammar, timeout, prefixes):
        # Compilation uses the frontend's bounded preparation slots. Share
        # identical misses without blocking unrelated immutable templates.
        with self.lock:
            cached = self.cache.get(grammar)
            if cached is not None:
                self.hits += 1
                return cached
            pending = self.pending.get(grammar)
            owner = pending is None
            if owner:
                pending = self.pending[grammar] = Future()
        if not owner:
            if not wait((pending,), timeout=timeout).done:
                raise APIError(504, "request timed out", "request_timeout")
            matcher = pending.result()
            with self.lock:
                # The shared matcher, if kept, is now the most recently used.
                self.cache.get(grammar)
                self.hits += 1
            return matcher
        try:
            error = LLMatcher.validate_grammar(grammar, self.tokenizer)
            if error:
                raise _grammar_error(error)
            matcher = LLMatcher(self.tokenizer, grammar, log_level=0)
            if matcher.is_error():
                raise _grammar_error(matcher.get_error())
            for tokens, message in prefixes() if prefixes else ():
                if not matcher.deep_copy().consume_tokens(tokens):
                    raise APIError(400, message)
            size = len(grammar.encode())
            # Oversized grammars remain usable without displacing the cache.
            # This bounds source bytes; LLGuidance bounds compiler complexity.
            with self.lock:
                self.misses += 1
                self.cache.put(grammar, matcher, size)
            pending.set_result(matcher)
            return matcher
        except BaseException as error:
            pending.set_exception(error)
            raise
        finally:
            with self.lock:
                del self.pending[grammar]

    def stats(self):
        with self.lock:
            return {
                "entries": len(self.cache),
                "capacity": self.cache.capacity,
                "source_bytes": self.cache.bytes,
                "source_budget_bytes": self.cache.budget_bytes,
                "hits": self.hits,
                "misses": self.misses,
            }
