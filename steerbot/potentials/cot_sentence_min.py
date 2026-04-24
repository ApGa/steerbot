"""Hard minimum on the number of sentences in the first ``<thought>`` block (Boolean potential)."""

from __future__ import annotations

from genlm.control.potential import Potential

from steerbot.potentials.cot_utils import (
    context_to_text,
    sentence_count_in_first_thought_from_context,
)


class COTSentenceMinPotential(Potential):
    """Reject (``-inf`` log weight) when the first ``<thought>`` block has fewer than ``min_sentences``.

    This is a **Boolean** potential (prefix in {0, -inf}) intended to be multiplied into
    an efficient steering condition, e.g. ``cfg * sentence_min``.

    Notes:
    - If the ``<thought>`` block is empty or missing, the sentence count is 0.
    - Counting is heuristic; see :func:`~steerbot.potentials.cot_utils.sentence_count_in_first_thought`.
    """

    def __init__(self, min_sentences: int, vocabulary: list):
        if min_sentences < 0:
            raise ValueError("min_sentences must be non-negative.")
        super().__init__(vocabulary)
        self.min_sentences = min_sentences

    @classmethod
    def aligned_with(cls, min_sentences: int, other: Potential) -> "COTSentenceMinPotential":
        return cls(min_sentences, vocabulary=other.vocab)

    def spawn(self):
        return type(self)(self.min_sentences, vocabulary=list(self.vocab))

    def _thought_closed(self, context) -> bool:
        # We only enforce the minimum once the first thought block is closed;
        # before that, the prefix can still grow to meet the minimum.
        return "</thought>" in context_to_text(context)

    def _ok_prefix(self, context) -> bool:
        if self.min_sentences == 0:
            return True
        # While the thought is still open, allow the prefix (it may still reach min_sentences).
        if not self._thought_closed(context):
            return True
        return sentence_count_in_first_thought_from_context(context) >= self.min_sentences

    def _ok_complete(self, context) -> bool:
        if self.min_sentences == 0:
            return True
        return sentence_count_in_first_thought_from_context(context) >= self.min_sentences

    async def prefix(self, context):
        return 0.0 if self._ok_prefix(context) else float("-inf")

    async def complete(self, context):
        return 0.0 if self._ok_complete(context) else float("-inf")

