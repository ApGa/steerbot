"""Hard cap on the number of sentences in the first ``<thought>`` block (Boolean potential)."""

from __future__ import annotations

from genlm.control.potential import Potential

from steerbot.potentials.cot_utils import sentence_count_in_first_thought_from_context


class COTSentenceCapPotential(Potential):
    """Reject (``-inf`` log weight) when the first ``<thought>`` block exceeds ``max_sentences``.

    Sentences are counted with a light-weight rule: split on whitespace after ``.`` ``!`` ``?``
    (see :func:`~steerbot.potentials.cot_utils.sentence_count_in_first_thought`). Abbreviations and
    decimals can be miscounted; use :class:`~steerbot.potentials.cot_word_cap.COTWordCapPotential`
    if you need tighter control.

    Use as a **Boolean** factor in :class:`genlm.control.potential.product.Product` (e.g.
    ``fsa * word_cap * sentence_cap``).

    Args:
        max_sentences: Maximum allowed sentences in the first thought block; ``0`` allows only an
            empty thought body (or no ``<thought>``).
        vocabulary: Must match the other potential(s), e.g. ``other.vocab`` from a :class:`BoolFSA`.
    """

    def __init__(self, max_sentences: int, vocabulary: list):
        if max_sentences < 0:
            raise ValueError("max_sentences must be non-negative.")
        super().__init__(vocabulary)
        self.max_sentences = max_sentences

    @classmethod
    def aligned_with(cls, max_sentences: int, other: Potential) -> "COTSentenceCapPotential":
        return cls(max_sentences, vocabulary=other.vocab)

    def spawn(self):
        return type(self)(self.max_sentences, vocabulary=list(self.vocab))

    def _ok(self, context) -> bool:
        return sentence_count_in_first_thought_from_context(context) <= self.max_sentences

    async def prefix(self, context):
        return 0.0 if self._ok(context) else float("-inf")

    async def complete(self, context):
        return await self.prefix(context)
