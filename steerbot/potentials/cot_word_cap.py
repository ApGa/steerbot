"""Hard cap on the number of words in the first ``<thought>`` block (Boolean potential)."""

from __future__ import annotations

from genlm.control.potential import Potential

from steerbot.potentials.cot_utils import word_count_in_first_thought_from_context


class COTWordCapPotential(Potential):
    """Reject (``-inf`` log weight) when the first ``<thought>`` block exceeds ``max_words`` words.

    Matches the word-count semantics of :class:`steerbot.potentials.cot_control.COTControlPotential`
    (whitespace-split words inside ``<thought>...(`` ``</thought>`` or end of string)).

    Use as the **Boolean** factor in :class:`genlm.control.potential.product.Product` or as the
    ``condition`` in :class:`genlm.control.sampler.token.AWRS` (not as a soft critic).

    Args:
        max_words: Maximum allowed words inside the first thought block; use ``0`` to forbid any
            non-whitespace content in ``<thought>`` (only empty thought or no ``<thought>``).
        vocabulary: Must match the other potential(s), e.g. ``other.vocab`` from a :class:`BoolFSA`.
    """

    def __init__(self, max_words: int, vocabulary: list):
        if max_words < 0:
            raise ValueError("max_words must be non-negative.")
        super().__init__(vocabulary)
        self.max_words = max_words

    @classmethod
    def aligned_with(cls, max_words: int, other: Potential) -> "COTWordCapPotential":
        """Same token vocabulary as ``other`` (e.g. ``fsa.vocab``)."""
        return cls(max_words, vocabulary=other.vocab)

    def spawn(self):
        return type(self)(self.max_words, vocabulary=list(self.vocab))

    def _ok(self, context) -> bool:
        return word_count_in_first_thought_from_context(context) <= self.max_words

    async def prefix(self, context):
        return 0.0 if self._ok(context) else float("-inf")

    async def complete(self, context):
        return await self.prefix(context)
