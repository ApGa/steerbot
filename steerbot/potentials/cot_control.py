import math
from genlm.control.potential import Potential

from steerbot.potentials.cot_utils import word_count_in_first_thought_from_context


class COTControlPotential(Potential):
    """Penalize CoT length (log discount per word inside `<thought>`).

    You must pass a ``vocabulary`` that matches the other potential(s) in a product
    (e.g. ``vocabulary=fsa.vocab`` for a byte-level ``BoolFSA``), because
    :class:`genlm.control.potential.base.Potential` requires ``token_type`` and
    ``vocab`` from ``super().__init__``.
    """

    def __init__(self, discount_factor: float, vocabulary: list):
        if not (0.0 < discount_factor <= 1.0):
            raise ValueError("discount_factor must be in (0, 1].")
        super().__init__(vocabulary)
        self.discount_factor = discount_factor
        self.log_discount_factor = math.log(discount_factor)

    @classmethod
    def aligned_with(cls, discount_factor: float, other: Potential) -> "COTControlPotential":
        """Build a CoT control potential on the same token vocabulary as ``other``."""
        return cls(discount_factor, vocabulary=other.vocab)

    def spawn(self):
        """Fresh copy for SMC / multiprocessing (matches other potentials' ``spawn`` API)."""
        return type(self)(self.discount_factor, vocabulary=list(self.vocab))

    def count_cot_words(self, context):
        """Count words inside the first `<thought>...</thought>` block.

        If the closing tag is missing (e.g. streaming output), count up to end-of-string.
        """
        return word_count_in_first_thought_from_context(context)

    
    async def prefix(self, context):
        return self.log_discount_factor * self.count_cot_words(context)
    
    async def complete(self, context):
        return await self.prefix(context)
