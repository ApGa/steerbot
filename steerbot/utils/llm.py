import asyncio
import litellm
import uuid
import warnings
from litellm.llms.custom_llm import CustomLLM
from litellm.utils import custom_llm_setup
from litellm.types.utils import Choices, ModelResponse, Usage, Message
from genlm.control.potential import Potential
from genlm.control import PromptedLLM, AWRS, direct_token_sampler
from genlm.control.constant import EndOfSequence
from typing import Any
from transformers import AutoTokenizer

cached_llms: dict[str, PromptedLLM] = {}

def get_llm(
    model_name_or_path: str,
    prompt_ids: list[int] | None = None,
    temperature: float = 1.0,
    engine_opts: dict[str, Any] | None = None,
    eos_byte_strings: list[bytes] | None = None,
) -> PromptedLLM:
    """Get a cached or new PromptedLLM instance."""
    
    if model_name_or_path not in cached_llms:
        cached_llms[model_name_or_path] = PromptedLLM.from_name(
            model_name_or_path,
            prompt_ids=prompt_ids,
            temperature=temperature,
            engine_opts=engine_opts,
            eos_byte_strings=eos_byte_strings,
        )
 
    elif engine_opts is not None:
        warnings.warn(
            "Found cached llm with existing engine opts. Ignoring new engine opts."
        )
    
    return cached_llms[model_name_or_path].spawn(
        prompt_ids=prompt_ids,
        temperature=temperature,
        eos_byte_strings=eos_byte_strings,
    )

def generate_id(prefix: str) -> str:
    """Generate a unique ID with the given prefix.

    Args:
        prefix: String prefix for the generated ID.

    Returns:
        A unique identifier string.
    """
    return prefix + str(uuid.uuid4())

class SteeredLM(CustomLLM):
    """LiteLLM Wrapper around genlm-control's PromptedLLM to support loading steered LMs 
    with an OpenAI chat completions compatible API.

    If ``potential_eff`` is ``None``, generation uses :func:`direct_token_sampler` on the
    base ``PromptedLLM`` (no grammar / rejection). Otherwise ``AWRS`` is used with the
    coerced efficient potential as the Boolean condition.
    """

    def __init__(
        self,
        model_name_or_path: str,
        potential_eff: Potential | None = None,
        potential_exp: Potential | None = None,
        extra_system_prompt: str | None = None,
        extra_prompt_keep_original_system: bool = False,
        temperature: float = 1.0,
        engine_opts: dict[str, Any] | None = None,
        n_particles: int = 10,
        ess_threshold: float = 0.5,
        max_tokens: int = 50,
        verbosity: int = 0,
    ):
        if engine_opts is None:
            engine_opts = {}
        
        self.model_name_or_path = model_name_or_path
        
        self.potential_eff = potential_eff
        self.potential_exp = potential_exp
        self.extra_system_prompt = extra_system_prompt
        self.extra_prompt_keep_original_system = extra_prompt_keep_original_system

        self.temperature = temperature
        self.engine_opts = engine_opts
        self.n_particles = n_particles
        self.ess_threshold = ess_threshold
        self.max_tokens = max_tokens
        self.verbosity = verbosity

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.llm = get_llm(
            model_name_or_path,
            temperature=temperature,
            engine_opts=engine_opts,
            eos_byte_strings=[
                bytes(self.tokenizer.eos_token, "utf-8"),
                bytes(self.tokenizer.pad_token, "utf-8"),
            ]
        )

        litellm.custom_provider_map.append({
            "provider": "steerbot",
            "custom_handler": self
        })

        custom_llm_setup()
    

    async def _prepare_model_input(self, *, messages: list[dict]) -> list[int]:
        """Messages -> token ids."""
        return await asyncio.to_thread(
            self.tokenizer.apply_chat_template,
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )

    async def acompletion(self, **kwargs):
        messages = kwargs.pop("messages")
        prompt_ids = await self._prepare_model_input(messages=messages)
        temperature = kwargs.pop('temperature', self.temperature)
        n_particles = kwargs.pop('n_particles', self.n_particles)
        ess_threshold = kwargs.pop('ess_threshold', self.ess_threshold)
        max_tokens = kwargs.pop('max_completion_tokens', self.max_tokens)
        verbosity = kwargs.pop('verbosity', self.verbosity)
        
        llm = get_llm(
            self.model_name_or_path,
            prompt_ids=prompt_ids,
            temperature=temperature,
        )

        # Prompt intersection: multiply the base prompted LM with a second prompted LM that uses
        # an extra system prompt prepended to the message list. This is useful when the upstream
        # caller (e.g. Platoon agent) constructs system messages internally.
        if self.extra_system_prompt:
            if self.potential_eff is not None:
                raise NotImplementedError(
                    "extra_system_prompt with potential_eff is not supported yet "
                    "(would require defining the AWRS condition over the product LM)."
                )

            # By default, we *exclude* upstream system prompts from the second LM, so it reflects
            # only the user's custom instruction plus the non-system conversation context.
            # (Otherwise you tend to double-count / entangle Platoon's own system policy.)
            extra_tail = (
                list(messages)
                if self.extra_prompt_keep_original_system
                else [m for m in messages if m.get("role") != "system"]
            )
            extra_messages = [{"role": "system", "content": self.extra_system_prompt}] + extra_tail
            extra_prompt_ids = await self._prepare_model_input(messages=extra_messages)
            extra_llm = get_llm(
                self.model_name_or_path,
                prompt_ids=extra_prompt_ids,
                temperature=temperature,
            )
            llm = llm * extra_llm

        # Unconstrained: sample directly from the prompted LM (no grammar / Boolean constraint).
        # Constrained: AWRS with a Boolean ``potential_eff`` (e.g. BoolFSA or product with caps).
        # https://arxiv.org/abs/2504.05410
        if self.potential_eff is None:
            token_sampler = direct_token_sampler(llm)
        else:
            token_sampler = AWRS(llm, self.potential_eff.coerce(llm, f=b"".join))

        sequences = await token_sampler.smc(
            n_particles=n_particles,
            ess_threshold=ess_threshold,
            max_tokens=max_tokens,
            verbosity=verbosity,
            # Expensive potential ("critic") used in importance weight correction as described in:
            # https://arxiv.org/abs/2504.13139
            critic=(
                self.potential_exp.spawn().coerce(llm, f=b"".join)
                if self.potential_exp is not None
                else None
            ),
        )

        choices = []

        # Note: we don't have access to log probs per sampled token here, 
        # but we still need to return something to be compatible with schema
        # so we just return log_weights instead. This is a hack to be compatible with 
        # output schema type. This is ok, because downstream code
        # ignores the returned log probs for now anyways.
        # To fix this, we would need to monkey patch genlm-control internals to expose logprobs
        for sequence, weights in zip(sequences.contexts, sequences.log_weights):
            reached_eos = bool(sequence) and isinstance(sequence[-1], EndOfSequence)
            output_tokens = sequence[:-1] if reached_eos else sequence
            decoded_sequence = b"".join(output_tokens).decode("utf-8")
            choices.append(
                Choices(
                    message=Message(
                        role="assistant",
                        content=decoded_sequence,

                    ),
                    finish_reason="stop" if reached_eos else "length",
                    logprobs=weights,
                    # Note: This might be slightly inacccurate in some edge cases,
                    # due tokenization inconsistencies of bpe but again this is ignored downstream.
                    # best effort to comply with output schema.
                    token_ids=self.tokenizer.encode(decoded_sequence)
                )
            )
        
        return ModelResponse(
            id=generate_id("steerbot-sampling-"),
            model=self.model_name_or_path,
            choices=choices,
            prompt_token_ids=prompt_ids,
            usage=Usage(
                prompt_tokens=len(prompt_ids),
                completion_tokens=len(choices[0].message.content),
                total_tokens=len(prompt_ids) + len(choices[0].message.content),
            ),
        )

    def completion(self, **kwargs):
        return asyncio.run(self.acompletion(**kwargs))


def register_steered_lm(
    **kwargs
) -> SteeredLM:
    """Register a SteeredLM instance with LiteLLM."""
    return SteeredLM(**kwargs)