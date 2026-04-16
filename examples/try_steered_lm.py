#%%
import os
os.environ["OTEL_SDK_DISABLED"] = "true"

from litellm import acompletion
from steerbot.utils.llm import register_steered_lm
from genlm.control import BoolFSA

#%%
fsa = BoolFSA.from_regex(r"SMC is (🔥🔥|😍😍|🤌🤌) with LMs")

#%%
register_steered_lm(
    model_name_or_path="Qwen/Qwen3-4B-Instruct-2507",
    potential_eff=fsa,
    max_tokens=2048,
    engine_opts={
        "max_model_len": 4096 * 8
    },
    ess_threshold=1,
    n_particles=5
)
#%%
response = await acompletion(
    model="steerbot/Qwen3-4B-Instruct-2507",
    messages=[
        {
            "role": "user",
            "content": "Tell me how you feel!"
        }
    ]
)
# %%
