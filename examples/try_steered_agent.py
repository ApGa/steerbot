#%%
import os
os.environ["OTEL_SDK_DISABLED"] = "true"

from litellm import acompletion
from steerbot.utils.llm import register_steered_lm
from genlm.control import BoolFSA
from platoon.config_defs import RolloutConfig
from platoon.textcraft.synth_rollout import run_synth_depth_aware_rollout
from platoon.textcraft.synth_tasks import get_synth_task, get_synth_task_ids_by_difficulty

#%%
fsa = BoolFSA.from_regex(
   r"<thought>[\s\S]*?</thought>\n<python>[\s\S]*?</python>"
)

#%%
register_steered_lm(
    model_name_or_path="Qwen/Qwen3-4B-Instruct-2507",
    potential_eff=fsa,
    max_tokens=512,
    engine_opts={
        "max_model_len": 4096 * 8
    },
)
#%%
config = RolloutConfig(
    model_name="steerbot/Qwen3-4B-Instruct-2507",
    model_endpoint="steerbot",
    model_api_key="NONE",
)
#%%
from platoon.textcraft.synth_tasks import Difficulty
task_ids = get_synth_task_ids_by_difficulty(split="val", difficulty=Difficulty.EASY)

# %%
result = await run_synth_depth_aware_rollout(
    task=get_synth_task(task_ids[0]),
    config=config
)
# %%
