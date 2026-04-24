import asyncio
import os
os.environ["OTEL_SDK_DISABLED"] = "true"
from genlm.control import BoolFSA
from steerbot.potentials.cot_control import COTControlPotential
fsa = BoolFSA.from_regex(r"x")  # minimal
cot = COTControlPotential.aligned_with(0.8, fsa)
text = b"<thought>one two three"
ctx = list(text)  # ints, like coerced byte stream
async def main():
    s = await cot.prefix(ctx)
    print(s, "expected", __import__("math").log(0.8) * 3)
asyncio.run(main())