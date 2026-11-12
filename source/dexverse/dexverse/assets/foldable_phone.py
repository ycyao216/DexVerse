"""Native asset bindings for the foldable phone.

The upstream asset archive retains its authored path and hinge identifiers.
Keep these stable so existing recorded articulation states remain compatible.
They are asset identifiers, not benchmark task names.
"""

from . import SYNTHESIS_DIR

FOLDABLE_PHONE_USD_PATH = str(SYNTHESIS_DIR / "huawei mate x5001" / "model_Huawei Mate01 X5.usd")
FOLDABLE_PHONE_LEFT_HINGE = "RevoluteJoint_Huawei_Mate01_X5_left"
FOLDABLE_PHONE_RIGHT_HINGE = "RevoluteJoint_Huawei_Mate01_X5_right"
