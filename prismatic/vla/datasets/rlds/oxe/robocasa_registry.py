"""
RoboCasa / GR1 RLDS dataset registry helpers.
"""

import re

from typing import Optional, Tuple


ROBOCASA_PRIMARY_IMAGE_KEY = "video.ego_view"
ROBOCASA_LEROBOT_IMAGE_KEY = "ego_view"
ROBOCASA_PRIMARY_IMAGE_FALLBACK_KEY = "video.ego_view_bg_crop_pad_res256_freq20"
ROBOCASA_LANGUAGE_KEY = "annotation.human.coarse_action"
ROBOCASA_LANGUAGE_FALLBACK_KEY = "language_instruction"

ROBOCASA_RAW_VECTOR_KEY = "state"
ROBOCASA_RAW_ACTION_DIM = 44

ROBOCASA_RAW_MODALITY_SLICES = {
    "left_arm": (0, 7),
    "left_hand": (7, 13),
    "left_leg": (13, 19),
    "neck": (19, 22),
    "right_arm": (22, 29),
    "right_hand": (29, 35),
    "right_leg": (35, 41),
    "waist": (41, 44),
}

ROBOCASA_STATE_KEYS: Tuple[str, ...] = (
    "state.left_arm",
    "state.right_arm",
    "state.left_hand",
    "state.right_hand",
    "state.waist",
)

ROBOCASA_ACTION_KEYS: Tuple[str, ...] = (
    "action.left_arm",
    "action.right_arm",
    "action.left_hand",
    "action.right_hand",
    "action.waist",
)

ROBOCASA_DATASET_SUFFIX = "_1000"


def robocasa_tfds_name(raw_dataset_name: str) -> str:
    dataset_name = raw_dataset_name.replace(".", "_").replace("-", "_")
    dataset_name = re.sub(r"(.)([A-Z][a-z0-9]+)", r"\1_\2", dataset_name)
    dataset_name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", dataset_name)
    return dataset_name.lower()


ROBOCASA_RAW_DATASETS: Tuple[str, ...] = (
    "gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000",
)

ROBOCASA_DATASETS: Tuple[str, ...] = tuple(robocasa_tfds_name(dataset_name) for dataset_name in ROBOCASA_RAW_DATASETS)

ROBOCASA_MIXTURE_ALIASES: Tuple[str, ...] = (
    "fourier_gr1_unified_1000",
    "robocasa_gr1_unified_1000",
)


ROBOCASA_MERGED_PANDA_DATASETS: Tuple[str, ...] = (
    "robocasa_merged_24_tasks_100demos_v1",
    "robocasa_merged_24_tasks_30demos_v3",
)

ROBOCASA_MERGED_PANDA_IMAGE_KEYS = {
    "primary": "right_image",
    "left_wrist": "left_image",
    "right_wrist": "wrist_image",
}

ROBOCASA_MERGED_PANDA_STATE_KEYS: Tuple[str, ...] = ("state",)
ROBOCASA_MERGED_PANDA_ACTION_DIM = 12
ROBOCASA_MERGED_PANDA_PROPRIO_DIM = 9

ROBOCASA_MERGED_PANDA_MIXTURE_ALIASES = {
    "robocasa_merged_panda_100demos": "robocasa_merged_24_tasks_100demos_v1",
    "robocasa_merged_panda_30demos": "robocasa_merged_24_tasks_30demos_v3",
}

ROBOCASA_COSMOS_DATASETS: Tuple[str, ...] = ("robocasa_cosmos_success_only",)

ROBOCASA_COSMOS_IMAGE_KEYS = {
    "primary": "left_image",
    "left_wrist": "right_image",
    "right_wrist": "wrist_image",
}

ROBOCASA_COSMOS_STATE_KEYS: Tuple[str, ...] = ("state",)
ROBOCASA_COSMOS_ACTION_DIM = 7
ROBOCASA_COSMOS_PROPRIO_DIM = 9

ROBOCASA_COSMOS_MIXTURE_ALIASES = {
    "robocasa_cosmos": "robocasa_cosmos_success_only",
    "robocasa_cosmos_success_only": "robocasa_cosmos_success_only",
}


def infer_robocasa_dataset_name_from_env(
    env_name: str,
    dataset_suffix: str = ROBOCASA_DATASET_SUFFIX,
) -> Optional[str]:
    if not env_name or "/" not in env_name or not env_name.endswith("_Env"):
        return None

    namespace, task_name = env_name.split("/", 1)
    raw_dataset_name = f"{namespace}.{task_name[:-4]}{dataset_suffix}"
    return robocasa_tfds_name(raw_dataset_name)
