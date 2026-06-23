from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


OBJECT_PROMPT_ROLES = {"whole", "whole_object", "whole_board", "target_object"}


@dataclass(frozen=True)
class Sam3Prompt:
    role: str
    text: str
    threshold: float = 0.4
    use_for_object_mask: bool = False
    use_for_refinement: bool = True
    source: str = "profile"


@dataclass(frozen=True)
class Sam3CategoryProfile:
    category: str
    dataset: str
    prompts: tuple[Sam3Prompt, ...]
    simple_copy: bool = False


@dataclass(frozen=True)
class ScoredSam3Prompt:
    prompt: Sam3Prompt
    relevance: float
    selection_reason: str = "selected"
    profile_order: int = 0


@dataclass(frozen=True)
class Sam3PromptSelectionPlan:
    selected: tuple[ScoredSam3Prompt, ...]
    audit: tuple[dict, ...]


def _prompt(
    role: str,
    text: str,
    *,
    source: str,
    object_mask: bool | None = None,
    use_for_refinement: bool | None = None,
) -> Sam3Prompt:
    if object_mask is None:
        object_mask = role in OBJECT_PROMPT_ROLES
    if use_for_refinement is None:
        use_for_refinement = source != PROFILE_SOURCE
    return Sam3Prompt(
        role=role,
        text=text,
        use_for_object_mask=object_mask,
        use_for_refinement=use_for_refinement,
        source=source,
    )


PROFILE_SOURCE = "sam3_profiles.py"
SAM3_PROMPT_SELECTION_POLICY = "role_family_task_prior_no_query_or_option_tokens"
PROMPT_FAMILY_PRIORS = {
    "localization": {
        "defect": 3.0,
        "part": 2.0,
        "object": 1.0,
    },
    "structural": {
        "defect": 3.0,
        "part": 2.0,
        "object": 2.0,
    },
    "semantic": {
        "defect": 3.0,
        "object": 2.0,
        "part": 1.0,
    },
    "generic": {
        "object": 1.0,
        "part": 0.75,
        "defect": 0.5,
    },
}

NONLOCAL_DEFECT_PROMPT_DISABLED = {
    "mvtec": {"toothbrush", "zipper"},
}


MVTEC_PROFILES = {
    "bottle": Sam3CategoryProfile(
        category="bottle",
        dataset="mvtec",
        prompts=(
            _prompt("whole", "the entire dark glass bottle object", source=PROFILE_SOURCE),
            _prompt("inner", "the circular hole in the very center", source=PROFILE_SOURCE),
            _prompt("rim", "the circular glass rim of the bottle opening", source=PROFILE_SOURCE, object_mask=False),
            _prompt("surface", "the dark glass bottle surface", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_contamination", "the contamination or stain on the bottle surface", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_broken", "the broken or chipped glass region on the bottle rim", source=PROFILE_SOURCE, object_mask=False),
        ),
    ),
    "capsule": Sam3CategoryProfile(
        category="capsule",
        dataset="mvtec",
        prompts=(
            _prompt("whole", "the entire capsule pill object", source=PROFILE_SOURCE),
            _prompt("part_left", "the dark black half of the capsule", source=PROFILE_SOURCE),
            _prompt("part_right", "the red half of the capsule pill", source=PROFILE_SOURCE, object_mask=False),
            _prompt("imprint", "the printed number on the capsule", source=PROFILE_SOURCE, object_mask=False),
            _prompt("seam", "the seam between the two halves of the capsule", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_crack", "the crack on the capsule shell", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_scratch", "the scratch on the capsule surface", source=PROFILE_SOURCE, object_mask=False),
        ),
    ),
    "cable": Sam3CategoryProfile(
        category="cable",
        dataset="mvtec",
        prompts=(
            _prompt("whole", "the entire cable cross section with colored wires", source=PROFILE_SOURCE),
            _prompt("outer_ring", "the large white circular object", source=PROFILE_SOURCE, object_mask=False),
            _prompt("copper_cores", "the inner copper strands inside the wire", source=PROFILE_SOURCE, object_mask=False),
            _prompt("blue_insulation", "the blue wire insulation region", source=PROFILE_SOURCE, object_mask=False),
            _prompt("brown_insulation", "the brown wire insulation region", source=PROFILE_SOURCE, object_mask=False),
            _prompt("yellow_insulation", "the yellow wire insulation region", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_misplaced", "the misplaced colored wire region", source=PROFILE_SOURCE, object_mask=False),
        ),
    ),
    "screw": Sam3CategoryProfile(
        category="screw",
        dataset="mvtec",
        prompts=(
            _prompt("whole", "the entire metal screw object", source=PROFILE_SOURCE),
            _prompt("thread", "the ridged and jagged section of the screw shaft", source=PROFILE_SOURCE),
            _prompt("head", "the flat head of the metal screw", source=PROFILE_SOURCE, object_mask=False),
            _prompt("shaft", "the straight shaft of the screw", source=PROFILE_SOURCE, object_mask=False),
            _prompt("tip", "the pointed tip of the screw", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_scratch", "the scratch on the screw surface", source=PROFILE_SOURCE, object_mask=False),
        ),
    ),
    "transistor": Sam3CategoryProfile(
        category="transistor",
        dataset="mvtec",
        prompts=(
            _prompt("whole", "the entire transistor with black body and silver metal legs", source=PROFILE_SOURCE),
            _prompt("body", "the black rectangular body", source=PROFILE_SOURCE, object_mask=False),
            _prompt("legs", "the silver metal legs extending from the black body", source=PROFILE_SOURCE, object_mask=False),
            _prompt("left_leg", "the left silver metal leg of the transistor", source=PROFILE_SOURCE, object_mask=False),
            _prompt("middle_leg", "the middle silver metal leg of the transistor", source=PROFILE_SOURCE, object_mask=False),
            _prompt("right_leg", "the right silver metal leg of the transistor", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_bent_lead", "the bent or misaligned metal lead", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_damaged_case", "the damaged black transistor case", source=PROFILE_SOURCE, object_mask=False),
        ),
    ),
    "zipper": Sam3CategoryProfile(
        category="zipper",
        dataset="mvtec",
        prompts=(
            _prompt("whole", "the entire black zipper tape object", source=PROFILE_SOURCE),
            _prompt("teeth", "the central interlocking zipper teeth", source=PROFILE_SOURCE),
            _prompt("fabric", "the black fabric tape on both sides of the zipper teeth", source=PROFILE_SOURCE, object_mask=False),
            _prompt("left_tape", "the left fabric tape beside the zipper teeth", source=PROFILE_SOURCE, object_mask=False),
            _prompt("right_tape", "the right fabric tape beside the zipper teeth", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_split_teeth", "the split or missing zipper teeth", source=PROFILE_SOURCE, object_mask=False),
        ),
    ),
    "toothbrush": Sam3CategoryProfile(
        category="toothbrush",
        dataset="mvtec",
        prompts=(
            _prompt("whole", "Toothbrush", source=PROFILE_SOURCE),
            _prompt("head", "the toothbrush head", source=PROFILE_SOURCE, object_mask=False),
            _prompt("bristles", "the blue and white toothbrush bristles", source=PROFILE_SOURCE, object_mask=False),
            _prompt("handle_neck", "the white toothbrush neck below the bristles", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_missing_bristles", "the missing or damaged toothbrush bristles", source=PROFILE_SOURCE, object_mask=False),
        ),
    ),
    "pill": Sam3CategoryProfile(
        category="pill",
        dataset="mvtec",
        prompts=(
            _prompt("whole", "the pill", source=PROFILE_SOURCE),
            _prompt("surface", "the pill surface", source=PROFILE_SOURCE, object_mask=False),
            _prompt("imprint", "the printed imprint on the pill", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_crack", "the crack on the pill", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_contamination", "the contamination spot on the pill", source=PROFILE_SOURCE, object_mask=False),
        ),
    ),
    "metal_nut": Sam3CategoryProfile(
        category="metal_nut",
        dataset="mvtec",
        prompts=(
            _prompt("whole", "the green part of the Metal_nut only", source=PROFILE_SOURCE),
            _prompt("hole", "the circular hole in the center of the metal nut", source=PROFILE_SOURCE, object_mask=False),
            _prompt("outer_lobes", "the outer green lobes of the metal nut", source=PROFILE_SOURCE, object_mask=False),
            _prompt("threaded_ring", "the inner threaded ring of the metal nut", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_scratch", "the scratch on the green metal nut surface", source=PROFILE_SOURCE, object_mask=False),
        ),
    ),
    "hazelnut": Sam3CategoryProfile(
        category="hazelnut",
        dataset="mvtec",
        prompts=(
            _prompt("whole", "the entire hazelnut object", source=PROFILE_SOURCE),
            _prompt("shell_surface", "the brown hazelnut shell surface", source=PROFILE_SOURCE, object_mask=False),
            _prompt("top_cap", "the natural rough beige cap area on the hazelnut end", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_crack", "the crack on the hazelnut shell", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_cut", "the cut region on the hazelnut shell", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_hole", "the hole in the hazelnut shell", source=PROFILE_SOURCE, object_mask=False),
        ),
        simple_copy=True,
    ),
    "carpet": Sam3CategoryProfile(
        category="carpet",
        dataset="mvtec",
        prompts=(
            _prompt("whole", "the entire woven carpet textile surface", source=PROFILE_SOURCE),
            _prompt("woven_fibers", "the woven carpet fiber pattern", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_hole", "the hole or torn region in the carpet", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_thread", "the loose thread defect in the carpet", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_color", "the discolored stain on the carpet", source=PROFILE_SOURCE, object_mask=False),
        ),
        simple_copy=True,
    ),
    "grid": Sam3CategoryProfile(
        category="grid",
        dataset="mvtec",
        prompts=(
            _prompt("whole", "the entire gray metal grid surface", source=PROFILE_SOURCE),
            _prompt("grid_lines", "the diagonal metal grid lines", source=PROFILE_SOURCE, object_mask=False),
            _prompt("grid_holes", "the regular diamond holes in the grid", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_broken", "the broken grid line", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_bent", "the bent or deformed grid line", source=PROFILE_SOURCE, object_mask=False),
        ),
        simple_copy=True,
    ),
    "leather": Sam3CategoryProfile(
        category="leather",
        dataset="mvtec",
        prompts=(
            _prompt("whole", "the entire brown leather surface", source=PROFILE_SOURCE),
            _prompt("grain", "the fine grain texture of the leather", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_cut", "the cut in the leather surface", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_poke", "the small puncture hole in the leather", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_glue", "the glue mark on the leather", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_fold", "the fold or wrinkle in the leather", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_color", "the color stain on the leather", source=PROFILE_SOURCE, object_mask=False),
        ),
        simple_copy=True,
    ),
    "tile": Sam3CategoryProfile(
        category="tile",
        dataset="mvtec",
        prompts=(
            _prompt("whole", "the entire speckled ceramic tile surface", source=PROFILE_SOURCE),
            _prompt("speckled_texture", "the speckled gray tile texture", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_crack", "the crack in the tile", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_oil", "the oil stain on the tile", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_gray_stroke", "the gray stroke mark on the tile", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_glue_strip", "the glue strip on the tile", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_rough", "the rough damaged patch on the tile", source=PROFILE_SOURCE, object_mask=False),
        ),
        simple_copy=True,
    ),
    "wood": Sam3CategoryProfile(
        category="wood",
        dataset="mvtec",
        prompts=(
            _prompt("whole", "the entire wooden surface with vertical grain", source=PROFILE_SOURCE),
            _prompt("wood_grain", "the vertical wood grain lines", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_scratch", "the scratch in the wood", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_hole", "the hole in the wood", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_color", "the discolored stain on the wood", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_liquid", "the liquid stain on the wood", source=PROFILE_SOURCE, object_mask=False),
        ),
        simple_copy=True,
    ),
}


VISA_PROFILES = {
    "pcb1": Sam3CategoryProfile(
        category="pcb1",
        dataset="visa",
        prompts=(
            _prompt("whole_object", "The entire blue circuit board, including silver cylinders and metal pins", source=PROFILE_SOURCE),
            _prompt("cylinders", "cylinders", source=PROFILE_SOURCE, object_mask=False),
            _prompt("metal_pins", "the row of silver metal pins", source=PROFILE_SOURCE, object_mask=False),
            _prompt("sensor_discs", "the two round silver ultrasonic sensor discs", source=PROFILE_SOURCE, object_mask=False),
            _prompt("blue_board", "the blue printed circuit board", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_missing_component", "the missing electronic component on the circuit board", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_bent_pin", "the bent metal pin on the circuit board", source=PROFILE_SOURCE, object_mask=False),
        ),
    ),
    "pcb2": Sam3CategoryProfile(
        category="pcb2",
        dataset="visa",
        prompts=(
            _prompt("whole_board", "the entire circuit board, including all electronic components", source=PROFILE_SOURCE),
            _prompt("metal_pins", "the row of silver header pins on the circuit board", source=PROFILE_SOURCE, object_mask=False),
            _prompt("central_chip", "the black integrated circuit chip on the board", source=PROFILE_SOURCE, object_mask=False),
            _prompt("white_connector", "the white connector on the circuit board", source=PROFILE_SOURCE, object_mask=False),
            _prompt("solder_pads", "the small silver solder pads on the circuit board", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_missing_component", "the missing component on the circuit board", source=PROFILE_SOURCE, object_mask=False),
        ),
    ),
    "pcb3": Sam3CategoryProfile(
        category="pcb3",
        dataset="visa",
        prompts=(
            _prompt("whole_object", "the circuit with metal and bulb", source=PROFILE_SOURCE),
            _prompt("blue_board", "the blue circuit board", source=PROFILE_SOURCE, object_mask=False),
            _prompt("metal_pins", "the three silver metal pins", source=PROFILE_SOURCE, object_mask=False),
            _prompt("clear_led", "the clear transparent bulb on the circuit board", source=PROFILE_SOURCE, object_mask=False),
            _prompt("black_sensor", "the black sensor cap on the circuit board", source=PROFILE_SOURCE, object_mask=False),
            _prompt("blue_potentiometer", "the blue square potentiometer on the circuit board", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_missing_pin", "the missing metal pin", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_bent_bulb", "the bent or misaligned bulb", source=PROFILE_SOURCE, object_mask=False),
        ),
    ),
    "pcb4": Sam3CategoryProfile(
        category="pcb4",
        dataset="visa",
        prompts=(
            _prompt("whole_board", "the entire circuit board, including all electronic components", source=PROFILE_SOURCE),
            _prompt("usb_port", "the silver USB connector on the circuit board", source=PROFILE_SOURCE, object_mask=False),
            _prompt("terminal_blocks", "the blue terminal blocks on the circuit board", source=PROFILE_SOURCE, object_mask=False),
            _prompt("resistors", "the small rectangular resistors on the circuit board", source=PROFILE_SOURCE, object_mask=False),
            _prompt("black_chip", "the black square chip on the circuit board", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_missing_component", "the missing electronic component on the circuit board", source=PROFILE_SOURCE, object_mask=False),
        ),
    ),
    "candle": Sam3CategoryProfile(
        category="candle",
        dataset="visa",
        prompts=(
            _prompt("whole_object", "all the round tea light candles", source=PROFILE_SOURCE),
            _prompt("wick", "the small white wick threads in the center of the tea light candles", source=PROFILE_SOURCE, object_mask=False),
            _prompt("wax_surface", "the flat yellow wax surface of the tea light candle", source=PROFILE_SOURCE, object_mask=False),
            _prompt("candle_rim", "the circular rim of the tea light candle cup", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_missing_wick", "the missing wick in the center of the candle", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_short_wick", "the short wick in the tea light candle", source=PROFILE_SOURCE, object_mask=False),
        ),
    ),
    "capsules": Sam3CategoryProfile(
        category="capsules",
        dataset="visa",
        prompts=(
            _prompt("target_object", "the capsules.", source=PROFILE_SOURCE),
            _prompt("single_capsule", "one green transparent capsule", source=PROFILE_SOURCE, object_mask=False),
            _prompt("capsule_shell", "the green transparent capsule shell", source=PROFILE_SOURCE, object_mask=False),
            _prompt("capsule_end", "the rounded end of the capsule", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_missing_capsule", "the missing capsule in the group", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_broken_capsule", "the broken green capsule", source=PROFILE_SOURCE, object_mask=False),
        ),
    ),
    "cashew": Sam3CategoryProfile(
        category="cashew",
        dataset="visa",
        prompts=(
            _prompt("target_object", "the kidney-shaped nut", source=PROFILE_SOURCE),
            _prompt("nut_surface", "the smooth surface of the cashew nut", source=PROFILE_SOURCE, object_mask=False),
            _prompt("nut_edge", "the curved outer edge of the cashew nut", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_crack", "the crack on the cashew nut", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_broken", "the broken part of the cashew nut", source=PROFILE_SOURCE, object_mask=False),
        ),
    ),
    "chewinggum": Sam3CategoryProfile(
        category="chewinggum",
        dataset="visa",
        prompts=(
            _prompt("target_object", "the white object", source=PROFILE_SOURCE),
            _prompt("gum_surface", "the flat white chewing gum surface", source=PROFILE_SOURCE, object_mask=False),
            _prompt("gum_edge", "the rectangular edge of the chewing gum", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_crack", "the crack in the chewing gum", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_bite", "the missing bite-shaped region in the chewing gum", source=PROFILE_SOURCE, object_mask=False),
        ),
    ),
    "fryum": Sam3CategoryProfile(
        category="fryum",
        dataset="visa",
        prompts=(
            _prompt("target_object", "light orange wheel", source=PROFILE_SOURCE),
            _prompt("outer_rim", "the outer circular rim of the fryum wheel", source=PROFILE_SOURCE, object_mask=False),
            _prompt("spokes", "the spokes inside the fryum wheel", source=PROFILE_SOURCE, object_mask=False),
            _prompt("central_hole", "the central hole in the fryum wheel", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_broken", "the broken part of the fryum wheel", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_missing_spoke", "the missing spoke of the fryum wheel", source=PROFILE_SOURCE, object_mask=False),
        ),
    ),
    "macaroni1": Sam3CategoryProfile(
        category="macaroni1",
        dataset="visa",
        prompts=(
            _prompt("target_object", "orange elbow macaroni", source=PROFILE_SOURCE),
            _prompt("single_macaroni", "one orange elbow macaroni piece", source=PROFILE_SOURCE, object_mask=False),
            _prompt("inner_hole", "the hollow opening inside the elbow macaroni", source=PROFILE_SOURCE, object_mask=False),
            _prompt("curved_edge", "the curved edge of the elbow macaroni", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_broken", "the broken orange macaroni piece", source=PROFILE_SOURCE, object_mask=False),
        ),
    ),
    "macaroni2": Sam3CategoryProfile(
        category="macaroni2",
        dataset="visa",
        prompts=(
            _prompt("target_object", "yellow elbow macaroni", source=PROFILE_SOURCE),
            _prompt("single_macaroni", "one yellow elbow macaroni piece", source=PROFILE_SOURCE, object_mask=False),
            _prompt("inner_hole", "the hollow opening inside the yellow elbow macaroni", source=PROFILE_SOURCE, object_mask=False),
            _prompt("curved_edge", "the curved edge of the yellow elbow macaroni", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_broken", "the broken yellow macaroni piece", source=PROFILE_SOURCE, object_mask=False),
        ),
    ),
    "pipe_fryum": Sam3CategoryProfile(
        category="pipe_fryum",
        dataset="visa",
        prompts=(
            _prompt("target_object", "the pipe fryum", source=PROFILE_SOURCE),
            _prompt("cylindrical_body", "the cylindrical body of the pipe fryum", source=PROFILE_SOURCE, object_mask=False),
            _prompt("hollow_opening", "the hollow circular opening of the pipe fryum", source=PROFILE_SOURCE, object_mask=False),
            _prompt("outer_wall", "the outer wall of the pipe fryum", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_crack", "the crack in the pipe fryum", source=PROFILE_SOURCE, object_mask=False),
            _prompt("defect_broken", "the broken edge of the pipe fryum", source=PROFILE_SOURCE, object_mask=False),
        ),
    ),
}


NORMAL_PART_VETO_CONFIGS = {
    "mvtec": {
        "hazelnut": {
            "normal_role": "top_cap",
            "competing_defect_roles": ("defect_crack", "defect_cut", "defect_hole"),
            "normal_min_score": 0.80,
            "normal_min_area_ratio": 0.03,
            "normal_max_area_ratio": 0.18,
            "score_advantage": 0.12,
            "policy": "sam3_normal_part_veto",
        },
    },
}


PROFILE_REGISTRY = {
    "mvtec": MVTEC_PROFILES,
    "visa": VISA_PROFILES,
}


AUDITED_AUTO_ROLES = {
    "mvtec": {
        "bottle": {"whole"},
        "cable": {"copper_cores"},
        "capsule": {"whole", "part_left", "part_right", "imprint", "defect_scratch"},
        "carpet": set(),
        "grid": {"grid_lines", "grid_holes", "defect_broken", "defect_bent"},
        "hazelnut": {"whole", "shell_surface", "defect_crack", "defect_cut", "defect_hole"},
        "leather": set(),
        "metal_nut": {"whole", "hole", "outer_lobes", "threaded_ring", "defect_scratch"},
        "pill": {"whole", "surface", "defect_contamination"},
        "screw": {"whole", "head", "shaft", "thread", "tip", "defect_scratch"},
        "tile": {"defect_crack", "defect_gray_stroke"},
        "toothbrush": {"whole", "defect_missing_bristles"},
        "transistor": {"whole", "legs", "defect_bent_lead"},
        "wood": set(),
        "zipper": {"whole", "teeth", "fabric", "left_tape", "defect_split_teeth"},
    },
    "visa": {
        "candle": {"whole_object", "wax_surface"},
        "capsules": {"target_object", "single_capsule", "capsule_shell"},
        "cashew": {"target_object", "nut_surface"},
        "chewinggum": {"target_object", "gum_surface", "defect_crack"},
        "fryum": {"target_object"},
        "macaroni1": {"target_object", "single_macaroni", "inner_hole", "defect_broken"},
        # Keep defect prompts available for manual inspection, but only audited
        # stable part/object prompts enter automatic refinement.
        "macaroni2": {"target_object", "single_macaroni", "inner_hole", "curved_edge"},
        "pcb1": set(),
        "pcb2": {"whole_board"},
        "pcb3": {"whole_object"},
        "pcb4": {"whole_board"},
        "pipe_fryum": set(),
    },
}


SAM3_LOCAL_REFINEMENT_DISABLED_TASKS = {
    "mvtec": {
        "transistor": {
            "defect analysis",
            "defect classification",
            "defect description",
        },
    },
}


def normalize_dataset_name(dataset: str | None) -> str | None:
    if not dataset:
        return None
    normalized = str(dataset).strip().lower().replace("_", "-")
    if normalized in {"mvtec", "ds-mvtec", "ds-mvtec-ad", "ds-mvtec-adaption"}:
        return "mvtec"
    if normalized in {"visa", "vis-a"}:
        return "visa"
    return normalized


def normalize_category_name(category: str | None) -> str:
    return str(category or "").strip().lower()


def get_sam3_profile(category: str, dataset: str | None = None) -> Sam3CategoryProfile:
    cat = normalize_category_name(category)
    ds = normalize_dataset_name(dataset)

    if ds and ds in PROFILE_REGISTRY and cat in PROFILE_REGISTRY[ds]:
        return PROFILE_REGISTRY[ds][cat]

    matches = [registry[cat] for registry in PROFILE_REGISTRY.values() if cat in registry]
    if matches:
        return matches[0]

    return Sam3CategoryProfile(category=cat, dataset=ds or "unknown", prompts=())


def get_prompt_dict(category: str, dataset: str | None = None) -> dict[str, str]:
    profile = get_sam3_profile(category, dataset)
    return {prompt.role: prompt.text for prompt in profile.prompts}


def get_object_prompt(category: str, dataset: str | None = None) -> str:
    cat = normalize_category_name(category)
    profile = get_sam3_profile(cat, dataset)
    for prompt in profile.prompts:
        if prompt.use_for_object_mask:
            return prompt.text
    return f"the {cat}"


def is_auto_refinement_prompt(
    category: str,
    dataset: str | None,
    prompt: Sam3Prompt,
) -> bool:
    profile = get_sam3_profile(category, dataset)
    allowed_roles = AUDITED_AUTO_ROLES.get(profile.dataset, {}).get(profile.category)
    if allowed_roles is None:
        return False
    return prompt.role in allowed_roles


def should_use_sam3_local_refinement(
    category: str,
    dataset: str | None,
    task_type: str | None,
) -> bool:
    profile = get_sam3_profile(category, dataset)
    task = str(task_type or "").strip().lower()
    disabled_tasks = SAM3_LOCAL_REFINEMENT_DISABLED_TASKS.get(profile.dataset, {}).get(profile.category, set())
    return task not in disabled_tasks


def normal_part_veto_config(category: str, dataset: str | None = None) -> dict | None:
    profile = get_sam3_profile(category, dataset)
    raw = NORMAL_PART_VETO_CONFIGS.get(profile.dataset, {}).get(profile.category)
    if not raw:
        return None
    prompts_by_role = {prompt.role: prompt for prompt in profile.prompts}
    normal_prompt = prompts_by_role.get(raw["normal_role"])
    if normal_prompt is None:
        return None
    defect_prompts = [
        prompts_by_role[role]
        for role in raw.get("competing_defect_roles", ())
        if role in prompts_by_role
    ]
    return {
        **raw,
        "dataset": profile.dataset,
        "category": profile.category,
        "normal_prompt": normal_prompt,
        "defect_prompts": tuple(defect_prompts),
    }


def iter_refinement_prompts(
    category: str,
    dataset: str | None = None,
    include_parts: bool = True,
) -> Iterable[Sam3Prompt]:
    profile = get_sam3_profile(category, dataset)
    for prompt in profile.prompts:
        if not is_auto_refinement_prompt(profile.category, profile.dataset, prompt):
            continue
        if not include_parts and prompt.role not in OBJECT_PROMPT_ROLES:
            continue
        yield prompt


def _is_defect_prompt(prompt: Sam3Prompt) -> bool:
    return prompt.role.startswith("defect_")


def prompt_role_family(prompt: Sam3Prompt) -> str:
    if prompt.role in OBJECT_PROMPT_ROLES:
        return "object"
    if _is_defect_prompt(prompt):
        return "defect"
    return "part"


def _task_prompt_mode(task_type: str | None) -> str:
    task = str(task_type or "").lower()
    if "localization" in task:
        return "localization"
    if "description" in task or "analysis" in task:
        return "structural"
    if "classification" in task or "detection" in task:
        return "semantic"
    return "generic"


def score_prompt_relevance(
    prompt: Sam3Prompt,
    query: str = "",
    task_type: str | None = None,
) -> float:
    del query
    mode = _task_prompt_mode(task_type)
    family = prompt_role_family(prompt)
    return PROMPT_FAMILY_PRIORS.get(mode, PROMPT_FAMILY_PRIORS["generic"]).get(family, 0.0)


def _allows_defect_prompt_for_task(
    task_type: str | None,
    category: str | None = None,
    dataset: str | None = None,
) -> bool:
    task = str(task_type or "").lower()
    if "localization" in task:
        return True
    if "classification" in task or "description" in task or "analysis" in task:
        dataset_key = str(dataset or "").lower()
        category_key = str(category or "").lower()
        if category_key in NONLOCAL_DEFECT_PROMPT_DISABLED.get(dataset_key, set()):
            return False
        return True
    return False


def _allows_part_prompt_for_task(task_type: str | None) -> bool:
    task = str(task_type or "").lower()
    if "localization" in task:
        return True
    if "description" in task or "analysis" in task:
        return True
    return False


def _selection_reason(prompt: Sam3Prompt, task_type: str | None) -> str:
    if _is_defect_prompt(prompt):
        task = str(task_type or "").lower()
        if "localization" in task:
            return "selected_defect_prompt_for_localization"
        return "selected_defect_prompt_for_defect_qa"
    if prompt.role in OBJECT_PROMPT_ROLES:
        return "selected_object_prompt"
    task = str(task_type or "").lower()
    if "description" in task or "analysis" in task:
        return "selected_audited_part_for_structural_task"
    return "selected_audited_part_for_localization"


def plan_refinement_prompts(
    category: str,
    dataset: str | None = None,
    query: str = "",
    max_prompts: int = 6,
    include_parts: bool = True,
    task_type: str | None = None,
) -> Sam3PromptSelectionPlan:
    profile = get_sam3_profile(category, dataset)
    scored: list[ScoredSam3Prompt] = []
    audit_by_role: dict[str, dict] = {}
    for profile_order, prompt in enumerate(profile.prompts):
        relevance = score_prompt_relevance(prompt, query, task_type=task_type)
        prompt_family = prompt_role_family(prompt)
        audit = {
            "role": prompt.role,
            "text": prompt.text,
            "source": prompt.source,
            "profile_order": profile_order,
            "prompt_family": prompt_family,
            "selection_policy": SAM3_PROMPT_SELECTION_POLICY,
            "relevance": round(float(relevance), 6),
            "status": "rejected",
            "reason": "",
        }
        if not is_auto_refinement_prompt(profile.category, profile.dataset, prompt):
            audit["reason"] = "not_in_audited_auto_roles"
        elif not include_parts and prompt.role not in OBJECT_PROMPT_ROLES:
            audit["reason"] = "part_prompts_disabled"
        elif _is_defect_prompt(prompt) and not _allows_defect_prompt_for_task(task_type, profile.category, profile.dataset):
            audit["reason"] = "defect_prompt_requires_localization_task"
        elif prompt.role not in OBJECT_PROMPT_ROLES and not _allows_part_prompt_for_task(task_type):
            audit["reason"] = "part_prompt_requires_localization_description_or_analysis"
        else:
            audit["status"] = "candidate"
            audit["reason"] = _selection_reason(prompt, task_type)
            scored.append(ScoredSam3Prompt(
                prompt=prompt,
                relevance=relevance,
                selection_reason=audit["reason"],
                profile_order=profile_order,
            ))
        audit_by_role[prompt.role] = audit

    scored.sort(key=lambda item: (-item.relevance, item.profile_order, item.prompt.role))
    selected = scored[:max_prompts]
    selected_roles = {item.prompt.role for item in selected}
    candidate_roles = {item.prompt.role for item in scored}
    for item in scored:
        audit = audit_by_role[item.prompt.role]
        if item.prompt.role in selected_roles:
            audit["status"] = "selected"
        elif item.prompt.role in candidate_roles:
            audit["status"] = "rejected"
            audit["reason"] = "ranked_below_prompt_budget"
    return Sam3PromptSelectionPlan(
        selected=tuple(selected),
        audit=tuple(audit_by_role[prompt.role] for prompt in profile.prompts),
    )


def select_refinement_prompts(
    category: str,
    dataset: str | None = None,
    query: str = "",
    max_prompts: int = 6,
    include_parts: bool = True,
    task_type: str | None = None,
) -> list[ScoredSam3Prompt]:
    return list(plan_refinement_prompts(
        category,
        dataset=dataset,
        query=query,
        max_prompts=max_prompts,
        include_parts=include_parts,
        task_type=task_type,
    ).selected)


def describe_profile(category: str, dataset: str | None = None) -> dict:
    profile = get_sam3_profile(category, dataset)
    return {
        "category": profile.category,
        "dataset": profile.dataset,
        "simple_copy": profile.simple_copy,
        "prompts": [
            {
                "role": prompt.role,
                "text": prompt.text,
                "threshold": prompt.threshold,
                "use_for_object_mask": prompt.use_for_object_mask,
                "use_for_refinement": is_auto_refinement_prompt(profile.category, profile.dataset, prompt),
                "source": prompt.source,
            }
            for prompt in profile.prompts
        ],
    }
