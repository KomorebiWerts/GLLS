from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


_STOP_TOKENS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "by",
    "defect",
    "defective",
    "good",
    "in",
    "is",
    "it",
    "normal",
    "of",
    "on",
    "only",
    "or",
    "the",
    "there",
    "to",
    "with",
}

_WEAK_GROUNDING_TOKENS = {
    "body",
    "color",
    "component",
    "core",
    "inner",
    "insulation",
    "layer",
    "material",
    "object",
    "outer",
    "part",
    "ring",
    "surface",
}

_SPELLING_NORMALIZATION = {
    "colour": "color",
    "coloured": "color",
    "colours": "color",
    "fibre": "fiber",
    "fibres": "fiber",
    "remov": "remove",
}


@dataclass(frozen=True)
class VisualPrimitiveRule:
    """Generic visual failure primitive used as an evidence bridge.

    These are not dataset label aliases. A primitive only contributes when both
    the option text and graph evidence express the same visual failure mode.
    """

    name: str
    terms: frozenset[str]
    rationale: str


_VISUAL_PRIMITIVE_RULES = (
    VisualPrimitiveRule(
        "shape_deformation",
        frozenset({
            "bend",
            "bent",
            "buckl",
            "collapsed",
            "collaps",
            "concave",
            "dent",
            "deform",
            "deformation",
            "deformity",
            "distort",
            "distortion",
            "flat",
            "flatten",
            "geometry",
            "contour",
            "outline",
            "misalign",
            "misshapen",
            "shape",
            "symmet",
            "symmetry",
            "symmetric",
            "symmetrical",
            "asymmet",
            "asymmetry",
            "asymmetric",
            "asymmetrical",
            "warp",
        }),
        "geometry changes such as bending, denting, buckling, or flattening",
    ),
    VisualPrimitiveRule(
        "material_split",
        frozenset({
            "break",
            "broken",
            "crack",
            "fissure",
            "fracture",
            "jagged",
            "rupture",
            "separation",
            "split",
            "tear",
        }),
        "structural discontinuities where material separates or tears",
    ),
    VisualPrimitiveRule(
        "material_cut_or_exposure",
        frozenset({
            "cut",
            "expos",
            "expose",
            "exposure",
            "reveal",
            "remove",
            "scrape",
            "slice",
            "slit",
        }),
        "cuts, scrapes, or openings that reveal underlying material",
    ),
    VisualPrimitiveRule(
        "surface_abrasion",
        frozenset({
            "abrasion",
            "line",
            "linear",
            "matte",
            "scratch",
            "scuff",
            "thin",
        }),
        "surface-level abrasion or thin scratch-like marks",
    ),
    VisualPrimitiveRule(
        "hole_or_puncture",
        frozenset({
            "circular",
            "dot",
            "gap",
            "hole",
            "indent",
            "penetration",
            "poke",
            "puncture",
            "round",
            "void",
        }),
        "round openings, punctures, voids, or local penetrations",
    ),
    VisualPrimitiveRule(
        "absence_or_missing",
        frozenset({
            "absence",
            "absent",
            "blank",
            "empty",
            "gap",
            "gone",
            "hollow",
            "lack",
            "miss",
            "missing",
            "sparse",
            "void",
        }),
        "missing material, absent parts, or hollow/empty areas",
    ),
    VisualPrimitiveRule(
        "color_or_marking",
        frozenset({
            "blemish",
            "chipped",
            "code",
            "discolor",
            "faded",
            "font",
            "identifier",
            "imprint",
            "ink",
            "print",
            "stain",
            "text",
        }),
        "printed, stained, chipped, or marked visual appearance changes",
    ),
    VisualPrimitiveRule(
        "exposed_material_contrast",
        frozenset({
            "contrast",
            "highlight",
            "inner",
            "light",
            "lighter",
            "reveal",
        }),
        "visible contrast caused by revealed inner or highlighted material",
    ),
    VisualPrimitiveRule(
        "fiber_or_thread",
        frozenset({
            "fabric",
            "fiber",
            "fuzzy",
            "string",
            "thread",
            "weav",
            "weave",
            "woven",
        }),
        "fabric, fiber, thread, or woven texture anomalies",
    ),
    VisualPrimitiveRule(
        "foreign_metal",
        frozenset({
            "contamination",
            "foreign",
            "hard",
            "metal",
            "metallic",
            "rigid",
            "shiny",
        }),
        "foreign rigid or metallic contamination cues",
    ),
    VisualPrimitiveRule(
        "color_layout_mismatch",
        frozenset({
            "coding",
            "incorrect",
            "mapping",
            "mismatch",
            "position",
            "swap",
            "wrong",
        }),
        "layout, position, or code mismatches such as swapped color order",
    ),
)

_VISUAL_PRIMITIVE_LEXICON = {rule.name: set(rule.terms) for rule in _VISUAL_PRIMITIVE_RULES}

_SOURCE_GROUNDED_CHANNELS = {
    "visual_signature",
    "visual_signature_attribute",
    "graph_visual_attribute",
    "semantic_embedding",
}


def _flatten_text_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        items: list[str] = []
        for key, item in value.items():
            items.extend(_flatten_text_items(key))
            items.extend(_flatten_text_items(item))
        return items
    if isinstance(value, Iterable):
        items = []
        for item in value:
            items.extend(_flatten_text_items(item))
        return items
    return [str(value)]


@dataclass(frozen=True)
class SemanticProfile:
    text: str
    tokens: frozenset[str]
    attributes: Dict[str, frozenset[str]]

    @property
    def attribute_names(self) -> set[str]:
        return {name for name, hits in self.attributes.items() if hits}


class SemanticOptionGrounder:
    """Ground answer choices to graph evidence through visual attributes.

    The matcher deliberately operates on evidence text emitted by the PVLA graph:
    defect name, visual signature, contrast-vs-normal, region checks, and
    distinction edges. It avoids mapping dataset label names directly to answer
    labels; any synonym-like behavior must be justified by those evidence fields
    or by a small category-agnostic visual attribute ontology.
    """

    def __init__(self, encoder: Any = None):
        self.encoder = encoder
        self._embedding_cache: Dict[str, Any] = {}

    @staticmethod
    def normalize_text(text: Any) -> str:
        return re.sub(r"\s+", " ", str(text or "").lower().replace("_", " ")).strip()

    @classmethod
    def normalize_token(cls, token: str) -> str:
        token = _SPELLING_NORMALIZATION.get(token, token)
        if len(token) > 6 and token.endswith("ations"):
            token = token[:-6]
        elif len(token) > 5 and token.endswith("ation"):
            token = token[:-5]
        elif len(token) > 5 and token.endswith("ities"):
            token = token[:-5] + "y"
        elif len(token) > 4 and token.endswith("ity"):
            token = token[:-3]
        elif len(token) > 5 and token.endswith("ness"):
            token = token[:-4]
        elif len(token) > 5 and token.endswith("ive"):
            token = token[:-3]
        elif len(token) > 5 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 4 and token.endswith("ed"):
            token = token[:-2]
            if token.endswith("c"):
                token += "e"
        elif len(token) > 4 and token.endswith("es") and token.endswith(("ches", "shes", "sses", "xes", "zes")):
            token = token[:-2]
        elif len(token) > 3 and token.endswith("es"):
            token = token[:-1]
        elif len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        return _SPELLING_NORMALIZATION.get(token, token)

    @classmethod
    def tokenize(cls, text: Any) -> set[str]:
        tokens = set()
        for raw in re.findall(r"[a-z0-9]+", cls.normalize_text(text)):
            if len(raw) <= 1 or raw in _STOP_TOKENS:
                continue
            token = cls.normalize_token(raw)
            if token and len(token) > 1 and token not in _STOP_TOKENS:
                tokens.add(token)
        return tokens

    @classmethod
    def profile(cls, text: Any) -> SemanticProfile:
        tokens = cls.tokenize(text)
        attr_hits = {}
        for attr, terms in _VISUAL_PRIMITIVE_LEXICON.items():
            normalized_terms = {cls.normalize_token(term) for term in terms}
            hits = normalized_terms & tokens
            if hits:
                attr_hits[attr] = frozenset(hits)
        return SemanticProfile(
            text=str(text or ""),
            tokens=frozenset(tokens),
            attributes=attr_hits,
        )

    @staticmethod
    def weak_tokens() -> set[str]:
        return {SemanticOptionGrounder.normalize_token(token) for token in _WEAK_GROUNDING_TOKENS}

    @staticmethod
    def evidence_text_for_defect(
        defect: Dict[str, Any],
        include_region: bool = True,
        include_contrast: bool = True,
        include_distinction_targets: bool = True,
        include_visual_attributes: bool = True,
    ) -> str:
        fields = [
            defect.get("defect", ""),
            defect.get("visual_signature", ""),
        ]
        if include_visual_attributes:
            fields.extend(_flatten_text_items(defect.get("visual_attributes")))
        if include_contrast:
            fields.append(defect.get("contrast_vs_normal", ""))
        if include_region:
            fields.extend([
                defect.get("region", ""),
                defect.get("region_definition", ""),
                defect.get("critical_check", ""),
            ])
        for dist in defect.get("distinctions") or []:
            if isinstance(dist, dict):
                if include_distinction_targets:
                    fields.append(dist.get("target_defect", ""))
                fields.append(dist.get("difference", ""))
        return " ".join(str(item) for item in fields if item)

    @classmethod
    def graph_attribute_profile(cls, defect: Dict[str, Any]) -> SemanticProfile:
        return cls.profile(" ".join(_flatten_text_items(defect.get("visual_attributes"))))

    @staticmethod
    def _attribute_overlap(left: SemanticProfile, right: SemanticProfile) -> Dict[str, list[str]]:
        overlap = {}
        for attr in sorted(left.attribute_names & right.attribute_names):
            overlap[attr] = sorted(set(left.attributes[attr]) | set(right.attributes[attr]))
        return overlap

    @staticmethod
    def _attribute_term_count(attribute_overlap: Dict[str, Iterable[str]]) -> int:
        return sum(len(set(hits)) for hits in attribute_overlap.values())

    @staticmethod
    def _support_channels(
        *,
        strong_name_overlap: Iterable[str] = (),
        strong_signature_overlap: Iterable[str] = (),
        attribute_overlap: Optional[Dict[str, Iterable[str]]] = None,
        name_attribute_overlap: Optional[Dict[str, Iterable[str]]] = None,
        graph_attribute_overlap: Optional[Dict[str, Iterable[str]]] = None,
        semantic_similarity: float = 0.0,
        semantic_similarity_floor: float = 0.58,
        name_channel: str = "defect_name",
    ) -> list[str]:
        channels = []
        if set(strong_name_overlap):
            channels.append(name_channel)
        if set(strong_signature_overlap):
            channels.append("visual_signature")
        if attribute_overlap:
            channels.append("visual_attribute")
        if name_attribute_overlap:
            channels.append("defect_name_attribute")
        if graph_attribute_overlap:
            channels.append("graph_visual_attribute")
        if semantic_similarity >= semantic_similarity_floor:
            channels.append("semantic_embedding")
        return channels

    @classmethod
    def has_evidence_consensus(cls, score_info: Dict[str, Any], min_channels: int = 2) -> bool:
        """Return True when a match is supported beyond a single lexical hit.

        A single rich visual-attribute bridge is allowed only when it carries
        multiple concrete attribute terms. This keeps option grounding usable
        for wording shifts such as "weaving irregularity" vs "loose fuzzy
        fiber" while rejecting one-word dataset-label shortcuts.
        """

        channels = set(score_info.get("support_channels") or [])
        if len(channels) >= min_channels:
            return True
        attribute_terms = int(score_info.get("attribute_term_count", 0) or 0)
        if channels == {"visual_attribute"} and attribute_terms >= 3:
            return True
        if channels == {"graph_visual_attribute"} and attribute_terms >= 2:
            return True
        if channels <= {"visual_attribute", "graph_visual_attribute"} and attribute_terms >= 3:
            return True
        if channels == {"semantic_embedding"} and float(score_info.get("semantic_similarity", 0.0)) >= 0.68:
            return True
        return False

    @classmethod
    def has_source_grounded_consensus(cls, score_info: Dict[str, Any]) -> bool:
        """Require graph-sourced visual evidence beyond option/defect labels."""

        if not cls.has_evidence_consensus(score_info):
            return False
        source_channels = set(score_info.get("source_evidence_channels") or [])
        return bool(source_channels & _SOURCE_GROUNDED_CHANNELS)

    def _embedding_similarity(self, left: str, right: str) -> float:
        if self.encoder is None:
            return 0.0
        left = left.strip()
        right = right.strip()
        if not left or not right:
            return 0.0
        try:
            import torch
            from sentence_transformers import util

            if left not in self._embedding_cache:
                self._embedding_cache[left] = self.encoder.encode(left, convert_to_tensor=True)
            if right not in self._embedding_cache:
                self._embedding_cache[right] = self.encoder.encode(right, convert_to_tensor=True)
            score = util.cos_sim(self._embedding_cache[left], self._embedding_cache[right])
            if torch.is_tensor(score):
                return float(score.reshape(-1)[0].item())
            return float(score)
        except Exception:
            return 0.0

    def score_text_pair(self, left: Any, right: Any) -> Dict[str, Any]:
        left_profile = self.profile(left)
        right_profile = self.profile(right)
        overlap = set(left_profile.tokens) & set(right_profile.tokens)
        weak_tokens = self.weak_tokens()
        strong_overlap = overlap - weak_tokens
        attribute_overlap = self._attribute_overlap(left_profile, right_profile)
        semantic_similarity = self._embedding_similarity(str(left or ""), str(right or ""))
        support_channels = self._support_channels(
            strong_name_overlap=strong_overlap,
            strong_signature_overlap=(),
            attribute_overlap=attribute_overlap,
            semantic_similarity=semantic_similarity,
            name_channel="lexical_overlap",
        )
        attribute_term_count = self._attribute_term_count(attribute_overlap)
        semantic_bonus = max(0.0, semantic_similarity - 0.45) * 4.0
        score = (
            1.2 * len(strong_overlap)
            + 0.25 * len(overlap & weak_tokens)
            + 1.35 * len(attribute_overlap)
            + 0.18 * attribute_term_count
            + semantic_bonus
        )
        return {
            "score": score,
            "signal_score": 1.2 * len(strong_overlap) + 1.35 * len(attribute_overlap) + 0.18 * attribute_term_count + semantic_bonus,
            "token_overlap": sorted(overlap),
            "strong_overlap": sorted(strong_overlap),
            "attribute_overlap": attribute_overlap,
            "attribute_term_count": attribute_term_count,
            "support_channels": support_channels,
            "evidence_consensus": self.has_evidence_consensus({
                "support_channels": support_channels,
                "attribute_term_count": attribute_term_count,
                "semantic_similarity": round(float(semantic_similarity), 4),
            }),
            "semantic_similarity": round(float(semantic_similarity), 4),
        }

    def score_option_against_defect(self, option_text: str, defect: Dict[str, Any]) -> Dict[str, Any]:
        option_profile = self.profile(option_text)
        name_profile = self.profile(defect.get("defect", ""))
        signature_profile = self.profile(
            f"{defect.get('visual_signature', '')} "
            + " ".join(
                str(dist.get("difference", ""))
                for dist in defect.get("distinctions") or []
                if isinstance(dist, dict)
            )
        )
        region_profile = self.profile(
            f"{defect.get('region', '')} {defect.get('region_definition', '')} {defect.get('critical_check', '')}"
        )
        evidence_text = self.evidence_text_for_defect(
            defect,
            include_region=False,
            include_contrast=False,
            include_visual_attributes=False,
        )
        semantic_evidence_text = self.evidence_text_for_defect(
            defect,
            include_region=False,
            include_contrast=False,
            include_visual_attributes=True,
        )
        evidence_profile = self.profile(evidence_text)
        graph_attribute_profile = self.graph_attribute_profile(defect)

        weak_tokens = self.weak_tokens()
        name_overlap = set(option_profile.tokens) & set(name_profile.tokens)
        signature_overlap = set(option_profile.tokens) & set(signature_profile.tokens)
        region_overlap = set(option_profile.tokens) & set(region_profile.tokens)
        strong_name_overlap = name_overlap - weak_tokens
        strong_signature_overlap = signature_overlap - weak_tokens
        attribute_overlap = self._attribute_overlap(option_profile, evidence_profile)
        signature_attribute_overlap = self._attribute_overlap(option_profile, signature_profile)
        name_attribute_overlap = self._attribute_overlap(option_profile, name_profile)
        graph_attribute_overlap = self._attribute_overlap(option_profile, graph_attribute_profile)
        semantic_similarity = self._embedding_similarity(option_text, semantic_evidence_text)
        support_channels = self._support_channels(
            strong_name_overlap=strong_name_overlap,
            strong_signature_overlap=strong_signature_overlap,
            attribute_overlap=attribute_overlap,
            name_attribute_overlap=name_attribute_overlap,
            graph_attribute_overlap=graph_attribute_overlap,
            semantic_similarity=semantic_similarity,
        )
        attribute_term_count = (
            self._attribute_term_count(attribute_overlap)
            + self._attribute_term_count(name_attribute_overlap)
            + self._attribute_term_count(graph_attribute_overlap)
        )
        source_attribute_term_count = (
            self._attribute_term_count(signature_attribute_overlap)
            + self._attribute_term_count(graph_attribute_overlap)
        )
        semantic_bonus = max(0.0, semantic_similarity - 0.48) * 3.0

        evidence_tokens = strong_name_overlap | strong_signature_overlap
        has_graph_visual_text = bool(
            str(defect.get("visual_signature", "")).strip()
            or _flatten_text_items(defect.get("visual_attributes"))
            or any(
                isinstance(dist, dict) and str(dist.get("difference", "")).strip()
                for dist in defect.get("distinctions") or []
            )
        )
        source_evidence_channels = []
        if strong_signature_overlap:
            source_evidence_channels.append("visual_signature")
        if signature_attribute_overlap:
            source_evidence_channels.append("visual_signature_attribute")
        if graph_attribute_overlap:
            source_evidence_channels.append("graph_visual_attribute")
        if semantic_similarity >= 0.58 and has_graph_visual_text:
            source_evidence_channels.append("semantic_embedding")

        score = (
            2.4 * len(strong_name_overlap)
            + 1.0 * len(strong_signature_overlap)
            + 1.45 * len(attribute_overlap)
            + 0.7 * len(name_attribute_overlap)
            + 1.2 * len(graph_attribute_overlap)
            + 0.18 * attribute_term_count
            + 0.55 * len(name_overlap & weak_tokens)
            + 0.3 * len(signature_overlap & weak_tokens)
            + 0.2 * len(region_overlap)
            + semantic_bonus
        )
        if option_profile.tokens and option_profile.tokens <= (
            name_profile.tokens | signature_profile.tokens | region_profile.tokens
        ):
            score += 0.4

        signal_score = (
            2.4 * len(strong_name_overlap)
            + 1.0 * len(strong_signature_overlap)
            + 1.45 * len(attribute_overlap)
            + 0.7 * len(name_attribute_overlap)
            + 1.2 * len(graph_attribute_overlap)
            + 0.18 * attribute_term_count
            + semantic_bonus
        )
        consensus_payload = {
            "support_channels": support_channels,
            "attribute_term_count": attribute_term_count,
            "source_evidence_channels": source_evidence_channels,
            "source_attribute_term_count": source_attribute_term_count,
            "semantic_similarity": round(float(semantic_similarity), 4),
        }
        return {
            "score": score,
            "signal_score": signal_score,
            "name_overlap": sorted(name_overlap),
            "signature_overlap": sorted(signature_overlap),
            "region_overlap": sorted(region_overlap),
            "evidence_tokens": sorted(evidence_tokens),
            "attribute_overlap": attribute_overlap,
            "signature_attribute_overlap": signature_attribute_overlap,
            "name_attribute_overlap": name_attribute_overlap,
            "graph_attribute_overlap": graph_attribute_overlap,
            "attribute_term_count": attribute_term_count,
            "source_attribute_term_count": source_attribute_term_count,
            "support_channels": support_channels,
            "source_evidence_channels": source_evidence_channels,
            "label_only_match": not bool(source_evidence_channels),
            "evidence_consensus": self.has_source_grounded_consensus(consensus_payload),
            "semantic_similarity": round(float(semantic_similarity), 4),
        }

    @staticmethod
    def format_attribute_overlap(attribute_overlap: Dict[str, Iterable[str]]) -> str:
        if not attribute_overlap:
            return ""
        chunks = []
        for attr, hits in sorted(attribute_overlap.items()):
            hit_text = ", ".join(sorted(set(hits)))
            chunks.append(f"{attr}({hit_text})" if hit_text else attr)
        return "; ".join(chunks)

    @staticmethod
    def is_decisive_pair(best: Dict[str, Any], runner_up: Optional[Dict[str, Any]], margin: float = 0.9) -> bool:
        if not best or float(best.get("signal_score", 0.0)) <= 0:
            return False
        if runner_up is None:
            return True
        return float(best.get("score", 0.0)) - float(runner_up.get("score", 0.0)) >= margin


class PVLAEvidenceHypothesizer:
    """Rank graph defect hypotheses from visual evidence text only.

    This module is intentionally option-agnostic. It compares the Phase-1
    visual report against PVLA defect evidence and returns graph-backed defect
    hypotheses. Answer options are handled later by the VLM as a formatting and
    final-selection step, avoiding prompt-visible option-to-label shortcuts.
    """

    def __init__(self, grounder: Optional[SemanticOptionGrounder] = None):
        self.grounder = grounder or SemanticOptionGrounder()

    @staticmethod
    def _defect_key(defect: Dict[str, Any]) -> str:
        return str(defect.get("defect_node") or defect.get("graph_path") or defect.get("defect") or "")

    def rank_phase_report(
        self,
        phase_report: str,
        defects: Iterable[Dict[str, Any]],
        *,
        max_candidates: int = 3,
        min_signal_score: float = 1.6,
    ) -> list[Dict[str, Any]]:
        if not phase_report or is_normal_or_no_defect_report(phase_report):
            return []

        ranked: list[Dict[str, Any]] = []
        seen: set[str] = set()
        for defect in defects or []:
            if not isinstance(defect, dict):
                continue
            key = self._defect_key(defect)
            if key and key in seen:
                continue
            seen.add(key)
            evidence_text = SemanticOptionGrounder.evidence_text_for_defect(
                defect,
                include_region=False,
                include_contrast=False,
                include_distinction_targets=False,
            )
            score_info = self.grounder.score_text_pair(phase_report, evidence_text)
            if not score_info.get("evidence_consensus"):
                continue
            support_channels = set(score_info.get("support_channels") or [])
            if support_channels and not (support_channels & {"lexical_overlap", "semantic_embedding"}):
                continue
            if float(score_info.get("signal_score", 0.0)) < min_signal_score:
                continue
            ranked.append({
                **defect,
                "score": round(float(score_info.get("score", 0.0)), 3),
                "signal_score": round(float(score_info.get("signal_score", 0.0)), 3),
                "token_overlap": score_info.get("token_overlap", []),
                "attribute_overlap": score_info.get("attribute_overlap", {}),
                "attribute_term_count": score_info.get("attribute_term_count", 0),
                "support_channels": score_info.get("support_channels", []),
                "evidence_consensus": bool(score_info.get("evidence_consensus", False)),
                "semantic_similarity": score_info.get("semantic_similarity", 0.0),
            })

        ranked.sort(
            key=lambda item: (
                float(item.get("score", 0.0)),
                float(item.get("signal_score", 0.0)),
                int(item.get("attribute_term_count", 0) or 0),
            ),
            reverse=True,
        )
        return ranked[:max_candidates]


_SURFACE_ONLY_ANOMALY_TERMS = {
    "cut",
    "edge",
    "irregular",
    "layer",
    "lighter",
    "natural",
    "scrape",
    "seam",
    "separ",
    "separation",
    "shallow",
    "shell",
    "surface",
    "texture",
}

_DECISIVE_ANOMALY_TERMS = {
    "absence",
    "absent",
    "broken",
    "contamination",
    "crack",
    "deep",
    "foreign",
    "fracture",
    "hole",
    "ink",
    "metal",
    "missing",
    "print",
    "puncture",
    "rupture",
    "split",
    "structural",
    "tear",
    "void",
}

_NEGATION_TERMS = {"no", "not", "without", "absent", "absence", "neither", "nor"}
_NEGATION_RESET_TERMS = {"and", "but", "however", "although", "though", "except"}
_NORMAL_REPORT_PATTERNS = (
    re.compile(r"\bno\s+(?:logical\s+)?defects?\b"),
    re.compile(r"\bno\s+(?:visible\s+)?defects?\b"),
    re.compile(r"\bno\s+defects?\s+detected\b"),
    re.compile(r"\bno\s+(?:visible\s+)?anomal(?:y|ies)\b"),
    re.compile(r"\bwithout\s+(?:any\s+)?(?:visible\s+)?(?:defects?|anomal(?:y|ies))\b"),
    re.compile(r"\bmatches?\s+(?:the\s+)?normal\s+standard\b"),
    re.compile(r"\bmatching\s+(?:the\s+)?normal\b"),
    re.compile(r"\bnormal\s+standard\b"),
    re.compile(r"\bglobal\s+status\s*:\s*normal\b"),
)
_PHASE_PRIOR_ANOMALY_TERMS = set().union(*_VISUAL_PRIMITIVE_LEXICON.values()) | {
    "abnormal",
    "anomaly",
    "damage",
    "damaged",
    "defect",
    "defective",
}

_BENIGN_HOLE_CONTEXT_TERMS = {"board", "hole", "holes", "insert", "inserted", "seat", "seated", "solder", "soldered"}


def _is_negated_context(context: list[str]) -> bool:
    last_negation = -1
    last_reset = -1
    for idx, token in enumerate(context):
        if token in _NEGATION_TERMS:
            last_negation = idx
        if token in _NEGATION_RESET_TERMS:
            last_reset = idx
    return last_negation >= 0 and last_negation > last_reset


def _has_affirmed_term(tokens: list[str], target_terms: set[str], window: int = 18) -> bool:
    for idx, token in enumerate(tokens):
        if token not in target_terms:
            continue
        context = tokens[max(0, idx - window):idx]
        if token == "hole" and any(term in _BENIGN_HOLE_CONTEXT_TERMS for term in context[-8:]):
            continue
        if _is_negated_context(context):
            continue
        return True
    return False


def is_normal_or_no_defect_report(phase_report: str) -> bool:
    """Return True when Phase 1 explicitly reports a normal/no-defect state.

    This is deliberately not a category-label matcher. It only suppresses
    downstream defect priors when the report has a normal/no-defect phrase and
    no affirmed visual anomaly term outside a negated span, so reports such as
    "normal, but a visible puncture is present" can still contribute evidence.
    """

    normalized = SemanticOptionGrounder.normalize_text(phase_report)
    if not normalized:
        return False
    has_normal_phrase = re.search(r"\bnormal\b", normalized) is not None or any(
        pattern.search(normalized) for pattern in _NORMAL_REPORT_PATTERNS
    )
    if not has_normal_phrase:
        return False

    tokens = [
        SemanticOptionGrounder.normalize_token(t)
        for t in re.findall(r"[a-z0-9]+", normalized)
    ]
    return not _has_affirmed_term(tokens, _PHASE_PRIOR_ANOMALY_TERMS)


def should_prefer_normal_without_local_evidence(
    *,
    task_type: str,
    options: Optional[Dict[str, str]],
    phase_report: str,
    heatmap_peak_score: float,
    image_threshold: float,
    local_evidence_available: Optional[bool] = None,
) -> bool:
    """Gate binary anomaly claims that are unsupported by local evidence.

    The gate is intentionally semantic rather than category-specific. Callers
    can pass whether their local evidence selector found a stable region. The
    peak/threshold check remains only as a compatibility fallback for older
    callers.
    """

    if str(task_type or "").lower() != "anomaly detection":
        return False
    if local_evidence_available is None:
        local_evidence_available = heatmap_peak_score >= image_threshold
    if local_evidence_available:
        return False
    if not isinstance(options, dict):
        return False
    option_text = " ".join(str(v).lower() for v in options.values())
    if "yes" not in option_text or "no" not in option_text:
        return False

    tokens = [SemanticOptionGrounder.normalize_token(t) for t in re.findall(r"[a-z0-9]+", str(phase_report).lower())]
    if not tokens:
        return False
    has_surface_only_claim = any(term in set(tokens) for term in _SURFACE_ONLY_ANOMALY_TERMS)
    has_affirmed_decisive_claim = _has_affirmed_term(tokens, _DECISIVE_ANOMALY_TERMS)
    return has_surface_only_claim and not has_affirmed_decisive_claim
