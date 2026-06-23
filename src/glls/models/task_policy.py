from __future__ import annotations

from dataclasses import dataclass


OBJECT_TASK_TYPES = frozenset({
    "object classification",
    "object structure",
    "object details",
    "object analysis",
})


@dataclass(frozen=True)
class QuestionTaskPolicy:
    """Execution policy derived from the benchmark QA task taxonomy.

    The policy is intentionally keyed by task type, not answer-option wording.
    It decides which evidence streams are appropriate before any option text is
    considered, so global object questions cannot be pulled into local anomaly
    search just because an option contains a defect-like word.
    """

    task_type: str
    task_group: str
    use_local_anomaly_stream: bool
    use_mcts_search: bool
    use_local_evidence_crops: bool
    show_local_crops_in_final_prompt: bool
    use_sam3_local_refinement: bool
    use_raw_heatmap_fill: bool
    use_no_local_anomaly_gate: bool
    include_global_reference_in_final: bool
    show_rag_blocks_in_final_prompt: bool
    show_phase1_report_in_final_prompt: bool
    use_pvla_defect_hypothesis_prompt: bool
    phase1_logic_role: str

    @property
    def is_object_task(self) -> bool:
        return self.task_group == "object"


def normalize_task_type(task_type: str | None) -> str:
    return str(task_type or "").strip().lower()


def policy_for_task(task_type: str | None, dataset: str | None = None) -> QuestionTaskPolicy:
    task = normalize_task_type(task_type)
    if task in OBJECT_TASK_TYPES:
        return QuestionTaskPolicy(
            task_type=task,
            task_group="object",
            use_local_anomaly_stream=False,
            use_mcts_search=False,
            use_local_evidence_crops=False,
            show_local_crops_in_final_prompt=False,
            use_sam3_local_refinement=False,
            use_raw_heatmap_fill=False,
            use_no_local_anomaly_gate=False,
            include_global_reference_in_final=True,
            show_rag_blocks_in_final_prompt=True,
            show_phase1_report_in_final_prompt=True,
            use_pvla_defect_hypothesis_prompt=False,
            phase1_logic_role="global_reference",
        )

    if task == "anomaly detection":
        return QuestionTaskPolicy(
            task_type=task,
            task_group="defect",
            use_local_anomaly_stream=True,
            use_mcts_search=True,
            use_local_evidence_crops=True,
            show_local_crops_in_final_prompt=True,
            use_sam3_local_refinement=True,
            use_raw_heatmap_fill=True,
            use_no_local_anomaly_gate=True,
            include_global_reference_in_final=False,
            show_rag_blocks_in_final_prompt=True,
            show_phase1_report_in_final_prompt=True,
            use_pvla_defect_hypothesis_prompt=False,
            phase1_logic_role="global_reference",
        )

    if task == "defect localization":
        return QuestionTaskPolicy(
            task_type=task,
            task_group="defect",
            use_local_anomaly_stream=True,
            use_mcts_search=True,
            use_local_evidence_crops=True,
            # Localization is a global spatial question. Keep the old effective
            # two-stage prompt anchored on Phase-1 + the red-contour global view;
            # the final prompt only shows stable heatmap/MCTS local evidence,
            # with SAM3 visible only for high-confidence structural masks.
            show_local_crops_in_final_prompt=True,
            use_sam3_local_refinement=True,
            use_raw_heatmap_fill=True,
            use_no_local_anomaly_gate=False,
            include_global_reference_in_final=False,
            show_rag_blocks_in_final_prompt=True,
            show_phase1_report_in_final_prompt=True,
            use_pvla_defect_hypothesis_prompt=False,
            phase1_logic_role="global_spatial_reference",
        )

    show_local_crops = True
    return QuestionTaskPolicy(
        task_type=task,
        task_group="defect",
        use_local_anomaly_stream=True,
        use_mcts_search=True,
        use_local_evidence_crops=True,
        show_local_crops_in_final_prompt=show_local_crops,
        use_sam3_local_refinement=True,
        use_raw_heatmap_fill=True,
        use_no_local_anomaly_gate=True,
        include_global_reference_in_final=False,
        show_rag_blocks_in_final_prompt=True,
        show_phase1_report_in_final_prompt=True,
        # Graph/PVLA evidence remains available through source-backed RAG blocks
        # and trace provenance. Do not inject graph-derived hypotheses as final
        # answer priors; they are too coarse when multiple options share similar
        # visual primitives or defect families.
        use_pvla_defect_hypothesis_prompt=False,
        phase1_logic_role="logic_guided_primary",
    )
