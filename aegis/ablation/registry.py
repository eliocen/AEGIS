"""
AEGIS Ablation Variant Registry.

Version: 0.21.0
"""

from .spec import (
    AblationSpec,
)


def default_ablation_registry():
    """
    Return the standard AEGIS ablation study
    configuration set.
    """

    return {
        "baseline": (
            AblationSpec(
                name="baseline",

                description=(
                    "Full multimodal AEGIS "
                    "training configuration."
                ),

                use_text=True,

                use_vision=True,

                alignment_loss_weight=1.0,

                classification_loss_weight=1.0,
            )
        ),

        "text_only": (
            AblationSpec(
                name="text_only",

                description=(
                    "Remove the visual modality."
                ),

                use_text=True,

                use_vision=False,

                alignment_loss_weight=1.0,

                classification_loss_weight=1.0,
            )
        ),

        "vision_only": (
            AblationSpec(
                name="vision_only",

                description=(
                    "Remove the textual modality."
                ),

                use_text=False,

                use_vision=True,

                alignment_loss_weight=1.0,

                classification_loss_weight=1.0,
            )
        ),

        "no_alignment_loss": (
            AblationSpec(
                name="no_alignment_loss",

                description=(
                    "Disable the cross-modal "
                    "alignment objective while "
                    "retaining classification."
                ),

                use_text=True,

                use_vision=True,

                alignment_loss_weight=0.0,

                classification_loss_weight=1.0,
            )
        ),

        "reduced_alignment": (
            AblationSpec(
                name="reduced_alignment",

                description=(
                    "Reduce the alignment-loss "
                    "contribution by 50 percent."
                ),

                use_text=True,

                use_vision=True,

                alignment_loss_weight=0.5,

                classification_loss_weight=1.0,
            )
        ),

        "reduced_classification": (
            AblationSpec(
                name="reduced_classification",

                description=(
                    "Reduce the hierarchical "
                    "classification-loss contribution "
                    "by 50 percent."
                ),

                use_text=True,

                use_vision=True,

                alignment_loss_weight=1.0,

                classification_loss_weight=0.5,
            )
        ),
    }