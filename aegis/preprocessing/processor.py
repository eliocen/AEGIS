"""
AEGIS canonical preprocessing processor.

Version: 0.6.0
"""

from aegis.acquisition import AcquisitionRecord
from aegis.pipeline import AEGISLayer

from .exceptions import InvalidSampleError
from .image import normalize_image_path
from .labels import normalize_label
from .language import normalize_language
from .metadata import normalize_metadata
from .sample import CanonicalSample
from .text import normalize_text


class PreprocessingLayer(AEGISLayer):
    """
    Layer 2 of the AEGIS architecture.

    Converts AcquisitionRecord objects into standardized,
    model-ready CanonicalSample objects.
    """

    layer_name = "multilingual_multimodal_preprocessing"

    def __init__(
        self,
        config=None,
        strict_labels: bool = False,
    ):
        super().__init__(config)

        self.strict_labels = strict_labels

    def process(
        self,
        data: AcquisitionRecord,
    ) -> CanonicalSample:

        if not isinstance(
            data,
            AcquisitionRecord,
        ):
            raise TypeError(
                "PreprocessingLayer expects "
                "an AcquisitionRecord."
            )

        text = normalize_text(
            data.text
        )

        image_path = normalize_image_path(
            data.image_path
        )

        has_text = bool(text)
        has_image = bool(image_path)

        if not has_text and not has_image:
            raise InvalidSampleError(
                f"Sample {data.sample_id} contains "
                "no usable text or image modality."
            )

        language = normalize_language(
            data.language
        )

        label = normalize_label(
            data.label,
            strict=self.strict_labels,
        )

        metadata = normalize_metadata(
            data.metadata
        )

        preprocessing_info = {
            "text_normalized": has_text,
            "image_path_valid": has_image,
            "language_normalized": True,
            "metadata_normalized": True,
            "label_normalized": (
                label is not None
            ),
        }

        return CanonicalSample(
            sample_id=data.sample_id,
            text=text,
            image_path=image_path,
            language=language,
            source=data.source,
            platform=data.platform,
            timestamp=data.timestamp,
            label=label,
            has_text=has_text,
            has_image=has_image,
            is_multimodal=(
                has_text and has_image
            ),
            metadata=metadata,
            preprocessing=preprocessing_info,
        )