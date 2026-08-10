"""
Transformer-based visual representation encoder for AEGIS.

Default backbone:
OpenAI CLIP ViT-B/32

Version: 0.8.0
"""

from pathlib import Path
from typing import Optional

from aegis.preprocessing import CanonicalSample

from .device import resolve_device
from .vision_base import VisionEncoder
from .vision_output import VisionRepresentation


class TransformerVisionEncoder(VisionEncoder):
    """
    Transformer-based image encoder.

    The default implementation uses the CLIP vision
    backbone but exposes a generic AEGIS VisionEncoder
    interface.
    """

    DEFAULT_MODEL = "openai/clip-vit-base-patch32"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "auto",
    ):

        try:
            import torch

            from transformers import (
                AutoImageProcessor,
                CLIPVisionModelWithProjection,
            )

        except ImportError as exc:

            raise RuntimeError(
                "TransformerVisionEncoder requires "
                "PyTorch and Hugging Face Transformers."
            ) from exc

        self.torch = torch

        self.model_name = (
            model_name
            or self.DEFAULT_MODEL
        )

        self.device = resolve_device(
            device
        )

        self.processor = (
            AutoImageProcessor.from_pretrained(
                self.model_name
            )
        )

        self.model = (
            CLIPVisionModelWithProjection
            .from_pretrained(
                self.model_name
            )
        )

        self.model.to(
            self.device
        )

        self.model.eval()

    def encode(
        self,
        sample: CanonicalSample,
    ) -> VisionRepresentation:

        if not isinstance(
            sample,
            CanonicalSample,
        ):
            raise TypeError(
                "TransformerVisionEncoder expects "
                "a CanonicalSample."
            )

        if not sample.has_image:
            raise ValueError(
                f"Sample {sample.sample_id} "
                "does not contain an image."
            )

        image_path = Path(
            sample.image_path
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image file not found: "
                f"{image_path}"
            )

        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Pillow is required for image loading."
            ) from exc

        with Image.open(
            image_path
        ) as image:

            image = image.convert(
                "RGB"
            )

            encoded = self.processor(
                images=image,
                return_tensors="pt",
            )

        encoded = {
            key: value.to(
                self.device
            )
            for key, value
            in encoded.items()
        }

        with self.torch.no_grad():

            outputs = self.model(
                **encoded
            )

        embedding = (
            outputs.image_embeds
            .squeeze(0)
            .detach()
            .cpu()
        )

        dimension = int(
            embedding.shape[-1]
        )

        return VisionRepresentation(
            sample_id=sample.sample_id,
            embedding=embedding,
            model_name=self.model_name,
            dimension=dimension,
            image_path=str(image_path),
            metadata={
                "device": self.device,
                "encoder_type": "clip_vision",
            },
        )