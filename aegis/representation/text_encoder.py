"""
Multilingual transformer text encoder for AEGIS.

Default backbone:
XLM-RoBERTa Base

Version: 0.7.0
"""

from typing import Optional

from aegis.preprocessing import (
    CanonicalSample,
)

from .base import TextEncoder
from .device import resolve_device
from .output import TextRepresentation
from .pooling import masked_mean_pooling


class TransformerTextEncoder(TextEncoder):
    """
    Multilingual transformer-based text encoder.

    The default backbone is XLM-RoBERTa, but the
    implementation uses Hugging Face AutoClasses so
    alternative compatible models may be substituted.
    """

    DEFAULT_MODEL = "FacebookAI/xlm-roberta-base"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "auto",
        max_length: int = 256,
    ):

        try:
            import torch
            from transformers import (
                AutoModel,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "TransformerTextEncoder requires "
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

        self.max_length = max_length

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                self.model_name
            )
        )

        self.model = (
            AutoModel.from_pretrained(
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
    ) -> TextRepresentation:

        if not isinstance(
            sample,
            CanonicalSample,
        ):
            raise TypeError(
                "TransformerTextEncoder expects "
                "a CanonicalSample."
            )

        if not sample.has_text:
            raise ValueError(
                f"Sample {sample.sample_id} "
                "does not contain text."
            )

        encoded = self.tokenizer(
            sample.text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
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

        pooled = masked_mean_pooling(
            outputs.last_hidden_state,
            encoded["attention_mask"],
        )

        embedding = (
            pooled
            .squeeze(0)
            .detach()
            .cpu()
        )

        dimension = int(
            embedding.shape[-1]
        )

        return TextRepresentation(
            sample_id=sample.sample_id,
            embedding=embedding,
            language=sample.language,
            model_name=self.model_name,
            dimension=dimension,
            text=sample.text,
            metadata={
                "device": self.device,
                "max_length": self.max_length,
                "pooling": "masked_mean",
            },
        )