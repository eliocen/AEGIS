"""
Default configuration values for AEGIS.

Version: 0.4.0
"""

DEFAULT_CONFIG = {
    "framework": {
        "name": "AEGIS",
        "version": "0.11.0",
        "device": "auto",
        "seed": 42,
    },
    "data": {
        "language": "auto",
        "max_text_length": 512,
        "image_size": 224,
    },
    "pipeline": {
        "enable_context": True,
        "enable_intelligence": True,
        "enable_attribution": True,
        "enable_explainability": True,
        "enable_decision_support": True,
    },
    "representation": {
    "text": {
        "model_name": (
            "FacebookAI/xlm-roberta-base"
        ),
        "max_length": 256,
        "pooling": "masked_mean",
         },

    "vision": {
            "model_name": (
                "openai/clip-vit-base-patch32"
            ),
        },
    },
    "alignment": {
    "text_dimension": 768,
    "vision_dimension": 512,
    "shared_dimension": 512,
    "dropout": 0.1,
    "temperature": 0.07,
    "fusion": "gated",
    },
    "classification": {
    "input_dimension": 512,
    "hidden_dimension": 256,
    "dropout": 0.2,

    "hierarchy": {
        "integrity_classes": 2,
        "threat_classes": 4,
    },

    "loss": {
        "integrity_weight": 1.0,
        "threat_weight": 1.0,
    },
    },
    "context": {
    "domain_dimension": 32,
    "auxiliary_dimension": 32,
    "context_dimension": 128,
    "content_dimension": 512,
    "output_dimension": 512,
  },
}
