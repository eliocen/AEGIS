"""
Default configuration values for AEGIS.

Version: 0.4.0
"""

DEFAULT_CONFIG = {
    "framework": {
        "name": "AEGIS",
        "version": "0.4.0",
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
}