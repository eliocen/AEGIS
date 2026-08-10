"""
Manual AEGIS vision encoder test.

Version: 0.8.0
"""

from aegis.preprocessing import (
    CanonicalSample,
)

from aegis.representation import (
    TransformerVisionEncoder,
)


encoder = TransformerVisionEncoder()


sample = CanonicalSample(
    sample_id="IMG001",
    image_path=(
        "data/test/"
        "aegis_test_image.jpg"
    ),
    has_image=True,
)


representation = encoder.encode(
    sample
)


print(
    "Model:",
    representation.model_name,
)

print(
    "Sample:",
    representation.sample_id,
)

print(
    "Dimension:",
    representation.dimension,
)

print(
    "Shape:",
    representation.embedding.shape,
)

print(
    "Device:",
    representation.metadata["device"],
)