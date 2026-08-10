"""
Create a local image for AEGIS vision testing.

Version: 0.8.0
"""

from pathlib import Path

from PIL import Image


output_dir = Path(
    "data/test"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

image_path = (
    output_dir
    / "aegis_test_image.jpg"
)

image = Image.new(
    "RGB",
    (224, 224),
    (128, 128, 128),
)

image.save(
    image_path
)

print(
    f"Created: {image_path}"
)