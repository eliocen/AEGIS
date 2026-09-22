# AEGIS T1 Step3D 鈥?Local Dependency and Weight-Acquisition Protocol

Parent Step3C commit: `b20ef6773070b9263c7c2e4ea0b7058cf68a4119`.

A read-only local environment probe was executed without loading or downloading any model. Observed environment:

`
{"python": "3.10.11", "torch": "2.13.0+cu132", "cuda_available": true, "cuda_runtime": "13.2", "gpu": "NVIDIA GeForce GTX 1660 SUPER", "torchvision": "0.28.0+cu132", "timm_error": "ModuleNotFoundError(\"No module named 'timm'\")"}
`

The selected artifact remains `timm/vit_small_patch16_224.dino` with expected SHA-256 `8D4795B8E4E327C6D841E15A2E3E6D88C45346256A43F8CDAE8C26D92C806F7A`.

Before model execution, `torch`, `torchvision` and `timm` must import successfully in the project virtual environment. Weight acquisition, when separately authorized, must use the official timm/Hugging Face model artifact, store it outside Git at `data/models/timm/vit_small_patch16_224.dino/model.safetensors`, and accept it only on exact SHA-256 match.

This step does not authorize weight download, model loading, forward execution, implementation, training, formal evaluation or official-test access.
