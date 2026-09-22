# AEGIS T1 Step3F 鈥?DINO Weight Acquisition Execution Authorization

Parent Step3E1 commit: `88011ff24a0e7705fce81d0958125486568c88e3`.

The dependency gate is closed with Python 3.10.11, PyTorch 2.13.0+cu132, torchvision 0.28.0+cu132, timm 1.0.29 and CUDA 13.2 on the GTX 1660 SUPER.

This step authorizes acquisition of only the official `timm/vit_small_patch16_224.dino` `model.safetensors` artifact. The expected SHA-256 is `8D4795B8E4E327C6D841E15A2E3E6D88C45346256A43F8CDAE8C26D92C806F7A`. The local destination is `data/models/timm/vit_small_patch16_224.dino/model.safetensors` and the artifact must remain outside Git.

Acquisition acceptance requires exact SHA-256 identity. This authorization does not permit model instantiation, tensor loading from the safetensors file, `load_state_dict`, a forward pass, implementation, training, formal evaluation or official-test access.
