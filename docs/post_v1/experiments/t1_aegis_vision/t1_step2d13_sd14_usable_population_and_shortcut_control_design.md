# AEGIS T1 Step2D13 鈥?SD1.4 Usable Population and Shortcut-Control Design

Parent disposition: `65fa3165d901612e7c3f326c84d1722401affba6`.

The raw GenImage SD1.4 payload remains immutable. The derived usable population shall exclude, by manifest only, the three decode-failing `train/ai` files and the 17 validation occurrences participating in exact train-validation collisions. Expected usable population: **323,997 training + 11,983 validation = 335,980 files**.

## Shortcut-control boundary

The observed PNG/JPEG class correlation and file-size association shall not be exposed directly to the model. Images will be decoded to pixels in memory and converted to RGB under an identical class-independent input path. Raw file bytes, extension, container format, file size, filename, path, and class-directory tokens are prohibited as model input features. The raw dataset will not be re-encoded or rewritten.

Input geometry and normalization depend on the eventual backbone and therefore will be frozen prospectively with T1 Step3 architecture design. Any augmentation must also be prospectively specified before use. This control removes direct container/metadata cues from the model interface but does **not** establish that all pixel-level generator or dataset artifacts are neutralized; residual shortcut risk remains part of the scientific interpretation.

Next: generate and verify the derived usable-population manifest, then freeze its provenance. Model execution, training, formal evaluation, and official-test access remain NOT_AUTHORIZED.
