# AEGIS T1 Step2D10 鈥?Prospective Manifest, Integrity and Shortcut-Audit Protocol

Frozen parent: `57fabf746506ad50c11fc734ff19876ae95e88e7`. Population: all **336,000** extracted Stable Diffusion v1.4 files under the preserved native `train/val 脳 ai/nature` structure.

The audit is prospective and read-only with respect to dataset payloads. Per-file manifest fields are relative path, native split/class, mapped label, extension, byte size, SHA-256, width, height, aspect ratio, image format, and decode status. Audits cover count reconciliation; extension/format; decode validity; dimensions/aspect ratio; file-size distributions; filename-pattern leakage; exact duplicates within cells; cross-class duplicates; train/validation duplicates; and collision inventories.

Anomalies are recorded and frozen; there is **no silent deletion, relabeling, repair, resizing, normalization, or split reassignment**. Raw data remains outside Git. Preprocessing, model execution, training, formal evaluation, and official-test access remain **NOT_AUTHORIZED**.
