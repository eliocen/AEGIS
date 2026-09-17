# AEGIS Post-v1 T1 Step2B 鈥?GenImage Acquisition, Split and Provenance Protocol

## Status

`POST_V1_T1_STEP2B_GENIMAGE_ACQUISITION_SPLIT_AND_PROVENANCE_PROTOCOL_FROZEN`

This is a prospective protocol only. It does not download or extract GenImage.

## Authoritative acquisition boundary

The selected dataset is **GenImage**.

Initial acquisition must use the official GenImage project distribution. The official repository currently documents a Google Drive distribution and also documents the historical Baidu Yunpan distribution. Third-party mirrors are not authorized by Step2B.

The intended initial acquisition covers all eight documented generator subsets where available:

- Midjourney
- Stable Diffusion V1.4
- Stable Diffusion V1.5
- ADM
- GLIDE
- Wukong
- VQDM
- BigGAN

If an official subset is unavailable, the acquisition must stop before the final scientific split is frozen. Missing subsets must be recorded explicitly; the experiment may not silently change population.

## Native structure and label preservation

Preserve the official per-generator structure:

`<generator>/{train,val}/{ai,nature}`

The native `train` and `val` boundaries are immutable acquisition metadata.

T1 mapping remains:

- `nature` -> `AUTHENTIC`
- `ai` -> `MANIPULATED`
- `MANIPULATED` -> operationally `AI_GENERATED` for this first experiment.

No files are to be renamed, merged, recompressed, or resaved before the canonical provenance manifest is built.

## Split governance

Native GenImage `train` is the only candidate training pool.

Native GenImage `val` is the only candidate held-out pool.

The final T1 validation/test derivation is deliberately deferred until after exact inventory, exact-duplicate audit, near-duplicate protocol freeze, and source/generator-family analysis.

No native-val sample may enter training.

No sample from the sealed AEGIS v1 official-test boundary may be accessed.

Whether T1 ultimately evaluates pooled-generator generalization, held-out-generator generalization, or both must be frozen prospectively after acquisition inventory. Step2B does not decide this from convenience.

## Canonical provenance manifest

Every acquired image must ultimately have:

`relative_path`, `generator`, `native_split`, `native_label`, `t1_label`, `file_size_bytes`, `sha256`, `extension`, `image_width`, `image_height`, `image_mode`, `decode_status`, `source_endpoint`, and `acquisition_batch`.

The raw dataset remains outside Git under `data/raw/genimage`. Compact manifests may be committed under `data/manifests/genimage` only when doing so does not redistribute dataset image bytes or violate dataset terms.

## Integrity and leakage controls

Each acquired file receives SHA256 identity.

Exact duplicate SHA256 values must be audited across generators, native splits, and labels.

Any exact file appearing under both `AUTHENTIC` and `MANIPULATED` is quarantined for review.

Corrupt or undecodable files are recorded and excluded from scientific pools; they are not silently deleted from the provenance record.

A perceptual near-duplicate algorithm and numerical threshold will be frozen only after the acquired format/resolution distribution is known. Step2B intentionally does not invent that threshold.

## Shortcut/bias audit requirement

Before training authorization, T1 must audit image dimensions, formats, and compression-related metadata.

This is mandatory because follow-up research on GenImage has reported that generated-image detectors can exploit image-size and JPEG/compression biases. AEGIS-Vision must therefore distinguish genuine forensic discrimination from trivial dataset shortcuts as far as the frozen evidence permits.

## Storage and failure rules

Before acquisition, record free disk capacity and ensure sufficient safety margin.

Raw dataset bytes are not committed to Git.

Partial network failure must leave completed verified artifacts intact and must not be represented as complete acquisition.

Current official terms must be checked at acquisition time. If the terms are incompatible with the intended academic research use, acquisition stops.

## Authorization effect

Authorized now:

- local disk-capacity probe;
- official endpoint reachability probe without downloading dataset payloads;
- acquisition-script design.

Still not authorized:

- dataset download or extraction;
- third-party mirror fallback;
- model/source implementation;
- model execution;
- training;
- formal evaluation;
- AEGIS v1 official-test access.

## Next

`POST_V1_T1_AEGIS_VISION_STEP2C_GENIMAGE_ACQUISITION_EXECUTION_AUTHORIZATION`
