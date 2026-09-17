# AEGIS Post-v1 T1 Step2C 鈥?GenImage Acquisition Execution Authorization

## Status

`POST_V1_T1_STEP2C_GENIMAGE_ACQUISITION_EXECUTION_AUTHORIZED`

This authorization is subordinate to the frozen Step2B acquisition, split, provenance, integrity, storage, and failure rules.

## Authorized acquisition

The next execution stage may acquire GenImage from the official project distribution into:

`data/raw/genimage`

Raw image/archive bytes remain outside Git.

Acquisition should attempt the eight frozen official generator subsets where available:

- Midjourney
- Stable Diffusion V1.4
- Stable Diffusion V1.5
- ADM
- GLIDE
- Wukong
- VQDM
- BigGAN

## Capacity rule

The Step2C environment probe records actual free space on the repository drive.

Acquisition execution must use that recorded capacity state conservatively. If full eight-subset acquisition cannot be demonstrated to fit with a safety margin, the acquisition stage must use a staged download/inventory strategy or stop before transferring large payloads.

Step2C does not authorize filling the disk to exhaustion.

## Required acquisition evidence

The acquisition execution must produce evidence sufficient to establish:

1. exact official source endpoint used;
2. acquisition timestamp/batch;
3. acquired generator subsets;
4. archive/file identities and SHA256 where technically applicable;
5. extraction success/failure;
6. native `train/val` and `ai/nature` structure;
7. file counts and byte counts;
8. decode/metadata inventory;
9. exact duplicates and cross-label collisions;
10. incomplete/missing subsets;
11. canonical provenance manifest identity;
12. disk usage before and after acquisition.

## Execution constraints

Do not silently use third-party mirrors.

Do not rename, resave, recompress, augment, resize, normalize, or otherwise preprocess scientific image content during acquisition.

Do not merge native `train` and `val`.

Do not construct the final T1 validation/test split during acquisition.

Do not execute an AEGIS model.

Do not train.

Do not perform formal evaluation.

Do not access the sealed AEGIS v1 official-test boundary.

If authentication, quota, terms, network, storage, or official-source availability prevents complete acquisition, stop cleanly and freeze the exact partial state rather than substituting data.

## Shortcut-risk boundary

Acquisition evidence must preserve file format, dimensions, and other metadata needed for the mandatory pre-training image-size/compression shortcut audit.

No preprocessing may erase those characteristics before the audit.

## Authorization effect

Authorized next activity: **GenImage acquisition and provenance/integrity inventory only**, under the Step2B protocol and the capacity/failure rules above.

Still not authorized:

- AEGIS-Vision architecture implementation;
- model/source changes;
- model execution;
- training;
- final validation/test construction;
- formal evaluation;
- official-test access.

## Next

`POST_V1_T1_AEGIS_VISION_STEP2D_GENIMAGE_ACQUISITION_AND_PROVENANCE_EVIDENCE_FREEZE`
