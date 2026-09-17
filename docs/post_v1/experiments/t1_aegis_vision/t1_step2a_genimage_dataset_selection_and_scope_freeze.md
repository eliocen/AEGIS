# AEGIS Post-v1 T1 Step2A 鈥?GenImage Dataset Selection and Scope Freeze

## Decision

**Selected dataset:** `GenImage`

**Decision state:** `SELECTED_FOR_T1_INITIAL_DATASET_ACQUISITION_FREEZE`

This is a documentary selection. Dataset download is not authorized by Step2A.

## Why GenImage fits the frozen first experiment

The official GenImage repository and project homepage describe a million-scale benchmark for AI-generated image detection with real/nature images and AI-generated images. The dataset spans the same 1,000 content classes used by ImageNet and includes images from multiple generator families, including Midjourney, Stable Diffusion, ADM, GLIDE, Wukong, VQDM, and BigGAN.

The official repository documents per-generator `train` and `val` directories containing `ai` and `nature` classes, and documents aggregation into a two-class full-dataset layout.

The project homepage states that, unless specifically labeled otherwise, the dataset is provided under CC BY-NC-SA 4.0 plus additional Dataset Terms and is restricted to non-commercial uses such as academic research, teaching, and scientific publication.

## Frozen T1 label mapping for this first experiment

| GenImage native organization | T1 label |
|---|---|
| `nature` | `AUTHENTIC` |
| `ai` | `MANIPULATED` |

For this first experiment, `MANIPULATED` is operationally narrowed to **`AI_GENERATED`**.

This does not mean that AI generation is the only kind of image manipulation relevant to AEGIS-Vision. Local editing, splicing, copy-move manipulation, face swaps, reenactment, and other forgery families remain outside this first experiment unless a later prospective protocol adds them.

## Population boundary

The research population is limited to the GenImage domains actually acquired and frozen: real/nature images derived from its documented source collection and AI-generated images from the frozen GenImage generator subsets.

No result may be generalized automatically to:

- all internet images;
- all manipulated images;
- all deepfakes;
- unseen future image generators;
- local-edit/splicing/copy-move forgeries;
- video or audio;
- contextual misinformation;
- production or operational deployment.

## Candidate comparison

FaceForensics++ is scientifically relevant and strongly documented for facial manipulation, but its population is face-centric and its images derive from video sequences, requiring careful sequence/family grouping.

ForgeryNet is also highly relevant for face forgery and provides very large image/video resources with multiple manipulation approaches, but its scope is substantially broader and more complex than the deliberately narrow first T1 experiment.

These datasets remain valid candidates for later AEGIS-Vision forgery-generalization research. Their non-selection here is not a scientific rejection.

## Acquisition prerequisites

Before downloading GenImage data, Step2B must freeze:

1. exact authoritative acquisition endpoint(s);
2. dataset terms/license acknowledgement;
3. exact generator subsets to acquire;
4. storage and expected-size planning;
5. archive/file/manifest hashing procedure;
6. class and split inventory procedure;
7. corruption/invalid-file handling;
8. duplicate and near-duplicate audit strategy;
9. split policy, including whether native train/val is retained and how a held-out T1 test is created without touching the AEGIS v1 official-test boundary;
10. deterministic manifest schema;
11. source-domain and generator limitations;
12. acquisition stop/failure rules.

## Authorization effect

Authorized: design and freeze the GenImage acquisition, split, and provenance protocol.

Not authorized: dataset download, extraction, preprocessing, model/source implementation, model execution, training, formal evaluation, or official-test access.

## Next

`POST_V1_T1_AEGIS_VISION_STEP2B_GENIMAGE_ACQUISITION_SPLIT_AND_PROVENANCE_PROTOCOL`
