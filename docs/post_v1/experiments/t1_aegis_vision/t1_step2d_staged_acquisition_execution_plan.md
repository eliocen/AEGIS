# AEGIS Post-v1 T1 Step2D 鈥?Capacity-Bounded Staged GenImage Acquisition Plan

## Status

`POST_V1_T1_STEP2D_STAGED_ACQUISITION_EXECUTION_PLAN_FROZEN`

Step2C measured **318.58 GiB free** on the repository drive and classified the environment as `STAGED_OR_SUBSET_ACQUISITION_RECOMMENDED`.

This Step2D therefore rejects a blind all-at-once acquisition.

## Storage envelope

A **100 GiB mandatory free-space reserve** is frozen.

The current maximum working budget for dataset acquisition is therefore **218 GiB**, based on the Step2C measurement. This is a safety envelope, not an assertion that GenImage requires that amount.

Before every acquisition batch, free capacity must be measured again.

## Staged strategy

Acquire **one official generator subset at a time**, in this frozen operational order:

1. Stable Diffusion V1.4
2. Stable Diffusion V1.5
3. ADM
4. GLIDE
5. Wukong
6. VQDM
7. BigGAN
8. Midjourney

The order is operational only. It is not a scientific ranking.

Each completed subset must be inventoried and frozen before proceeding to the next capacity decision.

## Mandatory post-batch evidence

For each generator subset record source endpoint, timestamps, free space before/after, transferred/extracted bytes, archive SHA256 where applicable, native train/val counts, ai/nature counts, decode failures, exact-duplicate summary, and completion state.

Native `train/val` and `ai/nature` structure must remain intact.

## Stop conditions

Stop immediately if the 100 GiB reserve would be breached or if authentication, quota, terms, network, integrity, storage, or authoritative-source availability prevents reliable completion.

A partial acquisition is evidence, not failure. It must be frozen exactly rather than silently replaced or relabeled.

## Still prohibited

Final validation/test construction, image preprocessing, model/source implementation, model execution, training, formal evaluation, and AEGIS v1 official-test access remain `NOT_AUTHORIZED`.

## Next

`POST_V1_T1_AEGIS_VISION_STEP2D1_FIRST_GENIMAGE_SUBSET_ACQUISITION_AND_EVIDENCE_FREEZE`
