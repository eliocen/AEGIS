# AEGIS Post-v1 T1 Step2 鈥?Dataset, Task, Population and Provenance Freeze

## Status

`POST_V1_T1_AEGIS_VISION_STEP2_DATASET_TASK_POPULATION_AND_PROVENANCE_CONTRACT_FROZEN`

This Step2 freezes the **selection criteria and scientific contract** for the first AEGIS-Vision experiment. It does not fabricate a dataset choice where repository evidence is insufficient.

## Primary task

The initial task is **binary image authenticity/manipulation classification**:

- `AUTHENTIC`
- `MANIPULATED`

The unit of analysis is one image.

`MANIPULATED` means the image pixels or image content have been intentionally altered/generated in a way covered by the subsequently selected dataset's documented ground-truth taxonomy.

This first task does **not** classify whether an authentic image is used with false, misleading, malicious, hateful, or out-of-context text. Contextual misuse belongs to a later cross-modal lineage.

## Population

The target research population is **digital images for which authenticity/manipulation ground truth is explicitly documented by the selected research dataset**.

No claim is made that results will generalize to all internet images, all deepfake methods, all geographic populations, all languages, all image generators, all editing tools, or future manipulation methods.

Any narrower source-domain, manipulation-family, geographic, demographic, platform, or temporal population imposed by the selected dataset must be inherited explicitly in the final dataset freeze.

## Dataset-selection requirements

A dataset may be selected for T1 only if its documentation supports all mandatory requirements:

1. explicit image-level authenticity/manipulation labels or a deterministic documented mapping to the frozen binary task;
2. accessible provenance describing source, collection or generation process;
3. licensing/terms compatible with research use;
4. sufficient image files or reproducible image references for the intended split;
5. a defensible train/validation/test partition or enough metadata to construct one prospectively;
6. no requirement to use the official AEGIS v1 test boundary;
7. manipulation categories documented well enough to state what `MANIPULATED` means;
8. duplicate/near-duplicate leakage can be assessed or controlled;
9. dataset version/release identity can be frozen;
10. any demographic, geographic, platform, generator, or manipulation-method limitations can be documented.

## Candidate-selection rule

Step2 does not select a dataset merely because its name or artifacts already occur in the repository.

Existing Fakeddit, Weibo, MFND, or other prior AEGIS data may be reused only if their authoritative dataset documentation independently satisfies the T1 image-authenticity/manipulation contract.

If no existing local dataset satisfies the contract, a separately documented external candidate review is required before acquisition. Dataset download remains unauthorized by this Step2.

## Label mapping

The final selected dataset must provide a frozen mapping table from native labels to:

- `AUTHENTIC`
- `MANIPULATED`
- `EXCLUDED_FROM_T1`

Ambiguous native labels must not be silently coerced into either class.

## Split governance

Before model training, the selected dataset freeze must specify exact train, validation, and held-out test identities or deterministic construction rules.

The split must address duplicate and near-duplicate leakage where technically feasible. Any grouping constraints available from the dataset, such as source identity, manipulation lineage, generator, original/derived image family, subject, event, or sequence, must be evaluated before random image-level splitting is accepted.

The held-out T1 test set is distinct from the sealed AEGIS v1 official-test boundary.

## Provenance freeze requirements

The final dataset freeze must record, where available:

- canonical dataset name and version;
- authoritative source/repository/publication;
- acquisition date and method;
- license/terms;
- archive/file hashes or manifest hashes;
- native label taxonomy;
- T1 label mapping;
- image counts by split and class;
- excluded/invalid samples;
- duplicate-control method;
- known population/domain limitations;
- any preprocessing required before model input.

## Step2 decision

The scientific **task/population/provenance contract is frozen**.

The concrete dataset identity is `NOT_YET_SELECTED`.

This is intentional: dataset selection requires authoritative candidate evidence rather than assumption from prior AEGIS datasets or repository filenames.

## Authorization effect

Authorized next activity: authoritative candidate-dataset review and evidence collection sufficient to select and freeze one T1 dataset under this contract.

Not authorized: dataset download/acquisition, model/source implementation, model execution, training, formal evaluation, or official-test access.

## Next

`POST_V1_T1_AEGIS_VISION_STEP2A_AUTHORITATIVE_CANDIDATE_DATASET_REVIEW_AND_SELECTION`
