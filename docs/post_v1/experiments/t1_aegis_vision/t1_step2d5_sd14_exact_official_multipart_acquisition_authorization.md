# AEGIS Post-v1 T1 Step2D5 鈥?SD V1.4 Exact Official Multipart Acquisition Authorization

## Authorization

The next acquisition-execution stage is authorized to transfer the **30 observed official Stable Diffusion V1.4 multipart components** from the resolved GenImage Google Drive child folder:

`https://drive.google.com/drive/folders/12xighYOtu-ryfYEUnNrSeZqrxT8P08Zy`

Destination:

`data/raw/genimage/stable_diffusion_v_1_4`

Raw data remains `DO_NOT_COMMIT_RAW_DATA`.

## Exact expected component set

The authorized component set is:

- `imagenet_ai_0419_sdv4.z01` through `imagenet_ai_0419_sdv4.z29`
- `imagenet_ai_0419_sdv4.zip`

No other dataset payload is authorized by this Step2D5 record.

The Google Drive interface displayed the first 29 components as 3 GB each and the terminal ZIP as 2.79 GB, approximately **89.79 GB** in aggregate. These are observational display values, not exact byte sizes.

## Capacity controls

Step2D established a 100 GiB mandatory free-space reserve. At this authorization gate the repository drive reports **318.58 GiB free**.

Before **every** component transfer, available disk must be measured again. Acquisition must stop before starting a component if the mandatory 100 GiB reserve cannot be preserved.

The compressed acquisition fitting on disk does **not** authorize extraction. Extraction requires a separate post-acquisition capacity assessment because the expanded dataset footprint is not yet known.

## Transfer and integrity controls

The execution stage must:

1. use only the resolved official GenImage SD1.4 Google Drive folder;
2. transfer sequentially/resumably where supported rather than starting 30 concurrent downloads;
3. preserve original component filenames and bytes;
4. capture exact byte size and SHA-256 for every completed component;
5. capture official Drive file ID when the transfer mechanism exposes it;
6. reject HTML/authentication/quota responses masquerading as files;
7. stop cleanly on quota, authentication, network, source, integrity, or storage failure;
8. preserve exact partial state if interrupted;
9. require all 30 expected components before acquisition completion can be declared;
10. perform **no extraction** during the acquisition stage.

## License and scientific boundary

Use remains `NON_COMMERCIAL_ACADEMIC_RESEARCH` under `CC BY-NC-SA 4.0 + GenImage Dataset Terms`.

This authorization changes only one boundary: official SD1.4 multipart **payload transfer in the next execution stage becomes AUTHORIZED_BOUNDED**.

Still `NOT_AUTHORIZED`: extraction, preprocessing, final validation/test construction, model/source implementation, model execution, training, formal evaluation, and AEGIS v1 official-test access.

## Next

`POST_V1_T1_AEGIS_VISION_STEP2D6_SD14_OFFICIAL_MULTIPART_ACQUISITION_EXECUTION_AND_INTEGRITY_EVIDENCE`
