# AEGIS Post-v1 T1 Step2D3 鈥?SD V1.4 Official Drive Child Resolution Decision

## Probe result

The official GenImage Google Drive parent folder was queried for public metadata without downloading a dataset archive.

- Metadata response bytes: **365284**
- SD V1.4 name tokens observed in the static response: **1**
- Resolution state: `NAME_VISIBLE_BUT_CHILD_ID_NOT_SAFELY_PARSED`
- Exact official SD V1.4 child ID: `NOT_SAFELY_RESOLVED`
- Exact official SD V1.4 size: `NOT_SAFELY_RESOLVED`

No child ID is inferred from unrelated opaque identifiers in the Google Drive page.

## Governance decision

The official parent distribution remains authoritative. A large payload transfer is not authorized from an unresolved child identity.

The next resolution path is `MANUAL_OFFICIAL_DRIVE_CHILD_SELECTION_AND_ID_CAPTURE`.

If manual resolution is used, the exact child name, Google Drive child URL/ID, displayed size where available, parent folder identity, and capture timestamp must be frozen before download.

A third-party transport source remains observational only unless a separately governed amendment verifies provenance equivalence and explicitly preserves the original GenImage CC BY-NC-SA 4.0 + Dataset Terms boundary.

## Still not authorized

Dataset archive transfer under an unresolved child identity, mirror substitution, extraction/preprocessing, final split construction, model/source implementation, model execution, training, formal evaluation, and AEGIS v1 official-test access remain `NOT_AUTHORIZED`.
