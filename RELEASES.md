# Releases -- vivijure-backend

Render backend for RunPod serverless. A release is an annotated git tag
`backend-v<semver>` **pushed to origin**; the fleet Jenkins (tag discovery) builds and
pushes a Docker image to `ghcr.io/skyphusion-labs/vivijure-backend:<semver>` (the image
tag drops the `backend-v` prefix).

> **Lesson (2026-06-12):** the release step MUST push tags to origin. See the 0.2.1-0.2.3
> gap below -- those tags were cut on mindcrimes local clone, never pushed, and lost when
> the box was released.

| git tag | GHCR image | source commit | built | notes |
|---|---|---|---|---|
| backend-v0.2.5 | 0.2.5 | 2e60829 | 2026-06-13 (fleet) | Re-land orphaned #37 finishing-stage deps (gfpgan/basicsr/facexlib + RIFE vendor); fix RIFE load_model path; CI import smoke gate (#51). |
|---|---|---|---|---|
| backend-v0.2.4 | 0.2.4 | 997568a | 2026-06-12 (fleet) | First release tagged AND pushed to origin post-mindcrime. Render-hardening batch (#40-#45) + deploy fixes (#34/#35/#38). |
| backend-v0.2.3 | 0.2.3 | ~8919c79 (#33)* | 2026-06-12 09:21Z (mindcrime) | **git tag LOST** (cut local, never pushed; box released). Pipeline iteration. Was the image running on RunPod. |
| backend-v0.2.2 | 0.2.2 | ~8919c79 (#33)* | 2026-06-12 08:08Z (mindcrime) | git tag lost (as above). Build iteration. |
| backend-v0.2.1 | 0.2.1 | ~8919c79 (#33)* | 2026-06-12 05:26Z (mindcrime) | git tag lost (as above). Build iteration. |
| backend-v0.2.0 | 0.2.0 | (on origin) | -- | last tag that reached origin before the gap. |

\* Commit inferred from image build-time vs the `main` commit timeline -- the images carry
no `org.opencontainers.image.revision` label. All three predate the #34/#35/#38 deploy
fixes, so they are the #33-era backend; 0.2.4 supersedes them.

## Fix-forward
- Release step pushes tags to origin (the bug behind the 0.2.1-0.2.3 gap).
- Add `org.opencontainers.image.revision=$GIT_SHA` to the Dockerfile (build ARG) so future
  images are self-documenting even if a tag is lost. (fast-follow)
