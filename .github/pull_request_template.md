<!--
Thanks for contributing. See CONTRIBUTING.md for the workflow and conventions.
This is an AGPL-3.0-only, clean-room project: do not include code, diffs, or
implementation details from any other render pipeline.
-->

## What changed and why

<!-- A sentence or two. The "why" matters more than the "what". -->

## How it was validated

- [ ] `pytest` passes (CPU suite, no GPU required)
- [ ] Anything that needs a GPU-validation pass (SDXL / LoRA / Wan i2v on real
      hardware) is called out below so it can be scheduled, not assumed

## Checklist

- [ ] No em-dashes or en-dashes (use commas, semicolons, parentheses, or `--`)
- [ ] No secrets in the diff (R2 tokens, RunPod keys, `.env`/`.dev.vars`)
- [ ] No third-party render-pipeline code or excerpts (keeps the clean room clean)
- [ ] Commits are signed off (`git commit -s`, DCO) for code changes
- [ ] Did not add a CHANGELOG release heading or cut a tag (maintainers cut releases)
