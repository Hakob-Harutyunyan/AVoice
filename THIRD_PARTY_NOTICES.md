# Third-Party Notices

## OmniVoice

This repository vendors the PyTorch implementation from:

- Project: `k2-fsa/OmniVoice`
- Source: https://github.com/k2-fsa/OmniVoice
- Vendored commit: `7a68a5cffa71b904a862f4870b246966deebadf7`
- License: Apache License 2.0

The vendored code lives in `omnivoice/`. Local Armenian-specific changes are
kept in this repository so training, inference, tokenization, and model changes
can be edited without depending on an installed `omnivoice` wheel.
