from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def _resolve_path(name_or_path: str) -> str:
    if os.path.isdir(name_or_path):
        return name_or_path

    from huggingface_hub import snapshot_download

    return snapshot_download(name_or_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnivoice-prefetch",
        description="Cache auxiliary OmniVoice assets for offline/container use.",
    )
    parser.add_argument(
        "--model-dir",
        default="/app/model",
        help="Directory containing the OmniVoice model files.",
    )
    parser.add_argument(
        "--audio-tokenizer",
        default="eustlb/higgs-audio-v2-tokenizer",
        help="Audio tokenizer repo id or local path.",
    )
    parser.add_argument(
        "--asr-model",
        default="openai/whisper-large-v3-turbo",
        help="ASR model repo id or local path.",
    )
    parser.add_argument(
        "--copy-audio-tokenizer",
        action="store_true",
        help="Copy the tokenizer into <model-dir>/audio_tokenizer.",
    )
    parser.add_argument(
        "--prefetch-asr",
        action="store_true",
        help="Download the ASR model into the Hugging Face cache.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()

    tokenizer_path = _resolve_path(args.audio_tokenizer)
    if args.copy_audio_tokenizer:
        model_dir = Path(args.model_dir)
        target_dir = model_dir / "audio_tokenizer"
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(tokenizer_path, target_dir, dirs_exist_ok=True)
        logger.info("Copied audio tokenizer to %s", target_dir)
    else:
        logger.info("Cached audio tokenizer at %s", tokenizer_path)

    if args.prefetch_asr:
        asr_path = _resolve_path(args.asr_model)
        logger.info("Cached ASR model at %s", asr_path)

