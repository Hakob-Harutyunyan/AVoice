from __future__ import annotations

import argparse
import base64
import binascii
import io
import logging
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal, Protocol

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from omnivoice import __version__
from omnivoice.utils.lang_map import LANG_NAME_TO_ID

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "/app/model"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_origins(value: str | None) -> list[str]:
    if not value or value.strip() == "*":
        return ["*"]
    return [item.strip() for item in value.split(",") if item.strip()]


def _clean_base64_audio(value: str) -> bytes:
    payload = value.strip()
    if "," in payload and payload.split(",", 1)[0].startswith("data:"):
        payload = payload.split(",", 1)[1]
    try:
        return base64.b64decode(payload, validate=True)
    except binascii.Error as exc:
        raise ValueError("reference audio must be valid base64 or data URI") from exc


@dataclass(slots=True)
class ServerSettings:
    model: str = DEFAULT_MODEL_PATH
    device: str = "auto"
    dtype: str = "auto"
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    log_level: str = "info"
    cors_origins: list[str] | None = None
    preload_model: bool = False
    load_asr: bool = False
    asr_model_name: str = "openai/whisper-large-v3-turbo"

    @classmethod
    def from_env(cls) -> "ServerSettings":
        port = int(os.getenv("OMNIVOICE_PORT", str(DEFAULT_PORT)))
        return cls(
            model=os.getenv("OMNIVOICE_MODEL", DEFAULT_MODEL_PATH),
            device=os.getenv("OMNIVOICE_DEVICE", "auto"),
            dtype=os.getenv("OMNIVOICE_DTYPE", "auto"),
            host=os.getenv("OMNIVOICE_HOST", DEFAULT_HOST),
            port=port,
            log_level=os.getenv("OMNIVOICE_LOG_LEVEL", "info"),
            cors_origins=_parse_origins(os.getenv("OMNIVOICE_CORS_ORIGINS", "*")),
            preload_model=_parse_bool(os.getenv("OMNIVOICE_PRELOAD", "0")),
            load_asr=_parse_bool(os.getenv("OMNIVOICE_LOAD_ASR", "0")),
            asr_model_name=os.getenv(
                "OMNIVOICE_ASR_MODEL",
                "openai/whisper-large-v3-turbo",
            ),
        )


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize.")
    language: str | None = Field(
        default=None,
        description="Optional language code or language name.",
    )
    instruct: str | None = Field(
        default=None,
        description="Optional voice-design instruction.",
    )
    ref_text: str | None = Field(
        default=None,
        description="Transcript for the reference audio used in voice cloning.",
    )
    ref_audio_base64: str | None = Field(
        default=None,
        description="Optional reference audio encoded as base64 or data URI.",
    )
    num_step: int = Field(default=32, ge=1, le=128)
    guidance_scale: float = Field(default=2.0, ge=0.0, le=20.0)
    speed: float = Field(default=1.0, gt=0.0, le=4.0)
    duration: float | None = Field(default=None, gt=0.0)
    denoise: bool = True
    preprocess_prompt: bool = True
    postprocess_output: bool = True


class RuntimeStatus(BaseModel):
    model_path: str
    model_loaded: bool
    device_preference: str
    device_resolved: str | None = None
    dtype_preference: str
    dtype_resolved: str | None = None
    load_asr: bool
    last_error: str | None = None


class SynthesizeResponse(BaseModel):
    audio_base64: str
    sample_rate: int
    duration_seconds: float
    device: str
    model_path: str


@dataclass(slots=True)
class SynthesisResult:
    audio_bytes: bytes
    sample_rate: int
    duration_seconds: float
    device: str
    model_path: str


class RuntimeLike(Protocol):
    settings: ServerSettings

    def get_status(self) -> RuntimeStatus:
        ...

    def list_languages(self) -> list[dict[str, str]]:
        ...

    def synthesize(self, request: SynthesizeRequest) -> SynthesisResult:
        ...

    def maybe_preload(self) -> None:
        ...


class OmniVoiceRuntime:
    def __init__(self, settings: ServerSettings):
        self.settings = settings
        self._model: Any | None = None
        self._torch: Any | None = None
        self._config_cls: Any | None = None
        self._device: str | None = None
        self._dtype_name: str | None = None
        self._last_error: str | None = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    def get_status(self) -> RuntimeStatus:
        return RuntimeStatus(
            model_path=self.settings.model,
            model_loaded=self._model is not None,
            device_preference=self.settings.device,
            device_resolved=self._device,
            dtype_preference=self.settings.dtype,
            dtype_resolved=self._dtype_name,
            load_asr=self.settings.load_asr,
            last_error=self._last_error,
        )

    def maybe_preload(self) -> None:
        if self.settings.preload_model:
            self._ensure_loaded()

    def list_languages(self) -> list[dict[str, str]]:
        languages = [
            {"id": code, "name": name.title()}
            for name, code in LANG_NAME_TO_ID.items()
        ]
        languages.sort(key=lambda item: (item["name"], item["id"]))
        return languages

    def synthesize(self, request: SynthesizeRequest) -> SynthesisResult:
        if not request.text.strip():
            raise ValueError("text must not be blank")
        self._ensure_loaded()
        assert self._model is not None
        assert self._torch is not None
        assert self._config_cls is not None

        prompt = None
        if request.ref_audio_base64:
            prompt = self._build_voice_clone_prompt(
                audio_base64=request.ref_audio_base64,
                ref_text=request.ref_text,
                preprocess_prompt=request.preprocess_prompt,
            )
        elif request.ref_text:
            raise ValueError("ref_text requires ref_audio_base64 as well")

        generation_config = self._config_cls(
            num_step=request.num_step,
            guidance_scale=request.guidance_scale,
            denoise=request.denoise,
            preprocess_prompt=request.preprocess_prompt,
            postprocess_output=request.postprocess_output,
        )

        with self._infer_lock:
            audios = self._model.generate(
                text=request.text,
                language=request.language,
                voice_clone_prompt=prompt,
                instruct=request.instruct,
                duration=request.duration,
                speed=request.speed,
                generation_config=generation_config,
            )

        audio = audios[0]
        wav_bytes = self._encode_wav(audio, self._model.sampling_rate)
        duration_seconds = float(len(audio) / self._model.sampling_rate)
        return SynthesisResult(
            audio_bytes=wav_bytes,
            sample_rate=int(self._model.sampling_rate),
            duration_seconds=duration_seconds,
            device=self._device or self.settings.device,
            model_path=self.settings.model,
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                torch_module = import_module("torch")
                omnivoice_module = import_module("omnivoice")
                model_cls = getattr(omnivoice_module, "OmniVoice")
                config_cls = getattr(omnivoice_module, "OmniVoiceGenerationConfig")
                device = self._resolve_device(torch_module)
                dtype_name, dtype_value = self._resolve_dtype(torch_module, device)
                logger.info(
                    "Loading OmniVoice model from %s on %s (%s)",
                    self.settings.model,
                    device,
                    dtype_name,
                )
                model = model_cls.from_pretrained(
                    self.settings.model,
                    device_map=device,
                    dtype=dtype_value,
                    load_asr=self.settings.load_asr,
                    asr_model_name=self.settings.asr_model_name,
                )
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                raise

            self._torch = torch_module
            self._config_cls = config_cls
            self._model = model
            self._device = device
            self._dtype_name = dtype_name
            self._last_error = None

    def _resolve_device(self, torch_module: Any) -> str:
        choice = self.settings.device.strip().lower()
        if choice == "auto":
            if torch_module.cuda.is_available():
                return "cuda"
            mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
            if mps_backend is not None and mps_backend.is_available():
                return "mps"
            return "cpu"
        if choice == "cuda":
            if not torch_module.cuda.is_available():
                raise RuntimeError("OMNIVOICE_DEVICE=cuda was requested but CUDA is unavailable")
            return "cuda"
        if choice == "mps":
            mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
            if mps_backend is None or not mps_backend.is_available():
                raise RuntimeError("OMNIVOICE_DEVICE=mps was requested but MPS is unavailable")
            return "mps"
        if choice == "cpu":
            return "cpu"
        raise RuntimeError(f"Unsupported device choice: {self.settings.device}")

    def _resolve_dtype(self, torch_module: Any, device: str) -> tuple[str, Any]:
        aliases = {
            "fp16": "float16",
            "half": "float16",
            "fp32": "float32",
            "float": "float32",
            "bf16": "bfloat16",
        }
        choice = self.settings.dtype.strip().lower()
        if choice == "auto":
            choice = "float16" if device == "cuda" else "float32"
        choice = aliases.get(choice, choice)
        valid = {"float16", "float32", "bfloat16"}
        if choice not in valid:
            raise RuntimeError(f"Unsupported dtype choice: {self.settings.dtype}")
        return choice, getattr(torch_module, choice)

    def _build_voice_clone_prompt(
        self,
        audio_base64: str,
        ref_text: str | None,
        preprocess_prompt: bool,
    ) -> Any:
        assert self._model is not None
        assert self._torch is not None
        waveform, sample_rate = self._decode_audio(audio_base64)
        return self._model.create_voice_clone_prompt(
            ref_audio=(self._torch.from_numpy(waveform), sample_rate),
            ref_text=ref_text,
            preprocess_prompt=preprocess_prompt,
        )

    def _decode_audio(self, audio_base64: str) -> tuple[Any, int]:
        import numpy as np
        import soundfile as sf

        raw_bytes = _clean_base64_audio(audio_base64)
        audio_buffer = io.BytesIO(raw_bytes)
        waveform, sample_rate = sf.read(audio_buffer, dtype="float32", always_2d=False)
        if waveform.ndim == 2:
            waveform = np.transpose(waveform)
        return waveform, int(sample_rate)

    def _encode_wav(self, audio: Any, sample_rate: int) -> bytes:
        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV")
        return buffer.getvalue()


def create_app(
    settings: ServerSettings | None = None,
    runtime: RuntimeLike | None = None,
) -> FastAPI:
    if settings is None:
        if runtime is not None and hasattr(runtime, "settings"):
            settings = runtime.settings
        else:
            settings = ServerSettings.from_env()
    runtime = runtime or OmniVoiceRuntime(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        runtime.maybe_preload()
        yield

    app = FastAPI(
        title="AVoice OmniVoice API",
        version=__version__,
        summary="Local HTTP API for OmniVoice speech generation.",
        lifespan=lifespan,
    )

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/healthz", response_model=RuntimeStatus)
    def healthz() -> RuntimeStatus:
        return runtime.get_status()

    @app.get("/v1/runtime", response_model=RuntimeStatus)
    def runtime_status() -> RuntimeStatus:
        return runtime.get_status()

    @app.get("/v1/languages")
    def languages() -> dict[str, list[dict[str, str]]]:
        return {"languages": runtime.list_languages()}

    @app.post("/v1/audio/speech")
    def speech(request: SynthesizeRequest) -> Response:
        try:
            result = runtime.synthesize(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("speech generation failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        headers = {
            "X-OmniVoice-Device": result.device,
            "X-OmniVoice-Sample-Rate": str(result.sample_rate),
            "X-OmniVoice-Model": result.model_path,
        }
        return Response(content=result.audio_bytes, media_type="audio/wav", headers=headers)

    @app.post("/v1/audio/speech/json", response_model=SynthesizeResponse)
    def speech_json(request: SynthesizeRequest) -> SynthesizeResponse:
        try:
            result = runtime.synthesize(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("speech generation failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return SynthesizeResponse(
            audio_base64=base64.b64encode(result.audio_bytes).decode("ascii"),
            sample_rate=result.sample_rate,
            duration_seconds=result.duration_seconds,
            device=result.device,
            model_path=result.model_path,
        )

    return app


def build_parser() -> argparse.ArgumentParser:
    env = ServerSettings.from_env()
    parser = argparse.ArgumentParser(
        prog="omnivoice-api",
        description="Serve OmniVoice inference as a local HTTP API.",
    )
    parser.add_argument("--model", default=env.model, help="Local model path or HuggingFace repo id.")
    parser.add_argument(
        "--device",
        default=env.device,
        help="Device selection: auto, cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--dtype",
        default=env.dtype,
        help="Precision selection: auto, float16, float32, or bfloat16.",
    )
    parser.add_argument("--host", default=env.host, help="Bind host.")
    parser.add_argument("--port", type=int, default=env.port, help="Bind port.")
    parser.add_argument("--log-level", default=env.log_level, help="Uvicorn log level.")
    parser.add_argument(
        "--preload-model",
        action="store_true",
        default=env.preload_model,
        help="Load the model during server startup instead of on first request.",
    )
    parser.add_argument(
        "--load-asr",
        action="store_true",
        default=env.load_asr,
        help="Load the Whisper ASR helper at startup for reference-audio transcription.",
    )
    parser.add_argument(
        "--asr-model-name",
        default=env.asr_model_name,
        help="Whisper model to use when ref_text is omitted.",
    )
    parser.add_argument(
        "--cors-origins",
        default=",".join(env.cors_origins or ["*"]),
        help="Comma-separated CORS allowlist or *.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = ServerSettings(
        model=args.model,
        device=args.device,
        dtype=args.dtype,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        cors_origins=_parse_origins(args.cors_origins),
        preload_model=args.preload_model,
        load_asr=args.load_asr,
        asr_model_name=args.asr_model_name,
    )

    import uvicorn

    uvicorn.run(
        create_app(settings=settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )


app = create_app()
