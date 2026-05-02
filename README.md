# AVoice

AVoice is an Armenian text-to-speech runtime. The GitHub repository contains the
small inference package and API server; the model weights are hosted separately
so the repository stays clean.

Model weights:

- Hugging Face: [`Hak5/AVoice-TTS`](https://huggingface.co/Hak5/AVoice-TTS)
- Kaggle Model: [`hakdevelopment/avoice-omnivoice-hy`](https://www.kaggle.com/models/hakdevelopment/avoice-omnivoice-hy)

## Quick Start

Use Python 3.10 or newer. A CUDA GPU is recommended for inference.

```bash
git clone https://github.com/Hakob-Harutyunyan/Avoice.git
cd Avoice

python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -e .
```

On Windows PowerShell:

```powershell
git clone https://github.com/Hakob-Harutyunyan/Avoice.git
cd Avoice

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -U pip
pip install -e .
```

If Hugging Face rate limits you, login once:

```bash
huggingface-cli login
```

## Generate A WAV

```bash
avoice \
  --model Hak5/AVoice-TTS \
  --text "Բարեւ, սա հայկական խոսքի սինթեզի փորձարկում է։" \
  --language hy \
  --output sample.wav
```

The first run downloads the model from Hugging Face. After that, it is cached on
your machine.

Useful optional flags:

- `--device cuda` to force GPU inference.
- `--device cpu` if you do not have a supported GPU.
- `--duration 4` to force the generated audio to be about 4 seconds long.
- `--guidance_scale 2.0` to control how strongly generation follows the text and
  voice prompt.

## Use A Kaggle Model Folder

If you downloaded the Kaggle model instead of using Hugging Face, pass the local
folder path:

```bash
avoice \
  --model /kaggle/input/models/hakdevelopment/avoice-omnivoice-hy/pytorch/armenian-tts/9 \
  --text "Այսօր 2026 թվականի մայիսի մեկն է։" \
  --language hy \
  --device cuda \
  --output sample.wav
```

## Run The API

Start a local server:

```bash
omnivoice-api \
  --model Hak5/AVoice-TTS \
  --device cuda \
  --host 0.0.0.0 \
  --port 8000
```

Send a request:

```bash
curl -X POST "http://127.0.0.1:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"text":"Բարեւ, սա փորձարկում է։","language":"hy"}' \
  --output sample.wav
```

On Windows PowerShell, put the JSON in a UTF-8 file so Armenian text is not
corrupted by shell quoting:

```powershell
@'
{
  "text": "Բարեւ, սա փորձարկում է։",
  "language": "hy"
}
'@ | Set-Content request.json -Encoding utf8

curl.exe -f -X POST "http://127.0.0.1:8000/v1/audio/speech" `
  -H "Content-Type: application/json; charset=utf-8" `
  --data-binary "@request.json" `
  --output sample.wav
```

## Docker

There are two Dockerfiles:

- `Dockerfile` builds the CUDA/GPU image.
- `Dockerfile.cpu` builds a smaller CPU-only image.

Build the GPU image:

```bash
docker build -t avoice-api .
```

Run the GPU image from Hugging Face:

```bash
docker run --gpus all --rm -p 8000:8000 \
  -e OMNIVOICE_MODEL=Hak5/AVoice-TTS \
  -e OMNIVOICE_DEVICE=cuda \
  avoice-api
```

Build the CPU-only image:

```bash
docker build -f Dockerfile.cpu -t avoice-api-cpu .
```

Run the CPU-only image from Hugging Face:

```bash
docker run --rm -p 8000:8000 \
  -e OMNIVOICE_MODEL=Hak5/AVoice-TTS \
  avoice-api-cpu
```

Run either image with a local model folder:

```bash
docker run --gpus all --rm -p 8000:8000 \
  -v /path/to/avoice-model:/app/model:ro \
  -e OMNIVOICE_MODEL=/app/model \
  -e OMNIVOICE_DEVICE=cuda \
  avoice-api
```

For the CPU image, remove `--gpus all` and use `avoice-api-cpu`:

```bash
docker run --rm -p 8000:8000 \
  -v /path/to/avoice-model:/app/model:ro \
  -e OMNIVOICE_MODEL=/app/model \
  avoice-api-cpu
```

Health check:

```bash
curl http://127.0.0.1:8000/healthz
```

## Hardware Notes

- NVIDIA T4, L4, A10, A100, RTX 20xx+, or newer GPUs are recommended.
- Kaggle P100 currently fails with recent PyTorch CUDA builds because the GPU is
  too old for the installed CUDA kernels. Use Kaggle T4 instead.
- CPU inference can work for debugging, but it is slow.

## Troubleshooting

If you see `CUDA error: no kernel image is available for execution on the
device`, your PyTorch build does not support your GPU. Use a newer GPU or install
a PyTorch build compatible with your hardware.

If you see Hugging Face unauthenticated request warnings, run
`huggingface-cli login` or set `HF_TOKEN`. The model is public, but login gives
better rate limits.

If you see `pydub` `SyntaxWarning` messages about invalid escape sequences, they
are harmless warnings from a dependency.

If the API returns an empty or invalid WAV, check the server logs first. Most
often the model failed to load, CUDA is unavailable, or the request JSON was
corrupted by shell quoting.

## License

AVoice is released under GPL-2.0. Some bundled or referenced components have
their own licenses; see `THIRD_PARTY_NOTICES.md`.
