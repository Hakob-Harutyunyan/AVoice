# AVoice

AVoice is an Armenian text-to-speech runtime for the hosted AVoice OmniVoice
model. This repository is intentionally small: it contains only the code needed
to load the model, synthesize speech, and serve a local HTTP API.

Model weights are hosted outside this repository:

- Hugging Face: `Hak5/AVoice`
- Kaggle Model: `hakdevelopment/avoice-omnivoice-hy`

## Install

Use Python 3.10+.

```bash
git clone https://github.com/Hakob-Harutyunyan/Avoice.git
cd Avoice
pip install -U pip
pip install -e .
```

If your model repository is private, login first:

```bash
huggingface-cli login
```

## Generate Audio

```bash
omnivoice-infer \
  --model Hak5/AVoice \
  --text "Բարեւ աշխարհ։" \
  --language hy \
  --duration 4 \
  --output sample.wav
```

Use a local Kaggle model path the same way:

```bash
omnivoice-infer \
  --model /kaggle/input/models/hakdevelopment/avoice-omnivoice-hy/pytorch/armenian-tts/6 \
  --text "Բարեւ աշխարհ։" \
  --language hy \
  --duration 4 \
  --output sample.wav
```

## Run The API

```bash
omnivoice-api --model Hak5/AVoice --device cuda --host 0.0.0.0 --port 8000
```

Request audio:

```bash
curl -X POST "http://127.0.0.1:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"text":"Բարեւ աշխարհ։","language":"hy","duration":4}' \
  --output sample.wav
```

On Windows PowerShell, write JSON from a file so Armenian text stays UTF-8:

```powershell
@'
{
  "text": "Բարեւ աշխարհ։",
  "language": "hy",
  "duration": 4
}
'@ | Set-Content request.json -Encoding utf8

curl.exe -f -X POST "http://127.0.0.1:8000/v1/audio/speech" `
  -H "Content-Type: application/json; charset=utf-8" `
  --data-binary "@request.json" `
  --output sample.wav
```

## Docker

Build:

```bash
docker build -t avoice-api .
```

Run from Hugging Face:

```bash
docker run --gpus all --rm -p 8000:8000 \
  -e OMNIVOICE_MODEL=Hak5/AVoice \
  -e HF_TOKEN=your_huggingface_token \
  -e OMNIVOICE_DEVICE=cuda \
  avoice-api
```

Run with a local model folder:

```bash
docker run --gpus all --rm -p 8000:8000 \
  -v /path/to/avoice-model:/app/model:ro \
  -e OMNIVOICE_MODEL=/app/model \
  -e OMNIVOICE_DEVICE=cuda \
  avoice-api
```

Health check:

```bash
curl http://127.0.0.1:8000/healthz
```

## Notes

- This repository is for inference and serving only.
- The model and code are released under GPL-2.0 unless noted otherwise in
  `THIRD_PARTY_NOTICES.md`.
