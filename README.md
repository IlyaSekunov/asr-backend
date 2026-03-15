# Speech Processing API

An asynchronous speech-to-text service built with FastAPI, Faster-Whisper, and Redis Queue (RQ). Audio files are uploaded via a REST API, preprocessed, and transcribed in a separate worker process. Results are retrieved by polling.

---

## Architecture

![ASR System Architecture](assets/asr_system.png)

The API and worker run as **separate processes**. The Whisper model is loaded only in the worker, keeping the API process lightweight.

### Preprocessing pipeline

Preprocessors are applied in this order before transcription:

1. **Loudness Normalizer** — LUFS, Peak, or RMS normalisation (configurable)
2. **Noise Reducer** — stationary or adaptive noise reduction via `noisereduce`

Either step can be disabled independently via environment variables.

---

## Requirements

- Docker and Docker Compose
- NVIDIA GPU + drivers (GPU build only)

---

## Running

**CPU build**
```bash
docker compose up --build
```

**GPU build**
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

The API will be available at `http://localhost:8000`.

---

## API

### `POST /api/v1/transcribe/`

Upload an audio file for transcription. Returns a `task_id` immediately.

**Accepted formats:** `.mp3`, `.wav`

```bash
curl -X POST http://localhost:8000/api/v1/transcribe/ \
  -F "file=@recording.wav"
```

**Response**
```json
{
  "task_id": "e3b0c442-98fc-4c14-9afb-ed8b1b3b0a1f"
}
```

---

### `GET /api/v1/transcribe/{task_id}`

Poll for the result of a transcription job.

```bash
curl http://localhost:8000/api/v1/transcribe/e3b0c442-98fc-4c14-9afb-ed8b1b3b0a1f
```

**Response — job in progress**
```json
{
  "status": "QUEUED",
  "result": null
}
```

**Response — job complete**
```json
{
  "status": "READY",
  "result": {
    "text": "Hello, this is a transcription.",
    "language": "en",
    "language_probability": 0.99
  }
}
```

**Task statuses**

| Status    | Meaning                              |
|-----------|--------------------------------------|
| `QUEUED`  | Waiting in the queue                 |
| `STARTED` | Worker is actively processing        |
| `READY`   | Transcription complete, result available |
| `FAILED`  | Processing failed                    |

---

## Configuration

All settings are read from environment variables (or a `.env` file). Defaults are production-ready for CPU inference.

### Model

| Variable                       | Default     | Description                                              |
|--------------------------------|-------------|----------------------------------------------------------|
| `MODEL_SIZE`                   | `small`     | Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3` |
| `COMPUTE_DEVICE`               | `cpu`       | `cpu` or `cuda`                                          |
| `QUANTIZATION`                 | `int8`      | `int8`, `int8_float16`, `float16`, `float32`             |
| `VAD_ENABLED`                  | `true`      | Voice activity detection — skips silent segments         |
| `TARGET_SAMPLE_RATE`           | `16000`     | Audio sample rate in Hz (Whisper minimum)                |

### Loudness normalisation

| Variable                       | Default     | Description                                              |
|--------------------------------|-------------|----------------------------------------------------------|
| `LOUDNESS_NORMALIZATION_ENABLED` | `true`    | Enable/disable loudness normalisation                    |
| `LOUDNESS_METHOD`              | `lufs`      | `lufs`, `peak`, or `rms`                                 |
| `LOUDNESS_TARGET`              | `-23.0`     | Target loudness in LUFS (EBU R128 broadcast standard)    |

### Noise reduction

| Variable                       | Default     | Description                                              |
|--------------------------------|-------------|----------------------------------------------------------|
| `DENOISING_ENABLED`            | `true`      | Enable/disable noise reduction                           |
| `DENOISE_STATIONARY`           | `true`      | `true` for constant noise profile; `false` for adaptive  |
| `DENOISE_PROP_DECREASE`        | `0.9`       | Fraction of noise energy to remove [0.0, 1.0]            |

### Redis

| Variable                       | Default     | Description                                              |
|--------------------------------|-------------|----------------------------------------------------------|
| `REDIS_HOST`                   | `localhost` |                                                          |
| `REDIS_PORT`                   | `6379`      |                                                          |
| `REDIS_QUEUE`                  | `asr`       | RQ queue name                                            |
| `REDIS_QUEUE_RESULT_TTL`       | `300`       | Seconds to retain a completed result in Redis            |
| `REDIS_QUEUE_FAILURE_TTL`      | `300`       | Seconds to retain a failed job for inspection            |

---

## Project structure

```
app/
├── api/
│   └── routes/
│       └── transcription.py     # REST endpoints
├── asyncqueue/
│   ├── redis_queue.py           # Redis connection and RQ queue
│   ├── redis_queue_manager.py   # Job lifecycle helpers
│   ├── tasks.py                 # RQ task — runs in worker process only
│   └── worker.py                # Worker entry point
├── pipeline/
│   ├── asr_pipeline.py          # Chains preprocessors → transcriber
│   └── asr_pipeline_factory.py  # Builds the pipeline from settings
├── preprocessing/
│   ├── audio_preprocessor.py    # Abstract base class
│   ├── loudness_normalizer.py   # LUFS / Peak / RMS normalisation
│   └── noise_reducer.py         # Stationary and adaptive denoising
├── schemas/
│   └── transcription.py         # Pydantic request/response models
├── transcribers/
│   ├── audio_transcriber.py     # Abstract base class
│   ├── transcription_result.py  # Immutable result dataclass
│   └── whisper_transcriber.py   # Faster-Whisper implementation
├── util/
│   ├── io.py                    # Audio file streaming and loading
│   └── tasks.py                 # Task ID generation
├── config.py                    # All settings (pydantic-settings)
└── main.py                      # FastAPI app factory
```