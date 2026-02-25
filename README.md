# Pocket TTS Playground (simplified)

This version follows your findings:

- **`serve` backend** uses only **simple mode** (one `/tts` request, no text splitting in our code).
- **`inproc` backend** uses only **batch mode** (text splitting + multiple `generate_audio` calls).
- CLI now only asks for:
  - backend (optional)
  - text

If backend is omitted:
1. Try `serve` first (default)
2. If server is not reachable, fallback to `inproc`

## Setup

```bash
uv sync
```

## Start server manually (for `serve`)

Default server URL expected by the app:

- `http://localhost:8000`

Start it with:

```bash
uv run pocket-tts serve --host localhost --port 8000 --voice alba
```

## Usage

### 1) Auto mode (recommended)

```bash
uv run python main.py --text "Hello world"
```

- Uses `serve` if available
- Falls back to `inproc` if not

### 2) Force serve

```bash
uv run python main.py --backend serve --text "Hello world"
```

### 3) Force inproc

```bash
uv run python main.py --backend inproc --text "Hello world"
```

## Useful flags

```bash
--serve-url http://localhost:8000
--max-chars 140         # only affects inproc (batch splitting)
--no-playback
--voice alba
--timing-report          # include timing breakdown section
```

Timing notes:
- `startup_time` for `serve` is the `/health` probe time only (server boot is external).
- `startup_time` for `inproc` is model load + voice prep (plus probe time if fallback from serve happened).
- The phase-by-phase timing breakdown is shown only when `--timing-report` is enabled.

## Output files

- `serve_simple_output.wav`
- `inproc_batch_output.wav`

## Code structure

- `main.py` - thin orchestrator
- `pocket_tts_playground/cli.py` - CLI args
- `pocket_tts_playground/backends/serve.py` - serve/simple path
- `pocket_tts_playground/backends/inproc.py` - inproc/batch path
- `pocket_tts_playground/audio.py` - playback utilities
- `pocket_tts_playground/text.py` - text input + batch split
- `pocket_tts_playground/metrics.py` - run metrics + summary
