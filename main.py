from __future__ import annotations

import argparse
import queue
import re
import sys
import threading
import time
import wave
from dataclasses import dataclass
from typing import Optional

import numpy as np
import requests
import scipy.io.wavfile
import sounddevice as sd
import torch
from pocket_tts import TTSModel

DEFAULT_VOICE = "alba"
DEFAULT_SERVE_URL = "http://localhost:8000"


def log(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {message}")


def get_input_text(cli_text: Optional[str]) -> str:
    if cli_text and cli_text.strip():
        return cli_text.strip()

    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            return piped

    print("Paste your text below. Press ENTER twice to finish:\n")
    lines: list[str] = []
    while True:
        line = input()
        if not line and lines:
            break
        lines.append(line)

    return "\n".join(lines).strip()


def split_text_for_batch(text: str, max_chars: int = 140) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if not sentences:
        stripped = text.strip()
        return [stripped] if stripped else []

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not current:
            current = sentence
            continue

        candidate = f"{current} {sentence}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks


def normalize_audio(audio: torch.Tensor) -> np.ndarray:
    audio_np = audio.detach().cpu().numpy().astype(np.float32, copy=False)
    if audio_np.ndim > 1:
        audio_np = np.squeeze(audio_np)

    peak = float(np.max(np.abs(audio_np))) if audio_np.size else 0.0
    if peak > 1.0:
        audio_np = audio_np / peak

    return audio_np


class FloatChunkPlayer:
    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._playback_error: Optional[Exception] = None

    def start(self) -> None:
        self._thread.start()

    def enqueue(self, chunk: np.ndarray) -> None:
        self._queue.put(chunk)

    def stop(self) -> None:
        self._queue.put(None)
        self._thread.join()

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return

            if self._playback_error is not None:
                continue

            try:
                sd.play(item, self.sample_rate, blocking=True)
            except Exception as exc:  # pragma: no cover (device dependent)
                self._playback_error = exc
                log(f"⚠ Playback disabled: {exc}")


class RawPCMPlayer:
    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self._stream: Optional[sd.RawOutputStream] = None
        self._playback_error: Optional[Exception] = None

    def start(self) -> None:
        try:
            self._stream = sd.RawOutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
            )
            self._stream.start()
        except Exception as exc:  # pragma: no cover (device dependent)
            self._playback_error = exc
            log(f"⚠ Playback disabled: {exc}")

    def write(self, data: bytes) -> None:
        if self._stream is None or self._playback_error is not None:
            return
        try:
            self._stream.write(data)
        except Exception as exc:  # pragma: no cover (device dependent)
            self._playback_error = exc
            log(f"⚠ Playback disabled: {exc}")

    def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None


@dataclass
class RunMetrics:
    backend: str
    strategy: str
    startup_time: float
    text_chunks: int
    audio_units: int
    time_to_first_audio: float
    generation_time: float
    wall_time: float
    output_file: str


def check_serve_health(base_url: str, timeout: float = 2.0) -> bool:
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/health", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def run_serve_simple(
    text: str,
    base_url: str,
    voice: str,
    output_file: str,
    playback: bool,
) -> RunMetrics:
    start_wall = time.perf_counter()
    base_url = base_url.rstrip("/")

    first_audio: Optional[float] = None
    network_chunks = 0
    player: Optional[RawPCMPlayer] = None
    header = bytearray()
    sample_rate: Optional[int] = None
    pcm_data = bytearray()
    remainder = b""

    log("Running SERVE strategy (simple request to /tts)...")

    with requests.post(
        f"{base_url}/tts",
        files={"text": (None, text), "voice_url": (None, voice)},
        stream=True,
        timeout=(10, 300),
    ) as resp:
        resp.raise_for_status()

        for net_chunk in resp.iter_content(chunk_size=4096):
            if not net_chunk:
                continue

            data = net_chunk

            if len(header) < 44:
                need = 44 - len(header)
                header += data[:need]
                data = data[need:]

                if len(header) == 44:
                    sample_rate = int.from_bytes(header[24:28], byteorder="little")
                    channels = int.from_bytes(header[22:24], byteorder="little")
                    bits = int.from_bytes(header[34:36], byteorder="little")
                    if channels != 1 or bits != 16:
                        raise RuntimeError(
                            f"Expected mono 16-bit WAV stream, got channels={channels}, bits={bits}"
                        )

                    if playback:
                        player = RawPCMPlayer(sample_rate)
                        player.start()

            if not data:
                continue

            if first_audio is None:
                first_audio = time.perf_counter() - start_wall
                log(f"First audio ready after {first_audio:.2f}s")

            data = remainder + data
            even_len = len(data) // 2 * 2
            payload = data[:even_len]
            remainder = data[even_len:]

            if payload:
                network_chunks += 1
                pcm_data.extend(payload)
                if player is not None:
                    player.write(payload)

    generation_time = time.perf_counter() - start_wall

    if player is not None:
        player.stop()

    if sample_rate is None:
        raise RuntimeError("No audio returned from server")

    with wave.open(output_file, "wb") as wav_out:
        wav_out.setnchannels(1)
        wav_out.setsampwidth(2)
        wav_out.setframerate(sample_rate)
        wav_out.writeframes(bytes(pcm_data))

    wall_time = time.perf_counter() - start_wall

    return RunMetrics(
        backend="serve",
        strategy="simple",
        startup_time=0.0,
        text_chunks=1,
        audio_units=network_chunks,
        time_to_first_audio=first_audio or 0.0,
        generation_time=generation_time,
        wall_time=wall_time,
        output_file=output_file,
    )


def run_inproc_batch(
    text: str,
    voice: str,
    output_file: str,
    playback: bool,
    max_chars: int,
) -> RunMetrics:
    log("Running INPROC strategy (batch text splitting)...")

    startup_start = time.perf_counter()
    model = TTSModel.load_model()
    voice_state = model.get_state_for_audio_prompt(voice)
    startup_time = time.perf_counter() - startup_start

    text_chunks = split_text_for_batch(text, max_chars=max_chars)
    if not text_chunks:
        raise RuntimeError("No text chunks were generated")

    start_wall = time.perf_counter()
    first_audio: Optional[float] = None

    player: Optional[FloatChunkPlayer] = None
    if playback:
        player = FloatChunkPlayer(model.sample_rate)
        player.start()

    audio_tensors: list[torch.Tensor] = []

    for idx, text_chunk in enumerate(text_chunks, start=1):
        log(f"Generating text chunk {idx}/{len(text_chunks)}")
        audio = model.generate_audio(voice_state, text_chunk)
        audio_tensors.append(audio)

        if first_audio is None:
            first_audio = time.perf_counter() - start_wall
            log(f"First audio ready after {first_audio:.2f}s")

        if player is not None:
            player.enqueue(normalize_audio(audio))

    generation_time = time.perf_counter() - start_wall

    if player is not None:
        player.stop()

    full_audio = torch.cat(audio_tensors, dim=0)
    scipy.io.wavfile.write(output_file, model.sample_rate, full_audio.detach().cpu().numpy())

    wall_time = time.perf_counter() - start_wall

    return RunMetrics(
        backend="inproc",
        strategy="batch",
        startup_time=startup_time,
        text_chunks=len(text_chunks),
        audio_units=len(audio_tensors),
        time_to_first_audio=first_audio or 0.0,
        generation_time=generation_time,
        wall_time=wall_time,
        output_file=output_file,
    )


def print_summary(metrics: RunMetrics) -> None:
    print("\n=== Run summary ===")
    print(f"Backend: {metrics.backend}")
    print(f"Strategy: {metrics.strategy}")
    print(f"Startup time: {metrics.startup_time:.2f}s")
    print(f"Text chunks: {metrics.text_chunks}")
    if metrics.backend == "serve":
        print(f"Network stream chunks: {metrics.audio_units}")
    else:
        print(f"Generated audio segments: {metrics.audio_units}")
    print(f"Time to first audio: {metrics.time_to_first_audio:.2f}s")
    print(f"Generation time: {metrics.generation_time:.2f}s")
    print(f"Total wall time: {metrics.wall_time:.2f}s")
    print(f"Saved WAV: {metrics.output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pocket TTS backend selector")
    parser.add_argument(
        "--backend",
        choices=["serve", "inproc"],
        default=None,
        help="Choose backend explicitly. If omitted: try serve first, then fallback to inproc.",
    )
    parser.add_argument("--text", type=str, help="Text to synthesize.")
    parser.add_argument("--voice", type=str, default=DEFAULT_VOICE, help="Voice prompt (default: alba).")
    parser.add_argument(
        "--serve-url",
        type=str,
        default=DEFAULT_SERVE_URL,
        help="Pocket TTS server URL (default: http://localhost:8000).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=140,
        help="Inproc batch split target size in chars.",
    )
    parser.add_argument(
        "--no-playback",
        action="store_true",
        help="Generate and save WAV without real-time playback.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = get_input_text(args.text)

    if not text:
        raise SystemExit("No input text provided.")

    playback = not args.no_playback

    if args.backend == "inproc":
        metrics = run_inproc_batch(
            text=text,
            voice=args.voice,
            output_file="inproc_batch_output.wav",
            playback=playback,
            max_chars=args.max_chars,
        )
        print_summary(metrics)
        return

    if args.backend == "serve":
        if not check_serve_health(args.serve_url):
            raise SystemExit(
                f"Server not reachable at {args.serve_url}. "
                "Start it manually with: uv run pocket-tts serve --host localhost --port 8000"
            )
        metrics = run_serve_simple(
            text=text,
            base_url=args.serve_url,
            voice=args.voice,
            output_file="serve_simple_output.wav",
            playback=playback,
        )
        print_summary(metrics)
        return

    # backend omitted => default to serve with fallback to inproc
    if check_serve_health(args.serve_url):
        log(f"No backend provided. Using serve at {args.serve_url}.")
        metrics = run_serve_simple(
            text=text,
            base_url=args.serve_url,
            voice=args.voice,
            output_file="serve_simple_output.wav",
            playback=playback,
        )
    else:
        log(f"No backend provided and serve unavailable at {args.serve_url}. Falling back to inproc.")
        metrics = run_inproc_batch(
            text=text,
            voice=args.voice,
            output_file="inproc_batch_output.wav",
            playback=playback,
            max_chars=args.max_chars,
        )

    print_summary(metrics)


if __name__ == "__main__":
    main()
