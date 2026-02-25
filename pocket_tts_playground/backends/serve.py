from __future__ import annotations

import time
import wave
from typing import Optional

import requests

from ..audio import RawPCMPlayer
from ..console import log
from ..metrics import RunMetrics


def probe_serve_health(base_url: str, timeout: float = 2.0) -> tuple[bool, float, Optional[str]]:
    start = time.perf_counter()
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/health", timeout=timeout)
        ok = resp.status_code == 200
        return ok, time.perf_counter() - start, None if ok else f"status={resp.status_code}"
    except Exception as exc:
        return False, time.perf_counter() - start, str(exc)


def check_serve_health(base_url: str, timeout: float = 2.0) -> bool:
    ok, _, _ = probe_serve_health(base_url, timeout=timeout)
    return ok


def run_serve_simple(
    text: str,
    base_url: str,
    voice: str,
    output_file: str,
    playback: bool,
    startup_time: float = 0.0,
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

    timing: dict[str, float] = {}

    log("Running SERVE strategy (simple request to /tts)...")

    request_start = time.perf_counter()
    with requests.post(
        f"{base_url}/tts",
        files={"text": (None, text), "voice_url": (None, voice)},
        stream=True,
        timeout=(10, 300),
    ) as resp:
        resp.raise_for_status()
        timing["serve.request_open"] = time.perf_counter() - request_start

        for net_chunk in resp.iter_content(chunk_size=4096):
            if not net_chunk:
                continue

            if "serve.first_network_chunk" not in timing:
                timing["serve.first_network_chunk"] = time.perf_counter() - request_start

            data = net_chunk

            if len(header) < 44:
                need = 44 - len(header)
                header += data[:need]
                data = data[need:]

                if len(header) == 44:
                    timing["serve.wav_header_ready"] = time.perf_counter() - request_start
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
                        timing["serve.player_start"] = time.perf_counter() - request_start

            if not data:
                continue

            if first_audio is None:
                first_audio = time.perf_counter() - start_wall
                timing["serve.first_audio_payload"] = time.perf_counter() - request_start
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

    save_start = time.perf_counter()
    with wave.open(output_file, "wb") as wav_out:
        wav_out.setnchannels(1)
        wav_out.setsampwidth(2)
        wav_out.setframerate(sample_rate)
        wav_out.writeframes(bytes(pcm_data))
    timing["serve.write_wav"] = time.perf_counter() - save_start

    wall_time = time.perf_counter() - start_wall
    timing["serve.total_request_stream"] = generation_time

    return RunMetrics(
        backend="serve",
        strategy="simple",
        startup_time=startup_time,
        text_chunks=1,
        audio_units=network_chunks,
        time_to_first_audio=first_audio or 0.0,
        generation_time=generation_time,
        wall_time=wall_time,
        output_file=output_file,
        timings=timing,
        notes=["Serve startup time measures health probe only; server boot time is external."],
    )
