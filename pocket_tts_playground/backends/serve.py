from __future__ import annotations

import time
import wave
from typing import Optional

import requests

from ..audio import RawPCMPlayer
from ..console import log
from ..metrics import RunMetrics


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
