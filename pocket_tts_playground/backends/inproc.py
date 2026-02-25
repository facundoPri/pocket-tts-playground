from __future__ import annotations

import time
from typing import Optional

import scipy.io.wavfile
import torch
from pocket_tts import TTSModel

from ..audio import FloatChunkPlayer, normalize_audio
from ..console import log
from ..metrics import RunMetrics
from ..text import split_text_for_batch


def run_inproc_batch(
    text: str,
    voice: str,
    output_file: str,
    playback: bool,
    max_chars: int,
    startup_time_offset: float = 0.0,
) -> RunMetrics:
    log("Running INPROC strategy (batch text splitting)...")

    timing: dict[str, float] = {}

    model_start = time.perf_counter()
    model = TTSModel.load_model()
    timing["inproc.model_load"] = time.perf_counter() - model_start

    voice_start = time.perf_counter()
    voice_state = model.get_state_for_audio_prompt(voice)
    timing["inproc.voice_prep"] = time.perf_counter() - voice_start

    split_start = time.perf_counter()
    text_chunks = split_text_for_batch(text, max_chars=max_chars)
    timing["inproc.text_split"] = time.perf_counter() - split_start

    if not text_chunks:
        raise RuntimeError("No text chunks were generated")

    startup_time = startup_time_offset + timing["inproc.model_load"] + timing["inproc.voice_prep"]

    start_wall = time.perf_counter()
    first_audio: Optional[float] = None

    player: Optional[FloatChunkPlayer] = None
    if playback:
        player_start = time.perf_counter()
        player = FloatChunkPlayer(model.sample_rate)
        player.start()
        timing["inproc.player_start"] = time.perf_counter() - player_start

    audio_tensors: list[torch.Tensor] = []

    for idx, text_chunk in enumerate(text_chunks, start=1):
        chunk_start = time.perf_counter()
        log(f"Generating text chunk {idx}/{len(text_chunks)}")
        audio = model.generate_audio(voice_state, text_chunk)
        timing[f"inproc.generate_chunk_{idx}"] = time.perf_counter() - chunk_start
        audio_tensors.append(audio)

        if first_audio is None:
            first_audio = time.perf_counter() - start_wall
            timing["inproc.first_audio"] = first_audio
            log(f"First audio ready after {first_audio:.2f}s")

        if player is not None:
            player.enqueue(normalize_audio(audio))

    generation_time = time.perf_counter() - start_wall
    timing["inproc.generation_loop"] = generation_time

    if player is not None:
        stop_start = time.perf_counter()
        player.stop()
        timing["inproc.player_stop_wait"] = time.perf_counter() - stop_start

    save_start = time.perf_counter()
    full_audio = torch.cat(audio_tensors, dim=0)
    scipy.io.wavfile.write(output_file, model.sample_rate, full_audio.detach().cpu().numpy())
    timing["inproc.write_wav"] = time.perf_counter() - save_start

    wall_time = time.perf_counter() - start_wall

    notes: list[str] = []
    if startup_time_offset > 0:
        notes.append("Startup includes serve health probe time before fallback.")

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
        timings=timing,
        notes=notes,
    )
