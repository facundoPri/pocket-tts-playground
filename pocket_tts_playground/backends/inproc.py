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
