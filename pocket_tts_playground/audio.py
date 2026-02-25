from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd
import torch

from .console import log


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
