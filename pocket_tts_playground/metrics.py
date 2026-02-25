from dataclasses import dataclass


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
