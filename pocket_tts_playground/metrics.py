from dataclasses import dataclass, field


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
    timings: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def print_summary(metrics: RunMetrics, show_timing_breakdown: bool = False) -> None:
    print("\n=== Run summary ===")
    print(f"Backend: {metrics.backend}")
    print(f"Strategy: {metrics.strategy}")
    print(f"Startup time: {metrics.startup_time:.3f}s")
    print(f"Text chunks: {metrics.text_chunks}")
    if metrics.backend == "serve":
        print(f"Network stream chunks: {metrics.audio_units}")
    else:
        print(f"Generated audio segments: {metrics.audio_units}")
    print(f"Time to first audio: {metrics.time_to_first_audio:.3f}s")
    print(f"Generation time: {metrics.generation_time:.3f}s")
    print(f"Total wall time: {metrics.wall_time:.3f}s")
    print(f"Saved WAV: {metrics.output_file}")

    if metrics.notes:
        print("Notes:")
        for note in metrics.notes:
            print(f"- {note}")

    if show_timing_breakdown and metrics.timings:
        print("\nTiming breakdown:")
        for key in sorted(metrics.timings):
            print(f"- {key}: {metrics.timings[key]:.3f}s")
