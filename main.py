from __future__ import annotations

from pocket_tts_playground.backends import check_serve_health, run_inproc_batch, run_serve_simple
from pocket_tts_playground.cli import parse_args
from pocket_tts_playground.console import log
from pocket_tts_playground.constants import INPROC_OUTPUT_FILE, SERVE_OUTPUT_FILE
from pocket_tts_playground.metrics import print_summary
from pocket_tts_playground.text import get_input_text


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
            output_file=INPROC_OUTPUT_FILE,
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
            output_file=SERVE_OUTPUT_FILE,
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
            output_file=SERVE_OUTPUT_FILE,
            playback=playback,
        )
    else:
        log(f"No backend provided and serve unavailable at {args.serve_url}. Falling back to inproc.")
        metrics = run_inproc_batch(
            text=text,
            voice=args.voice,
            output_file=INPROC_OUTPUT_FILE,
            playback=playback,
            max_chars=args.max_chars,
        )

    print_summary(metrics)


if __name__ == "__main__":
    main()
