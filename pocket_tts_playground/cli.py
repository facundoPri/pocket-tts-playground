import argparse

from .constants import DEFAULT_SERVE_URL, DEFAULT_VOICE


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
