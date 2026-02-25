import time


def log(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {message}")
