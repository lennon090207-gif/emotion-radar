"""Allow `python -m emotion_radar <command>`."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
