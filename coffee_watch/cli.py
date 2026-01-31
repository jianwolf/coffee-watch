from __future__ import annotations

import asyncio
import sys

from .config import build_settings, load_config_file, parse_args
from .runner import run


def main() -> None:
    try:
        args = parse_args()
        config = load_config_file(args.config)
        settings = build_settings(args, config)
        exit_code = asyncio.run(run(settings))
    except KeyboardInterrupt:
        exit_code = 130
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
