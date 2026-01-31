import logging


def setup_logging(level: str) -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())
    while root.handlers:
        root.handlers.pop()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    root.addHandler(stream)
