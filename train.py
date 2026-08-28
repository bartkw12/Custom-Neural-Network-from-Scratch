import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from custom_nn.cli import main as cli_main

_KNOWN_SUBCOMMANDS = {"custom", "pytorch", "compare", "analyze"}


def _compat_argv(argv: list[str]) -> list[str]:
    if not argv:
        return ["custom"]

    if argv[0] in _KNOWN_SUBCOMMANDS:
        return argv

    return ["custom", *argv]


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    return cli_main(_compat_argv(arguments))


if __name__ == "__main__":
    raise SystemExit(main())