"""Backward-compatible launcher for the shared GLLS demo in weld mode."""

from __future__ import annotations

import sys

from glls.cli.visualize import main as shared_main


def main() -> None:
    args = list(sys.argv[1:])
    if not any(arg == "--initial_dataset" or arg.startswith("--initial_dataset=") for arg in args):
        args.extend(["--initial_dataset", "weld"])
    if not any(arg in {"--port", "--start_port"} or arg.startswith(("--port=", "--start_port=")) for arg in args):
        args.extend(["--start_port", "7865"])
    sys.argv = [sys.argv[0], *args]
    shared_main()


if __name__ == "__main__":
    main()
