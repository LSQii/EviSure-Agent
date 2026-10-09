"""Unified command-line entry point for the supported video QA datasets.

This module only dispatches to the existing dataset runner. All dataset loading,
inference, caching, resuming, and output logic remains in the original runner.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class RunnerSpec:
    module: str
    description: str
    default_args: tuple[str, ...] = ()


RUNNERS = {
    "videomme": RunnerSpec(
        module="scripts.run_our_method_videomme",
        description="Video-MME nested JSON annotations.",
        default_args=("--dataset_format", "videomme"),
    ),
    "longvideobench": RunnerSpec(
        module="scripts.run_our_method_longvideobench",
        description="LongVideoBench JSON or JSONL annotations.",
    ),
    "mlvu": RunnerSpec(
        module="scripts.run_our_method_mlvu",
        description="MLVU task JSON directory and task video folders.",
    ),
}

DATASET_ALIASES = {
    "videomme": "videomme",
    "video-mme": "videomme",
    "video_mme": "videomme",
    "longvideobench": "longvideobench",
    "long-video-bench": "longvideobench",
    "longvideo_bench": "longvideobench",
    "lvb": "longvideobench",
    "mlvu": "mlvu",
}


def build_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    dataset_lines = "\n".join(
        f"  {name:<15} {spec.description}" for name, spec in RUNNERS.items()
    )
    return argparse.ArgumentParser(
        add_help=add_help,
        description=(
            "Run EviSure Agent on a supported dataset. Arguments after "
            "--dataset are forwarded to the selected dataset runner."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Datasets:\n"
            f"{dataset_lines}\n\n"
            "Examples:\n"
            "  python main.py --dataset videomme --help\n"
            "  python main.py --dataset longvideobench --help\n"
            "  python main.py --dataset mlvu --help"
        ),
    )


def normalize_dataset(value: str) -> str:
    normalized = DATASET_ALIASES.get(value.strip().lower())
    if normalized is None:
        supported = ", ".join(RUNNERS)
        raise argparse.ArgumentTypeError(
            f"unsupported dataset {value!r}; choose one of: {supported}"
        )
    return normalized


def contains_option(arguments: list[str], option: str) -> bool:
    """Return whether an argparse option is present in split or --key=value form."""

    return any(value == option or value.startswith(option + "=") for value in arguments)


def parse_dispatch_args(argv: list[str]) -> tuple[str, list[str]]:
    has_dataset = contains_option(argv, "--dataset")
    if not has_dataset and any(value in {"-h", "--help"} for value in argv):
        help_parser = build_parser()
        help_parser.add_argument(
            "--dataset", metavar="NAME",
            help="Dataset adapter: videomme, longvideobench, or mlvu.",
        )
        help_parser.parse_args(argv)

    parser = build_parser(add_help=False)
    parser.add_argument(
        "--dataset",
        type=normalize_dataset,
        metavar="NAME",
        help="Dataset adapter: videomme, longvideobench, or mlvu.",
    )
    namespace, forwarded = parser.parse_known_args(argv)
    if namespace.dataset is None:
        parser.error("--dataset is required")
    return namespace.dataset, forwarded


def dispatch(dataset: str, forwarded: list[str]) -> int:
    spec = RUNNERS[dataset]
    arguments = list(forwarded)
    for index in range(0, len(spec.default_args), 2):
        option, value = spec.default_args[index : index + 2]
        if not contains_option(arguments, option):
            arguments.extend((option, value))

    runner = importlib.import_module(spec.module)
    original_argv = sys.argv
    try:
        sys.argv = [f"{spec.module}.py", *arguments]
        result = runner.main()
    finally:
        sys.argv = original_argv
    return int(result or 0)


def main(argv: list[str] | None = None) -> int:
    dataset, forwarded = parse_dispatch_args(
        list(sys.argv[1:] if argv is None else argv)
    )
    return dispatch(dataset, forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
