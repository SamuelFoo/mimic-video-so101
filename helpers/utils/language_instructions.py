#!/usr/bin/env python3
"""Look up language instructions by experiment type."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_INSTRUCTIONS_PATH = Path(__file__).resolve().parents[2] / "config/language_instructions.json"


def get_language_instruction(ex_type: str, instructions_path: Path = DEFAULT_INSTRUCTIONS_PATH) -> str:
    instructions = json.loads(instructions_path.read_text())
    if ex_type not in instructions:
        available = ", ".join(sorted(instructions)) or "<none>"
        raise KeyError(f"No instruction for '{ex_type}' in {instructions_path}. Available: {available}")

    instruction = str(instructions[ex_type]).strip()
    if not instruction:
        raise ValueError(f"Instruction for '{ex_type}' in {instructions_path} is empty")
    return instruction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ex_type", help="Instruction key, e.g. ex1 or ex2")
    parser.add_argument("--instructions", type=Path, default=DEFAULT_INSTRUCTIONS_PATH)
    args = parser.parse_args()
    print(get_language_instruction(args.ex_type, args.instructions))


if __name__ == "__main__":
    main()
