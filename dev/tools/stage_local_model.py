#!/usr/bin/env python3
# Modified by Red Hills Media on 2026-10-10 for this M1 Ultra source release.
"""Stage local target and draft checkpoints without publishing them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from install import models, paths, upstream


def stage_local_model(
    models_root: str | Path,
    name: str,
    target: str | Path,
    draft: str | Path,
    *,
    language_only: bool = False,
) -> dict[str, str]:
    staging_root = Path(models_root).expanduser().resolve()
    protected_roots = {
        paths.MODELS.expanduser().resolve(),
        (Path.home() / "Library/Application Support/Splash/models").resolve(),
    }
    if any(
        staging_root == protected or protected in staging_root.parents
        for protected in protected_roots
    ):
        raise models.ModelError("local staging must not use the production models root")
    target_root = Path(target).expanduser().resolve()
    draft_root = Path(draft).expanduser().resolve()
    selection = models.Selection.of(
        staging_root,
        name,
        language_only=language_only,
        draft_model=str(draft_root),
    )
    link = upstream.install_local(selection, target_root, draft_root)
    return {
        "draft": str(draft_root),
        "model_root": str(link),
        "models_root": str(staging_root),
        "target": str(target_root),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--draft", required=True, type=Path)
    parser.add_argument("--language-only", action="store_true")
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = stage_local_model(
            args.models,
            args.name,
            args.target,
            args.draft,
            language_only=args.language_only,
        )
    except (models.ModelError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
