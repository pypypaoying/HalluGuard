#!/usr/bin/env python
"""Validate frozen core-table HalluGuard configs before server runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


DEFAULT_CONFIGS = [
    "experiments/halluguard/configs/halluguard_core_table_sp_frozen.yaml",
    "experiments/halluguard/configs/halluguard_core_table_stable_harm.yaml",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate HalluGuard core-table configs.")
    parser.add_argument("configs", nargs="*", default=DEFAULT_CONFIGS)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    import sys

    halluguard_dir = repo_root / "experiments" / "halluguard"
    sys.path.insert(0, str(halluguard_dir))
    import halluguard_router as router  # noqa: WPS433

    allowed_router_variants = set(router.DEPLOYABLE_ROUTERS) | {"oracle_test_ceiling"}
    failed = False
    for config in args.configs:
        path = (repo_root / config).resolve()
        with path.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        method = cfg.get("method", {}) or {}
        router_variants = list(method.get("router_variants", []))
        unknown = [name for name in router_variants if name not in allowed_router_variants]
        main_router = str(method.get("main_router", ""))
        if main_router and main_router not in allowed_router_variants:
            unknown.append(main_router)
        if unknown:
            failed = True
            print(f"ERROR {config}: unknown router variant(s): {', '.join(dict.fromkeys(unknown))}")
        else:
            print(f"ok {config}: {', '.join(router_variants)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
