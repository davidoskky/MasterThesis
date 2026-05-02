#!/usr/bin/env python3
"""
Generate OpenFisca-compatible YAML parameter files from the policy_db parquet files.

Output layout:
    parameters/rmi/{nuts_code}/{parameter_name}.yaml

Each file carries a 'metadata' block that records the source parquet, the
policy source year, and the assumption used to carry 2017 rules forward to
2018/2019, so every value can be traced back to the original CSV sources in
policy_sources/.

Usage (from the project root):
    uv run python generate_openfisca_parameters.py
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

POLICY_DB = Path("policy_db")
OUTPUT_ROOT = Path("parameters") / "rmi"

# ── Helpers ───────────────────────────────────────────────────────────────────


def _date(year: int) -> str:
    return f"{year}-01-01"


def _clean(value: Any) -> Any:
    """Convert numpy/pandas scalars to plain Python; map NaN/NA to None."""
    if value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):       # numpy scalar → Python scalar
        return value.item()
    return value


def _values_block(values_by_year: dict[int, Any]) -> dict:
    return {_date(y): {"value": _clean(v)} for y, v in sorted(values_by_year.items())}


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def _base_metadata(grp: pd.DataFrame, source_file: str) -> dict:
    return {
        "source": source_file,
        "source_year": int(grp["source_year"].iloc[0]),
        "assumption_no_change": bool(grp["assumption_no_change"].iloc[0]),
        "assumption_note": str(grp["assumption_note"].iloc[0]),
        "region": str(grp["region_name_policy"].iloc[0]),
        "program": str(grp["program_name"].iloc[0]),
    }


# ── Rule parameters ───────────────────────────────────────────────────────────

# (parquet column, output filename, human description, unit or None)
RULE_PARAMS: list[tuple[str, str, str, str | None]] = [
    (
        "baseline_age_threshold",
        "age_threshold",
        "Minimum age for the responsible-person claimant proxy",
        "year",
    ),
    (
        "baseline_wealth_test",
        "wealth_test",
        "Asset exclusion rule applied to the household"
        " ('proxy_asset_exclusion_strict', 'strict_proxy_exclusion', or 'none')",
        None,
    ),
    (
        "baseline_allowed_hh_types",
        "allowed_hh_types",
        "Household composition types eligible to claim",
        None,
    ),
    (
        "baseline_exclude_threeplus_adults",
        "exclude_threeplus_adults",
        "True if households with 3+ adults are always excluded (hard exclusion)",
        None,
    ),
    (
        "baseline_threeplus_rule",
        "threeplus_rule",
        "Three-plus-adults exclusion variant"
        " ('allow_all' or 'exclude_if_multi_nucleus_proxy')",
        None,
    ),
    (
        "baseline_conditionality_profile",
        "conditionality_profile",
        "Conditionality intensity profile"
        " (strict / standard / soft / guaranteed_soft / …)",
        None,
    ),
    (
        "baseline_apply_active_inclusion_gate",
        "apply_active_inclusion_gate",
        "True if the active-inclusion proxy gate is enforced before the income test",
        None,
    ),
    (
        "baseline_relax_labour_gate",
        "relax_labour_gate",
        "True if the labour gate checks only income, ignoring the labour-context condition",
        None,
    ),
    (
        "baseline_non_takeup_group",
        "non_takeup_group",
        "Non-take-up calibration group ('high', 'medium', or 'none')",
        None,
    ),
    (
        "baseline_main_included",
        "main_included",
        "True if this region is included in the main baseline simulation",
        None,
    ),
    (
        "baseline_amount_topup_factor",
        "amount_topup_factor",
        "Multiplier applied to the scheduled guaranteed amount (1.0 if not specified)",
        None,
    ),
    (
        "baseline_scheme_structure",
        "scheme_structure",
        "Overall scheme structure type"
        " (classic_rmi / guaranteed_income / coexisting_rmi_resoga / …)",
        None,
    ),
]


def _generate_rule_parameters(rules: pd.DataFrame) -> int:
    n = 0
    for nuts_code, grp in rules.groupby("nuts_code"):
        out_dir = OUTPUT_ROOT / str(nuts_code)
        metadata = _base_metadata(grp, "policy_db/rmi_baseline_rules.parquet")

        for col, filename, description, unit in RULE_PARAMS:
            if col not in grp.columns:
                continue

            values_by_year = {int(row["year"]): row[col] for _, row in grp.iterrows()}

            # Skip parameters that are not applicable to this region (all NaN/None)
            if all(_clean(v) is None for v in values_by_year.values()):
                continue

            meta = dict(metadata)
            if unit:
                meta["unit"] = unit

            data = {
                "description": f"{description} — {metadata['region']}",
                "metadata": meta,
                "values": _values_block(values_by_year),
            }
            _write_yaml(out_dir / f"{filename}.yaml", data)
            n += 1

    return n


# ── Schedule parameters ───────────────────────────────────────────────────────


def _generate_schedule_parameters(schedule: pd.DataFrame) -> int:
    n = 0
    for nuts_code, grp in schedule.groupby("nuts_code"):
        out_dir = OUTPUT_ROOT / str(nuts_code)
        metadata = _base_metadata(grp, "policy_db/rmi_baseline_schedule.parquet")
        metadata["unit"] = "currency-EUR"

        # Guaranteed-amount scale: one bracket per household size.
        # The threshold is the household size (lower bound of each bracket).
        # Amounts are keyed by year so future policy changes can be added per bracket.
        sizes = sorted(int(s) for s in grp["hh_size"].dropna().unique())
        brackets = []
        for hh_size in sizes:
            size_rows = grp[grp["hh_size"] == hh_size]
            guaranteed_by_year = {
                int(r["year"]): round(float(r["guaranteed_amount"]), 2)
                for _, r in size_rows.iterrows()
            }
            brackets.append(
                {
                    "threshold": {
                        "values": {
                            _date(y): {"value": hh_size}
                            for y in sorted(guaranteed_by_year)
                        }
                    },
                    "amount": {"values": _values_block(guaranteed_by_year)},
                }
            )

        _write_yaml(
            out_dir / "schedule.yaml",
            {
                "description": (
                    f"Monthly guaranteed amount by household size (EUR)"
                    f" — {metadata['region']}"
                ),
                "metadata": metadata,
                "brackets": brackets,
            },
        )
        n += 1

        # Max amount: flat cap per year, applied when household size exceeds the schedule.
        max_by_year = {
            int(year): round(float(val), 2)
            for year, val in grp.groupby("year")["max_amount"].first().dropna().items()
        }
        if max_by_year:
            _write_yaml(
                out_dir / "max_amount.yaml",
                {
                    "description": (
                        f"Maximum monthly benefit cap (EUR) — {metadata['region']}"
                    ),
                    "metadata": {
                        **metadata,
                        "note": (
                            "Applied when the household size exceeds the listed schedule"
                        ),
                    },
                    "values": _values_block(max_by_year),
                },
            )
            n += 1

    return n


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    rules = pd.read_parquet(POLICY_DB / "rmi_baseline_rules.parquet")
    schedule = pd.read_parquet(POLICY_DB / "rmi_baseline_schedule.parquet")

    n_rules = _generate_rule_parameters(rules)
    n_schedule = _generate_schedule_parameters(schedule)

    n_total = n_rules + n_schedule
    n_regions = sum(1 for p in OUTPUT_ROOT.iterdir() if p.is_dir())
    print(
        f"Generated {n_total} YAML files"
        f" ({n_rules} rule parameters, {n_schedule} schedule/cap files)"
        f" across {n_regions} regions → {OUTPUT_ROOT}/"
    )


if __name__ == "__main__":
    main()
