from __future__ import annotations

import numpy as np
import pandas as pd

from src.stats import weighted_quantile

LABOUR_INCOME_MONTHLY_LIMIT_DEFAULT = 600.0


def apply_age_rule(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    threshold = pd.to_numeric(out["baseline_age_threshold"], errors="coerce")

    rp1_candidate_ok = out["rp1_age"].ge(threshold).fillna(False) & out[
        "rp1_claimant_activity_eligible"
    ].eq(1).fillna(False)

    rp2_candidate_ok = out["rp2_age"].ge(threshold).fillna(False) & out[
        "rp2_claimant_activity_eligible"
    ].eq(1).fillna(False)

    out["rmi_age_eligible"] = np.where(
        out["responsible_person_proxy_available"].eq(1),
        (rp1_candidate_ok | rp2_candidate_ok).astype(float),
        np.nan,
    )

    out["rmi_age_rule_source"] = np.where(
        out["responsible_person_proxy_available"].eq(1),
        "responsible_person_claimant_age_proxy",
        "not_observed",
    )

    return out


def apply_claimant_proxy_rule(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["rmi_claimant_proxy_eligible"] = np.where(
        out["responsible_person_proxy_available"].eq(1),
        out["any_responsible_person_claimant_eligible"],
        np.nan,
    )

    out["rmi_claimant_proxy_source"] = np.where(
        out["responsible_person_proxy_available"].eq(1),
        "responsible_person_claimant_proxy",
        "not_observed",
    )

    return out


def add_multi_nucleus_proxy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["multi_nucleus_proxy"] = np.where(
        (out["n_adults_18plus"].fillna(0) >= 3)
        & (
            (out["n_working_18_64"].fillna(0) >= 2)
            | (out["n_unemployed_18_64"].fillna(0) >= 2)
        ),
        1.0,
        0.0,
    )

    return out


def apply_additional_institutional_rules(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["rmi_wealth_eligible"] = np.select(
        [
            out["baseline_wealth_test"].isin(
                ["proxy_asset_exclusion_strict", "strict_proxy_exclusion"]
            )
            & out["wealth_proxy_strict"].notna(),
            out["baseline_wealth_test"].eq("none"),
        ],
        [
            (out["wealth_proxy_strict"] == 0).astype(float),
            1.0,
        ],
        default=np.nan,
    )

    allowed_simple_types = (
        out["single_adult"].eq(1) | out["single_parent"].eq(1) | out["two_adults"].eq(1)
    )

    allowed_restricted_threeplus = out["threeplus_adults"].eq(1) & out[
        "multi_nucleus_proxy"
    ].eq(0)

    out["rmi_hhtype_eligible"] = np.select(
        [
            out["baseline_allowed_hh_types"].eq("all_household_types"),
            out["baseline_allowed_hh_types"].eq(
                "single_adult_single_parent_two_adults_only"
            ),
            out["baseline_allowed_hh_types"].eq(
                "single_adult_single_parent_two_adults_plus_restricted_threeplus"
            ),
        ],
        [
            1.0,
            allowed_simple_types.astype(float),
            (allowed_simple_types | allowed_restricted_threeplus).astype(float),
        ],
        default=np.nan,
    )

    out["rmi_threeplus_adults_allowed"] = np.select(
        [
            out["baseline_threeplus_rule"].eq("allow_all"),
            out["baseline_exclude_threeplus_adults"].eq(True),
            out["baseline_exclude_threeplus_adults"].eq(False),
        ],
        [
            1.0,
            (out["threeplus_adults"] != 1).astype(float),
            np.where(
                out["threeplus_adults"].eq(1),
                (out["multi_nucleus_proxy"] == 0).astype(float),
                1.0,
            ),
        ],
        default=np.nan,
    )

    return out


def apply_labour_rule(
    df: pd.DataFrame,
    labour_income_limit: float = LABOUR_INCOME_MONTHLY_LIMIT_DEFAULT,
) -> pd.DataFrame:
    out = df.copy()

    labour_income = pd.to_numeric(out["labour_income_hh_monthly"], errors="coerce")
    labour_income_ok = labour_income.le(labour_income_limit)

    labour_context_ok = (
        out["any_unemployed_18_64"].eq(1)
        | out["all_working_age_nonworking"].eq(1)
        | out["any_responsible_person_active_search"].eq(1)
        | out["any_social_assistance_income_hh"].eq(1)
    )

    out["rmi_labour_income_eligible"] = np.where(
        labour_income.notna(), labour_income_ok.astype(float), np.nan
    )

    out["rmi_labour_context_eligible"] = np.where(
        out["has_labour_composition"].eq(1)
        | out["responsible_person_proxy_available"].eq(1),
        labour_context_ok.astype(float),
        np.nan,
    )

    strict_labour_ok = np.where(
        out["rmi_labour_income_eligible"].eq(1)
        & out["rmi_labour_context_eligible"].eq(1),
        1.0,
        np.where(
            out["rmi_labour_income_eligible"].isna()
            | out["rmi_labour_context_eligible"].isna(),
            np.nan,
            0.0,
        ),
    )

    relaxed_labour_ok = np.where(
        out["rmi_labour_income_eligible"].eq(1),
        1.0,
        np.where(out["rmi_labour_income_eligible"].isna(), np.nan, 0.0),
    )

    out["rmi_labour_eligible"] = np.where(
        out["baseline_relax_labour_gate"].eq(True), relaxed_labour_ok, strict_labour_ok
    )

    out["rmi_labour_rule_source"] = np.select(
        [
            out["rmi_labour_income_eligible"].isna(),
            out["baseline_relax_labour_gate"].eq(True)
            & out["rmi_labour_income_eligible"].eq(1),
            out["rmi_labour_income_eligible"].eq(0),
            out["baseline_relax_labour_gate"].eq(False)
            & out["rmi_labour_context_eligible"].eq(0),
            out["baseline_relax_labour_gate"].eq(False)
            & out["rmi_labour_eligible"].eq(1),
        ],
        [
            "labour_rule_not_observable",
            "relaxed_labour_income_only_rule",
            "fails_labour_income_rule",
            "fails_labour_context_rule",
            "labour_income_and_context_rule",
        ],
        default="other",
    )

    return out


def apply_region_specific_insertion_rules(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["rmi_insertion_rule_eligible"] = 1.0
    out["rmi_insertion_rule_source"] = "not_applicable"

    # Andalusia
    mask = out["nuts_code"].eq("ES61")
    ok = out["any_responsible_person_active_search"].eq(1) | out[
        "any_social_assistance_income_hh"
    ].eq(1)
    out.loc[mask, "rmi_insertion_rule_eligible"] = np.where(ok[mask], 1.0, 0.0)
    out.loc[mask, "rmi_insertion_rule_source"] = np.where(
        ok[mask], "andalusia_insertion_proxy", "fails_andalusia_insertion_proxy"
    )

    # Castilla-La Mancha
    mask = out["nuts_code"].eq("ES42")
    ok = out["any_responsible_person_active_search"].eq(1) & out[
        "any_responsible_person_claimant_eligible"
    ].eq(1)
    out.loc[mask, "rmi_insertion_rule_eligible"] = np.where(ok[mask], 1.0, 0.0)
    out.loc[mask, "rmi_insertion_rule_source"] = np.where(
        ok[mask], "clm_insertion_proxy", "fails_clm_insertion_proxy"
    )

    # Extremadura
    mask = out["nuts_code"].eq("ES43")
    ok = out["any_responsible_person_active_search"].eq(1) | out[
        "any_social_assistance_income_hh"
    ].eq(1)
    out.loc[mask, "rmi_insertion_rule_eligible"] = np.where(ok[mask], 1.0, 0.0)
    out.loc[mask, "rmi_insertion_rule_source"] = np.where(
        ok[mask], "extremadura_insertion_proxy", "fails_extremadura_insertion_proxy"
    )

    # Madrid
    mask = out["nuts_code"].eq("ES30")
    ok = (
        out["any_responsible_person_active_search"].eq(1)
        | (out["any_unemployed_18_64"].eq(1) & out["all_unemployed_searching"].eq(1))
        | out["any_social_assistance_income_hh"].eq(1)
    )
    out.loc[mask, "rmi_insertion_rule_eligible"] = np.where(ok[mask], 1.0, 0.0)
    out.loc[mask, "rmi_insertion_rule_source"] = np.where(
        ok[mask], "madrid_insertion_proxy", "fails_madrid_insertion_proxy"
    )

    # Castilla y León
    mask = out["nuts_code"].eq("ES41")
    ok = out["any_responsible_person_active_search"].eq(1) | (
        out["any_unemployed_18_64"].eq(1) & out["all_unemployed_searching"].eq(1)
    )
    out.loc[mask, "rmi_insertion_rule_eligible"] = np.where(ok[mask], 1.0, 0.0)
    out.loc[mask, "rmi_insertion_rule_source"] = np.where(
        ok[mask], "cyl_insertion_proxy", "fails_cyl_insertion_proxy"
    )

    # Valencia
    mask = out["nuts_code"].eq("ES52")
    ok = out["any_responsible_person_active_search"].eq(1) | out[
        "any_social_assistance_income_hh"
    ].eq(1)
    out.loc[mask, "rmi_insertion_rule_eligible"] = np.where(ok[mask], 1.0, 0.0)
    out.loc[mask, "rmi_insertion_rule_source"] = np.where(
        ok[mask], "valencia_insertion_proxy", "fails_valencia_insertion_proxy"
    )

    return out


def add_active_inclusion_gate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    base_active_inclusion_ok = (
        out["rmi_claimant_proxy_eligible"].eq(1)
        & (
            out["any_responsible_person_active_search"].eq(1)
            | (
                out["any_unemployed_18_64"].eq(1)
                & out["all_unemployed_searching"].eq(1)
            )
            | out["any_social_assistance_income_hh"].eq(1)
        )
    ).astype(float)

    out["active_inclusion_ok"] = np.where(
        out["baseline_apply_active_inclusion_gate"].eq(True),
        base_active_inclusion_ok,
        1.0,
    )

    out["active_inclusion_gate_applied"] = np.where(
        out["baseline_apply_active_inclusion_gate"].eq(True), 1.0, 0.0
    )

    return out


def add_percentile_filter(df: pd.DataFrame, quantile: float) -> pd.DataFrame:
    out = df.copy()

    cutoff_map = (
        out.groupby("year")
        .apply(
            lambda g: weighted_quantile(
                g["pfilter_resources_monthly"], g["weight_hh"], quantile
            )
        )
        .to_dict()
    )

    out["percentile_cutoff_monthly"] = out["year"].map(cutoff_map)

    out["passes_percentile_filter"] = np.select(
        [
            out["pfilter_resources_monthly"].isna()
            | out["percentile_cutoff_monthly"].isna(),
            out["pfilter_resources_monthly"] <= out["percentile_cutoff_monthly"],
            out["pfilter_resources_monthly"] > out["percentile_cutoff_monthly"],
        ],
        [np.nan, 1.0, 0.0],
        default=np.nan,
    )

    out["percentile_rule"] = f"bottom_{int(quantile * 100)}pct"
    return out
