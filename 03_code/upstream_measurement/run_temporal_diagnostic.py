"""Read-only temporal diagnostic of the frozen 2000–2025 canopy resource panel."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
FROZEN = ROOT / "06_logs/canopy_cooling_equity_20261009_run01"
ROW = FROZEN / "annual_canopy_equity_row_manifest.csv.gz"
SOURCE = ROOT / "06_logs/national_canopy_streaming_summary_20260915_run01/national_tract_year_canopy_1999_2025.csv"
Y = "urban_canopy_percent_fixed_land"
ID = ["state", "tract_id"]
SOCIAL = ("income", "poverty", "age65plus")
YEARS = list(range(2000, 2026))


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_inputs() -> tuple[pd.DataFrame, dict]:
    m = pd.read_csv(ROW, low_memory=False)
    check(len(m) == 64999 * 26, "Frozen panel row count differs from 64,999 × 26")
    check(not m.duplicated([*ID, "year"]).any(), "Duplicate tract-year keys")
    check(sorted(m.year.unique().tolist()) == YEARS, "Unexpected year coverage")
    b = m.loc[m.balanced_26years].copy()
    check(len(b) == 46424 * 26, "Balanced tract-year count differs from frozen specification")
    check(b.groupby(ID).year.nunique().eq(26).all(), "Balanced tracts lack 26 observations")
    check(b.complete_fixed_land_year.all() and b.positive_land.all(), "Balanced rows lack fixed-land coverage")
    check(b[Y].notna().all() and b[Y].between(0, 100).all(), "Invalid canopy values")
    for field in ["single_city_id", *(f"{v}_city_quartile" for v in SOCIAL)]:
        check(b.groupby(ID)[field].nunique(dropna=False).eq(1).all(), f"Baseline field changed: {field}")
    base = b.loc[b.year.eq(2000), [*ID, Y]].rename(columns={Y: "baseline_cover"})
    merged = b.merge(base, on=ID, validate="many_to_one")
    delta_error = np.max(np.abs(merged.change_from_2000_pp - (merged[Y] - merged.baseline_cover)))
    check(delta_error <= 1e-9, "Frozen 2000-relative change disagrees with level series")

    s = pd.read_csv(SOURCE, usecols=[*ID, "year", "urban_fixed_land_m2", Y], low_memory=False)
    s = s.loc[s.year.between(2000, 2025)]
    check(len(s) == len(m) and not s.duplicated([*ID, "year"]).any(), "Source canopy panel keys changed")
    ids = base[ID]
    selected = s.merge(ids, on=ID, how="inner", validate="many_to_one")
    check(len(selected) == len(b), "Source panel does not cover balanced tract-years")
    area = selected.groupby(ID).urban_fixed_land_m2.agg(["min", "max", "count"])
    max_area_change = float((area["max"] - area["min"]).abs().max())
    check(area["count"].eq(26).all() and max_area_change <= 1e-8, "Fixed-land area changed across years")
    check(selected.urban_fixed_land_m2.gt(0).all(), "Nonpositive fixed land in balanced panel")
    linked = b[[*ID, "year", Y]].merge(
        selected[[*ID, "year", Y]], on=[*ID, "year"], validate="one_to_one", suffixes=("_manifest", "_source")
    )
    max_cover_diff = float(np.max(np.abs(linked[f"{Y}_manifest"] - linked[f"{Y}_source"])))
    check(max_cover_diff <= 1e-9, "Source and row-manifest canopy values differ")

    flow = pd.read_csv(FROZEN / "annual_canopy_coverage_flow.csv")
    checks = m.groupby("year").agg(
        input_rows=("tract_id", "size"),
        positive_land_rows=("positive_land", "sum"),
        full_fixed_land_rows=("complete_fixed_land_year", "sum"),
        balanced_rows=("balanced_26years", "sum"),
    ).reset_index()
    joined = checks.merge(flow, on="year", validate="one_to_one", suffixes=("_actual", "_frozen"))
    for col in ["input_rows", "positive_land_rows", "full_fixed_land_rows", "balanced_rows"]:
        check(joined[f"{col}_actual"].equals(joined[f"{col}_frozen"]), f"Coverage-flow mismatch: {col}")
    support = {
        "input_rows": len(m), "balanced_rows": len(b), "balanced_tracts": len(base),
        "balanced_years": len(YEARS), "source_rows_2000_2025": len(s),
        "duplicate_tract_year_keys": 0, "balanced_missing_cover": 0,
        "max_fixed_land_area_change_m2": max_area_change,
        "max_source_manifest_canopy_difference_pp": max_cover_diff,
        "max_change_field_difference_pp": float(delta_error),
        "city_count_with_group_support": int(b.single_city_id.nunique()),
        "quartile_unassigned_balanced_tracts": {
            v: int(b.loc[b.year.eq(2000), f"{v}_city_quartile"].isna().sum()) for v in SOCIAL
        },
    }
    return b, support


def calculate_targets(b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows = []
    national = b.groupby("year")[Y].agg(["mean", "size", "median"])
    for year, record in national.iterrows():
        rows.append({"target": "national_tract_mean", "year": int(year), "value": record["mean"],
                     "tracts": int(record["size"]), "cities": None, "unit": "percent_cover"})
    gaps = []
    for variable in SOCIAL:
        q = f"{variable}_city_quartile"
        grouped = b.loc[b[q].notna()].groupby(["year", q])[Y].agg(["mean", "size"])
        for (year, quartile), record in grouped.iterrows():
            rows.append({"target": f"{variable}_q{int(quartile)}_tract_mean", "year": int(year),
                         "value": record["mean"], "tracts": int(record["size"]),
                         "cities": None, "unit": "percent_cover"})
        city_means = b.loc[b[q].isin([1, 4])].groupby(["year", "single_city_id", q])[Y].agg(["mean", "size"])
        wide = city_means.unstack(q)
        required = [("mean", 1.0), ("mean", 4.0), ("size", 1.0), ("size", 4.0)]
        check(all(col in wide.columns for col in required), f"Missing city groups for {variable}")
        wide = wide.dropna(subset=[("mean", 1.0), ("mean", 4.0)])
        for (year, city), record in wide.iterrows():
            gaps.append({"year": int(year), "single_city_id": int(city), "variable": variable,
                         "high_minus_low_canopy_pp": float(record[("mean", 4.0)] - record[("mean", 1.0)]),
                         "low_tracts": int(record[("size", 1.0)]),
                         "high_tracts": int(record[("size", 4.0)])})
        qmeans = grouped["mean"].unstack(q)
        for year in YEARS:
            rows.append({"target": f"{variable}_q4_minus_q1_pooled", "year": year,
                         "value": float(qmeans.loc[year, 4.0] - qmeans.loc[year, 1.0]),
                         "tracts": None, "cities": None, "unit": "percentage_points"})
    gap_df = pd.DataFrame(gaps)
    equal_city = gap_df.groupby(["year", "variable"]).agg(
        value=("high_minus_low_canopy_pp", "mean"), cities=("single_city_id", "nunique")
    ).reset_index()
    for record in equal_city.itertuples():
        rows.append({"target": f"{record.variable}_q4_minus_q1_equal_city", "year": int(record.year),
                     "value": float(record.value), "tracts": None,
                     "cities": int(record.cities), "unit": "percentage_points"})
    result = pd.DataFrame(rows).sort_values(["target", "year"]).reset_index(drop=True)
    result["annual_delta"] = result.groupby("target").value.diff()
    check(result.groupby("target").year.nunique().eq(26).all(), "Incomplete annual target series")

    frozen_group = pd.read_csv(FROZEN / "annual_canopy_city_group_trajectories.csv")
    frozen_group = frozen_group.loc[frozen_group.panel.eq("balanced_26years")]
    calculated = result.loc[result.target.str.match(r"^(income|poverty|age65plus)_q[1-4]_tract_mean$")].copy()
    calculated[["variable", "quartile"]] = calculated.target.str.extract(r"^(income|poverty|age65plus)_q([1-4])_tract_mean$")
    calculated["quartile"] = calculated.quartile.astype(int)
    compare_group = calculated.merge(frozen_group, on=["year", "variable", "quartile"], validate="one_to_one")
    check(len(compare_group) == 3 * 4 * 26, "Frozen group trajectory mapping incomplete")
    group_difference = float(np.max(np.abs(compare_group.value - compare_group.canopy_percent_mean)))
    check(group_difference <= 1e-9, "Recomputed group means disagree with frozen trajectories")

    frozen_gaps = pd.read_csv(FROZEN / "annual_same_city_canopy_gap_summaries.csv")
    frozen_gaps = frozen_gaps.loc[frozen_gaps.panel.eq("balanced_26years")]
    compare_gap = equal_city.merge(frozen_gaps, on=["year", "variable"], validate="one_to_one")
    check(len(compare_gap) == 3 * 26, "Frozen city-gap mapping incomplete")
    gap_difference = float(np.max(np.abs(compare_gap.value - compare_gap.equal_city_mean_gap_pp)))
    check(gap_difference <= 1e-9, "Recomputed city gaps disagree with frozen summaries")
    compare_city = gap_df.merge(
        pd.read_csv(FROZEN / "annual_same_city_canopy_gaps.csv")
        .query("panel == 'balanced_26years'").drop(columns="panel"),
        on=["year", "variable", "single_city_id"], validate="one_to_one", suffixes=("_new", "_frozen")
    )
    check(len(compare_city) == len(gap_df), "City-level gap mapping incomplete")
    city_difference = float(np.max(np.abs(compare_city.high_minus_low_canopy_pp_new - compare_city.high_minus_low_canopy_pp_frozen)))
    check(city_difference <= 1e-9, "Recomputed city gaps disagree with frozen city rows")
    return result, gap_df, {
        "group_target_comparisons": len(compare_group), "max_group_mean_difference_pp": group_difference,
        "equal_city_target_comparisons": len(compare_gap), "max_equal_city_gap_difference_pp": gap_difference,
        "city_year_gap_comparisons": len(compare_city), "max_city_year_gap_difference_pp": city_difference,
    }


def summarize_periods(series: pd.DataFrame) -> pd.DataFrame:
    wide = series.pivot(index="target", columns="year", values="value")
    periods = [(2000, 2015, "pre_2016"), (2015, 2016, "transition_2015_2016"),
               (2016, 2025, "post_2016"), (2000, 2025, "full_period"),
               (2013, 2015, "pre_transition_window"), (2016, 2018, "post_transition_window")]
    rows = []
    for target, vals in wide.iterrows():
        for start, end, label in periods:
            rows.append({"target": target, "period": label, "year_start": start,
                         "year_end": end, "start_value": vals[start], "end_value": vals[end],
                         "change_pp": vals[end] - vals[start]})
    result = pd.DataFrame(rows)
    pivot = result.pivot(index="target", columns="period", values="change_pp")
    closure = (pivot.pre_2016 + pivot.transition_2015_2016 + pivot.post_2016 - pivot.full_period).abs().max()
    check(closure <= 1e-12, "Pre/transition/post decomposition does not close")
    return result


def draw(series: pd.DataFrame) -> None:
    display = [
        ("national_tract_mean", "National mean canopy", "Cover (%)", "#2A788E"),
        ("income_q4_minus_q1_equal_city", "Income: Q4 − Q1", "City-mean gap (pp)", "#B36A3B"),
        ("poverty_q4_minus_q1_equal_city", "Poverty: Q4 − Q1", "City-mean gap (pp)", "#7C5180"),
        ("age65plus_q4_minus_q1_equal_city", "Age 65+: Q4 − Q1", "City-mean gap (pp)", "#4F866B"),
    ]
    fig = plt.figure(figsize=(11, 8))
    grid = fig.add_gridspec(2, 2, left=.09, right=.98, top=.95, bottom=.10, hspace=.36, wspace=.24)
    for j, (target, title, unit, color) in enumerate(display):
        sub = grid[j // 2, j % 2].subgridspec(2, 1, height_ratios=[3, 1], hspace=.05)
        ax = fig.add_subplot(sub[0]); change = fig.add_subplot(sub[1], sharex=ax)
        t = series.loc[series.target.eq(target)].sort_values("year")
        ax.plot(t.year, t.value, color=color, lw=1.8)
        ax.axvline(2015.5, color=".45", ls="--", lw=.9)
        ax.set_title(f"{chr(97+j)}  {title}", loc="left", fontsize=10, fontweight="bold")
        ax.set_ylabel(unit, fontsize=8)
        ax.tick_params(labelbottom=False, labelsize=8)
        change.bar(t.year.iloc[1:], t.annual_delta.iloc[1:], width=.75, color=color, alpha=.65)
        change.axhline(0, color=".4", lw=.7)
        change.axvline(2015.5, color=".45", ls="--", lw=.9)
        change.set_ylabel("Δ / yr", fontsize=8)
        change.set_xlim(1999.5, 2025.5)
        change.set_xticks([2000, 2005, 2010, 2015, 2020, 2025])
        change.tick_params(labelsize=8)
        if j // 2 == 1:
            change.set_xlabel("Year", fontsize=8)
    fig.text(.09, .025, "Dashed line: change in the source-product compositing window (2016). Fixed tracts and baseline groups.", fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"temporal_consistency_four_panel.{ext}", dpi=220)
    plt.close(fig)


def main() -> None:
    b, support = read_inputs()
    series, city, comparison = calculate_targets(b)
    periods = summarize_periods(series)
    series.to_csv(OUT / "annual_target_series.csv", index=False)
    city.to_csv(OUT / "city_year_group_gaps.csv", index=False)
    periods.to_csv(OUT / "stage_contributions.csv", index=False)
    draw(series)
    selected = ["national_tract_mean", "income_q4_minus_q1_equal_city",
                "poverty_q4_minus_q1_equal_city", "age65plus_q4_minus_q1_equal_city"]
    anchors = series.loc[series.target.isin(selected) & series.year.isin([2000, 2013, 2015, 2016, 2018, 2025])]
    anchors.to_csv(OUT / "anchor_years.csv", index=False)
    transitions = periods.loc[periods.period.eq("transition_2015_2016") & periods.target.isin(selected),
                              ["target", "change_pp"]].set_index("target").change_pp.to_dict()
    report = {"status": "PASS_TEMPORAL_DIAGNOSTIC", "input_support": support,
              "frozen_output_comparison": comparison, "stage_closure_max_abs_pp": float(
                  (periods.pivot(index="target", columns="period", values="change_pp")
                   .eval("pre_2016 + transition_2015_2016 + post_2016 - full_period").abs().max())),
              "change_2015_2016_pp": transitions,
              "measurement_conclusion": "Annual patterns alone cannot identify ecological change versus product-processing change.",
              "source_data_read_only": True, "formal_model_or_manuscript_changed": False}
    (OUT / "temporal_diagnostic.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
