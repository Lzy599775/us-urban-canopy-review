"""Recalculate canopy-product and local-reference summary tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


def check(actual: float, reported: float, label: str, records: list[dict]) -> None:
    difference = abs(float(actual) - float(reported))
    records.append({"item": label, "recalculated": actual, "reported": reported, "difference": difference})
    if difference > 0.00016:
        raise AssertionError(f"{label}: {actual} != {reported}")


def category(theta: float, change: float, threshold: float) -> str:
    if theta > 0:
        return "positive_response"
    if change == 0:
        return "zero_change"
    direction = "loss" if change < 0 else "gain"
    return direction + ("_stronger" if theta <= threshold else "_weaker")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.package.resolve()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    data = root / "02_source_data/measurement"
    tables = root / "02_source_data/paper_tables"
    records: list[dict] = []
    read_ids = {"dtype": {"state": str, "tract_id": str, "single_city_id": str}}
    product = pd.read_csv(data / "tract_product_comparison.csv", **read_ids)
    assert len(product) == 46424
    assert not product.duplicated(["state", "tract_id"]).any()
    reported = pd.read_csv(tables / "Supplementary_Table_11.csv")
    dimensions = {"Income": "income_city_quartile", "Poverty": "poverty_city_quartile", "Age ≥65 share": "age65plus_city_quartile"}
    for row in reported.itertuples(index=False):
        title = row[0]
        for source, column in (("NLCD", "nlcd_change_full_pp"), ("Science", "science_change_full_pp")):
            if title == "National tract mean":
                actual = product[column].mean()
            else:
                dimension, group_text = title.split(" ", 1)
                if title.startswith("Age ≥65 share"):
                    dimension = "Age ≥65 share"
                    group_text = title[len(dimension) + 1:]
                field = dimensions[dimension]
                if "equal areas" in group_text:
                    subset = product.loc[product[field].isin([1, 4]) & product.single_city_id.notna()]
                    city_group = subset.groupby(["single_city_id", field])[column].mean().unstack()
                    assert city_group.shape == (448, 2)
                    actual = (city_group[4] - city_group[1]).mean()
                elif "pooled tracts" in group_text and "Q4−Q1" in group_text:
                    actual = product.loc[product[field].eq(4), column].mean() - product.loc[product[field].eq(1), column].mean()
                else:
                    quartile = 1 if group_text.startswith("Q1") else 4
                    actual = product.loc[product[field].eq(quartile), column].mean()
            check(actual, row[1] if source == "NLCD" else row[2], f"Supplementary Table 11 | {title} | {source}", records)

    figure = pd.read_csv(root / "02_source_data/project/06_logs/figure_attachment_revision_20261020_run01/Figure05/source_data/map_paired_tracts.csv", dtype={"state": str, "tract_id": str})
    figure.state = figure.state.str.zfill(2)
    figure.tract_id = figure.tract_id.str.zfill(11)
    product.state = product.state.str.zfill(2)
    product.tract_id = product.tract_id.str.zfill(11)
    pair = figure.merge(product[["state", "tract_id", "nlcd_change_full_pp", "science_change_full_pp"]], on=["state", "tract_id"], how="left", validate="one_to_one", indicator=True)
    assert len(pair) == 42369 and pair._merge.eq("both").all()
    assert np.allclose(pair.change_pp, pair.nlcd_change_full_pp, rtol=0, atol=1e-12)
    threshold = float(pair.loc[pair.theta.lt(0), "theta"].median())
    assert pair.category.to_list() == [category(t, c, threshold) for t, c in zip(pair.theta, pair.change_pp)]
    alternative = pd.Series([category(t, c, threshold) for t, c in zip(pair.theta, pair.science_change_full_pp)])
    assert int((alternative.to_numpy() != pair.category.to_numpy()).sum()) == 9205
    names = {
        "Gain / stronger negative response": "gain_stronger",
        "Loss / stronger negative response": "loss_stronger",
        "Gain / weaker negative response": "gain_weaker",
        "Loss / weaker negative response": "loss_weaker",
        "Zero net change": "zero_change",
        "Positive predicted response": "positive_response",
    }
    for row in pd.read_csv(tables / "Supplementary_Table_12.csv").itertuples(index=False):
        if row[0] == "Total":
            check(len(pair), int(str(row[1]).replace(",", "")), "Supplementary Table 12 | total | NLCD n", records)
            check(len(pair), int(str(row[3]).replace(",", "")), "Supplementary Table 12 | total | Science n", records)
            continue
        key = names[row[0]]
        check((pair.category == key).sum(), int(str(row[1]).replace(",", "")), f"Supplementary Table 12 | {key} | NLCD n", records)
        check((alternative == key).sum(), int(str(row[3]).replace(",", "")), f"Supplementary Table 12 | {key} | Science n", records)

    city = pd.read_csv(data / "local_city_comparison.csv")
    for number, table, target in ((14, "Supplementary_Table_14.csv", "late"), (15, "Supplementary_Table_15.csv", "change")):
        published = pd.read_csv(tables / table)
        source = city.loc[city.target.eq(target)]
        assert len(source) == len(published)
        for source_row, published_row in zip(source.itertuples(index=False), published.itertuples(index=False)):
            assert source_row.city == published_row[0]
            check(source_row.tracts, int(str(published_row[2]).replace(",", "")), f"Supplementary Table {number} | {source_row.city} | n", records)
            check(source_row.reference, float(published_row[3]), f"Supplementary Table {number} | {source_row.city} | reference", records)
            check(source_row.nlcd, float(published_row[4]), f"Supplementary Table {number} | {source_row.city} | NLCD", records)

    social = pd.read_csv(data / "local_social_gap_comparison.csv")
    published = pd.read_csv(tables / "Supplementary_Table_16.csv")
    dimension_names = {"Income quartile": "income_city_quartile", "Poverty quartile": "poverty_city_quartile", "Older-population-share quartile": "age65plus_city_quartile"}
    target_names = {"Early endpoint": "early", "Late endpoint": "late", "Net change": "change"}
    for row in published.itertuples(index=False):
        matched = social.loc[social.city.eq(row[0]) & social.group.eq(dimension_names[row[1]]) & social.target.eq(target_names[row[2]])]
        assert len(matched) == 1
        source_row = matched.iloc[0]
        for source_field, column in (("q1_n", 3), ("q4_n", 4), ("reference_q4_minus_q1", 5), ("nlcd_q4_minus_q1", 6)):
            check(source_row[source_field], float(str(row[column]).replace(",", "")), f"Supplementary Table 16 | {row[0]} | {row[1]} | {row[2]} | {source_field}", records)
        assert bool(source_row.same_direction) == (row[7] == "Yes")

    stages = pd.read_csv(data / "annual_stage_results.csv")
    published = pd.read_csv(tables / "Supplementary_Table_17.csv")
    stage_names = {"Pre-transition": "pre_2016", "Transition year": "transition_2015_2016", "Post-transition": "post_2016", "Full period": "full_period"}
    target_names = {
        "Older-population-share Q4−Q1 gap, equal-area urbanized areas": "age65plus_q4_minus_q1_equal_city",
        "Older-population-share Q4 pooled tract mean": "age65plus_q4_tract_mean",
        "Income Q4−Q1 gap, equal-area urbanized areas": "income_q4_minus_q1_equal_city",
        "National pooled tract mean": "national_tract_mean",
        "Poverty Q4−Q1 gap, equal-area urbanized areas": "poverty_q4_minus_q1_equal_city",
    }
    for row in published.itertuples(index=False):
        matched = stages.loc[stages.target.eq(target_names[row[0]]) & stages.period.eq(stage_names[row[1]])]
        assert len(matched) == 1
        for field, index in (("year_start", 2), ("year_end", 3), ("start_value", 4), ("end_value", 5), ("change_pp", 6)):
            check(matched.iloc[0][field], float(row[index]), f"Supplementary Table 17 | {row[0]} | {row[1]} | {field}", records)

    with (out / "comparisons.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item", "recalculated", "reported", "difference"])
        writer.writeheader()
        writer.writerows(records)
    (out / "summary.json").write_text(json.dumps({"status": "PASS", "comparisons": len(records), "product_tracts": len(product), "figure_5_tracts": len(pair), "figure_5_changed_classes": 9205}, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(records)} measurement-table comparisons; {len(pair)} Figure 5 tracts.")


if __name__ == "__main__":
    main()
