"""Provisional effect heterogeneity and descriptive fixed-boundary canopy equity.

No forest refit, formal confidence intervals, sample-authority mutation or figures.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from audit_temperature_measurement_equity_20261009 import ROOT, LOG, digest, read, require, write_json

BASE = LOG / "national_analysis_baseline_join_20260914_run01/national_analysis_baseline_readiness.csv"
GEOMETRY = LOG / "national_tract_urban_geometry_20260914_run01"
SAMPLE = LOG / "geographic_measurement_repair_20261009_run01/revised_analysis_sample.csv"
RUN = LOG / "geo_adjusted_national_orthogonal_20261009_run01"
VALIDATION = LOG / "geo_adjusted_national_orthogonal_validation_20261009_run01/validation.json"
CANOPY = LOG / "national_canopy_streaming_summary_20260915_run01/national_tract_year_canopy_1999_2025.csv"
KEY = ["state", "tract_id", "exposure_year", "outcome_year"]
ID = ["state", "tract_id"]
SOCIAL = {
    "income": ("income_point_usd", "income_point_usd_2000", 10000.),
    "poverty": ("poverty_share", "poverty_share_2000", .1),
    "age65plus": ("age65plus_share_sf3_p8", "age65plus_share_2000", .1),
}
Y = "urban_canopy_percent_fixed_land"


def tied_quartiles(values):
    values = np.asarray(values, dtype=float)
    require(len(values) > 0 and np.isfinite(values).all(), "Undefined quartile values")
    edges = np.quantile(values, [.25, .5, .75], method="linear")
    labels = np.searchsorted(edges, values, side="left")+1
    return labels, edges


def self_test():
    labels, edges = tied_quartiles([0., 0., 0., 1., 1., 1., 2., 2.])
    require(labels[0] == labels[1] == labels[2] and labels[3] == labels[4] == labels[5], "Ties split")
    labels, _ = tied_quartiles(np.ones(8))
    require((labels == 1).all(), "Degenerate quartiles silently randomized")
    x = np.arange(12, dtype=float)
    city = np.repeat(["a", "b", "c"], 4)
    f = pd.DataFrame({"x": x, "city": city, "y": 3*x+np.repeat([9., -4., 7.], 4)})
    xc = f.x-f.groupby("city").x.transform("mean")
    yc = f.y-f.groupby("city").y.transform("mean")
    require(np.isclose(np.linalg.lstsq(xc.to_numpy()[:,None], yc, rcond=None)[0][0], 3), "Within-city projection")
    print("PASS TEST_ONLY_SIMULATION: tied quartiles and within-city projection; no study data", flush=True)


def prepare_baseline():
    baseline = read(BASE)
    paths = sorted(GEOMETRY.glob("tract_urban_geometry_*.csv"))
    require(len(paths) == 49, "Urban geometry state count")
    geometry = pd.concat([read(p, usecols=[*ID, "positive_urban_count", "urban_ids_positive_area", "geometry_urban_fraction"],
                                converters={"urban_ids_positive_area": str}) for p in paths], ignore_index=True)
    require(len(geometry) == len(baseline) == 64999 and not geometry.duplicated(ID).any() and not baseline.duplicated(ID).any(), "Baseline identity")
    b = baseline.merge(geometry, on=ID, how="left", validate="one_to_one", indicator=True, sort=False)
    require(b._merge.eq("both").all() and b[ID].equals(baseline[ID]), "Urban geometry linkage")
    b = b.drop(columns="_merge")
    b["single_city_id"] = b.urban_ids_positive_area.where(b.positive_urban_count.eq(1))
    b["positive_land"] = b.urban_fixed_land_m2.gt(0)
    b["canopy_equity_balanced_26years"] = False
    thresholds = []
    for label, (field, _, _) in SOCIAL.items():
        b[f"{label}_city_quartile"] = np.nan
        b[f"{label}_group_status"] = "NOT_DEFINED_NO_POSITIVE_FIXED_LAND"
        b.loc[b.positive_land, f"{label}_group_status"] = "NOT_DEFINED_NOT_SINGLE_CITY"
        candidate = b.positive_land & b.positive_urban_count.eq(1)
        if label == "income":
            valid = b.income_status.eq("REPORTED_POINT_ESTIMATE") & b[field].notna()
        else:
            valid = b[field].notna() & np.isfinite(b[field])
        b.loc[candidate & ~valid, f"{label}_group_status"] = "NOT_DEFINED_MISSING_OR_INTERVAL_VALUE"
        for city, g in b.loc[candidate & valid].groupby("single_city_id", sort=True):
            labels, edges = tied_quartiles(g[field].to_numpy(float))
            b.loc[g.index, f"{label}_city_quartile"] = labels
            b.loc[g.index, f"{label}_group_status"] = "BASELINE_FIXED_CITY_GROUP_ASSIGNED"
            thresholds.append({"variable": label, "single_city_id": city, "baseline_tracts": len(g),
                               "q25": edges[0], "q50": edges[1], "q75": edges[2],
                               "groups_present": ";".join(map(str, sorted(set(labels)))),
                               "ties_split_by_id": False, "basis": "FULL_BASELINE_NOT_TEMPERATURE_SAMPLE"})
    return b, pd.DataFrame(thresholds), paths


def point_summary(g, value="orthogonal_score"):
    require(len(g) > 0, "Empty summary")
    return {"rows": len(g), "tracts": int(g.tract_id.nunique()),
            "effect_c_per_pp_mean": float(g[value].mean()),
            "forest_effect_c_per_pp_mean": float(g.theta.mean()),
            "mean_treatment_change_pp": float(g.treatment_pp_change_from_2000.mean()),
            "mean_baseline_canopy_percent": float(g.baseline_canopy_percent_fixed_land_2000.mean()),
            "role": "PROVISIONAL_EFFECT_POINT_ESTIMATE_NO_FORMAL_INTERVAL"}


def projections(f, label):
    records = []
    years = [2014, 2019, 2024]
    year_columns = [f.outcome_year.eq(y).to_numpy(float) for y in years[1:] if f.outcome_year.eq(y).any()]
    # The population term matches the fourth frozen effect modifier.
    continuous = [f[field].to_numpy(float)/scale for _,field,scale in SOCIAL.values()]
    population = f.population_2000.to_numpy(float)/10000.
    joint = np.column_stack([np.ones(len(f)), *year_columns, *continuous, population])
    require(np.linalg.matrix_rank(joint) == joint.shape[1], "Rank-deficient continuous effect projection")
    for outcome in ["orthogonal_score", "theta"]:
        y = f[outcome].to_numpy(float)
        beta = np.linalg.lstsq(joint, y, rcond=None)[0]
        start = 1+len(year_columns)
        for j,name in enumerate([*SOCIAL, "population"]):
            records.append({"sample": label, "projection": "JOINT_CONTINUOUS_WITH_YEAR_AND_POPULATION",
                            "outcome": outcome, "variable": name, "coefficient": float(beta[start+j]),
                            "rows": len(f), "tracts": int(f.tract_id.nunique()),
                            "unit": "10000_usd" if name == "income" else "10000_people" if name == "population" else "10_percentage_points",
                            "role": "SOCIAL_GRADIENT_OF_CONDITIONAL_EFFECT_NOT_SOCIAL_VARIABLE_CAUSAL_EFFECT"})
        for name,x in zip(SOCIAL,continuous):
            design = np.column_stack([np.ones(len(f)), *year_columns, x])
            require(np.linalg.matrix_rank(design) == design.shape[1], "Univariate projection rank")
            beta = np.linalg.lstsq(design, y, rcond=None)[0]
            records.append({"sample": label, "projection": "ONE_SOCIAL_VARIABLE_WITH_YEAR", "outcome": outcome,
                            "variable": name, "coefficient": float(beta[-1]), "rows": len(f),
                            "tracts": int(f.tract_id.nunique()), "unit": "10000_usd" if name == "income" else "10_percentage_points",
                            "role": "SOCIAL_GRADIENT_OF_CONDITIONAL_EFFECT_NOT_SOCIAL_VARIABLE_CAUSAL_EFFECT"})
    return records


def summarize_heterogeneity(f, out):
    sets = {"all_frozen_rows": f}
    complete_ids = f.groupby(ID).outcome_year.nunique().loc[lambda n: n.eq(3)].index
    sets["same_tracts_all_three_outcomes"] = f.loc[pd.MultiIndex.from_frame(f[ID]).isin(complete_ids)]
    for cutoff in [1000, 2000]:
        sets[f"baseline_population_at_least_{cutoff}"] = f.loc[f.population_2000.ge(cutoff)]
    summary, group_rows, projection_rows = [], [], []
    for name,g in sets.items():
        require(len(g) > 0, f"Empty predeclared sensitivity {name}")
        summary.append({"sample": name, "outcome_year": "all", **point_summary(g)})
        projection_rows.extend(projections(g, name))
        for year,year_frame in g.groupby("outcome_year"):
            summary.append({"sample": name, "outcome_year": int(year), **point_summary(year_frame)})
        for variable in SOCIAL:
            field = f"{variable}_city_quartile"
            for year in ["all", 2014, 2019, 2024]:
                frame = g if year == "all" else g.loc[g.outcome_year.eq(year)]
                for quartile,q in frame.loc[frame[field].notna()].groupby(field):
                    group_rows.append({"sample": name, "outcome_year": year, "variable": variable,
                                       "quartile": int(quartile), "cities": int(q.single_city_id.nunique()), **point_summary(q)})
                unassigned = frame.loc[frame[field].isna()]
                group_rows.append({"sample": name, "outcome_year": year, "variable": variable,
                                   "quartile": "unassigned_display_only", "cities": None,
                                   **point_summary(unassigned)} if len(unassigned) else
                                  {"sample": name, "outcome_year": year, "variable": variable,
                                   "quartile": "unassigned_display_only", "rows": 0, "tracts": 0})
    pd.DataFrame(summary).to_csv(out / "effect_sample_summaries.csv", index=False)
    pd.DataFrame(group_rows).to_csv(out / "city_quartile_effect_summaries_PROVISIONAL.csv", index=False)
    pd.DataFrame(projection_rows).to_csv(out / "continuous_effect_gradients_PROVISIONAL.csv", index=False)
    # Same-city contrasts distinguish relative local rank from national composition.
    contrasts = []
    for variable in SOCIAL:
        field = f"{variable}_city_quartile"
        for year in [2014, 2019, 2024]:
            frame = f.loc[f.outcome_year.eq(year) & f[field].isin([1,4])]
            table = frame.groupby(["single_city_id", field]).agg(effect=("orthogonal_score","mean"),
                        forest=("theta","mean"), rows=("tract_id","size")).unstack(field)
            if 1. not in table.effect.columns or 4. not in table.effect.columns:
                continue
            for city,row in table.dropna(subset=[("effect",1.),("effect",4.)]).iterrows():
                contrasts.append({"variable": variable, "outcome_year": year, "single_city_id": city,
                                  "high_minus_low_effect_c_per_pp": row["effect",4.]-row["effect",1.],
                                  "high_minus_low_forest_c_per_pp": row["forest",4.]-row["forest",1.],
                                  "low_group_rows": int(row["rows",1.]), "high_group_rows": int(row["rows",4.]),
                                  "role": "SAME_CITY_DIFFERENCE_POINT_ONLY_NO_FORMAL_INFERENCE"})
    pd.DataFrame(contrasts).to_csv(out / "same_city_effect_contrasts_PROVISIONAL.csv", index=False)
    return {name:{"rows":len(g),"tracts":int(g.tract_id.nunique())} for name,g in sets.items()}


def canopy_trajectories(b, out):
    f = read(CANOPY)
    require(len(f) == 1754973 and not f.duplicated([*ID,"year"]).any(), "Canopy identity")
    f = f.loc[f.year.between(2000,2025)].copy()
    require(len(f) == len(b)*26, "Canopy main-year accounting")
    keep = [*ID,"positive_land","single_city_id","positive_urban_count","income_status",
            "population_2000_100_percent", "income_point_usd", "poverty_share", "age65plus_share_sf3_p8",
            *[f"{v}_city_quartile" for v in SOCIAL]]
    f = f.merge(b[keep],on=ID,how="left",validate="many_to_one",indicator=True,sort=False)
    require(len(f)==len(b)*26 and f._merge.eq("both").all(), "Canopy baseline join")
    eligible = f.positive_land & f.publication_status.eq("FULL_FIXED_LAND_ESTIMATE_AVAILABLE") & f[Y].notna()
    require(np.isfinite(f.loc[eligible,Y]).all() and f.loc[eligible,Y].between(0,100).all(), "Canopy physical range")
    count = f.loc[eligible].groupby(ID).year.nunique()
    balanced = count.loc[count.eq(26)].index
    base_year = f.loc[f.year.eq(2000),[*ID,Y,"publication_status"]].rename(
        columns={Y:"canopy_baseline_2000", "publication_status":"baseline_publication_status"})
    f = f.drop(columns="_merge").merge(base_year,on=ID,how="left",validate="many_to_one",sort=False)
    f["complete_fixed_land_year"] = eligible.to_numpy()
    f["balanced_26years"] = pd.MultiIndex.from_frame(f[ID]).isin(balanced)
    f["change_from_2000_pp"] = (f[Y]-f.canopy_baseline_2000).where(
        f.complete_fixed_land_year & f.baseline_publication_status.eq("FULL_FIXED_LAND_ESTIMATE_AVAILABLE"))
    flow = f.groupby("year").agg(input_rows=("tract_id","size"), positive_land_rows=("positive_land","sum"),
                   full_fixed_land_rows=("complete_fixed_land_year","sum"), balanced_rows=("balanced_26years","sum"))
    flow["positive_land_incomplete"] = flow.positive_land_rows-flow.full_fixed_land_rows
    flow["zero_land_rows"] = flow.input_rows-flow.positive_land_rows
    flow.reset_index().to_csv(out / "annual_canopy_coverage_flow.csv",index=False)
    series, continuous, city_gaps, fe_rows = [],[],[],[]
    for panel in ["year_available", "balanced_26years"]:
        d = f.loc[f.complete_fixed_land_year & (f.balanced_26years if panel=="balanced_26years" else True)].copy()
        for year,g in d.groupby("year"):
            for variable,(field,_,scale) in SOCIAL.items():
                for q,h in g.loc[g[f"{variable}_city_quartile"].notna()].groupby(f"{variable}_city_quartile"):
                    series.append({"panel":panel,"year":int(year),"variable":variable,"quartile":int(q),
                                   "tracts":len(h),"cities":int(h.single_city_id.nunique()),
                                   "canopy_percent_mean":float(h[Y].mean()),
                                   "change_from_2000_pp_mean":float(h.change_from_2000_pp.mean()),
                                   "baseline_to_year_complete_pairs":int(h.change_from_2000_pp.notna().sum()),
                                   "fraction_positive_net_change":float(h.change_from_2000_pp.dropna().gt(0).mean()),
                                   "role":"TREE_RESOURCE_TRAJECTORY_NOT_TEMPERATURE_OR_HEALTH_BENEFIT"})
                good = g[field].notna() & (g.income_status.eq("REPORTED_POINT_ESTIMATE") if variable=="income" else True)
                h=g.loc[good]
                x=h[field].to_numpy(float)/scale; a=np.column_stack([np.ones(len(h)),x])
                require(np.linalg.matrix_rank(a)==2,"Canopy continuous projection rank")
                beta=np.linalg.lstsq(a,h[Y].to_numpy(float),rcond=None)[0]
                continuous.append({"panel":panel,"year":int(year),"variable":variable,"tracts":len(h),
                                   "continuous_canopy_gradient":float(beta[1]),"projection":"NATIONAL_ONE_SOCIAL_VARIABLE",
                                   "role":"DESCRIPTIVE_RESOURCE_SOCIAL_GRADIENT"})
                qfield=f"{variable}_city_quartile"
                for city,h in g.loc[g[qfield].isin([1,4])].groupby("single_city_id"):
                    means=h.groupby(qfield)[Y].mean()
                    if 1 in means.index and 4 in means.index:
                        city_gaps.append({"panel":panel,"year":int(year),"variable":variable,"single_city_id":city,
                                          "high_minus_low_canopy_pp":float(means.loc[4]-means.loc[1]),
                                          "low_tracts":int(h[qfield].eq(1).sum()),"high_tracts":int(h[qfield].eq(4).sum())})
        if panel=="balanced_26years":
            # In a balanced panel this reproduces social-by-year contrasts of the
            # tract FE + city-by-year FE model relative to 2000, without huge dummies.
            complete=d.single_city_id.notna() & d[[v[0] for v in SOCIAL.values()]].notna().all(axis=1) & d.income_status.eq("REPORTED_POINT_ESTIMATE")
            z=d.loc[complete].copy()
            for variable,(field,_,scale) in SOCIAL.items():
                z[f"scaled_{variable}"]=z[field]/scale
            fields=[f"scaled_{v}" for v in SOCIAL]
            for year,h in z.groupby("year"):
                x=(h[fields]-h.groupby("single_city_id")[fields].transform("mean")).to_numpy(float)
                y=(h.change_from_2000_pp-h.groupby("single_city_id").change_from_2000_pp.transform("mean")).to_numpy(float)
                require(np.linalg.matrix_rank(x)==3 and np.isfinite(y).all(),"Within-city balanced projection")
                beta=np.linalg.lstsq(x,y,rcond=None)[0]
                for j,variable in enumerate(SOCIAL):
                    fe_rows.append({"year":int(year),"variable":variable,"coefficient_change_gap_pp":float(beta[j]),
                                    "tracts":len(h),"cities":int(h.single_city_id.nunique()),
                                    "estimator":"BALANCED_TRACT_FE_CITY_BY_YEAR_FE_SOCIAL_BY_YEAR_RELATIVE_2000",
                                    "role":"DESCRIPTIVE_RESOURCE_ALLOCATION_POINT_ESTIMATE_NO_CAUSAL_POLICY_CLAIM"})
    pd.DataFrame(series).to_csv(out / "annual_canopy_city_group_trajectories.csv",index=False)
    pd.DataFrame(continuous).to_csv(out / "annual_canopy_continuous_gradients.csv",index=False)
    gaps=pd.DataFrame(city_gaps)
    gaps.to_csv(out / "annual_same_city_canopy_gaps.csv",index=False)
    gaps.groupby(["panel","year","variable"]).agg(equal_city_mean_gap_pp=("high_minus_low_canopy_pp","mean"),
                   median_city_gap_pp=("high_minus_low_canopy_pp","median"),cities=("single_city_id","nunique")).reset_index().to_csv(
                   out / "annual_same_city_canopy_gap_summaries.csv",index=False)
    pd.DataFrame(fe_rows).to_csv(out / "annual_canopy_fixed_effect_social_trajectories.csv",index=False)
    # Full row manifest preserves ineligible and unassigned records explicitly.
    f[[*ID,"year","publication_status","positive_land","complete_fixed_land_year","balanced_26years",
       "single_city_id",*[f"{v}_city_quartile" for v in SOCIAL],Y,"change_from_2000_pp"]].to_csv(
           out / "annual_canopy_equity_row_manifest.csv.gz",index=False,compression="gzip")
    b["canopy_equity_balanced_26years"]=pd.MultiIndex.from_frame(b[ID]).isin(balanced)
    return {"input_rows_27years":1754973,"main_rows_26years":len(f),"positive_land_tracts":int(b.positive_land.sum()),
            "balanced_26year_tracts":len(balanced),"eligible_tract_year_rows":int(f.complete_fixed_land_year.sum()),
            "zero_land_tracts":int((~b.positive_land).sum()),"duplicate_keys":0,"missing_as_zero":False,
            "fixed_effect_years":int(pd.DataFrame(fe_rows).year.nunique()),"fixed_effect_joint_social_variables":3}


def execute(args):
    start=time.monotonic()
    require(not args.out.exists(),"Refuse output overwrite")
    measurement=json.loads(args.measurement.read_text(encoding="utf-8"))
    require(measurement["status"]=="COMPLETED_TARGETED_MEASUREMENT_APPLICABILITY_ADJUDICATION", "Measurement adjudication required")
    require(all(digest(args.measurement.parent/p)==h for p,h in measurement["output_file_sha256"].items()), "Measurement outputs changed")
    validator=json.loads(VALIDATION.read_text(encoding="utf-8"))
    require(validator["maximum_recompute_error"]==0 and validator["input_audit_sha256"]==digest(RUN/"audit.json"), "Model reload gate")
    receipt=json.loads((RUN/"audit.json").read_text(encoding="utf-8"))
    require(all(digest(RUN/p)==h for p,h in receipt["outputs"].items()),"Model outputs changed")
    sources=[Path(__file__),args.measurement,BASE,SAMPLE,CANOPY,RUN/"crossfit_scores.csv",RUN/"audit.json",VALIDATION,
             ROOT/"00_admin/project_state.yaml",ROOT/"05_scripts/primary_canopy_model_geo_adjusted_20261009.json",
             ROOT/"03_protocol/national_age_source_adjudication_20260914_zh.md"]
    b,thresholds,geo=prepare_baseline()
    sources.extend(geo)
    hashes={str(p):digest(p) for p in sources}
    sample=read(SAMPLE); scores=read(RUN/"crossfit_scores.csv")
    require(len(sample)==len(scores)==104371 and sample[KEY].equals(scores[KEY]),"Effect sample keys/order")
    require(np.array_equal(scores.analysis_row.to_numpy(),np.arange(len(sample))),"Effect score row order")
    require(sample.spatial_fold.equals(scores.spatial_fold),"Frozen fold identity")
    score=scores.theta+scores.residual_t*(scores.residual_y-scores.intercept-scores.theta*scores.residual_t)/scores.conditional_variance
    require(np.allclose(score,scores.orthogonal_score,atol=1e-12,rtol=1e-12),"Orthogonal score recompute")
    f=sample.copy()
    for field in ["theta","orthogonal_score"]:
        f[field]=scores[field].to_numpy()
    group_fields=["single_city_id","positive_urban_count",*[f"{v}_city_quartile" for v in SOCIAL],*[f"{v}_group_status" for v in SOCIAL]]
    f=f.merge(b[[*ID,*group_fields]],on=ID,how="left",validate="many_to_one",indicator=True,sort=False)
    require(len(f)==len(sample) and f[KEY].equals(sample[KEY]) and f._merge.eq("both").all(),"Effect baseline group join")
    f=f.drop(columns="_merge")
    args.out.mkdir()
    thresholds.to_csv(args.out/"fixed_baseline_city_quartile_thresholds.csv",index=False)
    sets=summarize_heterogeneity(f,args.out)
    print("COMPLETED provisional cooling-effect social projections and display groups",flush=True)
    f[[*KEY,"spatial_fold","spatial_block_id","theta","orthogonal_score","treatment_pp_change_from_2000",
       "income_point_usd_2000","poverty_share_2000","age65plus_share_2000","population_2000",*group_fields]].to_csv(
           args.out/"effect_equity_row_manifest.csv",index=False)
    canopy=canopy_trajectories(b,args.out)
    b[[*ID,"positive_land","positive_urban_count","single_city_id","income_status", "canopy_equity_balanced_26years",
       *[f"{v}_group_status" for v in SOCIAL],*[f"{v}_city_quartile" for v in SOCIAL]]].to_csv(args.out/"baseline_equity_group_manifest.csv",index=False)
    require(all(digest(p)==h for p,h in hashes.items()),"Source mutation during equity analysis")
    audit={"status":"COMPLETED_PROVISIONAL_EFFECT_HETEROGENEITY_AND_DESCRIPTIVE_CANOPY_EQUITY",
           "effect_sample_rows":len(f),"effect_sample_tracts":int(f.tract_id.nunique()),"baseline_tracts":len(b),
           "effect_rows_dropped":0,"effect_duplicates":int(f.duplicated(KEY).sum()),"baseline_unmatched":0,
           "display_unassigned_rows":{v:int(f[f"{v}_city_quartile"].isna().sum()) for v in SOCIAL},
           "baseline_multi_city_tracts":int(b.positive_urban_count.gt(1).sum()),"effect_sensitivity_samples":sets,
           "canopy":canopy,"input_file_sha256":hashes,"new_forest_fits":0,"formal_intervals_computed":False,
           "formal_causal_equity_release":False,"independent_temperature_anomaly_ruled_out":False,
           "temperature_applicability":measurement["measurement_applicability"],
           "individual_health_benefits_estimated":False,"realized_cooling_equals_effect_times_cumulative_change":False,
           "raw_or_formal_inputs_modified":False,"administrative_state_modified":False,
           "elapsed_seconds":time.monotonic()-start}
    audit["output_file_sha256"]={p.name:digest(p) for p in args.out.iterdir() if p.is_file()}
    write_json(args.out/"audit.json",audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ["input_file_sha256","output_file_sha256"]},ensure_ascii=False),flush=True)


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    p.add_argument("--measurement",type=Path);p.add_argument("--out",type=Path)
    a=p.parse_args()
    if a.self_test:self_test()
    else:
        require(a.measurement is not None and a.out is not None,"Missing paths")
        execute(a)
