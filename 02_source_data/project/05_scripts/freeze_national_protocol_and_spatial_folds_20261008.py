"""Freeze candidate support/adjustment rules and create national spatial folds.

The artifact is a protocol/fold freeze candidate. It does not alter project
authority or select the formal analysis sample.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
MATCH = ROOT / "06_logs/national_gridmet_date_matching_20261008_run01"
SUPPORT = ROOT / "06_logs/weather_adjustment_support_qualification_20261008_run01"
CATALOG = ROOT / "06_logs/national_urban_weights_summary_20260914_run01/national_weight_catalog.csv"
STATES = ["01", "04", "05", "06", "08", "09", "10", "11", "12", "13", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "44", "45", "46", "47", "48", "49", "50", "51", "53", "54", "55", "56"]
FOLDS = 5
BLOCK_M = 25_000.0


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def csv(path, **kwargs):
    return pd.read_csv(path, dtype={"state": str, "tract_id": str}, float_precision="round_trip", **kwargs)


def run(out):
    out = out.resolve()
    require(not out.exists(), "Refusing output overwrite")
    out.mkdir(); (out / "folds").mkdir()
    panel = csv(MATCH / "national_tract_year_lst_weather_unqualified.csv")
    require(len(panel) == 140181 and not panel.duplicated(["state", "tract_id", "year"]).any(), "Panel keys")
    support_audit = json.loads((SUPPORT / "audit.json").read_text(encoding="utf-8"))
    match_audit = json.loads((MATCH / "audit.json").read_text(encoding="utf-8"))
    require(match_audit["status"] == "PASS_DATE_MATCHED_WEATHER_AND_DESCRIPTIVE_MISSINGNESS_NOT_FORMAL_OUTCOME", "Weather gate")
    catalog = csv(CATALOG)
    require(catalog.state.tolist() == STATES, "State catalog order")
    with h5py_file(ROOT / "04_pilot/gridMET/raw/tmmx_2014.nc") as f:
        lon, lat = f["lon"][:], f["lat"][:]
    ncell = len(lon) * len(lat)
    to_albers = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    rows = []
    sources = [MATCH / "national_tract_year_lst_weather_unqualified.csv", MATCH / "audit.json", SUPPORT / "audit.json", CATALOG]
    for state in STATES:
        matrix = sparse.load_npz(MATCH / "weights" / f"state_{state}_fixed_land_weather_weights.npz").tocsr()
        row_table = csv(MATCH / "weights" / f"state_{state}_weather_weight_rows.csv")
        require(matrix.shape[0] == len(row_table), f"Weight row count {state}")
        require(row_table.tract_id.tolist() == sorted(row_table.tract_id.tolist()), f"Tract order {state}")
        require(np.allclose(np.asarray(matrix.sum(axis=1)).ravel(), row_table.fixed_land_area_m2, rtol=1e-10, atol=.01), f"Weight closure {state}")
        valid_cell = matrix.indices < ncell
        rr, cc = np.divmod(np.minimum(matrix.indices, ncell - 1), len(lon))
        cell_lon, cell_lat = lon[cc], lat[rr]
        x, y = to_albers.transform(cell_lon, cell_lat)
        start = matrix.indptr[:-1]
        end = matrix.indptr[1:]
        # Sparse rows are traversed deterministically without dense expansion.
        for i, tract in enumerate(row_table.tract_id):
            lo, hi = start[i], end[i]
            sel = valid_cell[lo:hi]
            if not sel.any():
                continue
            row_data = matrix.data[lo:hi]
            wx = row_data[sel]; xx = x[lo:hi][sel]; yy = y[lo:hi][sel]
            total = wx.sum()
            cx, cy = float((wx * xx).sum() / total), float((wx * yy).sum() / total)
            bx, by = int(np.floor(cx / BLOCK_M)), int(np.floor(cy / BLOCK_M))
            rows.append({"state": state, "tract_id": tract, "fixed_land_area_m2": float(total),
                         "centroid_epsg5070_x_m": cx, "centroid_epsg5070_y_m": cy,
                         "block_size_m": BLOCK_M, "block_x": bx, "block_y": by,
                         "spatial_block_id": f"EPSG5070_{bx}_{by}"})
    folds = pd.DataFrame(rows)
    require(len(folds) == 46727 and not folds.duplicated(["state", "tract_id"]).any(), "National fold rows")
    require(set(zip(folds.state, folds.tract_id)) == set(zip(panel.state, panel.tract_id)), "Fold/panel key set")
    block_order = (folds[["spatial_block_id", "block_y", "block_x"]].drop_duplicates()
                   .sort_values(["block_y", "block_x", "spatial_block_id"], kind="mergesort").reset_index(drop=True))
    block_order["spatial_fold"] = np.arange(len(block_order)) % FOLDS
    folds = folds.merge(block_order[["spatial_block_id", "spatial_fold"]], on="spatial_block_id", how="left", validate="many_to_one")
    require(folds.spatial_fold.notna().all() and folds.spatial_fold.astype(int).between(0, FOLDS-1).all(), "Fold assignment")
    folds.spatial_fold = folds.spatial_fold.astype(int)
    folds = folds.sort_values(["state", "tract_id"], kind="mergesort").reset_index(drop=True)
    folds.to_csv(out / "folds" / "national_tract_spatial_fold_map.csv", index=False)
    block_order.to_csv(out / "folds" / "national_spatial_block_order.csv", index=False)
    # Year expansion is for fitting joins; fold membership is tract-level and
    # identical in all years, avoiding time-dependent fold leakage.
    panel_keys = panel[["state", "tract_id", "year"]].merge(folds[["state", "tract_id", "spatial_block_id", "spatial_fold"]], on=["state", "tract_id"], how="left", validate="many_to_one")
    require(len(panel_keys) == len(panel) and not panel_keys.spatial_fold.isna().any(), "Panel fold join")
    panel_keys.to_csv(out / "folds" / "national_tract_year_spatial_fold_manifest.csv", index=False)
    fold_counts = folds.spatial_fold.value_counts().sort_index().to_dict()
    block_counts = folds.groupby("spatial_fold").spatial_block_id.nunique().sort_index().to_dict()
    source_hashes = {str(p): digest(p) for p in sources}
    protocol = out / "冻结方案_天气调整_观测资格_空间折_20261008.md"
    protocol.write_text("\n".join([
        "# 冻结方案：天气调整、观测支持资格与全国空间折", "",
        f"冻结日期：{dt.date.today().isoformat()}。版本：NATIONAL_FREEZE_CANDIDATE_V1。",
        "", "## 冻结范围", "",
        "本文件冻结测量和交叉拟合输入口径，不冻结正式样本结果，不运行全国正式因果模型，不修改项目正式状态文件。",
        "", "## 观测支持资格标准", "",
        "小区—年至少6个温度有效采集日；至少覆盖4个月；自身几何机会日平均有效温度面积比例至少25%；"
        "5月1日、9月30日及相邻有效日期计算的最大空缺不超过64天；气象格网对固定陆地完整支持。",
        "资格标准不依赖处理变量、结局、显著性或模型性能。未达标记录保留在数据集中，并作为完整记录描述和缺测敏感性分析的输入。",
        "", "## 天气标准化估计量", "",
        "日层面采用最高气温、最低气温、降水、日期正弦项和日期余弦项，在州—年份内估计地表温度的天气构成关系。"
        "每个有效采集日的地表温度减去该日天气相对于同州—年份完整5—9月参考天气的预测差异，再对有效日期等权平均。",
        "水汽压亏缺不纳入主要标准化变量，保留为敏感性分析变量，因为它可能代表植被蒸散路径的一部分。",
        "调整结果解释为天气标准化的卫星过境时段地表温度，不解释为全天候温度或个人热暴露。",
        "", "## 全国空间折", "",
        "以冻结城市内固定陆地面积权重计算小区面积加权中心；将中心投影到EPSG:5070（美国本土阿尔伯斯等面积圆锥投影），"
        "按25,000米网格形成空间块。空间块按固定的南北、东西坐标顺序编号后，以编号对5取模分配到5折。",
        "同一空间块的小区永远在同一折；所有年份使用同一小区折，避免时间变化导致折泄漏。"
        "折分配没有使用温度、树冠、天气、收入、年龄或效应结果。",
        f"共有{len(block_order):,}个空间块、{len(folds):,}个小区；折数量：{fold_counts}；每折空间块数量：{block_counts}。",
        "该空间折用于分组交叉拟合和空间留出检验；它不能证明没有空间混杂。",
        "", "## 预试跑边界", "",
        "正式模型前的预试跑只检查数据连接、折完整性、变量缺失、模型接口和输出审计。"
        "不使用预试跑的效应大小、显著性或拟合指标调整资格标准、天气变量、折或随机种子。",
        "", "## 变更控制", "",
        "若原始文件、树冠指标、固定土地权重、天气文件或年度温度表发生变化，必须重新生成版本并重新计算全部哈希；"
        "不得在原版本上静默续跑。"
    ]), encoding="utf-8")
    audit = {"status": "PASS_NATIONAL_FREEZE_CANDIDATE_NOT_PROJECT_AUTHORITY", "version": "NATIONAL_FREEZE_CANDIDATE_V1",
             "support_rule": "valid_dates>=6;valid_months>=4;mean_valid_fraction_geometric>=0.25;maximum_gap<=64;full_weather_support",
             "weather_primary_variables": ["tmmx_c_mean_available_land", "tmmn_c_mean_available_land", "pr_mm_mean_available_land", "sin_doy", "cos_doy"],
             "weather_sensitivity_variable": "vpd_kpa_mean_available_land", "folds": FOLDS, "block_size_m": BLOCK_M,
             "tracts": len(folds), "blocks": len(block_order), "fold_counts": {str(k): int(v) for k,v in fold_counts.items()},
             "block_counts": {str(k): int(v) for k,v in block_counts.items()}, "years": [2014, 2019, 2024],
             "formal_sample_selected": False, "formal_model": "NOT_RUN", "project_authority_modified": False,
             "source_hashes": source_hashes, "protocol_sha256_file_bytes": digest(protocol)}
    (out / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))


class h5py_file:
    def __init__(self, path): self.path = path
    def __enter__(self):
        import h5py
        self.f = h5py.File(self.path, "r"); return self.f
    def __exit__(self, *args): self.f.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True)
    run(ap.parse_args().out)
