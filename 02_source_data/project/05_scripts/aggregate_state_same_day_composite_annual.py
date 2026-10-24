"""Aggregate same-day composite LST to tract-year without dropping rows."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--state", required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    input_path = args.input.resolve()
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    with input_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["state"] != args.state:
                raise RuntimeError(f"Unexpected state: {row['state']}")
            groups[(row["tract_id"], int(row["year"]))].append(row)
    if not groups:
        raise RuntimeError("Input contains no rows")
    fields = [
        "state", "tract_id", "year", "annual_lst_c_mean", "valid_temperature_dates",
        "candidate_dates", "valid_date_fraction", "valid_month_count", "months",
        "mean_valid_fraction_across_dates", "mean_valid_fraction_on_valid_dates",
        "first_valid_date", "last_valid_date", "longest_gap_days", "annual_support_status",
    ]
    output_rows = []
    for (tract_id, year), rows in sorted(groups.items()):
        dates = sorted({row["observation_date"] for row in rows})
        valid = [row for row in rows if row["status"] == "VALID_TEMPERATURE" and row["temperature_c_mean_valid"] != ""]
        valid_dates = sorted({row["observation_date"] for row in valid})
        months = sorted({int(row["observation_date"][5:7]) for row in rows})
        valid_months = sorted({int(row["observation_date"][5:7]) for row in valid})
        values = [float(row["temperature_c_mean_valid"]) for row in valid]
        fractions = [float(row["valid_fraction"]) for row in rows]
        valid_fractions = [float(row["valid_fraction"]) for row in valid]
        gaps = [(__import__("datetime").date.fromisoformat(b) - __import__("datetime").date.fromisoformat(a)).days for a, b in zip(valid_dates, valid_dates[1:])]
        mean_lst = sum(values) / len(values) if values else None
        mean_fraction = sum(fractions) / len(fractions)
        mean_valid_fraction = sum(valid_fractions) / len(valid_fractions) if valid_fractions else None
        # Descriptive support flag only; no rows are removed and no causal sample is selected.
        support = "QUALIFIED_SUPPORT_DIAGNOSTIC" if len(valid_dates) >= 4 and len(valid_months) >= 3 and mean_fraction >= 0.2 else "INSUFFICIENT_SUPPORT"
        output_rows.append({
            "state": args.state, "tract_id": tract_id, "year": year,
            "annual_lst_c_mean": mean_lst, "valid_temperature_dates": len(valid_dates),
            "candidate_dates": len(dates), "valid_date_fraction": len(valid_dates) / len(dates),
            "valid_month_count": len(valid_months), "months": ";".join(map(str, months)),
            "mean_valid_fraction_across_dates": mean_fraction,
            "mean_valid_fraction_on_valid_dates": mean_valid_fraction,
            "first_valid_date": valid_dates[0] if valid_dates else "",
            "last_valid_date": valid_dates[-1] if valid_dates else "",
            "longest_gap_days": max(gaps) if gaps else None,
            "annual_support_status": support,
        })
    output_path = args.out / f"state_{args.state}_tract_year_lst_outcome_diagnostic.csv"
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(output_rows)
    from collections import Counter
    audit = {
        "status": "PASS_STATE_ANNUAL_LST_OUTCOME_DIAGNOSTIC_NOT_FORMAL_OUTCOME",
        "state": args.state, "input_sha256_file_bytes": sha256(input_path),
        "tract_year_rows": len(output_rows), "years": sorted({r["year"] for r in output_rows}),
        "support_status_counts": dict(Counter(r["annual_support_status"] for r in output_rows)),
        "annual_aggregation": "equal-weight mean of valid same-day pixel composites; no imputation; no row deletion",
        "support_flag_rule": "valid dates >=4, valid months >=3, mean candidate-date valid fraction >=0.2; diagnostic only",
        "formal_sample_selected": False, "formal_model": "NOT_RUN", "raw_data_modified": False,
        "output_sha256_file_bytes": sha256(output_path),
    }
    (args.out / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
