"""Independent deterministic recomputation using csv, math.fsum and bisect only."""
from pathlib import Path
from collections import defaultdict
from bisect import bisect_right
import csv
import math
import json
import hashlib

OUT = Path(__file__).resolve().parent
ATOL = 1e-10
DIMS = ('income', 'poverty', 'age65plus')
GROUPS = ('Q1', 'Q2', 'Q3', 'Q4', 'unassigned')
comparisons = 0
max_error = 0.0
failures = []

def read(name):
    with (OUT / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def compare(key, actual, expected):
    global comparisons, max_error
    comparisons += 1
    if expected is None:
        ok = actual == ''
    elif isinstance(expected, str):
        ok = actual == expected
    elif isinstance(expected, bool):
        ok = actual == str(expected)
    else:
        err = abs(float(actual) - expected) if actual != '' else float('inf')
        max_error = max(max_error, err)
        ok = err == 0 if isinstance(expected, int) else err <= ATOL
    if not ok:
        failures.append(dict(key=str(key), actual=actual, expected=expected))

def mean(values):
    return math.fsum(values) / len(values)

def stats(values):
    n = len(values)
    result = dict(n_tracts=n)
    for i, col in enumerate(('canopy_2000_mean', 'canopy_2025_mean', 'delta_pp_mean')):
        result[col] = mean([v[i] for v in values]) if n else None
    deltas = sorted(v[2] for v in values)
    for q, name in ((.25, 'p25'), (.5, 'median'), (.75, 'p75')):
        if not n:
            result['delta_' + name + '_pp'] = None
        else:
            pos = (n - 1) * q
            lo, hi = math.floor(pos), math.ceil(pos)
            result['delta_' + name + '_pp'] = deltas[lo] + (deltas[hi] - deltas[lo]) * (pos - lo)
    for name, predicate in (('gain', lambda x:x > 0), ('zero', lambda x:x == 0), ('loss', lambda x:x < 0)):
        count = sum(predicate(x) for x in deltas)
        result['n_' + name] = count
        result[name + '_share'] = count / n if n else None
    return result

def diffs(low, high):
    return dict(D_2000_pp=high['canopy_2000_mean'] - low['canopy_2000_mean'],
                D_2025_pp=high['canopy_2025_mean'] - low['canopy_2025_mean'],
                D_delta_pp=high['delta_pp_mean'] - low['delta_pp_mean'],
                loss_share_difference_pp=100 * (high['loss_share'] - low['loss_share']))

def main():
    prov = json.loads((OUT / 'J1_input_provenance.json').read_text(encoding='utf-8'))
    src = Path(next(x['path'] for x in prov['inputs'] if x['role'] == 'paired'))
    national = defaultdict(list)
    local = defaultdict(list)
    cities = set()
    seen = set()
    with src.open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            key = (r['state'], r['tract_id'])
            if key in seen:
                raise RuntimeError('Duplicate source key')
            seen.add(key)
            c0, c25 = float(r['canopy_2000']), float(r['canopy_2025'])
            values = (c0, c25, c25 - c0)
            if not 0 <= c25 <= 100:
                raise RuntimeError('Out-of-range endpoint')
            b = 19 if c25 == 100 else bisect_right(list(range(0, 101, 5)), c25) - 1
            c = r['single_city_id']
            if c:
                cities.add(c)
            for d in DIMS:
                label = r[d + '_city_quartile']
                g = 'unassigned' if label == '' else 'Q' + str(int(float(label)))
                national[(d, b, g)].append(values)
                if c:
                    local[(d, b, c, g)].append(values)
    ns = {k:stats(v) for k,v in national.items()}
    ls = {k:stats(v) for k,v in local.items()}
    empty = stats([])
    groups = read('J1_group_by_end2025_band.csv')
    expected_group_keys = {(d,b,g) for d in DIMS for b in range(20) for g in GROUPS}
    actual_group_keys = {(r['dimension'],int(r['end2025_band_id']),r['group']) for r in groups}
    if len(groups) != 300 or actual_group_keys != expected_group_keys:
        raise RuntimeError('National group key set mismatch')
    for r in groups:
        key = (r['dimension'],int(r['end2025_band_id']),r['group'])
        for col, v in ns.get(key,empty).items():
            compare((key,col),r[col],v)
    cg = read('J1_city_group_by_end2025_band.csv')
    actual_local_keys = {(r['dimension'],int(r['end2025_band_id']),r['city_id'],r['group']) for r in cg}
    if len(cg) != len(ls) or actual_local_keys != set(ls):
        raise RuntimeError('City group key set mismatch')
    for r in cg:
        key = (r['dimension'],int(r['end2025_band_id']),r['city_id'],r['group'])
        for col,v in ls[key].items():
            compare((key,col),r[col],v)
    for r in read('J1_national_Q4_Q1.csv'):
        key = (r['dimension'],int(r['end2025_band_id']))
        low, high = ns.get((*key,'Q1'),empty), ns.get((*key,'Q4'),empty)
        available = low['n_tracts'] > 0 and high['n_tracts'] > 0
        compare((key,'n_Q1'),r['n_Q1'],low['n_tracts'])
        compare((key,'n_Q4'),r['n_Q4'],high['n_tracts'])
        compare((key,'available'),r['available'],available)
        expected = diffs(low,high) if available else {c:None for c in ('D_2000_pp','D_2025_pp','D_delta_pp','loss_share_difference_pp')}
        for col,v in expected.items():
            compare((key,col),r[col],v)
    cells = read('J1_city_cell_Q4_Q1.csv')
    cell_keys = {(r['dimension'],int(r['end2025_band_id']),r['city_id']) for r in cells}
    expected_cells = {(d,b,c) for d in DIMS for b in range(20) for c in cities}
    if len(cells) != len(expected_cells) or cell_keys != expected_cells:
        raise RuntimeError('City cell domain mismatch')
    common = defaultdict(list)
    for r in cells:
        d,b,c = r['dimension'],int(r['end2025_band_id']),r['city_id']
        key = (d,b,c)
        low, high = ls.get((*key,'Q1'),empty), ls.get((*key,'Q4'),empty)
        n1,n4 = low['n_tracts'],high['n_tracts']
        available = n1 > 0 and n4 > 0
        for col,v in dict(n_Q1=n1,n_Q4=n4,available=available,smaller_group_n=min(n1,n4),
                           smaller_group_n_is_one=available and min(n1,n4)==1).items():
            compare((key,col),r[col],v)
        expected = diffs(low,high) if available else {col:None for col in ('D_2000_pp','D_2025_pp','D_delta_pp','loss_share_difference_pp')}
        for col,v in expected.items():
            compare((key,col),r[col],v)
        if available:
            common[(d,b)].append(dict(city_id=c,n_Q1=n1,n_Q4=n4,**expected))
    for r in read('J1_city_equal_Q4_Q1.csv'):
        key = (r['dimension'],int(r['end2025_band_id']))
        vals = common[key]
        for col,v in dict(n_common_cities=len(vals),n_Q1=sum(v['n_Q1'] for v in vals),n_Q4=sum(v['n_Q4'] for v in vals),
                          n_city_cells_smaller_group_n1=sum(min(v['n_Q1'],v['n_Q4'])==1 for v in vals),available=bool(vals)).items():
            compare((key,col),r[col],v)
        for col in ('D_2000_pp','D_2025_pp','D_delta_pp','loss_share_difference_pp'):
            compare((key,col),r[col],mean([v[col] for v in vals]) if vals else None)
    residuals = []
    for r in read('J1_identity_checks.csv'):
        resid = float(r['D_delta_pp']) - (float(r['D_2025_pp']) - float(r['D_2000_pp']))
        compare(('identity',r['level'],r['dimension'],r['end2025_band_id'],r['city_id']),r['identity_pass'],abs(resid)<=ATOL)
        residuals.append(abs(resid))
    protected = []
    for item in prov['protected_assets']:
        path = Path(item['path'])
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        compare(('frozen_asset',str(path)),h,item['sha256_before'])
        protected.append(dict(path=str(path),unchanged=h==item['sha256_before'],sha256=h))
    summary = dict(status='PASS' if not failures else 'FAIL', method='independent Python csv + bisect + math.fsum; no pandas/numpy',
                   unique_tracts=len(seen), independent_scalar_comparisons=comparisons, failed_comparisons=len(failures),
                   max_absolute_numeric_error=max_error, tolerance_pp_or_share=ATOL,
                   identity_rows=len(residuals), max_recomputed_identity_residual_pp=max(residuals),
                   frozen_assets=protected, failures=failures,
                   source_files_read=[str(src)], models_loaded=0, model_fits=0)
    (OUT / 'J1_independent_validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if failures:
        raise RuntimeError('Independent validation failed; see JSON, no downstream interpretation authorized.')

if __name__ == '__main__':
    main()
