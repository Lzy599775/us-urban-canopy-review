"""Authorized J1 only: frozen 5 pp end bands, descriptive statistics, no models."""
from nc_paths import project_root as _nc_project_root, external_path as _nc_external_path
from pathlib import Path
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT = _nc_project_root()
OUT = Path(__file__).resolve().parent
G0 = ROOT / '06_logs/NLCD_joint_information_support_preflight_20261015_run01'
TASK = ROOT / 'Codex_J1_9CE9908FE8B0AFE88389E88C8CE4A4_20261015.md'
DIMS = ('income', 'poverty', 'age65plus')
GROUPS = ('Q1', 'Q2', 'Q3', 'Q4', 'unassigned')
ATOL = 1e-10
CHECKS = []

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()

def save_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def check(name, actual, expected):
    passed = bool(actual == expected)
    CHECKS.append(dict(check=name, actual=actual, expected=expected, passed=passed))
    pd.DataFrame(CHECKS).to_csv(OUT / 'J1_checks.csv', index=False, encoding='utf-8-sig')
    if not passed:
        raise RuntimeError(f'{name}: {actual!r} != {expected!r}; STOP')

def band_label(b):
    return f'[{5*b},{5*b+5}' + (']' if b == 19 else ')')

def describe(g):
    n = len(g)
    result = dict(n_tracts=n)
    for col in ('canopy_2000', 'canopy_2025', 'delta_pp'):
        result[col + '_mean'] = float(g[col].mean()) if n else np.nan
    for q, label in ((.25, 'p25'), (.5, 'median'), (.75, 'p75')):
        result['delta_' + label + '_pp'] = float(g.delta_pp.quantile(q, interpolation='linear')) if n else np.nan
    for name, mask in (('gain', g.delta_pp > 0), ('zero', g.delta_pp == 0), ('loss', g.delta_pp < 0)):
        result['n_' + name] = int(mask.sum())
        result[name + '_share'] = float(mask.mean()) if n else np.nan
    return result

def contrast(low, high):
    result = dict(n_Q1=int(low['n_tracts']), n_Q4=int(high['n_tracts']))
    for outcol, incol in (('D_2000_pp', 'canopy_2000_mean'), ('D_2025_pp', 'canopy_2025_mean'),
                           ('D_delta_pp', 'delta_pp_mean')):
        result[outcol] = high[incol] - low[incol]
    result['loss_share_difference_pp'] = 100 * (high['loss_share'] - low['loss_share'])
    result['identity_residual_pp'] = result['D_delta_pp'] - (result['D_2025_pp'] - result['D_2000_pp'])
    result['identity_atol_pp'] = ATOL
    result['identity_pass'] = bool(abs(result['identity_residual_pp']) <= ATOL)
    return result

def unavailable(reason):
    return dict(available=False, availability_reason=reason, D_2000_pp=np.nan,
                D_2025_pp=np.nan, D_delta_pp=np.nan, loss_share_difference_pp=np.nan,
                identity_residual_pp=np.nan, identity_atol_pp=ATOL, identity_pass=None)

def main():
    started = time.perf_counter()
    if (OUT / 'J1_run_receipt.json').exists():
        raise RuntimeError('Run receipt already exists; no silent overwrite/resume.')
    prov = json.loads((G0 / 'input_provenance.json').read_text(encoding='utf-8'))
    pair_info = next(x for x in prov['inputs'] if x['role'] == 'paired')
    paired_path = Path(pair_info['path'])
    inputs = []
    for role, path in [('task', TASK), ('G0_provenance', G0 / 'input_provenance.json'),
                       ('G0_end_support', G0 / 'end2025_5pp_support.csv'),
                       ('G0_membership', G0 / 'resource_pairing_support_membership.csv'),
                       ('paired', paired_path)]:
        inputs.append(dict(role=role, path=str(path), bytes=path.stat().st_size, sha256=sha(path), basis='exact_file_bytes'))
    check('paired_hash_matches_G0', inputs[-1]['sha256'], pair_info['sha256'])
    protected = []
    for item in prov['protected_assets']:
        actual = sha(item['path'])
        check('frozen_before:' + Path(item['path']).name, actual, item['sha256'])
        protected.append(dict(path=item['path'], sha256_before=actual))
    save_json('J1_input_provenance.json', dict(inputs=inputs, protected_assets=protected,
                  authorization='User approved J1 after G0; prior G0 pending status superseded',
                  mode='AUTHORIZED_FINITE_DESCRIPTIVE_J1', atol_pp=ATOL, python=sys.version,
                  platform=platform.platform(), pandas=pd.__version__, numpy=np.__version__))
    df = pd.read_csv(paired_path, dtype={'state': str, 'tract_id': str, 'single_city_id': str,
                     **{d + '_city_quartile': str for d in DIMS}}, keep_default_na=False, float_precision='round_trip')
    check('input_rows', len(df), 42369)
    check('duplicate_state_tract', int(df.duplicated(['state', 'tract_id']).sum()), 0)
    check('valid_state_strings', bool(df.state.str.fullmatch(r'\d{2}').all()), True)
    check('valid_tract_strings', bool(df.tract_id.str.fullmatch(r'\d{11}').all()), True)
    check('tract_state_prefix', bool((df.tract_id.str[:2] == df.state).all()), True)
    check('city_strings', bool(df.single_city_id.str.fullmatch(r'\d{5}|').all()), True)
    check('unique_cities', int(df.loc[df.single_city_id != '', 'single_city_id'].nunique()), 448)
    check('nonunique_city_rows', int((df.single_city_id == '').sum()), 290)
    for col in ('canopy_2000', 'canopy_2025', 'change_pp'):
        check('finite:' + col, bool(np.isfinite(df[col].to_numpy(dtype=float)).all()), True)
    for col in ('canopy_2000', 'canopy_2025'):
        check('range:' + col, bool(df[col].between(0, 100).all()), True)
    df['delta_pp'] = df.canopy_2025 - df.canopy_2000
    source_delta_residual = float(np.abs(df.delta_pp - df.change_pp).max())
    check('stored_delta_identity_atol', source_delta_residual <= ATOL, True)
    df['band_id'] = np.floor(df.canopy_2025 / 5).astype(int)
    df.loc[df.canopy_2025 == 100, 'band_id'] = 19
    check('band_range', bool(df.band_id.between(0, 19).all()), True)
    member = pd.read_csv(G0 / 'resource_pairing_support_membership.csv', dtype=str, keep_default_na=False)
    member = member.loc[member.in_paired == 'True'].copy()
    check('G0_membership_rows', len(member), 42369)
    check('G0_membership_duplicates', int(member.duplicated(['state', 'tract_id']).sum()), 0)
    linked = df.merge(member, on=['state', 'tract_id'], how='outer', validate='one_to_one', indicator=True, suffixes=('', '_G0'))
    check('G0_unmatched', int((linked['_merge'] != 'both').sum()), 0)
    check('G0_band_matches', bool((linked.band_id == linked.b25.astype(int)).all()), True)
    check('G0_city_matches', bool((linked.single_city_id == linked.single_city_id_G0).all()), True)
    for d in DIMS:
        col = d + '_city_quartile'
        check('group_labels:' + d, bool(df[col].isin(['1.0', '2.0', '3.0', '4.0', '']).all()), True)
        mapping = {'1.0': 'Q1', '2.0': 'Q2', '3.0': 'Q3', '4.0': 'Q4', '': 'unassigned'}
        df[d] = df[col].map(mapping)
        check('assigned:' + d, int((df[d] != 'unassigned').sum()), 42079)
        check('unassigned:' + d, int((df[d] == 'unassigned').sum()), 290)
        expected_groups = linked[col + '_G0'].replace({'1': 'Q1', '2': 'Q2', '3': 'Q3', '4': 'Q4', '': 'unassigned'})
        check('G0_groups:' + d, bool((linked[col].map(mapping) == expected_groups).all()), True)
    g0_support = pd.read_csv(G0 / 'end2025_5pp_support.csv')
    group_rows, national_rows, city_group_rows, city_rows, equal_rows = [], [], [], [], []
    cities = sorted(df.loc[df.single_city_id != '', 'single_city_id'].unique())
    for d in DIMS:
        grouped = {(int(b), g): describe(part) for (b, g), part in df.groupby(['band_id', d], sort=True)}
        city_grouped = {(c, int(b), g): describe(part) for (c, b, g), part in
                        df.loc[df.single_city_id != ''].groupby(['single_city_id', 'band_id', d], sort=True)}
        empty = describe(df.iloc[:0])
        for (c, b, g), stats in city_grouped.items():
            city_group_rows.append(dict(dimension=d, city_id=c, end2025_band_id=b, end2025_band=band_label(b), group=g, **stats))
        for b in range(20):
            base = dict(dimension=d, end2025_band_id=b, end2025_band=band_label(b))
            ref = g0_support.loc[(g0_support.dimension == d) & (g0_support.end2025_band_id == b)].iloc[0]
            for g in GROUPS:
                stats = grouped.get((b, g), empty)
                group_rows.append(dict(**base, group=g, **stats))
                check(f'G0_n:{d}:{b}:{g}', stats['n_tracts'], int(ref['n_unassigned' if g == 'unassigned' else 'n_' + g]))
                check(f'sign_counts:{d}:{b}:{g}', stats['n_gain'] + stats['n_zero'] + stats['n_loss'], stats['n_tracts'])
            low, high = grouped.get((b, 'Q1'), empty), grouped.get((b, 'Q4'), empty)
            if low['n_tracts'] and high['n_tracts']:
                national_rows.append(dict(**base, available=True, availability_reason='BOTH_GROUPS_PRESENT', **contrast(low, high)))
            else:
                national_rows.append(dict(**base, n_Q1=low['n_tracts'], n_Q4=high['n_tracts'],
                                          **unavailable('EMPTY_BAND' if int(ref.n_paired_tracts) == 0 else 'MISSING_Q1_OR_Q4')))
            common = []
            for c in cities:
                q1, q4 = city_grouped.get((c, b, 'Q1'), empty), city_grouped.get((c, b, 'Q4'), empty)
                n1, n4 = q1['n_tracts'], q4['n_tracts']
                if n1 and n4:
                    row = dict(**base, city_id=c, available=True, availability_reason='BOTH_GROUPS_PRESENT',
                               smaller_group_n=min(n1, n4), smaller_group_n_is_one=min(n1, n4) == 1, **contrast(q1, q4))
                    common.append(row)
                else:
                    row = dict(**base, city_id=c, n_Q1=n1, n_Q4=n4, smaller_group_n=min(n1, n4),
                               smaller_group_n_is_one=False, **unavailable('NO_Q1_OR_Q4' if n1 + n4 == 0 else 'MISSING_Q1_OR_Q4'))
                city_rows.append(row)
            n_common = len(common)
            sum_n1 = sum(x['n_Q1'] for x in common)
            sum_n4 = sum(x['n_Q4'] for x in common)
            check(f'G0_common_cities:{d}:{b}', n_common, int(ref.n_common_Q1_Q4_cities))
            check(f'G0_common_nQ1:{d}:{b}', sum_n1, int(ref.n_Q1_tracts_in_common_cities))
            check(f'G0_common_nQ4:{d}:{b}', sum_n4, int(ref.n_Q4_tracts_in_common_cities))
            row = dict(**base, n_common_cities=n_common, n_Q1=sum_n1, n_Q4=sum_n4,
                       n_city_cells_smaller_group_n1=sum(x['smaller_group_n_is_one'] for x in common))
            if common:
                row.update(available=True, availability_reason='COMMON_CITY_Q1_Q4')
                for col in ('D_2000_pp', 'D_2025_pp', 'D_delta_pp', 'loss_share_difference_pp'):
                    row[col] = float(np.mean([x[col] for x in common]))
                row['identity_residual_pp'] = row['D_delta_pp'] - (row['D_2025_pp'] - row['D_2000_pp'])
                row.update(identity_atol_pp=ATOL, identity_pass=abs(row['identity_residual_pp']) <= ATOL)
            else:
                row.update(unavailable('EMPTY_BAND' if int(ref.n_paired_tracts) == 0 else 'NO_COMMON_CITY_Q1_Q4'))
            equal_rows.append(row)
    outputs = {'J1_group_by_end2025_band.csv': group_rows, 'J1_national_Q4_Q1.csv': national_rows,
               'J1_city_group_by_end2025_band.csv': city_group_rows,
               'J1_city_cell_Q4_Q1.csv': city_rows, 'J1_city_equal_Q4_Q1.csv': equal_rows}
    identities = []
    for name, rows in outputs.items():
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding='utf-8-sig', float_format='%.17g', na_rep='')
        if 'group_by' not in name:
            for r in rows:
                if r['available']:
                    identities.append(dict(level=name, dimension=r['dimension'], end2025_band_id=r['end2025_band_id'],
                       city_id=r.get('city_id', ''), D_2000_pp=r['D_2000_pp'], D_2025_pp=r['D_2025_pp'],
                       D_delta_pp=r['D_delta_pp'], identity_residual_pp=r['identity_residual_pp'],
                       identity_atol_pp=ATOL, identity_pass=r['identity_pass']))
    pd.DataFrame(identities).to_csv(OUT / 'J1_identity_checks.csv', index=False, encoding='utf-8-sig', float_format='%.17g')
    check('all_contrast_identities', all(r['identity_pass'] for r in identities), True)
    for item in protected:
        item['sha256_after'] = sha(item['path'])
        item['unchanged'] = item['sha256_after'] == item['sha256_before']
        check('frozen_after:' + Path(item['path']).name, item['unchanged'], True)
    check('source_unchanged_after', sha(paired_path), pair_info['sha256'])
    save_json('J1_frozen_asset_checks.json', protected)
    receipt = dict(status='J1_DESCRIPTIVE_COMPUTATION_PASS_PENDING_INDEPENDENT_VALIDATION',
                   created_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-started,
                   input_rows=len(df), unique_tracts=len(df), dimensions=list(DIMS), national_group_rows=len(group_rows),
                   city_group_rows=len(city_group_rows), city_cell_rows=len(city_rows),
                   applicable_identity_checks=len(identities), max_identity_residual_pp=max(abs(x['identity_residual_pp']) for x in identities),
                   max_stored_delta_residual_pp=source_delta_residual, atol_pp=ATOL,
                   quantile_method='linear interpolation (n-1)*p', sign_rule='strict >0, ==0, <0; tolerance not used for classification',
                   national_weight='unique tract equal', city_weight='equal city after within-city Q4 minus Q1',
                   model_fits=0, new_pvalues=0, new_intervals=0, formal_asset_writes=0,
                   J2='NOT RUN', J3='NOT RUN', completed_checks=len(CHECKS), failed_checks=0,
                   tables={n: len(v) for n,v in outputs.items()})
    save_json('J1_run_receipt.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
