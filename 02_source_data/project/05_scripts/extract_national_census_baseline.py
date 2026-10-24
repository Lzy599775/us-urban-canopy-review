"""Extract verified SF3 and available SF1 state archives; no network or model fits."""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import re
import struct
import traceback
import zipfile
from collections import Counter
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
STATES = '01 04 05 06 08 09 10 11 12 13 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 44 45 46 47 48 49 50 51 53 54 55 56'.split()
GEO = {'SUMLEV': (9, 3), 'GEOCOMP': (12, 2), 'CHARITER': (14, 3),
       'LOGRECNO': (19, 7), 'STATE': (30, 2), 'COUNTY': (32, 3),
       'TRACT': (56, 6), 'POP100': (293, 9)}
KEEP = {1: ['P001001', 'P002001', 'P002002', 'P002003', 'P002004',
            'P002005', 'P002006', 'P004001', 'P004002', 'P004003', 'P004006',
            *[f'P012{i:03}' for i in range(1, 50)], 'H001001', 'H003001',
            'H003002', 'H003003', 'H004001', 'H004002', 'H004003'],
        3: ['P052001', 'P053001', *[f'P087{i:03}' for i in range(1, 18)]]}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def write_csv(path, rows, fields):
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def unzip_one(data, expected):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        require(z.namelist() == [expected], f'Unexpected nested member: {expected}')
        return z.read(expected)  # verifies the uncompressed member CRC

def geography(data, state, abbr):
    ids, targets, records = set(), {}, {}
    for raw in data.splitlines():
        require(len(raw) == 400, f'{state}: geography record width')
        require(raw[:8] == ('uSF' + str(1 if data[:4] == b'uSF1' else 3) + '  ' + abbr.upper()).encode(), 'Geography file/state label')
        d = {k: raw[p-1:p-1+n].decode('ascii').strip() for k, (p, n) in GEO.items()}
        key = (d['CHARITER'], d['LOGRECNO'])
        require(key not in ids, f'{state}: duplicate geography key {key}')
        ids.add(key)
        require(d['STATE'] == state, f'{state}: state mismatch')
        if (d['SUMLEV'], d['GEOCOMP'], d['CHARITER']) != ('140', '00', '000'):
            continue
        tid = state + d['COUNTY'] + d['TRACT']
        require(len(tid) == 11 and tid.isdigit() and tid not in records, 'Invalid/duplicate tract')
        targets[key] = tid
        records[tid] = {'county_code': tid[:5], 'tract_id': tid,
                        'population_2000_geo': int(d['POP100']), 'record_id': d['LOGRECNO']}
    require(records, f'{state}: no tracts')
    return ids, targets, records

def segment(data, series, segment_id, abbr, schema, ids, targets, records):
    selected = [(r['csv_column_one_based']-1, r['field']) for r in schema if r['field'] in KEEP[series]]
    seen, matched = set(), set()
    for row in csv.reader(io.StringIO(data.decode('ascii'))):
        require(len(row) == 5 + len(schema), f'{abbr}/{series}/{segment_id}: column count')
        require(row[:4] == [f'uSF{series}', abbr.upper(), '000', segment_id], 'Linkage fields differ')
        key = (row[2], row[4])
        require(key not in seen, 'Duplicate segment record')
        seen.add(key)
        if key in targets:
            records[targets[key]].update({field: row[col] for col, field in selected})
            matched.add(key)
    require(seen == ids, 'Full-state segment/geography keys differ')
    require(matched == set(targets), 'Target tract lost during linkage')
    return {'series': series, 'segment': segment_id, 'input_rows': len(seen),
            'output_tract_rows': len(matched), 'other_geographies': len(seen)-len(matched),
            'target_dropped': 0, 'duplicate_keys': 0, 'unmatched': 0}

def boundary_ids(path):
    # dBASE table read only; this is a key audit, not a geometry-validity audit.
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.lower().endswith('.dbf')]
        require(len(names) == 1, 'Expected one boundary dBASE table')
        b = z.read(names[0])
    count = struct.unpack_from('<I', b, 4)[0]
    hlen, rlen = struct.unpack_from('<HH', b, 8)
    fields, offset = {}, 1
    for pos in range(32, hlen-1, 32):
        name = b[pos:pos+11].split(b'\0')[0].decode('ascii')
        size = b[pos+16]
        fields[name] = (offset, size)
        offset += size
    require(offset == rlen, 'Boundary record length differs')
    require({'STATEFP00','COUNTYFP00','TRACTCE00'}.issubset(fields), '2000 boundary keys absent')
    ids = []
    for i in range(count):
        row = b[hlen+i*rlen:hlen+(i+1)*rlen]
        require(len(row) == rlen and row[:1] == b' ', 'Boundary deleted/truncated record')
        parts = [row[fields[k][0]:sum(fields[k])].decode('ascii').strip() for k in ['STATEFP00','COUNTYFP00','TRACTCE00']]
        tid = ''.join(parts)
        require(len(tid) == 11 and tid.isdigit(), 'Invalid boundary key')
        ids.append(tid)
    require(len(ids) == len(set(ids)), 'Duplicate boundary tract')
    return set(ids), digest(b)

def derive(records, series):
    for r in records.values():
        require(set(KEEP[series]).issubset(r), 'Missing raw fields')
        v = {k: int(r[k]) for k in KEEP[series]}
        require(all(n >= 0 for n in v.values()), 'Unrecognized negative value')
        if series == 3:
            require(v['P087002'] + v['P087010'] == v['P087001'], 'Poverty identity')
            require(sum(v[f'P087{i:03}'] for i in range(3,10)) == v['P087002'], 'Poor age subtotal')
            require(sum(v[f'P087{i:03}'] for i in range(11,18)) == v['P087010'], 'Nonpoor age subtotal')
            r['poverty_share'] = v['P087002']/v['P087001'] if v['P087001'] else None
            r['income_reference_year'] = 1999
            r['income_point_usd'] = r['income_lower_inclusive_usd'] = r['income_upper_exclusive_usd'] = None
            income, hh = v['P053001'], v['P052001']
            if hh == 0:
                require(income == 0, 'Income despite zero SF3 households')
                status = 'UNDEFINED_SF3_HOUSEHOLD_UNIVERSE_ZERO'
            elif income == 200001:
                status = 'UPPER_INTERVAL_200000_OR_MORE'
                r['income_lower_inclusive_usd'] = 200000
            elif income == 2499:
                status = 'LOWER_INTERVAL_BELOW_2500'
                r['income_upper_exclusive_usd'] = 2500
            elif 2500 <= income < 200000:
                status = 'REPORTED_POINT_ESTIMATE'
                r['income_point_usd'] = income
            else:
                raise ValueError(f'Unrecognized income code {r["tract_id"]}: {income}, households {hh}')
            r['income_status'] = status
        else:
            require(v['P001001'] == v['P002001'] == v['P004001'] == v['P012001'] == r['population_2000_geo'], 'Population identity')
            require(v['P012002']+v['P012026'] == v['P012001'], 'Sex identity')
            require(sum(v[f'P012{i:03}'] for i in range(3,26)) == v['P012002'], 'Male age sum')
            require(sum(v[f'P012{i:03}'] for i in range(27,50)) == v['P012026'], 'Female age sum')
            require(v['H004002']+v['H004003'] == v['H004001'] == v['H003002'], 'Tenure identity')
            require(v['H003002']+v['H003003'] == v['H003001'] == v['H001001'], 'Housing identity')
            r['age65plus_count'] = sum(v[f'P012{i:03}'] for i in [*range(20,26),*range(44,50)])
            r['age65plus_share'] = r['age65plus_count']/v['P012001'] if v['P012001'] else None
            r['renter_occupied_unit_share'] = v['H004003']/v['H004001'] if v['H004001'] else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--states', nargs='+', default=STATES)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    require(set(args.states).issubset(STATES), 'Outside contiguous US + DC')
    out = ROOT / args.output
    out.mkdir(exist_ok=False, parents=True)
    evidence, audits, all_sf3, all_sf1, pending, mismatches = {}, [], {}, {}, [], []
    try:
        schema_path = ROOT/'06_logs/census_fields_validation_20260911_run01/official_column_schemas.json'
        schemas = json.loads(schema_path.read_text())
        evidence[str(schema_path)] = digest(schema_path.read_bytes())
        for series in [1,3]:
            pdf = ROOT/f'04_pilot/census_2000/documentation/sf{series}_technical_documentation.pdf'
            evidence[str(pdf)] = digest(pdf.read_bytes())
            reader = PdfReader(pdf)
            for seg, schema in schemas[str(series)].items():
                regenerated = []
                for page in sorted({r['pdf_page'] for r in schema}):
                    text = reader.pages[page-1].extract_text()
                    regenerated.extend(code for code,s in re.findall(r'\b([PH]\d{3}[A-Z]?\d{3})\s+(\d{2})\s+\d+',text) if s == seg)
                require(regenerated == [r['field'] for r in schema], f'Official schema mismatch {series}/{seg}')
                require([r['csv_column_one_based'] for r in schema] == list(range(6,6+len(schema))), 'Column offsets')
        sources = json.loads((ROOT/'04_national/census_2000/sf3/manifest_progress.json').read_text())['files']
        require({r['state'] for r in sources} == set(STATES) and len(sources)==49, 'State inventory')
        for src in sources:
            st = src['state']
            if st not in args.states:
                continue
            path = Path(src['path'])
            require(path.stat().st_size == src['bytes'], 'Archive size changed')
            with zipfile.ZipFile(path) as z:
                geo_name, = [n for n in z.namelist() if n.endswith('geo_uf3.zip')]
                abbr = geo_name[:2]
                b = z.read(geo_name)
                evidence[str(path)+'!'+geo_name] = digest(b)
                ids, targets, rec3 = geography(unzip_one(b,abbr+'geo.uf3'),st,abbr)
                for seg in ['06','07']:
                    name = f'{abbr}000{seg}_uf3.zip'
                    b = z.read(name)
                    evidence[str(path)+'!'+name] = digest(b)
                    a = segment(unzip_one(b,f'{abbr}000{seg}.uf3'),3,seg,abbr,schemas['3'][seg],ids,targets,rec3)
                    audits.append(dict(state=st,**a))
            derive(rec3,3)
            bound = ROOT/f'04_national/boundaries_2000/tract_by_state/tl_2010_{st}_tract00.zip'
            bids, bdigest = boundary_ids(bound)
            evidence[str(bound)+'!dbf_uncompressed'] = bdigest
            for tid in sorted(bids-set(rec3)):
                mismatches.append({'state':st,'tract_id':tid,'mismatch':'BOUNDARY_ONLY'})
            for tid in sorted(set(rec3)-bids):
                mismatches.append({'state':st,'tract_id':tid,'mismatch':'SF3_ONLY'})
            require(not set(rec3).intersection(all_sf3), 'Cross-state duplicate')
            all_sf3.update(rec3)
            selected = {}
            for seg in ['geo','01','02','37']:
                name = f'{abbr}geo_uf1.zip' if seg=='geo' else f'{abbr}000{seg}_uf1.zip'
                candidates = [ROOT/'04_national/census_2000/sf1'/name,
                              ROOT/'04_pilot/census_2000'/('geography' if seg=='geo' else 'sf1')/name]
                available = [p for p in candidates if p.exists()]
                if available:
                    hashes = {digest(p.read_bytes()) for p in available}
                    require(len(hashes)==1, f'Existing SF1 copies differ: {name}')
                    selected[seg] = available[0]
                else:
                    pending.append({'state':st,'state_name':src['name'],'file':name,
                        'url':f'https://www2.census.gov/census_2000/datasets/Summary_File_1/{src["name"]}/{name}'})
            if len(selected)==4:
                p = selected['geo']; b=p.read_bytes(); evidence[str(p)]=digest(b)
                ids1,targets1,rec1 = geography(unzip_one(b,abbr+'geo.uf1'),st,abbr)
                require(set(rec1)==set(rec3),'SF1/SF3 target keys differ')
                for seg in ['01','02','37']:
                    p=selected[seg]; b=p.read_bytes(); evidence[str(p)]=digest(b)
                    a=segment(unzip_one(b,f'{abbr}000{seg}.uf1'),1,seg,abbr,schemas['1'][seg],ids1,targets1,rec1)
                    audits.append(dict(state=st,**a))
                derive(rec1,1)
                require(all(rec1[k]['population_2000_geo']==rec3[k]['population_2000_geo'] for k in rec1),'SF1/SF3 census counts differ')
                all_sf1.update(rec1)
            print(st, 'SF3_TRACTS',len(rec3),'BOUNDARY_ONLY',len(bids-set(rec3)),
                  'SF3_ONLY',len(set(rec3)-bids),'SF1', 'EXTRACTED' if len(selected)==4 else 'FILES_MISSING',flush=True)
        pilot=list(csv.DictReader((ROOT/'06_logs/census_fields_validation_20260911_run01/tract_socioeconomic_2000_diagnostic.csv').open(encoding='utf-8-sig')))
        regressions=Counter()
        for r in pilot:
            for series,records in [(3,all_sf3),(1,all_sf1)]:
                if r['tract_id'] in records:
                    new=records[r['tract_id']]
                    require(all(new[k]==r[k] for k in KEEP[series]), f'Pilot raw-field regression {series}: {r["tract_id"]}')
                    regressions[str(series)]+=1
        rows3=[all_sf3[k] for k in sorted(all_sf3)]
        write_csv(out/'tract_income_poverty_2000.csv', rows3,list(rows3[0]))
        if all_sf1:
            rows1=[all_sf1[k] for k in sorted(all_sf1)]
            write_csv(out/'tract_population_age_available_states.csv', rows1,list(rows1[0]))
        write_csv(out/'boundary_key_mismatches.csv',mismatches,['state','tract_id','mismatch'])
        write_json(out/'missing_sf1_files.json',pending)
        write_json(out/'segment_audits.json',audits)
        write_json(out/'source_hashes.json',{'basis':'Named nested ZIP bytes for SF3; whole ZIP bytes for SF1; uncompressed dBASE bytes for boundary key audit; whole documentation and schema bytes','hashes':evidence})
        scope_is_national = set(args.states) == set(STATES)
        report={'status':'PASS_EXTRACTION_WITH_EXPLICIT_PENDING_ITEMS','states_requested':len(args.states),
            'states_requested_codes':sorted(set(args.states)), 'scope_is_national':scope_is_national,
            'sf3_rows':len(all_sf3),'sf3_unique_tracts':len(all_sf3),'sf1_rows':len(all_sf1),
            'sf1_states':sorted({k[:2] for k in all_sf1}),'missing_sf1_files':len(pending),
            'boundary_mismatches':dict(Counter(r['mismatch'] for r in mismatches)),
            'income_status_counts':dict(Counter(r['income_status'] for r in rows3)),
            'zero_poverty_universe':sum(int(r['P087001'])==0 for r in rows3),
            'sf3_zero_households_positive_census_population':sum(int(r['P052001'])==0 and r['population_2000_geo']>0 for r in rows3),
            'pilot_raw_field_regression_rows':dict(regressions),'target_dropped':0,
            'geometry_validity':'NOT RUN; dBASE identifiers only','national_sample_frozen':False,
            'requested_scope_census_gate_passed':not pending and not mismatches,
            'national_census_gate_passed':scope_is_national and not pending and not mismatches,
            'formal_model':'NOT RUN','script_sha256':digest(Path(__file__).read_bytes())}
        write_json(out/'audit.json',report)
        print(json.dumps(report,ensure_ascii=True),flush=True)
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise

if __name__=='__main__':
    main()
