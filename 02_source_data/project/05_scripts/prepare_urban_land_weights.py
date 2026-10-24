"""Exact urban/tract/pixel/fixed-land intersections; no statistical sample filter."""
from pathlib import Path
import csv
import json
import time
import traceback
import warnings
import numpy as np
import geopandas as gpd
import rasterio
import shapely
from shapely.affinity import affine_transform
from shapely.ops import unary_union
from scipy.sparse import load_npz, save_npz, csr_matrix, vstack
from affine import Affine
from compute_tract_canopy_exact import pixel_weights, self_test, require, sha, save_csv, save_json, COUNTIES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '06_logs/urban_fixed_land_weights_20260911_run01'
WHOLE = ROOT / '06_logs/tract_exact_weights_20260911_run02'
URBAN = ROOT / '06_logs/urban_2000_boundary_validation_20260911_run03'


def polygonal(g):
    if g.geom_type in ('Polygon', 'MultiPolygon'):
        return g
    parts = []
    if hasattr(g, 'geoms'):
        for part in g.geoms:
            p = polygonal(part)
            if not p.is_empty:
                parts.append(p)
    return unary_union(parts)


def main():
    OUT.mkdir(exist_ok=False)
    started = time.time()
    try:
        require(json.loads((WHOLE/'audit.json').read_text())['status']=='PASS_WEIGHTS_ONLY', 'Whole weight gate')
        require(json.loads((URBAN/'audit.json').read_text())['status']=='PASS_SOURCE_AND_PILOT_GEOMETRY', 'Urban gate')
        tests = self_test()
        inputs = [Path(__file__), ROOT/'05_scripts/compute_tract_canopy_exact.py',
                  WHOLE/'audit.json', WHOLE/'tract_area_audit.csv', URBAN/'audit.json',
                  URBAN/'pilot_intersecting_urbanized_areas_2000.gpkg',
                  ROOT/'03_protocol/tract_pixel_area_rule_v1_20260911.md',
                  ROOT/'03_protocol/tract_pixel_area_rule_v1_correction_254_20260911.md',
                  ROOT/'00_admin/project_state.yaml']
        with (WHOLE/'tract_area_audit.csv').open(encoding='utf-8-sig') as f:
            whole_area = {r['tract_id']:r for r in csv.DictReader(f)}
        city = gpd.read_file(URBAN/'pilot_intersecting_urbanized_areas_2000.gpkg')
        rows, county_audits, notice = [], [], []
        matrices = []
        for county in COUNTIES:
            gp, wp, mp = [WHOLE/f'{stem}_{county}.{suffix}' for stem,suffix in
                          [('grid','json'),('land_area_weights','npz'),('fixed_landmask','tif')]]
            tp = ROOT/f'04_pilot/boundaries_2000/tract/tl_2010_{county}_tract00.zip'
            inputs += [gp,wp,mp,tp]
        hashes = {str(p):sha(p) for p in inputs}
        for county in COUNTIES:
            grid = json.loads((WHOLE/f'grid_{county}.json').read_text())
            full = load_npz(WHOLE/f'land_area_weights_{county}.npz')
            tr = gpd.read_file('zip://'+(ROOT/f'04_pilot/boundaries_2000/tract/tl_2010_{county}_tract00.zip').as_posix())
            tr = tr.set_index('CTIDFP00').loc[grid['tract_order']].to_crs(grid['crs_wkt'])
            require(tr.geometry.is_valid.all(), 'Invalid projected tract')
            cu = city.to_crs(tr.crs)
            cu = cu[cu.intersects(unary_union(tr.geometry))]
            urban_union = unary_union(cu.geometry)
            with rasterio.open(WHOLE/f'fixed_landmask_{county}.tif') as m:
                require(m.crs == tr.crs and list(m.transform)==grid['transform'] and list(m.shape)==grid['shape'], 'Fixed mask alignment')
                mask = m.read(1).ravel()
            require(set(np.unique(mask)).issubset({0,1,255}), 'Unexpected fixed mask values')
            inv = ~Affine(*grid['transform'][:6])
            nr, nc = grid['shape']
            localrows, parts, exact_count = [], [], 0
            for index, (tid, row) in enumerate(tr.iterrows()):
                geom = row.geometry
                fullrow = full.getrow(index)
                land_total = float(fullrow.sum())
                covered = urban_union.covers(geom)
                if covered:
                    part = fullrow
                    geom_area = geom.area
                    water = float(whole_area[tid]['open_water_m2'])
                    missing_landcover = float(whole_area[tid]['landcover_nodata_m2'])
                    closure = 0.0
                    mode = 'EXACT_FULL_TRACT_REUSE'
                elif not geom.intersects(urban_union):
                    part = csr_matrix((1,full.shape[1]))
                    geom_area = water = missing_landcover = closure = 0.0
                    mode = 'NO_URBAN_INTERSECTION'
                else:
                    with warnings.catch_warnings(record=True) as ws:
                        warnings.simplefilter('always')
                        clip = polygonal(geom.intersection(urban_union))
                    notice.extend({'tract_id':tid,'operation':'urban_intersection','message':str(w.message)} for w in ws)
                    require(clip.is_valid, 'Invalid clipped polygon')
                    geom_area = clip.area
                    if clip.is_empty or geom_area == 0:
                        part = csr_matrix((1,full.shape[1]))
                        water = missing_landcover = closure = 0.0
                        mode = 'ZERO_AREA_TOUCH'
                    else:
                        px = affine_transform(clip,[inv.a,inv.b,inv.d,inv.e,inv.c,inv.f])
                        with warnings.catch_warnings(record=True) as ws:
                            warnings.simplefilter('always')
                            x0,y0,w,_ = pixel_weights(px)
                        notice.extend({'tract_id':tid,'operation':'pixel_intersection','message':str(w.message)} for w in ws)
                        rr,cc = np.nonzero(w>0)
                        require(np.all(rr+y0>=0) and np.all(rr+y0<nr) and np.all(cc+x0>=0) and np.all(cc+x0<nc), 'Clip outside county window')
                        indices=(rr+y0)*nc+cc+x0
                        values=w[rr,cc]*900
                        closure=float(values.sum()-geom_area)
                        require(abs(closure)<=max(.01,geom_area*1e-10), 'Exact clipped area closure')
                        water=float(values[mask[indices]==0].sum())
                        missing_landcover=float(values[mask[indices]==255].sum())
                        keep=mask[indices]==1
                        part=csr_matrix((values[keep],indices[keep],np.array([0,int(keep.sum())])),shape=(1,full.shape[1]))
                        mode='EXACT_URBAN_CLIP'
                        exact_count+=1
                area=float(part.sum())
                delta=part-fullrow
                excess=max(0.0,float(delta.data.max())) if delta.nnz else 0.0
                require(excess<=1e-4,'Urban weight exceeds whole-tract weight beyond numerical tolerance')
                require(area<=land_total+max(.01,land_total*1e-10),'Urban land exceeds whole land')
                require(abs(area+water+missing_landcover-geom_area)<=max(.01,geom_area*1e-10),'Land/water/missing closure')
                parts.append(part)
                localrows.append({'county_code':county,'tract_id':tid,'whole_fixed_land_m2':land_total,
                                  'urban_geometry_m2':geom_area,'urban_fixed_land_m2':area,
                                  'urban_open_water_m2':water,'urban_landcover_nodata_m2':missing_landcover,
                                  'urban_fraction_of_whole_fixed_land':area/land_total,
                                  'pixel_area_closure_error_m2':closure,'max_weight_excess_m2':excess,
                                  'weight_mode':mode,'urban_land_status':'POSITIVE_LAND' if area>0 else 'NO_URBAN_LAND'})
                if (index+1)%200==0:
                    print(county,index+1,'/',len(tr),'weights',flush=True)
            matrix=vstack(parts,format='csr')
            require(matrix.shape==full.shape and matrix.nnz>0,'Matrix extent mismatch')
            column_sum=np.asarray(matrix.sum(axis=0)).ravel()
            require(float(column_sum.max())<=900.0001,'Overlapping urban tract weights')
            save_npz(OUT/f'urban_land_area_weights_{county}.npz',matrix,compressed=True)
            save_json(OUT/f'grid_{county}.json',grid)
            save_csv(OUT/f'area_{county}.csv',localrows)
            rows.extend(localrows)
            county_audits.append({'county_code':county,'tract_count':len(tr),'exact_clips':exact_count,
                                  'positive_urban_land_tracts':sum(r['urban_fixed_land_m2']>0 for r in localrows),
                                  'zero_urban_land_tracts':sum(r['urban_fixed_land_m2']==0 for r in localrows),
                                  'whole_fixed_land_m2':sum(r['whole_fixed_land_m2'] for r in localrows),
                                  'urban_fixed_land_m2':float(matrix.sum()),
                                  'max_area_closure_error_m2':max(abs(r['pixel_area_closure_error_m2']) for r in localrows)})
            print(county,'weights finished',flush=True)
        require(len(rows)==2656 and len({r['tract_id'] for r in rows})==2656,'Tract accounting')
        require(all(sha(Path(p))==h for p,h in hashes.items()),'Input changed')
        save_csv(OUT/'tract_urban_fixed_land.csv',rows)
        save_json(OUT/'warnings.json',notice)
        save_json(OUT/'audit.json',{'status':'PASS_URBAN_FIXED_LAND_WEIGHTS','rows':2656,'dropped':0,
                                   'duplicates':0,'counties':county_audits,'self_test':tests,
                                   'inputs_sha256':hashes,'warnings_count':len(notice),
                                   'resampling':'NONE','sample_selection':'NOT RUN',
                                   'land_definition':'Fixed 2001 mask: excludes open water class 11 and landcover NoData class 250; wetlands retained',
                                   'numerical_tolerance_note':'Area and per-cell tolerances check floating arithmetic only; no sample exclusion or weight clipping',
                                   'elapsed_seconds':time.time()-started})
    except BaseException:
        (OUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise


if __name__=='__main__':
    main()
