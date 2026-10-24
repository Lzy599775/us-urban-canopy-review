"""Three-county diagnostic panel using exact polygon/pixel intersection areas.
No urban sample selection, imputation, model fitting, or raw-data mutations.
"""
import argparse, csv, hashlib, json, math, sys, time, traceback
from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from affine import Affine
import shapely
from shapely.affinity import affine_transform, translate
from shapely.geometry import box, Polygon, MultiPolygon, mapping
from scipy.sparse import csr_matrix, save_npz
from scipy.ndimage import binary_dilation

COUNTIES=('04013','17031','48201')

def require(ok,msg):
 if not ok: raise ValueError(msg)

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()

def save_json(path,obj):
 with path.open('x',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)

def save_csv(path,rows):
 with path.open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def pixel_weights(g):
 """Input geometry in unit pixel coordinates. Interior 1, boundary exact area."""
 minx,miny,maxx,maxy=g.bounds
 x0,y0=math.floor(minx),math.floor(miny); x1,y1=math.ceil(maxx),math.ceil(maxy)
 shape=(y1-y0,x1-x0); local=translate(g,xoff=-x0,yoff=-y0)
 center=rasterize([(mapping(local),1)],out_shape=shape,transform=Affine.identity(),all_touched=False,dtype='uint8')
 edge=rasterize([(mapping(local.boundary),1)],out_shape=shape,transform=Affine.identity(),all_touched=True,dtype='uint8')
 # Conservative candidate ring; exact intersections, not buffered-area weights.
 edge=binary_dilation(edge,structure=np.ones((3,3),dtype=bool))
 weights=center.astype('float64'); rr,cc=np.nonzero(edge)
 for k in range(0,len(rr),20000):
  ry,cx=rr[k:k+20000],cc[k:k+20000]
  cells=shapely.box(cx,ry,cx+1,ry+1)
  weights[ry,cx]=shapely.area(shapely.intersection(cells,local))
 require(np.all((weights>=0)&(weights<=1+1e-10)),'Invalid cell weights')
 require(abs(weights.sum()-g.area)<=max(1e-8,g.area*1e-10),f'Pixel area closure failed: sum={weights.sum()}, geometry={g.area}, delta={weights.sum()-g.area}, bounds={g.bounds}')
 return x0,y0,weights,center

def self_test():
 # Analytic geometry verification only; these are not research observations.
 cases=[box(0,0,3,3),box(.2,.3,3.7,2.8),Polygon([(0,0),(5,0),(.1,4)]),
  Polygon([(0,0),(5,0),(5,5),(0,5)],holes=[[(1,1),(1,4),(4,4),(4,1)]]),
  MultiPolygon([box(.1,.1,.2,3),box(2,2,3,3)]),Polygon([(0,0),(.01,0),(4,4),(3.99,4)])]
 for g in cases:
  x0,y0,w,c=pixel_weights(g); r,co=np.indices(w.shape)
  brute=shapely.area(shapely.intersection(shapely.box(co+x0,r+y0,co+x0+1,r+y0+1),g))
  require(np.allclose(w,brute,rtol=0,atol=1e-10),'Exact weights differ from exhaustive intersection')
 return {'cases':len(cases),'status':'PASS','comparison':'exhaustive cell intersections incl aligned boundaries, holes, slivers, multipolygons'}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--source-audit',type=Path);ap.add_argument('--weights-only',action='store_true');ap.add_argument('--self-test',action='store_true'); a=ap.parse_args()
 test=self_test()
 if a.self_test: print(json.dumps(test));return
 require(a.weights_only,'This entry point is restricted to weights preparation; use aggregate_canopy_from_weights.py for audited annual aggregation')
 root=a.root.resolve(); out=a.out.resolve();out.mkdir(exist_ok=False); start=time.time()
 try:
  source=json.loads(a.source_audit.read_text(encoding='utf-8'))
  require(source['status']==('HEADERS_ONLY' if a.weights_only else 'PASS_MAIN_YEARS'),'Source audit not passed')
  sources={r['year']:r for r in source['sources'] if 2000<=r['year']<=2025}; require(len(sources)==26,'Main year count')
  land_path=root/'04_pilot/landcover_2001/intermediate_2001/Annual_NLCD_LndCov_2001_CU_C1V2.tif'
  land_xml=land_path.with_suffix('.xml')
  import xml.etree.ElementTree as ET
  definitions={e.findtext('edomv'):e.findtext('edomvd') for e in ET.parse(land_xml).findall('.//edom')}
  require('Open Water' in definitions.get('11','') and definitions.get('250')=='NoData','Water/nodata definitions unverified')
  classes=set(map(int,definitions)); require(classes, 'Missing legend')
  paths=[land_path,land_xml,Path(__file__).resolve(),a.source_audit.resolve(),root/'03_protocol/research_protocol_review_v1.md']
  paths += [root/f'04_pilot/boundaries_2000/tract/tl_2010_{c}_tract00.zip' for c in COUNTIES]
  hashes={str(p):sha(p) for p in paths}; stat_before={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in paths}
  rows=[]; spatial=[]; counties=[]
  with rasterio.open(sources[2025]['uri']) as ref, rasterio.open(land_path) as ld:
   require(ref.crs==ld.crs and ref.res==ld.res==(30,30),'Different grid projections/resolutions')
   inv=~ref.transform
   dc=(ref.transform.c-ld.transform.c)/30; dr=(ld.transform.f-ref.transform.f)/30
   require(dc==round(dc) and dr==round(dr),'Land-cover grid is not integer aligned')
   for county in COUNTIES:
    tr=gpd.read_file(f'zip://{root.as_posix()}/04_pilot/boundaries_2000/tract/tl_2010_{county}_tract00.zip').sort_values('CTIDFP00').reset_index(drop=True)
    require(tr.CTIDFP00.is_unique,'Duplicate tract IDs'); require(tr.geometry.notna().all() and (~tr.geometry.is_empty).all() and tr.geometry.is_valid.all(),'Invalid source geometry')
    projected=tr.to_crs(ref.crs); require(projected.geometry.is_valid.all(),'Invalid projected geometry')
    geoms=[affine_transform(g,[inv.a,inv.b,inv.d,inv.e,inv.c,inv.f]) for g in projected.geometry]
    bx=np.array([g.bounds for g in geoms]);x0,y0=np.floor(bx[:,:2].min(axis=0)).astype(int);x1,y1=np.ceil(bx[:,2:].max(axis=0)).astype(int)
    require(0<=x0<x1<=ref.width and 0<=y0<y1<=ref.height,'County outside canopy raster')
    shape=(int(y1-y0),int(x1-x0));win=Window(int(x0),int(y0),int(x1-x0),int(y1-y0));ncell=int(np.prod(shape))
    land=ld.read(1,window=Window(int(x0+dc),int(y0+dr),int(x1-x0),int(y1-y0)))
    require(land.shape==shape,'Land-cover bounds invalid'); require(set(map(int,np.unique(land))).issubset(classes),'Unknown land-cover class')
    flat=land.ravel(); allidx=[];allweights=[];indptr=[0]; localrows=[]
    for i,g in enumerate(geoms):
     gx,gy,w,center=pixel_weights(g);ry,cx=np.nonzero(w>0);weights=w[ry,cx]*900
     idx=(ry+gy-y0)*shape[1]+cx+gx-x0; cl=flat[idx]
     geo=float(projected.geometry.iloc[i].area);total=float(weights.sum())
     require(abs(total-geo)<=max(.01,geo*1e-10),'Projected area does not close')
     rec={'county_code':county,'tract_id':str(tr.CTIDFP00.iloc[i]),'geometry_area_m2':geo,'pixel_intersection_area_m2':total,'closure_error_m2':total-geo,
      'fixed_land_m2':float(weights[(cl!=11)&(cl!=250)].sum()),'open_water_m2':float(weights[cl==11].sum()),'landcover_nodata_m2':float(weights[cl==250].sum()),
      'census_aland00_m2':float(tr.ALAND00.iloc[i]),'center_rule_geometry_area_m2':int(center.sum())*900,'boundary_partial_pixels':int(((w>0)&(w<1)).sum())}
     cy,cc=np.nonzero(center);ci=(cy+gy-y0)*shape[1]+cc+gx-x0;cclass=flat[ci]
     rec['center_rule_land_area_m2']=int(((cclass!=11)&(cclass!=250)).sum())*900
     rec['center_minus_exact_land_m2']=rec['center_rule_land_area_m2']-rec['fixed_land_m2']
     localrows.append(rec);allidx.append(idx.astype('int32'));allweights.append(weights);indptr.append(indptr[-1]+len(idx))
     if (i+1)%200==0: print(f'{county} exact weights {i+1}/{len(tr)}',flush=True)
    matrix=csr_matrix((np.concatenate(allweights),np.concatenate(allidx),np.array(indptr)),shape=(len(tr),ncell))
    del allidx,allweights
    column_sum=np.asarray(matrix.sum(axis=0)).ravel()
    require(float(column_sum.max())<=900.0001,'Overlapping tract cell areas')
    union_area=float(shapely.union_all(projected.geometry.values).area)
    require(abs(float(matrix.sum())-union_area)<=max(.1,union_area*1e-10),'Tract areas versus union mismatch')
    full_total=float(matrix.sum());del column_sum
    # Retain only the fixed 2001 non-open-water, non-nodata domain.
    matrix.data[(flat[matrix.indices]==11)|(flat[matrix.indices]==250)]=0; matrix.eliminate_zeros()
    save_npz(out/f'land_area_weights_{county}.npz',matrix,compressed=True)
    fixed=np.asarray(matrix.sum(axis=1)).ravel();require(np.all(fixed>0),'Zero-land tract requires adjudication')
    save_json(out/f'grid_{county}.json',{'shape':shape,'window':[int(x0),int(y0),int(x1-x0),int(y1-y0)],'crs_wkt':ref.crs.to_wkt(),'transform':list(ref.window_transform(win)),'tract_order':[r['tract_id'] for r in localrows]})
    mask=np.where(flat==250,255,np.where(flat==11,0,1)).astype('uint8').reshape(shape)
    with rasterio.open(out/f'fixed_landmask_{county}.tif','w',driver='GTiff',width=shape[1],height=shape[0],count=1,dtype='uint8',crs=ref.crs,transform=ref.window_transform(win),nodata=255,compress='DEFLATE',tiled=True) as ds:ds.write(mask,1)
    common=np.ones(ncell,dtype=bool); all_complete=True; base_rates=None
    for year,entry in ([] if a.weights_only else sorted(sources.items())):
     src=Path(entry['path']); st=src.stat();require((st.st_size,st.st_mtime_ns)==(entry['bytes'],entry['mtime_ns']),'Source changed since CRC audit')
     with rasterio.open(entry['uri']) as ds:
      require(ds.transform==ref.transform and ds.crs==ref.crs and ds.shape==ref.shape,'Annual grid changed'); vals=ds.read(1,window=win).ravel()
     require(np.all((vals<=100)|(vals==254)|(vals==255)),f'Unexpected canopy codes: {year}')
     valid=vals<=100;common&=valid
     area=np.asarray(matrix@valid.astype('float64')).ravel(); numerator=np.asarray(matrix@np.where(valid,vals,0).astype('float64')).ravel()
     complete=np.abs(area-fixed)<=1e-6;all_complete=all_complete and bool(complete.all())
     rates=np.divide(numerator,area,out=np.full(len(tr),np.nan),where=area>0)
     if year==2000:base_rates=rates.copy();base_complete=complete.copy()
     for j,r in enumerate(localrows):
      rows.append({'county_code':county,'tract_id':r['tract_id'],'year':year,'fixed_land_m2':float(fixed[j]),'valid_canopy_land_m2':float(area[j]),'missing_canopy_land_m2':float(fixed[j]-area[j]),'valid_fraction':float(area[j]/fixed[j]),
       'canopy_percent_observed_land':float(rates[j]) if np.isfinite(rates[j]) else None,
       'canopy_percent_fixed_land':float(rates[j]) if complete[j] else None,
       'canopy_area_m2_observed':float(numerator[j]/100),
       'change_from_2000_pp':float(rates[j]-base_rates[j]) if complete[j] and base_complete[j] and np.isfinite(base_rates[j]) else None,
       'status':'COMPLETE_FIXED_LAND' if complete[j] else 'INCOMPLETE_NO_IMPUTATION','scope':'WHOLE_TRACT_DIAGNOSTIC_NOT_FINAL_URBAN_SAMPLE'})
     print(f'{county} {year} pixels decoded; complete tracts {int(complete.sum())}/{len(tr)}',flush=True)
    common_area=np.asarray(matrix@common.astype('float64')).ravel()
    for j,r in enumerate(localrows):r['common_valid_land_2000_2025_m2']=None if a.weights_only else float(common_area[j])
    spatial.extend(localrows)
    counties.append({'county_code':county,'tract_count':len(tr),'geometry_area_m2':union_area,'intersection_area_m2':full_total,'fixed_land_m2':float(fixed.sum()),'water_m2':sum(r['open_water_m2'] for r in localrows),'landcover_nodata_m2':sum(r['landcover_nodata_m2'] for r in localrows),'all_years_complete':None if a.weights_only else all_complete,'common_valid_land_m2':None if a.weights_only else float(common_area.sum())})
    if not a.weights_only: save_csv(out/f'tract_year_{county}.csv',[r for r in rows if r['county_code']==county])
    print(f'{county} done',flush=True)
  require(len(spatial)==2656 and len(rows)==(0 if a.weights_only else 2656*26),'Unexpected output counts')
  require(len({(r['tract_id'],r['year']) for r in rows})==len(rows),'Duplicate tract-year')
  for p in paths: require(stat_before[str(p)]==(p.stat().st_size,p.stat().st_mtime_ns),f'Input changed: {p}')
  save_csv(out/'tract_area_audit.csv',spatial)
  if not a.weights_only: save_csv(out/'tract_year_canopy_2000_2025.csv',rows)
  save_json(out/'audit.json',{'status':'PASS_WEIGHTS_ONLY' if a.weights_only else 'PASS_DIAGNOSTIC' if all(c['all_years_complete'] and c['landcover_nodata_m2']==0 for c in counties) else 'REVIEW_MISSING_COVERAGE','scope':'all complete tract geometries in 3 counties; urban clipping and population correspondence NOT resolved','years':list(range(2000,2026)),'tract_count':len(spatial),'tract_year_count':len(rows),'duplicates':0,'dropped':0,'counties':counties,'source_file_sha256':hashes,'archive_source_audit':str(a.source_audit.resolve()),'self_test':test,'landcover_offset_pixels':[dc,dr],'resampling':'NONE: same projection, resolution and integer-aligned grid; window translation only','formal_model':'NOT_RUN','project_state_modified':False,'runtime_seconds':time.time()-start,'software':{'python':sys.version,'rasterio':rasterio.__version__,'shapely':shapely.__version__,'geopandas':gpd.__version__}})
  print(json.dumps({'complete':True,'rows':len(rows),'runtime_seconds':time.time()-start}),flush=True)
 except BaseException:
  (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
if __name__=='__main__':main()
