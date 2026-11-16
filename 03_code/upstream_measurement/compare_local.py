"""Exact categorical-pixel/geometry areas on frozen complete 30 m land cells.
No pixel-centre approximation, reference resampling, grouping changes or training.
"""
import argparse,csv,json,sys,math,time
from pathlib import Path
from collections import defaultdict
from zipfile import ZipFile
import numpy as np,pandas as pd,rasterio,pyogrio,shapely
from rasterio import features
from rasterio.windows import Window,from_bounds,transform as window_transform
from affine import Affine
from pyproj import Transformer,CRS
from scipy.sparse import load_npz
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
SRC=ROOT/'06_logs/canopy_multicity_reference_20261026_run01'
CAT=ROOT/'06_logs/national_urban_weights_summary_20260914_run01/national_weight_catalog.csv'
def require(v,msg):
    if not v:raise ValueError(msg)
def jr(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def csvread(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def jwrite(p,obj):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8');temp.replace(p)
def projected(g,src,dst):
    if CRS(src)==CRS(dst):return g
    tr=Transformer.from_crs(src,dst,always_xy=True)
    return shapely.transform(g,tr.transform,interleaved=False)
def area_only(g):
    if g.geom_type!='GeometryCollection':return g
    polygons=[]
    for part in g.geoms:
        if part.geom_type in ['Polygon','MultiPolygon']:polygons.append(part)
        elif part.geom_type=='GeometryCollection':polygons.append(area_only(part))
        else:require(part.area==0,'Nonpolygon area')
    result=shapely.union_all(polygons)
    require(abs(result.area-g.area)<=max(.001,g.area*1e-12),'Polygon-part extraction changed area')
    return result
def source(year):
    p=ROOT/f'04_pilot/tcc_annual/nlcd_tcc_conus_{year}_v2025-6_wgs84.zip'
    with ZipFile(p) as z:n=[n for n in z.namelist() if n.lower().endswith('.tif')]
    require(len(n)==1,'Ambiguous national raster')
    return rasterio.open(f'/vsizip/{p.as_posix()}/{n[0]}')
def subwindows(ds,geom,size=1024):
    if geom.is_empty or geom.area==0:return
    w=from_bounds(*geom.bounds,transform=ds.transform)
    c0=max(0,math.floor(w.col_off));r0=max(0,math.floor(w.row_off))
    c1=min(ds.width,math.ceil(w.col_off+w.width));r1=min(ds.height,math.ceil(w.row_off+w.height))
    for r in range(r0,r1,size):
        for c in range(c0,c1,size):
            win=Window(c,r,min(size,c1-c),min(size,r1-r))
            bounds=rasterio.windows.bounds(win,ds.transform)
            g=area_only(shapely.intersection(geom,shapely.box(*bounds)))
            if not g.is_empty and g.area>0:yield win,g
def exact_hist(ds,geom):
    totals=defaultdict(float);touched=0
    require(ds.transform.b==0 and ds.transform.d==0 and ds.transform.a>0 and ds.transform.e<0,'Rotated source grid')
    for win,g in subwindows(ds,geom):
        a=ds.read(1,window=win);tf=window_transform(win,ds.transform);shape=a.shape
        mask=features.rasterize([(g,1)],out_shape=shape,transform=tf,dtype='uint8').astype(bool)
        edge=features.rasterize([(g.boundary,1)],out_shape=shape,transform=tf,all_touched=True,dtype='uint8').astype(bool)
        vals,counts=np.unique(a[mask&~edge],return_counts=True);pixel=abs(tf.a*tf.e)
        for v,n in zip(vals,counts):totals[int(v)]+=float(n)*pixel
        rr,cc=np.where(edge);touched+=len(rr)
        if len(rr):
            left=tf.c+cc*tf.a;top=tf.f+rr*tf.e
            cells=shapely.box(left,top+tf.e,left+tf.a,top)
            areas=shapely.area(shapely.intersection(cells,g))
            for v in np.unique(a[rr,cc]):totals[int(v)]+=float(areas[a[rr,cc]==v].sum())
    require(abs(sum(totals.values())-geom.area)<=max(.02,geom.area*2e-8),'Reference area closure failed')
    return dict(totals),touched
def valid_domain(ds,geom,invalid):
    g=shapely.intersection(geom,shapely.box(*ds.bounds));bad=[]
    if g.is_empty or g.area==0:return g
    for win,piece in subwindows(ds,g):
        a=ds.read(1,window=win);mask=np.isin(a,invalid)
        if mask.any():
            tf=window_transform(win,ds.transform)
            bad.extend(shapely.geometry.shape(s) for s,v in features.shapes(mask.astype('uint8'),mask=mask,transform=tf))
    if bad:g=shapely.difference(g,shapely.union_all(bad))
    return area_only(g)
def build_geometry(indices,grid):
    h,w=grid['shape'];tf=Affine(*grid['transform'][:6]);rr,cc=np.divmod(indices,w)
    require(tf.a==30 and tf.e==-30 and tf.b==tf.d==0,'Frozen grid differs')
    left=tf.c+30*cc;top=tf.f-30*rr
    cells=shapely.box(left,top-30,left+30,top)
    return cells
def union_cells(indices,grid):
    width=grid['shape'][1];rr,cc=np.divmod(indices,width);r0,r1=int(rr.min()),int(rr.max());c0,c1=int(cc.min()),int(cc.max())
    arr=np.zeros((r1-r0+1,c1-c0+1),dtype='uint8');arr[rr-r0,cc-c0]=1
    tf=Affine(*grid['transform'][:6])*Affine.translation(c0,r0)
    polys=[shapely.geometry.shape(g) for g,v in features.shapes(arr,mask=arr.astype(bool),transform=tf)]
    return shapely.union_all(polys)
def nlcd_at_cells(ds,indices,grid):
    require(ds.crs==rasterio.crs.CRS.from_wkt(grid['crs_wkt']) and ds.res==(30.,30.),'NLCD/grid CRS mismatch')
    tf=Affine(*grid['transform'][:6]);rr,cc=np.divmod(indices,grid['shape'][1]);x=tf.c+30*(cc+.5);y=tf.f-30*(rr+.5)
    cols=(x-ds.transform.c)/30-.5;rows=(ds.transform.f-y)/30-.5
    require(np.allclose(cols,np.rint(cols),atol=1e-6,rtol=0) and np.allclose(rows,np.rint(rows),atol=1e-6,rtol=0),'NLCD alignment mismatch')
    rows=np.rint(rows).astype(int);cols=np.rint(cols).astype(int)
    r0,r1=rows.min(),rows.max();c0,c1=cols.min(),cols.max()
    a=ds.read(1,window=Window(int(c0),int(r0),int(c1-c0+1),int(r1-r0+1)))
    v=a[rows-r0,cols-c0];require(np.all(v<=100),'Invalid NLCD within frozen land')
    return v
def aggregate(cfg,limit=None):
    run=OUT/'comparisons_exact_v2'/cfg['key'];run.mkdir(parents=True,exist_ok=True)
    recdir=run/'records';recdir.mkdir(exist_ok=True)
    # Distinguish validation runs from complete analyses by their separate key/output.
    jwrite(run/'configuration.json',cfg)
    frame=pd.read_csv(SRC/'city45_frozen_product_tracts.csv',dtype={'tract_id':str,'state':str},float_precision='round_trip')
    frame['tract_id']=frame.tract_id.str.zfill(11);frame['state']=frame.state.str.zfill(2)
    frame=frame.loc[frame.city_number.eq(cfg['city_number'])].copy()
    datasets=[rasterio.open(p) for p in cfg.get('rasters',[cfg.get('raster')])]
    ds=datasets[0];refcrs=CRS(ds.crs);require(refcrs.is_projected,'Reference CRS must be projected')
    require(all(CRS(d.crs)==refcrs for d in datasets),'Endpoint reference CRS differ; explicit spatial mapping required')
    require(len(datasets)==1 or len(datasets)==len(cfg['years']),'Endpoint count differs')
    unit=refcrs.axis_info[0].unit_conversion_factor;require(unit>0,'Unknown reference length unit')
    if cfg.get('aoi'):
        aoi=pyogrio.read_dataframe(cfg['aoi']);require(aoi.geometry.notna().all() and aoi.is_valid.all(),'Invalid AOI geometry')
        domain=projected(shapely.union_all(aoi.geometry.to_numpy()),aoi.crs,refcrs)
        domain=shapely.intersection(domain,shapely.box(*ds.bounds))
    else:domain=shapely.box(*ds.bounds)
    for d in datasets[1:]:domain=shapely.intersection(domain,shapely.box(*d.bounds))
    require(domain.is_valid and domain.area>0,'Invalid reference domain')
    national={y:source(y) for y in cfg['years']}
    maps=[{int(k):v for k,v in m.items()} for m in cfg['canopy_maps']]
    catalogs={r['state']:r for r in csvread(CAT)};processed=0;outside=[]
    for state,ff in frame.groupby('state',sort=True):
        cat=catalogs[state];grid=jr(cat['grid_path']);audit=jr(cat['audit_path'])
        require(audit['status'] in ['PASS_DIAGNOSTIC_STRUCTURE','PASS_DIAGNOSTIC_STRUCTURE_AND_REGRESSION'],'Frozen weight gate')
        domain_grid=projected(domain,refcrs,grid['crs_wkt']);b=domain_grid.bounds
        shardgroups=ff.groupby('weight_shard',sort=True)
        for shard,subset in shardgroups:
            wd=Path(cat['weights_dir']);matrix=load_npz(wd/f'urban_weights_{shard}.npz').tocsr()
            table=csvread(wd/f'urban_area_{shard}.csv');lookup={r['tract_id'].zfill(11):i for i,r in enumerate(table)}
            require(matrix.shape[0]==len(table) and matrix.has_sorted_indices,'Sparse weights shape')
            for _,tr in subset.iterrows():
                tid=tr.tract_id;record=recdir/f'{tid}.json'
                if record.exists():processed+=1;continue
                i=lookup[tid];st,en=matrix.indptr[i:i+2];ind=matrix.indices[st:en];weights=matrix.data[st:en]
                require(abs(weights.sum()-tr.fixed_land_m2)<max(.02,tr.fixed_land_m2*1e-9),'Frozen land mismatch')
                full=ind[np.isclose(weights,900.,rtol=0,atol=.01)]
                if not len(full):outside.append({'tract_id':tid,'reason':'no_complete_land_cell'});continue
                tf=Affine(*grid['transform'][:6]);rr,cc=np.divmod(full,grid['shape'][1]);x=tf.c+30*cc;y=tf.f-30*rr
                candidate=full[(x<b[2])&(x+30>b[0])&(y>b[1])&(y-30<b[3])]
                if not len(candidate):outside.append({'tract_id':tid,'reason':'outside_reference_bbox'});continue
                # Project the same individual cell edges used by both numerators.
                # Projecting simplified long union edges is not geometrically identical.
                cells=projected(build_geometry(candidate,grid),grid['crs_wkt'],refcrs)
                geom=shapely.coverage_union_all(cells)
                geom=shapely.intersection(geom,domain)
                if cfg.get('invalid_codes'):
                    for d in datasets:geom=valid_domain(d,geom,cfg['invalid_codes'])
                if geom.is_empty or geom.area<=.0001:outside.append({'tract_id':tid,'reason':'no_common_valid_area'});continue
                require(geom.is_valid,'Invalid common geometry')
                areas=shapely.area(shapely.intersection(cells,geom));keep=areas>1e-8;areas=areas[keep];ids=candidate[keep]
                require(abs(areas.sum()-geom.area)<max(.05,geom.area*2e-7),'NLCD/reference support closure')
                hists=[];boundary=0
                for di,d in enumerate(datasets):
                    hist,nb=exact_hist(d,geom);hists.append(hist);boundary+=nb
                    unknown=set(k for k,a in hist.items() if a>1e-6)-set(maps[di])
                    require(not unknown,'Unmapped reference codes '+str(unknown))
                result={'tract_id':tid,'state':state,'city_number':int(cfg['city_number']),'dataset':cfg['key'],
                        'fixed_land_m2':float(tr.fixed_land_m2),'complete_land_m2':len(full)*900.,
                        'common_area_m2':geom.area*unit**2,'support_fraction':geom.area*unit**2/tr.fixed_land_m2,
                        'intersecting_nlcd_cells':len(ids),'boundary_pixels_exact':boundary,
                        **{v:float(tr[v]) for v in ['income_city_quartile','poverty_city_quartile','age65plus_city_quartile']},
                        'reference_class_area_source_units':hists}
                for k,y in enumerate(cfg['years']):
                    hist=hists[k] if len(hists)>1 else hists[0]
                    require(all(code in maps[k] for code,a in hist.items() if a>1e-6),'Endpoint canopy crosswalk incomplete')
                    result[f'reference_{y}']=100*sum(a*maps[k][code] for code,a in hist.items() if a>1e-6)/geom.area
                    v=nlcd_at_cells(national[y],ids,grid)
                    result[f'nlcd_{y}']=float(np.dot(v,areas)/areas.sum())
                if len(cfg['years'])==2:
                    y0,y1=cfg['years'];result['reference_change']=result[f'reference_{y1}']-result[f'reference_{y0}'];result['nlcd_change']=result[f'nlcd_{y1}']-result[f'nlcd_{y0}']
                jwrite(record,result);processed+=1
                if processed%10==0:print(cfg['key'],processed,'tracts saved',flush=True)
                if limit and processed>=limit:
                    jwrite(run/'status.json',{'status':'BOUNDED_PREFLIGHT','saved':processed});return
    for d in datasets:d.close()
    for s in national.values():s.close()
    records=[jr(p) for p in sorted(recdir.glob('*.json'))]
    result=pd.DataFrame(records).drop(columns=['reference_class_area_source_units']);result.to_csv(run/'tract_comparison.csv',index=False)
    pd.DataFrame(outside).to_csv(run/'excluded_tracts.csv',index=False)
    require(len(result)+len(outside)==len(frame),'Tract accounting mismatch')
    jwrite(run/'status.json',{'status':'COMPLETE','input_city_tracts':len(frame),'paired_tracts':len(result),'outside':len(outside),
                           'common_area_m2':float(result.common_area_m2.sum()),'input_fixed_area_m2':float(frame.fixed_land_m2.sum()),
                           'area_retained':float(result.common_area_m2.sum()/frame.fixed_land_m2.sum()),'formal_outputs_modified':False})
    print(cfg['key'],'COMPLETE',len(result),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--limit',type=int);a=p.parse_args();aggregate(jr(a.config),a.limit)
if __name__=='__main__':main()
