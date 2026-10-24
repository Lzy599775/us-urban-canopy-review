"""45-city response atlas. Read-only scientific inputs; exact inherited geometry.

Run in sportpark-ee: python figure04_atlas.py all
Explicit stages: download, prepare, render, validate.
No modelling, requalification, or result-dependent selection. One explicitly
authorized numerical repair is restricted to a recorded display component.
"""
from nc_paths import project_root as _nc_project_root, external_path as _nc_external_path
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import math
import sys
import time
import zipfile
from contextlib import contextmanager

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages, PdfFile
import numpy as np
import pandas as pd
import pymupdf
from pyproj import CRS, Transformer, Geod
import requests
import shapely
from scipy.optimize import brentq
from shapely.geometry import box

ROOT=_nc_project_root()
OUT=Path(__file__).resolve().parent
FIG1=ROOT/'08_figure图/Fig01_全国主分析样本空间分布'
ROSTER=ROOT/'08_figure图/replanning_20261013/revision_45cities/proposed_45_city_inventory.csv'
SCORES=ROOT/'06_logs/geo_adjusted_national_orthogonal_20261009_run01/crossfit_scores.csv'
BASELINE=ROOT/'06_logs/canopy_cooling_equity_20261009_run01/baseline_equity_group_manifest.csv'
URBAN=ROOT/'04_pilot/boundaries_2000/urban_area/tl_2009_us_uac00.zip'
REFERENCE=ROOT/'02_literature/benchmark_papers_20261012_run01/pdf/08_US_vulnerable_tree_cooling_2021.pdf'
MAP_CRS=CRS.from_epsg(5070)
TO_LL=Transformer.from_crs(MAP_CRS,4326,always_xy=True)
FROM_LL=Transformer.from_crs(4326,MAP_CRS,always_xy=True)
GEOD=Geod(ellps='WGS84')
ASPECT=1.6
DIVISIONS=['New England','Middle Atlantic','East North Central','West North Central','South Atlantic','East South Central','West South Central','Mountain','Pacific']
SHORT=['Boston','Providence','Hartford','Portland, ME','Manchester',
       'New York–Newark','Philadelphia','Pittsburgh','Buffalo','Rochester, NY',
       'Chicago','Detroit','Cleveland','Indianapolis','Milwaukee',
       'Minneapolis–St. Paul','Kansas City','St. Louis','Omaha','Des Moines',
       'Washington, DC','Richmond','Atlanta','Miami','Charlotte',
       'Louisville','Nashville','Memphis','Birmingham','Jackson, MS',
       'Houston','Dallas–Fort Worth','New Orleans','Oklahoma City','Little Rock',
       'Phoenix–Mesa','Denver–Aurora','Salt Lake City','Las Vegas','Albuquerque',
       'Los Angeles','San Francisco–Oakland','Seattle','Portland, OR','San Diego']
# Fixed numeric bins, shared across every panel, covering the full national range.
# Zero is a true sign boundary; these are display classes, not significance levels.
BREAKS=np.array([-.55,-.30,-.24,-.20,-.16,-.12,-.08,0,.05])
COLORS=['#377F7B','#559A91','#79B3A6','#A1C9B7','#C5DCC9','#E1E9D7','#F4F1DF','#E7B39B']
CMAP=ListedColormap(COLORS)
NORM=BoundaryNorm(BREAKS,CMAP.N,clip=False)
LAND='#FFFFFF';WATER='#FFFFFF';UNMAPPED='#DADCDD';INK='#252D31';FAINT='#A8AEAE'

def emit(x): print(time.strftime('%H:%M:%S'),x,flush=True)
def require(condition,message):
    if not condition: raise ValueError(message)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def rd(p):return pd.read_csv(p,dtype={'state':str,'tract_id':str,'single_city_id':str},float_precision='round_trip')
def init():
    require(Path(sys.executable).parent.name=='sportpark-ee','Use sportpark-ee')
    for d in ['raw_basemap','prepared','source_data','exports','qa','schemas']:(OUT/d).mkdir(exist_ok=True)
    font=font_manager.findfont(font_manager.FontProperties(family='Arial'),fallback_to_default=False)
    require('arial' in Path(font).name.lower(),'Arial font not found')
    plt.rcParams.update({'font.family':'Arial','font.size':7,'pdf.fonttype':42,'ps.fonttype':42,
                         'svg.fonttype':'none','savefig.dpi':600,'axes.linewidth':.4,
                         'text.color':INK,'axes.labelcolor':INK,'path.simplify':False})
    return font

def download():
    p=OUT/'raw_basemap/ne_10m_land.zip'
    side=p.with_suffix('.source.json')
    url='https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip'
    if p.exists():
        require(side.exists(),'Existing land layer lacks provenance')
        require(json.loads(side.read_text(encoding='utf8'))['sha256']==sha(p),'Land file changed')
        emit('Existing 1:10m land source verified');return
    session=requests.Session();session.trust_env=False
    emit('Downloading Natural Earth 1:10m land context')
    with session.get(url,stream=True,timeout=(20,120)) as r:
        r.raise_for_status()
        part=p.with_suffix('.part')
        with part.open('wb') as f:
            for chunk in r.iter_content(1024*1024):f.write(chunk)
        with zipfile.ZipFile(part) as z:require(z.testzip() is None,'Land archive corrupt')
        part.rename(p)
        dump(side,{'url':url,'resolved_url':r.url,'sha256':sha(p),'bytes':p.stat().st_size,
                   'download_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                   'role':'Cartographic land and coastline only; not analytical land mask',
                   'proxy_environment_used':False,'operating_system_vpn_route':'NOT VERIFIED'})

def window_for(u,g):
    ub=u.bounds; gb=g.total_bounds
    lo=np.minimum(ub[:2],gb[:2]);hi=np.maximum(ub[2:],gb[2:])
    cx,cy=(lo+hi)/2;w,h=(hi-lo)*1.10
    if w/h<ASPECT:w=h*ASPECT
    else:h=w/ASPECT
    return [float(cx-w/2),float(cy-h/2),float(cx+w/2),float(cy+h/2)]

def scale_north(bounds):
    x0,y0,x1,y1=bounds;w=x1-x0;h=y1-y0
    cx=(x0+x1)/2;cy=(y0+y1)/2
    x=cx-w*.1;y=cy
    def distance(dx):
        lon,lat=TO_LL.transform(x,y);lo2,la2=TO_LL.transform(x+dx,y)
        return GEOD.inv(lon,lat,lo2,la2)[2]
    target=distance(w*.23)
    choices=np.array([1,2,5,10,20,25,50,100,200,500])*1000
    target_m=float(choices[choices<=target][-1])
    dx=brentq(lambda q:distance(q)-target_m,.01,w*.4,xtol=1e-7)
    lon,lat=TO_LL.transform(cx,cy)
    nx,ny=FROM_LL.transform(lon,lat+.02)
    vec=np.array([nx-cx,ny-cy]);vec/=np.linalg.norm(vec)
    return {'scale_distance_m':target_m,'scale_projected_dx_m':float(dx),
            'scale_geodesic_verified_m':float(distance(dx)),
            'scale_reference_endpoints_epsg5070':[[x,y],[x+dx,y]],
            'north_unit_vector_epsg5070':vec.tolist(),'north_reference_lon_lat':[lon,lat]}

def clip_context(source,bounds,layer=None):
    w=box(*bounds)
    d=source.iloc[source.sindex.query(w,predicate='intersects')].copy()
    require(d.geometry.is_valid.all(),'Invalid context source geometry')
    d.geometry=d.geometry.intersection(w)
    require(d.geometry.is_valid.all() and (~d.geometry.is_empty).all(),'Invalid context clipping')
    return d

def prepare():
    t0=time.monotonic();sources={str(p):sha(p) for p in [ROSTER,SCORES,BASELINE,URBAN,REFERENCE]}
    r=rd(ROSTER).sort_values('city_number').reset_index(drop=True)
    require(len(r)==45 and r.single_city_id.is_unique,'45-city roster differs')
    s=rd(SCORES);b=rd(BASELINE)
    require(len(s)==104371 and s.tract_id.nunique()==42373,'National sample differs')
    check=s.groupby(['state','tract_id']).theta.agg(['min','max'])
    require(check['min'].eq(check['max']).all(),'Within-tract predictions differ; design needs revision')
    require(s.theta.between(BREAKS[0],BREAKS[-1]).all(),'Color scale does not cover national results')
    x=s.groupby(['state','tract_id']).agg(theta=('theta','first'),analysis_records=('theta','size')).reset_index()
    x=x.merge(b[['state','tract_id','single_city_id','canopy_equity_balanced_26years']],on=['state','tract_id'],validate='one_to_one')
    x=x[x.single_city_id.isin(r.single_city_id)].copy()
    require(len(x)==24493 and x.analysis_records.sum()==61135,'Selected-city sample differs')
    require(x.canopy_equity_balanced_26years.all(),'Figure 6 paired support differs')
    x['response_display_bin']=np.searchsorted(BREAKS,x.theta,side='right')-1
    x.to_csv(OUT/'source_data/figure04_tract_values.csv',index=False)
    urban=gpd.read_file('zip://'+URBAN.as_posix())
    require(urban.crs.to_epsg()==4269,'Unexpected urban source CRS')
    urban=urban[urban.UACE00.isin(r.single_city_id)].to_crs(MAP_CRS)
    require(len(urban)==45 and urban.UACE00.is_unique and urban.is_valid.all(),'Urban geometry mismatch')
    parts=[]
    for state in sorted(x.state.unique()):
        p=FIG1/f'prepared/states/{state}.gpkg';receipt=json.loads(p.with_suffix('.json').read_text(encoding='utf8'))
        sources[str(p)]=sha(p)
        require(sources[str(p)]==receipt['output_sha256'],f'Source geometry changed: {state}')
        g=gpd.read_file(p,layer='urban_intersection_parts')
        require(g.crs==MAP_CRS,'Source geometry CRS differs')
        g=g[g.tract_id.isin(set(x.loc[x.state.eq(state),'tract_id']))]
        parts.append(g)
        emit(f'Read inherited geometry state {state}: {g.tract_id.nunique()} tracts')
    g=gpd.GeoDataFrame(pd.concat(parts,ignore_index=True),crs=MAP_CRS)
    invalid=g.loc[~g.is_valid]
    require(len(invalid)==1 and invalid.iloc[0].tract_id=='24015030602' and int(invalid.iloc[0].part_rank)==4,
            'Inherited geometry issue set differs from the specifically authorized display-only repair')
    idx=invalid.index[0];before=g.loc[idx,'geometry'];after=shapely.make_valid(before)
    require(after.is_valid and abs(after.area-before.area)<1e-7 and before.hausdorff_distance(after)<1e-9,
            'Authorized numerical repair exceeds diagnosed size; STOP')
    g.loc[idx,'geometry']=after
    repair={'user_authorization':'Explicit reply: allow repair only in display copy',
            'tract_id':'24015030602','part_rank':4,'operation':'shapely.make_valid on one projected display component',
            'original_valid_before_projection':True,'original_area_m2':before.area,'display_repaired_area_m2':after.area,
            'area_change_m2':after.area-before.area,'hausdorff_distance_m':before.hausdorff_distance(after),
            'before_wkb_sha256':hashlib.sha256(before.wkb).hexdigest(),'after_wkb_sha256':hashlib.sha256(after.wkb).hexdigest(),
            'raw_geometry_modified':False,'model_or_area_weights_modified':False}
    dump(OUT/'qa/authorized_display_geometry_repair.json',repair)
    require(g.is_valid.all() and (~g.is_empty).all() and g.area.gt(0).all(),'Invalid sample geometry')
    g=g.merge(x,on=['state','tract_id'],validate='many_to_one',indicator=True)
    require(g._merge.eq('both').all() and g.tract_id.nunique()==24493,'Sample/geometry join mismatch')
    g=g.drop(columns='_merge')
    land_path=OUT/'raw_basemap/ne_10m_land.zip'
    lake_path=FIG1/'raw_basemap/ne_10m_lakes.zip'
    state_path=FIG1/'raw_basemap/cb_2024_us_state_500k.zip'
    sources.update({str(p):sha(p) for p in [land_path,lake_path,state_path]})
    land=gpd.read_file('zip://'+land_path.as_posix(),bbox=(-130,20,-60,55))
    require(land.crs.to_epsg()==4326 and land.is_valid.all(),'Invalid native land context')
    # Bbox reads return whole global polygons. Clip valid geographic context first
    # to avoid projecting remote/dateline portions outside the CONUS CRS domain.
    # This is viewport clipping of context, not geometry repair or sample clipping.
    land.geometry=land.geometry.intersection(box(-130,20,-60,55))
    land=land.loc[~land.is_empty].to_crs(MAP_CRS)
    require(land.is_valid.all(),'Invalid land context after domain-limited reprojection')
    lakes=gpd.read_file('zip://'+lake_path.as_posix(),bbox=(-130,20,-60,55)).to_crs(MAP_CRS)
    states=gpd.read_file('zip://'+state_path.as_posix())
    states=states[~states.STUSPS.isin(['AK','HI','PR','VI','GU','MP','AS'])].to_crs(MAP_CRS)
    require(len(states)==49,'Locator must contain 48 states and DC')
    locator=OUT/'prepared/locator_states.gpkg'
    states.to_file(locator,layer='states',driver='GPKG')
    windows=[];reports=[]
    for i,row in r.iterrows():
        city=row.single_city_id;cg=g[g.single_city_id.eq(city)].copy()
        ug=urban[urban.UACE00.eq(city)].copy();u=ug.geometry.iloc[0]
        require(cg.tract_id.nunique()==row.main_unique_tracts,'Per-city count differs')
        bounds=window_for(u,cg);w=box(*bounds)
        require(cg.covered_by(w).all() and w.covers(u),'Window crops selected geography')
        sc=scale_north(bounds)
        # Reprojection of separately intersected boundaries can cause sub-metre edge
        # differences. Record the area; never reclip or repair analytical geometry.
        outside=float(cg.geometry.difference(u).area.sum())
        report={'city_id':city,'city_number':int(row.city_number),'official_name':row.NAME00,
                'short_name':SHORT[i],'division':DIVISIONS[i//5],
                'main_tracts':int(row.main_unique_tracts),'records':int(row.main_records),
                'parts':len(cg),'positive_prediction_tracts':int(cg.loc[cg.theta.gt(0),'tract_id'].nunique()),
                'bounds_epsg5070_m':bounds,'urban_bounds_epsg5070_m':list(u.bounds),
                'geometry_outside_reprojected_urban_m2':outside,'sample_polygon_area_m2':float(cg.area.sum()),
                'window_crops_zero_features':True,'valid_geometries':True,
                'source_urban_crs':'EPSG:4269','display_crs':'EPSG:5070',**sc}
        windows.append(report)
        dest=OUT/f'prepared/city_{int(row.city_number):02d}_{city}.gpkg'
        cg.to_file(dest,layer='sample_parts',driver='GPKG')
        ug.to_file(dest,layer='urban_2000',driver='GPKG')
        for name,data in [('land_context',land),('lakes_context',lakes),('state_context',states)]:
            cl=clip_context(data,bounds)
            if len(cl):cl.to_file(dest,layer=name,driver='GPKG')
            report[name+'_features']=len(cl)
        report['prepared_file']=str(dest);report['prepared_sha256']=sha(dest)
        reports.append(report)
        emit(f'Prepared {i+1}/45 {row.NAME00}: {cg.tract_id.nunique()} tracts; {len(cg)} parts')
    dump(OUT/'source_data/city_windows_epsg5070.json',windows)
    flat=[{k:v for k,v in d.items() if not isinstance(v,(list,dict))} for d in reports]
    pd.DataFrame(flat).to_csv(OUT/'source_data/city_geometry_checks.csv',index=False)
    r.assign(short_name=SHORT,division=[DIVISIONS[i//5] for i in range(45)]).to_csv(OUT/'source_data/city_index.csv',index=False)
    dump(OUT/'qa/preparation.json',{'status':'PASS','national_records':104371,'national_unique_tracts':42373,
         'mapped_unique_tracts':24493,'mapped_records':61135,'mapped_parts':len(g),'cities':45,
         'unmatched_tracts':0,'empty_or_invalid_sample_parts':0,'silently_dropped_sample_parts':0,
         'sources_sha256':sources,'windows':reports,'locator_sha256':sha(locator),
         'display_rule':'All parts of inherited city-intersection geometry for uniquely city-assigned selected tracts',
         'unmapped_urban_background':'Other area within the 2000 urban polygon; includes locations outside the displayed formal sample',
         'not_a_land_mask':'Context water/land does not clip or alter analytical polygons',
         'elapsed_seconds':time.monotonic()-t0})

def geo_plot(ax,g,color=None,column=None,lw=0,edge='none',z=1):
    if not len(g):return
    kw={'ax':ax,'linewidth':lw,'edgecolor':edge,'zorder':z,'rasterized':True,'aspect':None}
    if column:kw.update(column=column,cmap=CMAP,norm=NORM)
    else:kw.update(color=color)
    g.plot(**kw)

def read_city(w):
    p=Path(w['prepared_file'])
    layers=set(gpd.list_layers(p).name)
    return {k:gpd.read_file(p,layer=k) for k in layers}

def map_panel(fig,rect,w,data,detail=False):
    ax=fig.add_axes(rect);ax.set_facecolor(WATER)
    # User requested urban-theme-only maps on pure white. Context source layers
    # remain available for traceability, but are not drawn or used to clip data.
    geo_plot(ax,data['urban_2000'],UNMAPPED,lw=.22 if detail else .10,edge='#9EA8A8',z=3)
    geo_plot(ax,data['sample_parts'],column='theta',lw=.11 if detail else .035,edge='#F8F8F3',z=4)
    bounds=w['bounds_epsg5070_m'];x0,y0,x1,y1=bounds
    ax.set_xlim(x0,x1);ax.set_ylim(y0,y1);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
    for s in ax.spines.values():s.set_visible(True);s.set_linewidth(.25);s.set_color('#D4D9D8')
    # A true-north arrow measured at panel centre. Direction corrected for map projection.
    vx,vy=w['north_unit_vector_epsg5070'];length=.09 if detail else .11
    # Axes units are anisotropic; convert projected direction into displayed axis fractions.
    dx=vx*length/ASPECT;dy=vy*length
    ax.annotate('',xy=(.93+dx,.77+dy),xytext=(.93,.77),xycoords='axes fraction',
                arrowprops={'arrowstyle':'-|>','color':INK,'lw':.5,'mutation_scale':6 if detail else 4.7},zorder=8)
    ax.text(.93+dx,.77+dy+.025,'N',transform=ax.transAxes,ha='center',va='bottom',fontsize=6,zorder=8)
    # Scale and count in a dedicated strip below the map: never cover source geometry.
    sx=.02;fraction=w['scale_projected_dx_m']/(x1-x0)
    y=-.055
    ax.plot([sx,sx+fraction],[y,y],transform=ax.transAxes,color=INK,lw=.65,clip_on=False)
    for v in [sx,sx+fraction]:ax.plot([v,v],[y-.015,y+.015],transform=ax.transAxes,color=INK,lw=.5,clip_on=False)
    size=6.1 if detail else 6
    ax.text(sx+fraction+.025,y,f"{w['scale_distance_m']/1000:g} km",transform=ax.transAxes,fontsize=size,va='center')
    ax.text(.995,y,f"n = {w['main_tracts']:,}",transform=ax.transAxes,fontsize=size,ha='right',va='center')
    return ax

def common_legend(fig,y=.025):
    ax=fig.add_axes([.15,y+.027,.70,.018])
    sm=matplotlib.cm.ScalarMappable(norm=NORM,cmap=CMAP)
    cb=fig.colorbar(sm,cax=ax,orientation='horizontal',boundaries=BREAKS,ticks=BREAKS,spacing='uniform')
    cb.ax.tick_params(labelsize=6,length=2,pad=2);cb.outline.set_linewidth(.3)
    fig.text(.5,y+.057,'Predicted conditional response (°C per canopy percentage point)',
             fontsize=7,ha='center',va='bottom')
    fig.legend(handles=[Patch(facecolor=UNMAPPED,edgecolor='none',label='Other urban area')],
               loc='lower center',bbox_to_anchor=(.5,.002),ncol=1,frameon=False,fontsize=6.4,
               handlelength=1.2,columnspacing=2)

def check_text(fig,name):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[]
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():continue
        b=text.get_window_extent(renderer)
        if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.x1+1 or b.y1 > fig.bbox.y1+1:
            outside.append(text.get_text())
    require(not outside,f'{name}: text clipped outside page: {outside}')
    return {'file':name,'texts_outside_canvas':outside}

@contextmanager
def compact_pdf_image_buffers():
    # Matplotlib retains a cropped view backed by an entire 600-dpi page for
    # each rasterized panel. Own only the crop, preserving every byte exactly.
    # Scoped to this single-process exporter; installed packages are unchanged.
    original=PdfFile.imageObject
    def own_crop(pdf,image):
        compact=np.array(image,copy=True,order='C')
        require(np.array_equal(compact,image),'PDF image copy changed pixels')
        return original(pdf,compact)
    PdfFile.imageObject=own_crop
    try:
        yield
    finally:
        PdfFile.imageObject=original

def save_figure(fig,name,combined=None):
    t=time.monotonic();qa=check_text(fig,name)
    for ext in ['pdf','svg']:
        with compact_pdf_image_buffers():
            fig.savefig(OUT/f'exports/{name}.{ext}',dpi=600,facecolor='white')
    fig.savefig(OUT/f'qa/{name}_preview.png',dpi=180,facecolor='white')
    if combined is not None:
        with compact_pdf_image_buffers():combined.savefig(fig,dpi=600)
    plt.close(fig);emit(f'Exported {name} ({time.monotonic()-t:.1f}s)');return qa

def locator(fig,rect,windows,states,division):
    ax=fig.add_axes(rect);ax.set_facecolor('white')
    states.plot(ax=ax,color='white',edgecolor='#C9CECC',linewidth=.25,rasterized=True,aspect=None)
    for w in windows:
        u=w['urban_bounds_epsg5070_m'];x=(u[0]+u[2])/2;y=(u[1]+u[3])/2
        active=w['division']==division
        ax.scatter(x,y,s=15 if active else 3,c='#548E87' if active else '#CED7D3',zorder=3)
        # True locator points stay fixed. Avoid illegible overlapping labels in
        # the Northeast; the five numbered map titles identify this page's cases.
    ax.set_xlim(-2450000,2300000);ax.set_ylim(230000,3200000);ax.set_aspect('equal');ax.axis('off')

def render():
    receipt=json.loads((OUT/'qa/preparation.json').read_text(encoding='utf8'))
    require(receipt['status']=='PASS','Preparation not passed')
    for p,h in receipt['sources_sha256'].items():require(sha(p)==h,'Source changed: '+p)
    windows=receipt['windows'];data={}
    for w in windows:
        require(sha(w['prepared_file'])==w['prepared_sha256'],'Prepared geometry changed')
        data[w['city_number']]=read_city(w)
        if w['city_number']%5==0:emit(f"Read map layers {w['city_number']}/45")
    results=[]
    # 180 x 270 mm: complete 45-city comparison, supported by larger atlas pages.
    fig=plt.figure(figsize=(180/25.4,270/25.4),dpi=150,facecolor='white')
    fig.text(.035,.986,'Conditional cooling responses across 45 urban areas',fontsize=10.5,fontweight='bold',va='top')
    fig.text(.035,.968,'Geographically adjusted forest  |  24,493 mapped census tracts',fontsize=7,color='#58625F',va='top')
    left=.061;cw=.176;gap=.010;mh=cw*180/(270*ASPECT);pitch=.0955
    for i,w in enumerate(windows):
        row=i//5;col=i%5;top=.945-row*pitch
        if col==0:fig.text(.022,top-.012-mh/2,DIVISIONS[row],rotation=90,fontsize=6.3,
                          fontweight='bold',color='#526660',va='center',ha='center')
        x=left+col*(cw+gap);title_y=top
        fig.text(x,title_y,f"{w['city_number']:02d}  {w['short_name']}",fontsize=6,va='top')
        map_top=title_y-.012
        map_panel(fig,[x,map_top-mh,cw,mh],w,data[w['city_number']])
    common_legend(fig,y=.015)
    results.append(save_figure(fig,'Figure04_45city_overview'))
    states=gpd.read_file(OUT/'prepared/locator_states.gpkg')
    combined_path=OUT/'exports/Figure04_regional_atlas_9pages.pdf'
    with PdfPages(combined_path) as combined:
        for di,division in enumerate(DIVISIONS):
            fig=plt.figure(figsize=(180/25.4,220/25.4),dpi=150,facecolor='white')
            fig.text(.045,.978,f'{di+1:02d}  {division}',fontsize=13,fontweight='bold',va='top')
            fig.text(.045,.949,'Conditional cooling responses · fixed 2000 urban boundaries',fontsize=7.5,color='#58625F',va='top')
            city_list=windows[di*5:di*5+5]
            cw=.425;mh=cw*180/(220*ASPECT)
            for k,w in enumerate(city_list):
                col=k%2;row=k//2;x=.045+col*.485;top=.906-row*.258
                fig.text(x,top,f"{w['city_number']:02d}  {w['short_name']}",fontsize=9,fontweight='bold',va='top')
                map_top=top-.021
                map_panel(fig,[x,map_top-mh,cw,mh],w,data[w['city_number']],detail=True)
            fig.text(.53,.390,'Study locations',fontsize=9,fontweight='bold',va='top')
            locator(fig,[.53,.178,.425,.185],windows,states,division)
            fig.text(.53,.164,'Highlighted locations are shown on this page.',fontsize=6.5,color='#68716E')
            common_legend(fig,y=.023)
            name=f'Figure04_atlas_{di+1:02d}_{division.replace(" ","_")}'
            results.append(save_figure(fig,name,combined))
    dump(OUT/'qa/render_receipt.json',{'status':'RENDERED','raster_layer_dpi':600,'text_font':'Arial',
         'map_data_simplification':False,'map_layers':'600-dpi rasterized polygons, vector text and scales',
         'shared_breaks':BREAKS.tolist(),'shared_colors':COLORS,'per_figure_text_checks':results,
         'main_dimensions_mm':[180,270],'regional_dimensions_mm':[180,220],
         'maps_rendered':90,'unique_cities':45,'overview_and_atlas_use_same_windows':True,
         'background':'pure white; no land or water context fill; user requested',
         'context_layers_drawn':False})

def validate():
    r=json.loads((OUT/'qa/preparation.json').read_text(encoding='utf8'))
    render=json.loads((OUT/'qa/render_receipt.json').read_text(encoding='utf8'))
    require(len(r['windows'])==45 and sum(w['main_tracts'] for w in r['windows'])==24493,'Counts differ')
    require(all(abs(w['scale_geodesic_verified_m']-w['scale_distance_m'])<.01 for w in r['windows']),'Scale error')
    outputs=[]
    for p in sorted((OUT/'exports').glob('*.pdf')):
        doc=pymupdf.open(p)
        expected=9 if p.name.endswith('9pages.pdf') else 1
        require(len(doc)==expected,'PDF page count differs')
        for page in doc:
            require(len(page.get_text())>80,'PDF missing text')
            fonts=page.get_fonts()
            require(any('Arial' in str(f) for f in fonts),'PDF missing Arial')
        if p.name=='Figure04_45city_overview.pdf':
            doc[0].get_pixmap(matrix=pymupdf.Matrix(2,2)).save(OUT/'qa/overview_pdf_render.png')
        outputs.append({'file':str(p),'pages':len(doc),'bytes':p.stat().st_size,'sha256':sha(p)})
    svgs=list((OUT/'exports').glob('*.svg'))
    require(len(svgs)==10 and len(outputs)==11,'Expected 10 SVG + 11 PDF exports')
    for p in svgs:
        txt=p.read_text(encoding='utf8')
        require('<text' in txt and '<image' in txt,'SVG must retain live text and raster map layers')
        outputs.append({'file':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
    dump(OUT/'qa/validation.json',{'status':'PASS_DATA_EXPORT_AND_SCALE_CHECKS','maps':90,'unique_cities':45,
        'main_tracts':24493,'main_records':61135,'national_model_unchanged':True,
        'sample_geometry_repaired':'One display component only, explicitly authorized; see repair record',
        'sample_geometry_simplified':False,'response_values_clipped':False,
        'max_scale_error_m':max(abs(w['scale_geodesic_verified_m']-w['scale_distance_m']) for w in r['windows']),
        'outputs':outputs,'visual_review':'PENDING_SEPARATE_IMAGE_INSPECTION'})
    emit('PASS data, exports, Arial, city counts, and geodesic scales')

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf8');init()
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['download','prepare','render','validate','all'])
    args=ap.parse_args()
    for stage,fn in [('download',download),('prepare',prepare),('render',render),('validate',validate)]:
        if args.stage in [stage,'all']:fn()
