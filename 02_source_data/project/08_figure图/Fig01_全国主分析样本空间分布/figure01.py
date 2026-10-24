"""Figure 1. Exact sample-to-2000-urban geometry lineage, then publication map.

Run with sportpark-ee. Commands: download, prepare, render, validate, all.
Inputs are read-only. Per-state outputs permit verified resume of preparation.
"""
from nc_paths import project_root as _nc_project_root, external_path as _nc_external_path
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import platform
import sys
import time
import zipfile

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer, Geod
import requests
import shapely
from shapely.geometry import box, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union

ROOT = _nc_project_root()
OUT = Path(__file__).resolve().parent
MANIFEST = ROOT/'06_logs/geo_adjusted_national_orthogonal_20261009_run01/sample_manifest.csv'
SOURCE_AUDIT = ROOT/'06_logs/national_urban_source_check_20260914_run01/audit.json'
LINKS = SOURCE_AUDIT.parent/'state_urban_candidates.csv'
URBAN = ROOT/'04_pilot/boundaries_2000/urban_area/tl_2009_us_uac00.zip'
CRS_MAP = CRS.from_epsg(5070)
STATES = '01 04 05 06 08 09 10 11 12 13 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 44 45 46 47 48 49 50 51 53 54 55 56'.split()
DOWNLOADS = {
    'cb_2024_us_state_500k.zip': {'url':'https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_state_500k.zip','role':'US Census 2024 cartographic state boundaries; locator background only; not analysis geography'},
    'ne_50m_land.zip': {'url':'https://naturalearth.s3.amazonaws.com/50m_physical/ne_50m_land.zip','role':'Natural Earth 1:50m land; neighbouring-country context only'},
    'ne_10m_lakes.zip': {'url':'https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_lakes.zip','role':'Natural Earth 1:10m lakes; visual hydrography only, not the fixed study land mask'}
}

def emit(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def dump(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def init():
    require(Path(sys.executable).parent.name.lower()=='sportpark-ee','Use the requested sportpark-ee environment')
    for folder in ['raw_basemap','prepared','prepared/states','source_data','exports','qa']:
        (OUT/folder).mkdir(parents=True,exist_ok=True)

def download():
    session=requests.Session()
    session.trust_env=False  # No HTTP(S) environment proxy; VPN routing is OS-managed.
    receipt={}
    for name,meta in DOWNLOADS.items():
        p=OUT/'raw_basemap'/name
        side=p.with_suffix('.source.json')
        if p.exists():
            require(side.exists(),'Existing basemap without provenance: '+name)
            saved=json.loads(side.read_text(encoding='utf-8'))
            require(saved['url']==meta['url'] and saved['sha256']==sha(p),'Basemap provenance/hash differs')
            receipt[name]=saved
            emit('Verified existing basemap '+name)
            continue
        emit('Downloading '+meta['url'])
        with session.get(meta['url'],timeout=(20,120),stream=True) as response:
            response.raise_for_status()
            part=p.with_suffix('.part')
            with part.open('wb') as f:
                for chunk in response.iter_content(1024*1024):
                    f.write(chunk)
            with zipfile.ZipFile(part) as z:
                require(z.testzip() is None,'Corrupt basemap zip '+name)
            part.rename(p)
            saved={**meta,'resolved_url':response.url,'bytes':p.stat().st_size,'sha256':sha(p),
                   'retrieved_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'http_status':response.status_code,'proxy_environment_used':False}
            dump(side,saved);receipt[name]=saved
            emit(f'Downloaded {name}: {p.stat().st_size/1e6:.1f} MB')
    dump(OUT/'qa/download_receipt.json',receipt)

def read_sample():
    d=pd.read_csv(MANIFEST,dtype={'state':str,'tract_id':str},float_precision='round_trip')
    require(len(d)==104371 and d.tract_id.nunique()==42373,'Formal sample count differs')
    require(not d.duplicated(['state','tract_id','exposure_year','outcome_year']).any(),'Duplicate panel keys')
    require(sorted(d.state.unique())==STATES,'State coverage differs')
    require(d.tract_id.str.fullmatch(r'\d{11}').all(),'Tract ID width differs')
    require((d.tract_id.str[:2]==d.state).all(),'Tract/state mismatch')
    d.groupby(['state','tract_id']).size().rename('analysis_records').to_csv(OUT/'source_data/sample_record_counts.csv')
    return d

def polygon_parts(geom):
    if isinstance(geom,Polygon):return [geom]
    if isinstance(geom,(MultiPolygon,GeometryCollection)):
        return [p for g in geom.geoms for p in polygon_parts(g)]
    return []

def prepare():
    started=time.monotonic()
    d=read_sample()
    audit=json.loads(SOURCE_AUDIT.read_text(encoding='utf-8'))
    require(audit['status']=='PASS_SOURCE_AND_CANDIDATE_LINKS','Urban source gate differs')
    require(sha(URBAN)==audit['source_sha256_archive_bytes'],'Urban archive hash differs')
    original_crs=CRS.from_wkt(audit['area_crs_wkt'])
    require(original_crs.is_projected and all(x.unit_name=='metre' for x in original_crs.axis_info),'Original CRS invalid')
    links=pd.read_csv(LINKS,dtype=str)
    city_ids=set(links.loc[links.lsad.eq('75'),'urban_id'])
    urban_native=gpd.read_file('zip://'+URBAN.as_posix())
    require(urban_native.crs==CRS.from_epsg(4269),'Urban source CRS differs')
    urban=urban_native.loc[urban_native.UACE00.isin(city_ids)].sort_values('UACE00').to_crs(original_crs)
    require(len(urban)==448 and urban.geometry.is_valid.all(),'Urban IDs/geometries differ')
    require(urban.UACE00.is_unique,'Duplicate urban IDs')
    source_hashes={str(p):sha(p) for p in [MANIFEST,SOURCE_AUDIT,LINKS,URBAN]}
    reports=[]
    for k,state in enumerate(STATES,1):
        t0=time.monotonic()
        boundary=ROOT/f'04_national/boundaries_2000/tract_by_state/tl_2010_{state}_tract00.zip'
        reference=ROOT/f'06_logs/national_tract_urban_geometry_20260914_run01/tract_urban_geometry_{state}.csv'
        source_hashes[str(boundary)]=sha(boundary)
        source_hashes[str(reference)]=sha(reference)
        require(source_hashes[str(boundary)]==audit['boundary_sha256_archive_bytes'][state],f'{state}: archive changed')
        state_inputs={p:v for p,v in source_hashes.items() if p in map(str,[MANIFEST,SOURCE_AUDIT,LINKS,URBAN,boundary,reference])}
        cache=OUT/'prepared/states'/f'{state}.gpkg'
        receipt=cache.with_suffix('.json')
        if receipt.exists():
            saved=json.loads(receipt.read_text(encoding='utf-8'))
            require(saved['inputs']==state_inputs and saved['geometry_rule_version']=='urban_internal_largest_component_v1','State resume inputs/rule differ')
            require(cache.exists() and sha(cache)==saved['output_sha256'],'State cache changed')
            reports.append(saved)
            emit(f'{k:02}/49 state {state}: verified resume, {saved["tracts"]} tracts')
            continue
        require(not cache.exists(),'Incomplete state cache must be inspected before rerun: '+str(cache))
        native=gpd.read_file('zip://'+boundary.as_posix())
        require(native.crs==urban_native.crs,f'{state}: source CRS differs')
        require(native.CTIDFP00.is_unique,f'{state}: duplicated boundary IDs')
        ids=set(d.loc[d.state.eq(state),'tract_id'])
        require(ids<=set(native.CTIDFP00),f'{state}: unmatched formal sample IDs')
        tracts=native.loc[native.CTIDFP00.isin(ids)].sort_values('CTIDFP00').to_crs(original_crs)
        ref=pd.read_csv(reference,dtype={'tract_id':str,'state':str}).set_index('tract_id')
        require(len(tracts)==len(ids) and tracts.geometry.is_valid.all(),f'{state}: geometry validity/count')
        part_rows=[];point_rows=[];max_area_error=0.0
        city_union_cache={}
        for row in tracts.itertuples():
            tid=row.CTIDFP00;geom=row.geometry
            require(not geom.is_empty and geom.area>0,tid+': empty boundary')
            ix=urban.sindex.query(geom,predicate='intersects')
            local=urban.iloc[ix].sort_values('UACE00')
            city_key=tuple(local.UACE00.tolist())
            if city_key not in city_union_cache:
                city_union_cache[city_key]=unary_union(local.geometry.tolist())
            clip=geom.intersection(city_union_cache[city_key])
            require(clip.is_valid and not clip.is_empty and clip.area>0,tid+': invalid/empty intersection')
            ref_area=float(ref.loc[tid,'urban_intersection_geometry_m2'])
            error=abs(clip.area-ref_area);max_area_error=max(max_area_error,error)
            require(error<=max(.01,ref_area*1e-10),tid+f': area differs from original evidence: {error}')
            parts=sorted(polygon_parts(clip),key=lambda g:(-g.area,g.bounds,g.wkb_hex))
            require(parts and abs(sum(p.area for p in parts)-clip.area)<=max(.01,clip.area*1e-10),tid+': component area closure')
            pt=parts[0].representative_point()
            require(parts[0].contains(pt) and geom.covers(pt),tid+': representative point outside geometry')
            for i,part in enumerate(parts):
                part_rows.append({'state':state,'tract_id':tid,'part_rank':i+1,'selected_for_point':i==0,'part_area_original_m2':part.area,'geometry':part})
            point_rows.append({'state':state,'tract_id':tid,'component_count':len(parts),'urban_intersection_area_original_m2':clip.area,'largest_part_area_original_m2':parts[0].area,'other_parts_area_original_m2':sum(p.area for p in parts[1:]),'area_check_error_m2':error,'geometry':pt})
        points=gpd.GeoDataFrame(point_rows,crs=original_crs).to_crs(CRS_MAP)
        parts_gdf=gpd.GeoDataFrame(part_rows,crs=original_crs).to_crs(CRS_MAP)
        points.to_file(cache,layer='display_points',driver='GPKG',index=False)
        parts_gdf.to_file(cache,layer='urban_intersection_parts',driver='GPKG',index=False)
        saved={'state':state,'tracts':len(points),'components':len(parts_gdf),'max_area_error_original_crs_m2':max_area_error,'inputs':state_inputs,'geometry_rule_version':'urban_internal_largest_component_v1','native_tract_crs':native.crs.to_wkt(),'native_urban_crs':urban_native.crs.to_wkt(),'original_area_crs':original_crs.to_wkt(),'display_crs':CRS_MAP.to_wkt(),'output_sha256':sha(cache),'elapsed_seconds':time.monotonic()-t0,'status':'PASS'}
        dump(receipt,saved);reports.append(saved)
        emit(f'{k:02}/49 state {state}: {len(points):,} tracts; {len(parts_gdf):,} parts; {saved["elapsed_seconds"]:.1f}s')
    points=gpd.GeoDataFrame(pd.concat([gpd.read_file(OUT/'prepared/states'/f'{s}.gpkg',layer='display_points') for s in STATES],ignore_index=True),crs=CRS_MAP).sort_values(['state','tract_id'])
    require(len(points)==42373 and points.tract_id.is_unique,'Final display point count differs')
    expected=set(zip(d.state,d.tract_id));actual=set(zip(points.state,points.tract_id))
    missing=sorted(expected-actual);extra=sorted(actual-expected)
    pd.DataFrame(missing+extra,columns=['state','tract_id']).to_csv(OUT/'qa/unmatched_sample_ids.csv',index=False)
    require(not missing and not extra,'Final display ID mismatch')
    counts=d.groupby('tract_id').size()
    points['analysis_records']=points.tract_id.map(counts)
    points.to_file(OUT/'prepared/display_points.gpkg',layer='display_points',driver='GPKG',index=False)
    csv=points.drop(columns='geometry').copy()
    csv['display_x_epsg5070_m']=points.geometry.x
    csv['display_y_epsg5070_m']=points.geometry.y
    lonlat=points.to_crs(4326)
    csv['longitude']=lonlat.geometry.x;csv['latitude']=lonlat.geometry.y
    require(csv.longitude.between(-125,-66).all() and csv.latitude.between(24,50).all(),'Unexpected continental extent')
    csv.to_csv(OUT/'source_data/figure01_display_points.csv',index=False)
    csv.groupby('state').agg(unique_tracts=('tract_id','size'),analysis_records=('analysis_records','sum')).to_csv(OUT/'source_data/figure01_state_counts.csv')
    for p,h in source_hashes.items():require(sha(p)==h,'Source changed during preparation: '+p)
    dump(OUT/'qa/geometry_validation.json',{'status':'PASS','sample_records':len(d),'unique_tracts':len(points),'state_and_dc_count':len(STATES),'duplicate_display_ids':0,'unmatched':0,'extra':0,'source_crs':urban_native.crs.to_wkt(),'original_area_crs':original_crs.to_wkt(),'display_crs':CRS_MAP.to_wkt(),'max_area_difference_m2':max(r['max_area_error_original_crs_m2'] for r in reports),'total_components':sum(r['components'] for r in reports),'multipart_tracts':int((points.component_count>1).sum()),'display_extent':points.total_bounds.tolist(),'longitude_latitude_extent':lonlat.total_bounds.tolist(),'source_hashes':source_hashes,'display_point_csv_sha256':sha(OUT/'source_data/figure01_display_points.csv'),'display_geopackage_sha256':sha(OUT/'prepared/display_points.gpkg'),'elapsed_seconds':time.monotonic()-started,'formal_model_coordinates_modified':False,'sampling_or_jitter':False})
    emit(f'Geometry PASS: {len(points):,} unique points; elapsed {(time.monotonic()-started)/60:.1f} min')

def basemaps():
    for name in DOWNLOADS:
        p=OUT/'raw_basemap'/name;meta=json.loads(p.with_suffix('.source.json').read_text(encoding='utf-8'))
        require(sha(p)==meta['sha256'],'Basemap changed: '+name)
    states=gpd.read_file('zip://'+(OUT/'raw_basemap/cb_2024_us_state_500k.zip').as_posix())
    states=states.loc[states.STATEFP.isin(STATES)].sort_values('STATEFP')
    require(len(states)==49 and states.STATEFP.is_unique and states.geometry.is_valid.all(),'State basemap validity/count')
    land=gpd.read_file('zip://'+(OUT/'raw_basemap/ne_50m_land.zip').as_posix(),bbox=(-130,20,-60,55))
    lakes=gpd.read_file('zip://'+(OUT/'raw_basemap/ne_10m_lakes.zip').as_posix(),bbox=(-130,20,-60,55))
    require(land.crs is not None and lakes.crs is not None,'Basemap CRS missing')
    return states.to_crs(CRS_MAP),land.to_crs(CRS_MAP),lakes.to_crs(CRS_MAP)

def draw_map(ax,states,land,lakes,points,extent,point_size=.58):
    ocean='#EDF4F5';land_color='#FAF7F0';context='#EEEBE5';line='#B2BAB8';sample='#88ADA6'
    ax.set_facecolor(ocean)
    land.plot(ax=ax,facecolor=context,edgecolor='none',zorder=0)
    ax.collections[-1].set_gid('neighbouring_land_context')
    states.plot(ax=ax,facecolor=land_color,edgecolor='none',zorder=1)
    ax.collections[-1].set_gid('census_state_land_fill')
    national_outline=unary_union(states.geometry.tolist()).boundary
    internal_boundaries=unary_union(states.geometry.boundary.tolist()).difference(national_outline)
    gpd.GeoSeries([internal_boundaries],crs=CRS_MAP).plot(ax=ax,color=line,linewidth=.30,zorder=2)
    ax.collections[-1].set_gid('internal_state_boundaries')
    gpd.GeoSeries([national_outline],crs=CRS_MAP).plot(ax=ax,color='#AAB6B3',linewidth=.35,zorder=2)
    ax.collections[-1].set_gid('national_outline_including_coastline')
    lakes.plot(ax=ax,facecolor=ocean,edgecolor='#DAE5E4',linewidth=.15,zorder=2)
    ax.collections[-1].set_gid('lake_water')
    artist=ax.scatter(points.geometry.x,points.geometry.y,s=point_size,c=sample,alpha=.75,edgecolors='none',linewidths=0,zorder=4,rasterized=False)
    ax.set_xlim(extent[0],extent[2]);ax.set_ylim(extent[1],extent[3]);ax.set_aspect('equal',adjustable='box')
    ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_visible(False)
    return artist

def scale_bar(ax,x,y,meters):
    # Horizontal map segment whose endpoint geodesic distance equals the label.
    from scipy.optimize import brentq
    to_geo=Transformer.from_crs(CRS_MAP,4326,always_xy=True)
    lon,lat=to_geo.transform(x,y);geod=Geod(ellps='GRS80')
    def ground_distance(dx):
        lo,la=to_geo.transform(x+dx,y)
        return geod.inv(lon,lat,lo,la)[2]
    dx=brentq(lambda dx:ground_distance(dx)-meters,0,meters*2,xtol=1e-6)
    ex,ey=x+dx,y
    lo,la=to_geo.transform(ex,ey)
    require(abs(ground_distance(dx)-meters)<.001,'Scale distance verification failed')
    require(ax.get_xlim()[0]<x<ex<ax.get_xlim()[1] and ax.get_ylim()[0]+100000<y<ax.get_ylim()[1]-100000,'Scale bar would clip')
    ax.plot([x,ex],[y,ey],color='#727B79',lw=1.1,zorder=8)
    for xx,yy in [(x,y),(ex,ey)]:ax.plot([xx,xx],[yy-14000,yy+14000],color='#727B79',lw=.65,zorder=8)
    ax.annotate(f'{int(meters/1000)} km',((x+ex)/2,(y+ey)/2),xytext=(0,-7),textcoords='offset points',ha='center',va='top',fontsize=7,color='#53615F',zorder=9)
    return {'origin_lonlat':[lon,lat],'distance_m':meters,'end_lonlat':[lo,la],'projected_endpoints':[[x,y],[ex,ey]]}

def render():
    t0=time.monotonic()
    g=json.loads((OUT/'qa/geometry_validation.json').read_text(encoding='utf-8'));require(g['status']=='PASS','Geometry not ready')
    require(sha(OUT/'prepared/display_points.gpkg')==g['display_geopackage_sha256'],'Display points changed')
    points=gpd.read_file(OUT/'prepared/display_points.gpkg',layer='display_points')
    states,land,lakes=basemaps()
    font=font_manager.findfont('Arial',fallback_to_default=False)
    plt.rcParams.update({'font.family':'Arial','font.size':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':600,'axes.unicode_minus':True})
    fig=plt.figure(figsize=(180/25.4,135/25.4),dpi=150,facecolor='white')
    ax=fig.add_axes([.025,.11,.95,.78])
    extent=(-2450000,190000,2350000,3350000)
    bounds=points.total_bounds
    require(bounds[0]>=extent[0] and bounds[1]>=extent[1] and bounds[2]<=extent[2] and bounds[3]<=extent[3],'Map frame clips sampled points')
    dots=draw_map(ax,states,land,lakes,points,extent)
    dots.set_gid('temperature_analysis_tracts')
    fig.text(.045,.975,'Contiguous United States',fontsize=11,fontweight='bold',color='#4D5A58',va='top')
    fig.text(.045,.933,'42,373 census tracts  |  104,371 tract-year records',fontsize=8,color='#697572',va='top')
    legend=fig.legend(handles=[Line2D([],[],marker='o',linestyle='none',markersize=4.0,markerfacecolor='#88ADA6',markeredgecolor='none',label='Temperature-analysis tracts')],loc='lower left',bbox_to_anchor=(.033,.055),frameon=False,fontsize=8,handletextpad=.5)
    scale=scale_bar(ax,-2100000,430000,500000)
    fig.text(.955,.040,'Albers equal-area projection',ha='right',fontsize=6.5,color='#7B8582')
    # Symbol collection retains every row, including coincident coordinates.
    require(len(dots.get_offsets())==42373,'Plotted marker count differs')
    for suffix in ['pdf','svg','png']:
        fig.savefig(OUT/f'exports/Figure01_national_sample_distribution.{suffix}',dpi=600,facecolor='white',metadata={'Creator':'Matplotlib; sportpark-ee'} if suffix in ['pdf','svg'] else None)
    # Final-size preview is intentionally separate from the 600 dpi delivery.
    fig.savefig(OUT/'qa/Figure01_preview_180mm.png',dpi=180,facecolor='white')
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    text_boxes=[]
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            b=text.get_window_extent(renderer)
            text_boxes.append({'text':text.get_text(),'bounds_pixels':list(b.bounds)})
    plt.close(fig)
    dump(OUT/'qa/render_receipt.json',{'status':'RENDERED_PENDING_VISUAL_REVIEW','plotted_points':42373,'point_size_pt2':.58,'sample_symbol_color':'#88ADA6','extent_epsg5070_m':list(extent),'projection':CRS_MAP.to_wkt(),'font_path':font,'font_family':'Arial','page_mm':[180,135],'export_dpi':600,'vector_geometry_and_points':True,'rasterization':False,'insets':0,'scale_bar':scale,'text_bounding_boxes':text_boxes,'environment':{'executable':sys.executable,'python':platform.python_version(),'geopandas':gpd.__version__,'shapely':shapely.__version__,'matplotlib':matplotlib.__version__},'script_sha256':sha(__file__),'elapsed_seconds':time.monotonic()-t0,'exports':{s:sha(OUT/f'exports/Figure01_national_sample_distribution.{s}') for s in ['pdf','svg','png']}})
    emit(f'Rendered Figure 1 in {time.monotonic()-t0:.1f}s')

def validate():
    import pymupdf
    import xml.etree.ElementTree as ET
    g=json.loads((OUT/'qa/geometry_validation.json').read_text(encoding='utf-8'))
    r=json.loads((OUT/'qa/render_receipt.json').read_text(encoding='utf-8'))
    require(g['unique_tracts']==r['plotted_points']==42373,'Count not reconciled')
    for p,h in g['source_hashes'].items():require(sha(p)==h,'Protected source changed: '+p)
    doc=pymupdf.open(OUT/'exports/Figure01_national_sample_distribution.pdf')
    require(len(doc)==1,'PDF pages differ')
    page=doc[0];size=[page.rect.width*25.4/72,page.rect.height*25.4/72]
    require(np.allclose(size,[180,135],atol=.02),'Page dimensions differ')
    fonts=page.get_fonts(full=True);require(fonts and all('Arial' in f[3] for f in fonts),'Non-Arial PDF font')
    root=ET.parse(OUT/'exports/Figure01_national_sample_distribution.svg').getroot()
    ns={'s':'http://www.w3.org/2000/svg'}
    texts=[''.join(e.itertext()) for e in root.findall('.//s:text',ns)]
    require(any('42,373' in s for s in texts),'SVG editable text absent')
    require(not any(any('\u4e00'<=c<='\u9fff' for c in s) for s in texts),'Non-English figure text')
    point_group=root.find('.//s:g[@id="temperature_analysis_tracts"]',ns)
    require(point_group is not None,'SVG sample layer is missing')
    svg_points=len(point_group.findall('.//s:use',ns))
    require(svg_points==42373,'SVG sample marker count differs')
    text_lines=[]
    for block in page.get_text('dict')['blocks']:
        for line in block.get('lines',[]):
            text=''.join(span['text'] for span in line['spans']).strip()
            if text:text_lines.append((text,pymupdf.Rect(line['bbox'])))
    text_overlaps=[]
    for i,(text,rect) in enumerate(text_lines):
        require(page.rect.contains(rect),'PDF text extends beyond page: '+text)
        for text2,rect2 in text_lines[i+1:]:
            overlap=rect & rect2
            if overlap.width>.1 and overlap.height>.1:text_overlaps.append([text,text2])
    require(not text_overlaps,'PDF text bounding boxes overlap')
    geometry_checks=[]
    for state in STATES:
        cache=OUT/'prepared/states'/f'{state}.gpkg'
        pts=gpd.read_file(cache,layer='display_points').set_index('tract_id')
        parts=gpd.read_file(cache,layer='urban_intersection_parts')
        selected=parts.loc[parts.selected_for_point.eq(True)].set_index('tract_id').reindex(pts.index)
        require(pts.crs==selected.crs==CRS_MAP,'Export geometry CRS differs')
        require(selected.geometry.is_valid.all() and pts.geometry.is_valid.all(),'Projected geometry validity failure')
        require(shapely.contains(selected.geometry.array,pts.geometry.array).all(),'Projected internal-point containment failure')
        geometry_checks.append({'state':state,'points':len(pts),'inside_selected_component':True})
    doc[0].get_pixmap(matrix=pymupdf.Matrix(2,2),alpha=False).save(str(OUT/'qa/Figure01_pdf_render.png'))
    dump(OUT/'qa/export_validation.json',{'status':'PASS_EXPORT_AND_SOURCE_INTEGRITY','pdf_pages':1,'pdf_dimensions_mm':size,'pdf_fonts':[f[3] for f in fonts],'svg_editable_text_items':len(texts),'svg_sample_markers':svg_points,'text_overlap_pairs':text_overlaps,'all_figure_text_english':True,'vector_points':True,'dpi_setting':600,'protected_sources_unchanged':len(g['source_hashes']),'geometries_match_original_area_evidence':True,'projected_point_containment':geometry_checks,'human_visual_review':'REQUIRED_AFTER_RENDER','formal_model_runs':0})
    emit('Export and protected source verification PASS')

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('step',choices=['download','prepare','render','validate','all']);args=parser.parse_args()
    init()
    operations={'download':download,'prepare':prepare,'render':render,'validate':validate}
    try:
        for name,fn in operations.items():
            if args.step in [name,'all']:fn()
    except Exception as error:
        import traceback
        dump(OUT/'qa/last_failure.json',{'step':args.step,'error_type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        raise
