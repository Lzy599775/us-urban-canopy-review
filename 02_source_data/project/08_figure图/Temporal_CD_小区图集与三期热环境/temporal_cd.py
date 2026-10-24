"""Read-only scientific inputs; descriptive temporal figures and gap accounting."""
from nc_paths import project_root as _nc_project_root, external_path as _nc_external_path
from pathlib import Path
import argparse, hashlib, json, sys, time, importlib.util
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import font_manager
import pymupdf
from pyproj import Transformer, Geod

R=_nc_project_root(); O=Path(__file__).resolve().parent
AB=R/'08_figure图/Temporal_AB_全国与45城年度轨迹'; F4=R/'08_figure图/Fig04_全国响应与估计对象比较/revision03_atlas45'
C=O/'TemporalC_45城小区时序图集'; D=O/'TemporalD_三期热环境'; E=O/'AB_review_and_gap_decomposition'
BASE=R/'06_logs/canopy_cooling_equity_20261009_run01'
SRC={'annual':BASE/'annual_canopy_equity_row_manifest.csv.gz','groups':BASE/'baseline_equity_group_manifest.csv','sample':R/'06_logs/geo_adjusted_national_orthogonal_20261009_run01/sample_manifest.csv','weather':R/'06_logs/primary_weather_no_vpd_20261008_run02/annual_primary_weather_standardized.csv','windows':F4/'source_data/city_windows_epsg5070.json','roster':F4/'source_data/city_index.csv','tracts':F4/'source_data/figure04_tract_values.csv','points':R/'08_figure图/Fig01_全国主分析样本空间分布/source_data/figure01_display_points.csv','states':F4/'prepared/locator_states.gpkg','decomposition':AB/'TemporalB2_45城社会差距/source_data/endpoint_gap_decomposition.csv','group_annual':AB/'TemporalB2_45城社会差距/source_data/city45_quartile_annual.csv'}
VARS=['income','poverty','age65plus'];LABELS=['Income','Poverty','Age 65+']; TY=[2014,2019,2024]; INK='#2D3940'
COL='urban_canopy_percent_fixed_land'; TEMP='weather_adjusted_lst_reference_mean'
SEQ=LinearSegmentedColormap.from_list('canopy',['#F0F3E4','#D3E1C0','#9BC8AC','#62AA9B','#367F80'])
DIV=LinearSegmentedColormap.from_list('canopy_change',[(0,'#806296'),(.25,'#AB91BE'),(.43,'#C5B0D0'),(.485,'#DDD0E3'),(.5,'#F0F0EC'),(.515,'#CFDDD7'),(.57,'#9EC3B6'),(.75,'#72A795'),(1,'#3F8778')])
HEAT=LinearSegmentedColormap.from_list('heat',['#689EB6','#B6CCD3','#F4EBDF','#E6B29C','#BE796B'])
HDIV=LinearSegmentedColormap.from_list('heat_change',['#658FAE','#BDCEDA','#FAF9F5','#E7C2B3','#B97468'])
GROUP=['#638DA5','#96BAB1','#CAA985','#A681A8']

def require(b,msg):
 if not b:raise ValueError(msg)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def rd(p):return pd.read_csv(p,dtype={'state':str,'tract_id':str,'single_city_id':str},float_precision='round_trip')
def ids(d):
 for k,n in [('state',2),('tract_id',11),('single_city_id',5)]:
  if k in d:d[k]=pd.to_numeric(d[k],errors='raise').astype('Int64').astype('string').str.zfill(n)
 return d
def save(d,p):d.to_csv(p,index=False,encoding='utf-8-sig')

def review_ab():
 report=json.loads((AB/'qa/preparation.json').read_text(encoding='utf8'))
 for p,h in report['source_hashes'].items():require(sha(p)==h,'AB input changed: '+p)
 v=json.loads((AB/'qa/export_validation.json').read_text(encoding='utf8')); require(v['status']=='PASS','AB export failure')
 for f in v['files']:require(sha(f['file'])==f['sha256'],'AB output changed')
 require(json.loads((AB/'qa/independent_data_validation.json').read_text(encoding='utf8'))['status']=='PASS','AB numeric failure')
 dump(E/'qa/review.json',{'status':'PASS','files_checked':14,'source_hashes_unchanged':True,'AB_rewritten':False,'interpretation':'City illustrations are purposive; gaps and colours are signed, not universal equity scores.','visual_assessment':'Overview suitable for pattern reading; single-dimension details needed for numeric reading.','journal_acceptance':'NOT DETERMINED'})

def decompose():
 d=ids(rd(SRC['decomposition']));g=ids(rd(SRC['group_annual']));rows=[]
 for r in d.itertuples():
  p=g[(g.single_city_id==r.single_city_id)&(g.variable==r.variable)].pivot(index='year',columns='quartile',values='mean')
  lo=float(p.loc[2025,1]-p.loc[2000,1]);hi=float(p.loc[2025,4]-p.loc[2000,4]);gap0=float(p.loc[2000,4]-p.loc[2000,1]);gap1=float(p.loc[2025,4]-p.loc[2025,1])
  require(np.allclose([lo,hi,hi-lo],[r.low_group_change_pp,r.high_group_change_pp,r.gap_change_pp],rtol=0,atol=1e-10),'Decomposition mismatch')
  a,b=(hi,lo) if r.variable=='poverty' else (lo,hi)
  lowname,highname=('高贫困组','低贫困组') if r.variable=='poverty' else (('低收入组','高收入组') if r.variable=='income' else ('低老年比例组','高老年比例组'))
  if a>0 and b>0:pattern='两组均增绿；'+(lowname if a>b else highname if b>a else '两组')+'增量较大'
  elif a<0 and b<0:pattern='两组均失树；'+(lowname if a<b else highname if b<a else '两组')+'损失较大'
  elif a>0 and b<0:pattern=lowname+'增绿、'+highname+'失树'
  elif a<0 and b>0:pattern=lowname+'失树、'+highname+'增绿'
  else:pattern='至少一组净变化为零'
  absolute=abs(gap1)-abs(gap0); orientation=-1 if r.variable=='poverty' else 1
  rows.append({**r._asdict(),'low_group_2000':p.loc[2000,1],'low_group_2025':p.loc[2025,1],'high_group_2000':p.loc[2000,4],'high_group_2025':p.loc[2025,4],'gap_2000_pp':gap0,'gap_2025_pp':gap1,'signed_gap_change_pp':hi-lo,'absolute_gap_change_pp':absolute,'absolute_gap_direction':'扩大' if absolute>0 else '缩小' if absolute<0 else '不变','gap_sign_reversal':gap0*gap1<0,'oriented_gap_change_pp':orientation*(hi-lo),'disadvantaged_or_low_age_change_pp':a,'advantaged_or_high_age_change_pp':b,'pattern_cn':pattern})
 out=pd.DataFrame(rows).sort_values(['variable','city_number']);save(out,E/'source_data/city45_gap_pathways.csv')
 counts=out.groupby(['variable','pattern_cn'],sort=False).size().rename('cities').reset_index();save(counts,E/'source_data/pathway_counts.csv')
 lines=['# 45城高低组增减与差距变化','', '所有结果使用固定24,493个展示小区和2000年分组，描述2000—2025端点变化。没有重新排序城市、检验显著性或归因于具体政策。收入以低收入组为经济弱势组，贫困以高贫困组为经济弱势组；年龄只称高低老年比例组，不预设哪组的树冠权利更高。','', '表内原始差距始终为最高组减最低组。另列绝对差距扩大或缩小及符号翻转，避免把越过零点后的变化误写成持续收敛。两个组的净变化共同解释差距，不把它误称唯一因果机制。','']
 for v,label in zip(VARS,['收入','贫困','老年人口比例']):
  sub=out[out.variable.eq(v)];lines+=['## '+label,'','|编号 城市|低组变化|高组变化|原差距→末差距|变化构成|绝对差距|','|---|---:|---:|---|---|---|']
  for r in sub.itertuples():lines.append(f'|{r.city_number:02d} {r.short_name}|{r.low_group_change_pp:.2f}|{r.high_group_change_pp:.2f}|{r.gap_2000_pp:.2f}→{r.gap_2025_pp:.2f}|{r.pattern_cn}|{r.absolute_gap_direction}'+('；组间顺序翻转' if r.gap_sign_reversal else '')+'|')
  lines+=['','所有数值单位为覆盖百分点。低组、高组指对应社会变量的最低、最高四分位。','']
 (E/'城市差距变化来源_完整说明.md').write_text('\n'.join(lines),encoding='utf8')
 return out

def prepare():
 review_ab();dec=decompose();h={str(p):sha(p) for p in SRC.values()}
 roster=ids(rd(SRC['roster'])).sort_values('city_number');f=ids(rd(SRC['tracts']));a=ids(rd(SRC['annual']));a=a[a.balanced_26years].copy()
 require(len(a)==1207024 and a.tract_id.nunique()==46424,'Resource panel')
 require(len(f)==24493 and f.tract_id.is_unique,'Display manifest')
 q=a.merge(f[['state','tract_id','single_city_id']],on=['state','tract_id'],validate='many_to_one',suffixes=('','_f'))
 require(len(q)==24493*26 and q.single_city_id.eq(q.single_city_id_f).all(),'C pairing')
 cs=q[q.year.isin([2000,2005,2010,2015,2025])][['state','tract_id','single_city_id','year',COL,'change_from_2000_pp']]
 require(cs[COL].between(0,100).all(),'Canopy range');save(cs,C/'source_data/tract_year_values.csv')
 windows=json.loads(SRC['windows'].read_text(encoding='utf8'));checks=[]
 to_ll=Transformer.from_crs(5070,4326,always_xy=True);geod=Geod(ellps='WGS84')
 for w in windows:
  p=Path(w['prepared_file']);require(sha(p)==w['prepared_sha256'],'F4 geometry hash changed');h[str(p)]=sha(p)
  z=gpd.read_file(p,layer='sample_parts');z=ids(z);require(z.crs.to_epsg()==5070 and z.is_valid.all() and (~z.is_empty).all() and z.area.gt(0).all(),'Invalid inherited geometry')
  expect=set(f.loc[f.single_city_id.eq(w['city_id']),'tract_id']);require(set(z.tract_id)==expect and len(expect)==w['main_tracts'],'City geometry IDs')
  bounds=w['bounds_epsg5070_m'];b=z.total_bounds;require(b[0]>=bounds[0] and b[1]>=bounds[1] and b[2]<=bounds[2] and b[3]<=bounds[3],'City cropped')
  ends=w['scale_reference_endpoints_epsg5070'];ll=[to_ll.transform(*v) for v in ends];distance=geod.inv(*ll[0],*ll[1])[2];require(abs(distance-w['scale_distance_m'])<.01,'Scale geodesic')
  checks.append({'city_number':w['city_number'],'single_city_id':w['city_id'],'tracts':len(expect),'parts':len(z),'invalid':0,'unmatched':0,'scale_error_m':distance-w['scale_distance_m'],'source':str(p)})
 save(pd.DataFrame(checks),C/'qa/city_geometry_checks.csv');save(roster,C/'source_data/city_roster.csv')
 m=ids(rd(SRC['sample']));require(len(m)==104371 and m.tract_id.nunique()==42373,'Main sample')
 require(not m.duplicated(['state','tract_id','outcome_year']).any() and set(m.outcome_year)==set(TY),'Main duplicates/years')
 weather=ids(rd(SRC['weather'])).rename(columns={'year':'outcome_year'});check=m.merge(weather[['state','tract_id','outcome_year',TEMP]],on=['state','tract_id','outcome_year'],validate='one_to_one',suffixes=('','_source'))
 require(len(check)==len(m) and np.allclose(check[TEMP],check[TEMP+'_source'],rtol=0,atol=1e-10),'Weather lineage')
 common=m.groupby('tract_id').outcome_year.nunique();common=common[common.eq(3)].index
 t=m[m.tract_id.isin(common)].copy();require(t.tract_id.nunique()==26477 and len(t)==79431,'Common panel')
 gr=ids(rd(SRC['groups']));t=t.merge(gr[['state','tract_id','single_city_id']+[v+'_city_quartile' for v in VARS]],on=['state','tract_id'],validate='many_to_one',indicator=True)
 require(t._merge.eq('both').all(),'Missing baseline');t=t.drop(columns='_merge')
 pts=ids(rd(SRC['points']));t=t.merge(pts[['state','tract_id','display_x_epsg5070_m','display_y_epsg5070_m']],on=['state','tract_id'],validate='many_to_one',indicator=True)
 require(t._merge.eq('both').all(),'Missing display point');t=t.drop(columns='_merge');require(t[TEMP].notna().all(),'Missing temperature')
 save(t,D/'source_data/common_tract_temperature_panel.csv')
 tw=t.pivot(index='tract_id',columns='outcome_year',values=TEMP);base=t[t.outcome_year.eq(2014)].set_index('tract_id');mp=base[['state','single_city_id','display_x_epsg5070_m','display_y_epsg5070_m']].join(tw.rename(columns=lambda y:'temp_'+str(y)))
 mp['change_2019_2014']=tw[2019]-tw[2014];mp['change_2024_2019']=tw[2024]-tw[2019];save(mp.reset_index(),D/'source_data/common_tract_map_values.csv')
 group_rows=[];cityrows=[];qual=[];candidates=sorted(t.single_city_id.dropna().unique())
 for v in VARS:
  k=v+'_city_quartile';u=t[t[k].notna()]
  p=u.groupby(['outcome_year',k])[TEMP].agg(mean='mean',n='size').reset_index().rename(columns={k:'quartile'});p['variable']=v;group_rows.append(p)
  cg=u.groupby(['single_city_id','outcome_year',k])[TEMP].agg(mean='mean',n='size').reset_index().rename(columns={k:'quartile'});cg['variable']=v;cityrows.append(cg)
  ct=cg[cg.outcome_year.eq(2014)].groupby('single_city_id').quartile.nunique()
  for cityid in candidates:qual.append({'single_city_id':cityid,'variable':v,'quartiles_present':int(ct.get(cityid,0)),'all_four_groups':ct.get(cityid,0)==4})
 qual=pd.DataFrame(qual);eligible=qual.groupby('single_city_id').all_four_groups.all();good=eligible[eligible].index
 require(len(good)>0,'No city supports all dimensions')
 save(qual,D/'source_data/city_group_support.csv');save(pd.DataFrame({'single_city_id':good}),D/'source_data/fixed_city_set_for_gap.csv')
 groups=pd.concat(group_rows);cg=pd.concat(cityrows);save(groups,D/'source_data/national_tract_weighted_groups.csv');save(cg,D/'source_data/city_group_temperature_means.csv')
 cg=cg[cg.single_city_id.isin(good)];wide=cg.pivot(index=['single_city_id','variable','outcome_year'],columns='quartile',values='mean');gap=(wide[4]-wide[1]).rename('gap_c').reset_index();require(len(gap)==len(good)*9,'City gap panel')
 save(gap,D/'source_data/fixed_city_signed_gaps.csv');summary=gap.groupby(['variable','outcome_year']).gap_c.agg(mean_gap_c='mean',n_cities='size').reset_index();save(summary,D/'source_data/equal_city_gap_summary.csv')
 national=t.groupby('outcome_year')[TEMP].agg(mean='mean',median='median',q25=lambda x:x.quantile(.25),q75=lambda x:x.quantile(.75),minimum='min',maximum='max',n='size').reset_index();save(national,D/'source_data/national_temperature_distribution.csv')
 hist=[];binlo=np.floor(t[TEMP].min());binhi=np.ceil(t[TEMP].max());edges=np.arange(binlo,binhi+1,1)
 for year in TY:
  count,_=np.histogram(t.loc[t.outcome_year.eq(year),TEMP],bins=edges)
  for left,right,n in zip(edges[:-1],edges[1:],count):hist.append({'year':year,'left':left,'right':right,'count':n,'share':n/26477})
 save(pd.DataFrame(hist),D/'source_data/temperature_histogram.csv')
 audit={'status':'PASS','source_hashes':h,'main_records':len(m),'main_tracts':42373,'common_temperature_tracts':26477,'common_temperature_rows':len(t),'common_unique_city_tracts':int(base.single_city_id.notna().sum()),'candidate_cities':len(candidates),'fixed_all_groups_cities':len(good),'display_tracts':24493,'city_geometry_parts':sum(z['parts'] for z in checks),'canopy_range':[float(cs[COL].min()),float(cs[COL].max())],'canopy_change_limit':float(np.ceil(cs.loc[cs.year.isin([2005,2015,2025]),'change_from_2000_pp'].abs().max()/5)*5),'temp_limits':[float(binlo),float(binhi)],'temp_change_limit':float(np.ceil(mp[['change_2019_2014','change_2024_2019']].abs().to_numpy().max())),'missing_group_tracts':{v:int(base[v+'_city_quartile'].isna().sum()) for v in VARS},'formal_model':'NOT RUN','inference':'descriptive; no confidence intervals','decomposition_rows':len(dec)}
 dump(O/'qa/preparation.json',audit);print(json.dumps({k:v for k,v in audit.items() if k!='source_hashes'},ensure_ascii=False),flush=True)

def figure(w,h):
 f=plt.figure(figsize=(w/25.4,h/25.4));f.mm=(w,h);return f
def axmm(f,x,y,w,h):return f.add_axes([x/f.mm[0],y/f.mm[1],w/f.mm[0],h/f.mm[1]])
def txt(f,x,y,text,size=7,ha='left',weight='normal',**kw):return f.text(x/f.mm[0],y/f.mm[1],text,fontsize=size,ha=ha,va='center',fontweight=weight,color=INK,**kw)
def bar(f,x,y,w,cmap,norm,label,ticks):
 a=axmm(f,x,y,w,2.6);cb=f.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=a,orientation='horizontal',ticks=ticks);cb.outline.set_visible(False);cb.solids.set_rasterized(False);cb.solids.set_edgecolor('face');cb.ax.tick_params(length=0,labelsize=6.5,pad=3);txt(f,x+w/2,y+5.5,label,7,ha='center')
def export(f,out,name):
 f.canvas.draw();renderer=f.canvas.get_renderer();bad=[]
 for t in f.findobj(matplotlib.text.Text):
  if t.get_visible() and t.get_text():
   b=t.get_window_extent(renderer)
   if b.x0<-.5 or b.y0<-.5 or b.x1>f.bbox.x1+.5 or b.y1>f.bbox.y1+.5:bad.append(t.get_text())
 require(not bad,'Clipped labels: '+str(bad))
 for ext in ['pdf','svg']:f.savefig(out/'exports'/f'{name}.{ext}',dpi=600,facecolor='white')
 plt.close(f)
 with pymupdf.open(out/'exports'/f'{name}.pdf') as doc:doc[0].get_pixmap(matrix=pymupdf.Matrix(1.7,1.7)).save(out/'qa'/f'{name}.png')
 print('Exported '+name,flush=True)
def clean(a):
 a.spines[['top','right']].set_visible(False)
 for s in ['bottom','left']:a.spines[s].set_linewidth(.5);a.spines[s].set_color('#A5B1B5')
 a.tick_params(labelsize=6.5,length=2,width=.5,pad=3)

def render_c(mode):
 values=ids(rd(C/'source_data/tract_year_values.csv'));windows=json.loads(SRC['windows'].read_text(encoding='utf8'));audit=json.loads((O/'qa/preparation.json').read_text(encoding='utf8'))
 years=[2000,2010,2025] if mode=='cover' else [2005,2015,2025];field=COL if mode=='cover' else 'change_from_2000_pp';limit=audit['canopy_change_limit']
 cmap=SEQ if mode=='cover' else DIV;norm=Normalize(0,100) if mode=='cover' else Normalize(-limit,limit)
 for region in range(9):
  subset=windows[region*5:region*5+5];f=figure(180,225);txt(f,6,218,subset[0]['division'],10,weight='bold')
  txt(f,174,218,'Canopy cover' if mode=='cover' else 'Cumulative change from 2000',7,ha='right')
  for j,year in enumerate(years):txt(f,32+j*57,209,str(year),8,ha='center',weight='bold')
  for i,w in enumerate(subset):
   y=166-i*36;g=ids(gpd.read_file(w['prepared_file'],layer='sample_parts'));x0,y0,x1,y1=w['bounds_epsg5070_m']
   txt(f,6,y+35,f"{w['city_number']:02d}  {w['short_name']}",7,weight='bold');txt(f,174,y+35,f"n = {w['main_tracts']:,}",6.5,ha='right')
   for j,year in enumerate(years):
    vals=values[values.year.eq(year)].set_index('tract_id')[field];g['value']=g.tract_id.map(vals);require(g.value.notna().all(),'Missing polygon values')
    a=axmm(f,6+j*57,y,54,33.75);g.plot(ax=a,column='value',cmap=cmap,norm=norm,edgecolor='#FFFFFF',linewidth=.045,rasterized=True,aspect=None)
    a.set_xlim(x0,x1);a.set_ylim(y0,y1);a.set_aspect('equal');a.set_xticks([]);a.set_yticks([]);a.set_axis_off()
   # Geodesically checked scale and true north in the row header, outside the maps.
   frac=w['scale_projected_dx_m']/(x1-x0);length=54*frac
   f.add_artist(plt.Line2D([113/180,(113+length)/180],[(y+35)/225]*2,transform=f.transFigure,color=INK,lw=.7))
   txt(f,114+length,y+35,f"{w['scale_distance_m']/1000:g} km",6.3)
   vx,vy=w['north_unit_vector_epsg5070'];a=axmm(f,156,y+32.5,4,5);a.annotate('',xy=(.5+vx*.75,.15+vy*.60),xytext=(.5,.15),xycoords='axes fraction',arrowprops={'arrowstyle':'-|>','lw':.5,'color':INK,'mutation_scale':4});a.set_xticks([]);a.set_yticks([]);a.set_axis_off()
   txt(f,161,y+35.5,'N',5.7)
  ticks=[0,25,50,75,100] if mode=='cover' else [-limit,-15,-5,0,5,15,limit]
  bar(f,46,7,88,cmap,norm,'Canopy cover (%)' if mode=='cover' else 'Cumulative canopy change (percentage points)',ticks)
  export(f,C,f'TemporalC_{mode}_{region+1:02d}')
 combined=pymupdf.open()
 for region in range(9):
  with pymupdf.open(C/'exports'/f'TemporalC_{mode}_{region+1:02d}.pdf') as d:combined.insert_pdf(d)
 combined.save(C/'exports'/f'TemporalC_{mode}_atlas_9pages.pdf');combined.close()

def map_d(a,mp,field,states,cmap,norm):
 states.plot(ax=a,facecolor='white',edgecolor='#D0D6D5',linewidth=.22,aspect=None)
 a.scatter(mp.display_x_epsg5070_m,mp.display_y_epsg5070_m,c=mp[field],s=1.25,edgecolors='none',cmap=cmap,norm=norm,rasterized=True,zorder=2)
 a.set_xlim(-2450000,2300000);a.set_ylim(230000,3200000);a.set_aspect('equal');a.set_xticks([]);a.set_yticks([]);a.set_axis_off()

def render_d():
 mp=rd(D/'source_data/common_tract_map_values.csv');t=rd(D/'source_data/common_tract_temperature_panel.csv');states=gpd.read_file(SRC['states']);audit=json.loads((O/'qa/preparation.json').read_text(encoding='utf8'));lo,hi=audit['temp_limits'];lim=audit['temp_change_limit']
 f=figure(180,208)
 for i,year in enumerate(TY):
  y=147-i*51;txt(f,6,y+48,chr(97+2*i),9,weight='bold');txt(f,12,y+48,str(year),8,weight='bold');map_d(axmm(f,6,y,79,46),mp,'temp_'+str(year),states,HEAT,Normalize(lo,hi))
 for i,(field,label) in enumerate([('change_2019_2014','2019 − 2014'),('change_2024_2019','2024 − 2019')]):
  y=147-i*51;txt(f,96,y+48,chr(98+2*i),9,weight='bold');txt(f,102,y+48,label,8,weight='bold');map_d(axmm(f,96,y,79,46),mp,field,states,HDIV,Normalize(-lim,lim))
 txt(f,96,93,'f',9,weight='bold');txt(f,102,93,'Common-tract distributions',7.5,weight='bold')
 a=axmm(f,106,49,67,34);clean(a);hist=rd(D/'source_data/temperature_histogram.csv')
 for year,color in zip(TY,['#7196AF','#A1B5A0','#CB9983']):
  q=hist[hist.year.eq(year)];a.stairs(q.share*100,np.r_[q.left.to_numpy(),q.right.iloc[-1]],color=color,lw=1,label=str(year))
 a.set_xlabel('Adjusted surface temperature (°C)',fontsize=6.5);a.set_ylabel('Tracts (%)',fontsize=6.5);a.legend(frameon=False,fontsize=6,ncol=3,loc='upper right',handlelength=1,columnspacing=.6)
 bar(f,15,24,65,HEAT,Normalize(lo,hi),'Adjusted temperature (°C)',[lo,(lo+hi)/2,hi]);bar(f,104,24,65,HDIV,Normalize(-lim,lim),'Temperature change (°C)',[-lim,0,lim]);txt(f,90,8,'26,477 fixed tracts · 2014, 2019 and 2024',7,ha='center')
 export(f,D,'TemporalD_spatial_temperature')
 groups=rd(D/'source_data/national_tract_weighted_groups.csv');gaps=rd(D/'source_data/equal_city_gap_summary.csv');f=figure(180,152)
 txt(f,7,145,'Tract-weighted group means',9,weight='bold');txt(f,7,73,'Equal-city within-city gaps',9,weight='bold')
 ymin=np.floor(groups['mean'].min());ymax=np.ceil(groups['mean'].max());gmin=min(0,np.floor(gaps.mean_gap_c.min()));gmax=max(0,np.ceil(gaps.mean_gap_c.max()))
 for j,(v,label) in enumerate(zip(VARS,LABELS)):
  x=13+j*57;txt(f,x-6,135,chr(97+j),9,weight='bold');txt(f,x+23,135,label,8,ha='center',weight='bold');a=axmm(f,x,91,46,38);clean(a)
  for q in range(1,5):
   z=groups[(groups.variable.eq(v))&(groups.quartile.eq(q))].sort_values('outcome_year');a.plot(z.outcome_year,z['mean'],marker='o',ms=3,lw=1,color=GROUP[q-1],label=f'Q{q}')
  a.set_xticks(TY);a.set_xlim(2013,2025);a.set_ylim(ymin,ymax);a.set_yticks(np.arange(ymin,ymax+1,1));
  if j==0:a.set_ylabel('Temperature (°C)',fontsize=7)
  else:a.tick_params(labelleft=False)
  txt(f,x-6,62,chr(100+j),9,weight='bold');b=axmm(f,x,20,46,37);clean(b);z=gaps[gaps.variable.eq(v)].sort_values('outcome_year');b.axhline(0,lw=.5,color='#B6BEC1',ls='--');b.plot(z.outcome_year,z.mean_gap_c,color='#637B89',marker='o',ms=3.2,lw=1.1);b.set_xticks(TY);b.set_xlim(2013,2025);b.set_ylim(gmin-.15,gmax+.15)
  if j==0:b.set_ylabel('Highest − lowest (°C)',fontsize=7)
  else:b.tick_params(labelleft=False)
  for year,value in zip(z.outcome_year,z.mean_gap_c):b.annotate(f'{value:.2f}',(year,value),xytext=(0,7),textcoords='offset points',ha='center',fontsize=6.3)
 # Shared group legend; Q1 to Q4 are baseline quartiles, explained in the caption.
 handles=[plt.Line2D([],[],color=c,marker='o',lw=1,ms=3,label=l) for c,l in zip(GROUP,['Q1 (lowest)','Q2','Q3','Q4 (highest)'])]
 f.legend(handles=handles,loc='center',bbox_to_anchor=(.51,.525),ncol=4,frameon=False,fontsize=6.5,columnspacing=1.5,handlelength=1.6)
 txt(f,90,7,f"Fixed baseline quartiles · {audit['fixed_all_groups_cities']} cities in within-city summaries",7,ha='center')
 export(f,D,'TemporalD_social_temperature_gaps')

def validate():
 audit=json.loads((O/'qa/preparation.json').read_text(encoding='utf8'))
 for p,h in audit['source_hashes'].items():require(sha(p)==h,'Input changed '+p)
 overlaps=[];files=[]
 for out in [C,D]:
  for p in (out/'exports').glob('*.pdf'):
   with pymupdf.open(p) as doc:
    require(len(doc)==(9 if 'atlas_9pages' in p.name else 1),'Page count')
    for page in doc:
     require(any('Arial' in str(font) for font in page.get_fonts()),'Arial missing');spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans'] if s['text'].strip()]
     for i,s in enumerate(spans):
      a=pymupdf.Rect(s['bbox'])
      require(a.x0>=0 and a.y0>=0 and a.x1<=page.rect.width and a.y1<=page.rect.height,'PDF text clipped')
      for z in spans[i+1:]:
       b=pymupdf.Rect(z['bbox']);r=a&b
       if not r.is_empty and r.get_area()/min(a.get_area(),b.get_area())>.12:overlaps.append({'file':p.name,'page':page.number+1,'a':s['text'],'b':z['text']})
    files.append({'path':str(p),'sha256':sha(p),'pages':len(doc)})
  for p in (out/'exports').glob('*.svg'):require('<text' in p.read_text(encoding='utf8'),'SVG lacks live text');files.append({'path':str(p),'sha256':sha(p)})
 require(len(files)==42,'Expected 18 atlas pages + 2 thermal pages in PDF and SVG, and two combined atlases')
 dump(O/'qa/export_validation.json',{'status':'PASS' if not overlaps else 'LAYOUT_FIX_REQUIRED','files':files,'text_overlaps':overlaps,'source_hashes_unchanged':True,'maps':'600 dpi raster within vector containers; text remains editable'})
 print('Files:',len(files),'overlaps:',len(overlaps),flush=True)

if __name__=='__main__':
 sys.stdout.reconfigure(encoding='utf8');require(Path(sys.executable).parent.name=='sportpark-ee','Wrong environment')
 for out in [O,C,D,E]:
  for sub in ['source_data','exports','qa','schemas']:(out/sub).mkdir(parents=True,exist_ok=True)
 font_manager.findfont(font_manager.FontProperties(family='Arial'),fallback_to_default=False)
 plt.rcParams.update({'font.family':'Arial','font.size':7,'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':600,'path.simplify':False,'text.color':INK,'axes.labelcolor':INK})
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','cover','change','thermal','validate']);args=p.parse_args();start=time.time()
 if args.stage=='prepare':prepare()
 elif args.stage=='cover':render_c('cover')
 elif args.stage=='change':render_c('change')
 elif args.stage=='thermal':render_d()
 elif args.stage=='validate':validate()
 print(f'Elapsed seconds: {time.time()-start:.1f}',flush=True)
