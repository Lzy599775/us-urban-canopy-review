from nc_paths import project_root as _nc_project_root, external_path as _nc_external_path
from pathlib import Path
import argparse,json,hashlib,sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib import font_manager
import pymupdf
R=_nc_project_root();O=Path(__file__).resolve().parent
A=O/'TemporalA_全国树冠空间演变';B=O/'TemporalB1_45城年度资源';C=O/'TemporalB2_45城社会差距'
BASE=R/'06_logs/canopy_cooling_equity_20261009_run01';F4=R/'08_figure图/Fig04_全国响应与估计对象比较/revision03_atlas45'
SRC={'annual':BASE/'annual_canopy_equity_row_manifest.csv.gz','groups':BASE/'baseline_equity_group_manifest.csv','roster':F4/'source_data/city_index.csv','tracts':F4/'source_data/figure04_tract_values.csv','urban':R/'04_pilot/boundaries_2000/urban_area/tl_2009_us_uac00.zip','states':F4/'prepared/locator_states.gpkg'}
YRS=list(range(2000,2026));SNAPS=[2000,2005,2010,2015,2020,2025];VARS=['income','poverty','age65plus'];LABELS=['Income','Poverty','Age 65+']
COL='urban_canopy_percent_fixed_land';INK='#2D3940';EXT=(-2450000,2300000,230000,3200000)
SEQ=LinearSegmentedColormap.from_list('canopy',['#F0F3E4','#D3E1C0','#9BC8AC','#62AA9B','#367F80'])
DIV=LinearSegmentedColormap.from_list('change',['#947AAC','#CEC0DC','#FAF9F5','#BEDBD2','#489C8E'])
def req(b,msg):
 if not b:raise ValueError(msg)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def rd(p):return pd.read_csv(p,dtype={'state':str,'tract_id':str,'single_city_id':str},float_precision='round_trip')
def ids(d):
 for k,n in [('state',2),('tract_id',11),('single_city_id',5)]:
  if k in d:d[k]=pd.to_numeric(d[k],errors='raise').astype('Int64').astype('string').str.zfill(n)
 return d

def prepare():
 hashes={str(p):sha(p) for p in SRC.values()};a=ids(rd(SRC['annual']));nraw=len(a);a=a[a.balanced_26years].copy()
 req(len(a)==1207024 and a.tract_id.nunique()==46424,'Resource sample differs');req(not a.duplicated(['state','tract_id','year']).any(),'Duplicate year')
 req(a.groupby('tract_id').year.nunique().eq(26).all(),'Incomplete resource panel');req(a[COL].between(0,100).all(),'Invalid canopy value')
 for q in ['single_city_id']+[v+'_city_quartile' for v in VARS]:req(a.groupby('tract_id')[q].nunique(dropna=False).eq(1).all(),'Time-varying group '+q)
 gr=ids(rd(SRC['groups'])); base=a[a.year.eq(2000)];checks=base.merge(gr,on=['state','tract_id'],validate='one_to_one',suffixes=('_a','_b'))
 req(len(checks)==46424,'Baseline unmatched')
 for v in VARS:
  k=v+'_city_quartile';req(checks[k+'_a'].fillna(-1).eq(checks[k+'_b'].fillna(-1)).all(),'Group mismatch')
 city=a.groupby(['single_city_id','year']).agg(canopy=(COL,'mean'),change=('change_from_2000_pp','mean'),n=('tract_id','size')).reset_index()
 req(city.single_city_id.nunique()==448 and len(city)==448*26,'Cities differ');req(city.groupby('single_city_id').n.nunique().eq(1).all(),'City count changes')
 nat=a.groupby('year')[COL].agg(mean='mean',q25=lambda x:x.quantile(.25),q75=lambda x:x.quantile(.75),n='size').reset_index()
 u=gpd.read_file('zip://'+SRC['urban'].as_posix());crs=str(u.crs);u=u[u.UACE00.isin(city.single_city_id.unique())].copy();req(len(u)==448 and u.is_valid.all(),'Original cities invalid')
 u=u.to_crs(5070);req(u.is_valid.all() and (~u.is_empty).all() and u.area.gt(0).all(),'Projected cities invalid');points=[]
 for row in u.itertuples():
  parts=list(row.geometry.geoms) if row.geometry.geom_type=='MultiPolygon' else [row.geometry]
  ix=max(range(len(parts)),key=lambda k:parts[k].area);pt=parts[ix].representative_point();req(parts[ix].contains(pt),'Point not interior')
  points.append({'single_city_id':row.UACE00,'name':row.NAME00,'x':pt.x,'y':pt.y,'parts':len(parts),'selected_part':ix,'part_area_m2':parts[ix].area})
 pts=pd.DataFrame(points);req(pts.single_city_id.is_unique,'Duplicate city location')
 city=city.merge(pts,on='single_city_id',validate='many_to_one');req(city.x.between(*EXT[:2]).all() and city.y.between(*EXT[2:]).all(),'Map crop')
 city.to_csv(A/'source_data/city_annual_448.csv',index=False);pts.to_csv(A/'source_data/city_display_points.csv',index=False);nat.to_csv(A/'source_data/national_annual_46424.csv',index=False)
 wide=city.pivot(index='single_city_id',columns='year',values='canopy');parts=[]
 for start,end in zip(SNAPS[:-1],SNAPS[1:]):
  d=pts.copy();d['start']=start;d['end']=end;d['change']=d.single_city_id.map(wide[end]-wide[start]);parts.append(d)
 intervals=pd.concat(parts);intervals.to_csv(A/'source_data/city_five_year_changes.csv',index=False)
 roster=ids(rd(SRC['roster'])).sort_values('city_number');f=ids(rd(SRC['tracts']));req(len(f)==24493 and not f.duplicated(['state','tract_id']).any(),'F4 scope differs')
 x=a.merge(f[['state','tract_id','single_city_id']],on=['state','tract_id'],validate='many_to_one',suffixes=('','_f'))
 req(len(x)==24493*26 and x.single_city_id.eq(x.single_city_id_f).all(),'45 city pairing mismatch')
 m=x.groupby(['single_city_id','year']).agg(canopy=(COL,'mean'),change=('change_from_2000_pp','mean'),n=('tract_id','size')).reset_index().merge(roster[['single_city_id','city_number','short_name','division']],on='single_city_id',validate='many_to_one')
 req(len(m)==1170 and m.groupby('single_city_id').n.nunique().eq(1).all(),'City panel invalid')
 m.to_csv(B/'source_data/city45_resource_annual.csv',index=False);roster.to_csv(B/'source_data/city_roster.csv',index=False)
 gs=[]
 for v in VARS:
  q=x.groupby(['single_city_id','year',v+'_city_quartile']).agg(mean=(COL,'mean'),n=('tract_id','size')).reset_index().rename(columns={v+'_city_quartile':'quartile'});q['variable']=v;gs.append(q)
 groups=pd.concat(gs);groups.to_csv(C/'source_data/city45_quartile_annual.csv',index=False)
 gaps=[]
 for v in VARS:
  q=groups[groups.variable.eq(v)];w=q.pivot(index=['single_city_id','year'],columns='quartile',values='mean');g=(w[4]-w[1]).rename('gap').reset_index();b=g[g.year.eq(2000)].set_index('single_city_id').gap;g['delta_gap']=g.gap-g.single_city_id.map(b);g['variable']=v
  for quart in [1,4]:g['n_q'+str(quart)]=g.set_index(['single_city_id','year']).index.map(q[q.quartile.eq(quart)].set_index(['single_city_id','year']).n)
  gaps.append(g)
 gaps=pd.concat(gaps).merge(roster[['single_city_id','city_number','short_name','division']],on='single_city_id',validate='many_to_one');req(len(gaps)==3510 and gaps[['gap','delta_gap','n_q1','n_q4']].notna().all().all(),'Missing group')
 gaps.to_csv(C/'source_data/city45_gap_annual.csv',index=False)
 # Independent endpoint and change algebra, avoiding rounded labels.
 for d in [m,city]:
  b=d[d.year.eq(2000)].set_index('single_city_id').canopy;req(np.allclose(d.change,d.canopy-d.single_city_id.map(b),atol=1e-10,rtol=0),'Change algebra mismatch')
 req(m.change.abs().max()<=6 and gaps.delta_gap.abs().max()<=8 and city.canopy.max()<=60,'Approved colour limits do not contain data')
 sourcehash=hashes
 dump(O/'qa/preparation.json',{'status':'PASS','source_hashes':sourcehash,'raw_rows':nraw,'balanced_rows':len(a),'balanced_tracts':46424,'cities':448,'unassigned_tracts':int(base.single_city_id.isna().sum()),'display45_tracts':24493,'city45_year_rows':len(m),'gap_rows':len(gaps),'group_rows':len(groups),'geometry_source_crs':crs,'display_crs':'EPSG:5070','geometry_invalid':0,'unmatched':0,'duplicates':0,'maximum_five_year_city_change':float(intervals.change.abs().max()),'five_year_limit':float(np.ceil(intervals.change.abs().max())),'resource_change_range':[float(m.change.min()),float(m.change.max())],'gap_change_range':[float(gaps.delta_gap.min()),float(gaps.delta_gap.max())],'model_refit':False})
 print('Prepared: 448 cities, 46424 resource tracts; 45 cities, 24493 display tracts.',flush=True)

def fig(w,h):
 f=plt.figure(figsize=(w/25.4,h/25.4));f.mm=(w,h);return f
def axmm(f,x,y,w,h):return f.add_axes([x/f.mm[0],y/f.mm[1],w/f.mm[0],h/f.mm[1]])
def txt(f,x,y,s,size=7,ha='left',weight='normal',**kw):return f.text(x/f.mm[0],y/f.mm[1],s,fontsize=size,ha=ha,va='center',fontweight=weight,color=INK,**kw)

def number(value):
 s=f'{value:.1f}'
 return '0.0' if s=='-0.0' else s
def clean(a):
 a.spines[['top','right']].set_visible(False)
 for s in ['left','bottom']:a.spines[s].set_color('#A8B3B4');a.spines[s].set_linewidth(.5)
 a.tick_params(labelsize=6.5,length=2,width=.5,pad=2)
def bar(f,x,y,w,label,cmap,norm,ticks):
 a=axmm(f,x,y,w,2.8);cb=plt.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=a,orientation='horizontal',ticks=ticks);cb.solids.set_rasterized(False);cb.solids.set_edgecolor('face');cb.outline.set_visible(False);cb.ax.tick_params(labelsize=6.5,length=0,pad=2);txt(f,x+w/2,y+6,label,7,ha='center')
def export(f,out,name):
 f.canvas.draw();ren=f.canvas.get_renderer();bad=[]
 for t in f.findobj(matplotlib.text.Text):
  if t.get_visible() and t.get_text():
   z=t.get_window_extent(ren)
   if z.x0<-.5 or z.y0<-.5 or z.x1>f.bbox.x1+.5 or z.y1>f.bbox.y1+.5:bad.append(t.get_text())
 req(not bad,'Clipped text '+str(bad))
 for ext in ['pdf','svg']:f.savefig(out/'exports'/f'{name}.{ext}',dpi=600,facecolor='white')
 plt.close(f)
 with pymupdf.open(out/'exports'/f'{name}.pdf') as d:d[0].get_pixmap(matrix=pymupdf.Matrix(2,2)).save(out/'qa'/f'{name}.png')
 print('Exported '+name,flush=True)
def mapping(a,d,field,states,cmap,norm):
 states.plot(ax=a,facecolor='white',edgecolor='#CFD5D3',linewidth=.24,aspect=None,zorder=0)
 a.scatter(d.x,d.y,c=d[field],cmap=cmap,norm=norm,s=6.0,edgecolors='#657A75',linewidths=.13,zorder=2,rasterized=False)
 a.set_xlim(EXT[:2]);a.set_ylim(EXT[2:]);a.set_aspect('equal');a.set_xticks([]);a.set_yticks([]);a.axis('off');req(len(d)==448,'Map point count')

def render_a():
 city=rd(A/'source_data/city_annual_448.csv');nat=rd(A/'source_data/national_annual_46424.csv');change=rd(A/'source_data/city_five_year_changes.csv');states=gpd.read_file(SRC['states']);req(states.crs.to_epsg()==5070 and len(states)==49,'State CRS')
 f=fig(180,220)
 for i,year in enumerate(SNAPS):
  x=7+(i%2)*88;y=164-(i//2)*51
  txt(f,x,y+48,chr(97+i),9,weight='bold');txt(f,x+5,y+48,str(year),8.5,weight='bold')
  mapping(axmm(f,x,y,78,46),city[city.year.eq(year)],'canopy',states,SEQ,Normalize(0,60))
 bar(f,48,55,84,'Mean tract canopy cover (%)',SEQ,Normalize(0,60),[0,15,30,45,60])
 txt(f,7,43,'g',9,weight='bold');txt(f,12,43,'National annual canopy distribution',8,weight='bold')
 a=axmm(f,19,12,153,25);clean(a);a.fill_between(nat.year,nat.q25,nat.q75,color='#DAE8DB',lw=0,label='Tract interquartile range');a.plot(nat.year,nat['mean'],color='#397D77',lw=1.35,label='Tract mean');a.set_xlim(2000,2025);a.set_xticks(SNAPS);a.set_ylabel('Canopy cover (%)',fontsize=7);a.legend(loc='upper left',bbox_to_anchor=(.02,1.03),frameon=False,ncol=2,fontsize=6.5,handlelength=1.6);a.set_ylim(0,40);a.set_yticks([0,20,40])
 a.tick_params(axis='x',pad=5)
 export(f,A,'TemporalA_national_canopy_six_years')
 limit=json.loads((O/'qa/preparation.json').read_text(encoding='utf8'))['five_year_limit'];norm=Normalize(-limit,limit)
 f=fig(180,188)
 for i,(start,end) in enumerate(zip(SNAPS[:-1],SNAPS[1:])):
  x=7+(i%2)*88;y=130-(i//2)*53
  txt(f,x,y+48,chr(97+i),9,weight='bold');txt(f,x+5,y+48,f'{start}–{end}',8,weight='bold');mapping(axmm(f,x,y,78,46),change[change.start.eq(start)],'change',states,DIV,norm)
 txt(f,95,72,'f',9,weight='bold');txt(f,100,72,'City change distributions',8,weight='bold')
 a=axmm(f,106,28,64,37);clean(a);arr=[change.loc[change.start.eq(s),'change'] for s in SNAPS[:-1]]
 bp=a.boxplot(arr,vert=False,positions=np.arange(5),widths=.48,patch_artist=True,whis=(0,100),showfliers=False,medianprops={'color':'#3D6B65','linewidth':1},boxprops={'facecolor':'#D0E2DD','edgecolor':'#859F98','linewidth':.6},whiskerprops={'color':'#A9B6B1','linewidth':.6},capprops={'color':'#A9B6B1','linewidth':.6})
 a.set_yticks(range(5),[f'{s}–{e}' for s,e in zip(SNAPS[:-1],SNAPS[1:])]);a.invert_yaxis();a.axvline(0,color='#ADB4B1',lw=.5,ls='--');a.set_xlim(-limit,limit);a.set_xticks([-limit,0,limit]);a.set_xlabel('Change (percentage points)',fontsize=6.5)
 bar(f,47,10,85,'Five-year canopy change (percentage points)',DIV,norm,[-limit,-limit/2,0,limit/2,limit])
 export(f,A,'TemporalA_five_year_changes')

def matrix_ax(f,x,y,w,h,values,lim):
 a=axmm(f,x,y,w,h);n=len(values);# Explicit vector cells, not raster image.
 a.pcolormesh(np.arange(27)-.5,np.arange(n+1)-.5,values,cmap=DIV,norm=Normalize(-lim,lim),shading='flat',rasterized=False,edgecolors='none')
 a.set_xlim(-.5,25.5);a.set_ylim(n-.5,-.5);a.set_yticks([]);a.set_xticks([0,5,10,15,20,25],['2000','2005','2010','2015','2020','2025']);a.tick_params(axis='x',length=0,labelsize=6.5,pad=4)
 for sp in a.spines.values():sp.set_visible(False)
 for z in [4.5,9.5,14.5,19.5,24.5,29.5,34.5,39.5]:a.axhline(z,color='white',lw=1.5)
 return a

def rowlabels(f,roster,y,h,x=39,size=6.7):
 for i,r in enumerate(roster.itertuples()):txt(f,x,y+h-(i+.5)*h/45,f'{r.city_number:02d}  {r.short_name}',size,ha='right')
 names={'East North Central':'E. North Central','West North Central':'W. North Central','East South Central':'E. South Central','West South Central':'W. South Central'}
 for j in range(9):
  block=roster.iloc[j*5:(j+1)*5];req(block.division.nunique()==1,'Division order mismatch')
  title=block.division.iloc[0]
  txt(f,3,y+h-(j*5+2.5)*h/45,names.get(title,title),5.7,ha='center',rotation=90)

def render_b():
 d=rd(B/'source_data/city45_resource_annual.csv');roster=rd(B/'source_data/city_roster.csv').sort_values('city_number');order=roster.single_city_id.tolist()
 vals=d.pivot(index='single_city_id',columns='year',values='change').loc[order,YRS].to_numpy();end=d.pivot(index='single_city_id',columns='year',values='canopy').loc[order];counts=d.groupby('single_city_id').n.first()
 f=fig(180,218);txt(f,7,211,'Annual canopy change across 45 cities',10,weight='bold');txt(f,7,203,'Fixed city order and 24,493 census tracts',7)
 y,h=34,157;rowlabels(f,roster,y,h,x=42);matrix_ax(f,44,y,87,h,vals,6)
 for x,title in [(140,'2000'),(153,'2025'),(169,'Tracts')]:txt(f,x,195,title,7,ha='center',weight='bold')
 txt(f,146.5,200,'Cover (%)',6.5,ha='center')
 for i,r in enumerate(roster.itertuples()):
  yy=y+h-(i+.5)*h/45
  for x,value in [(140,f'{end.loc[r.single_city_id,2000]:.1f}'),(153,f'{end.loc[r.single_city_id,2025]:.1f}'),(169,f'{counts[r.single_city_id]:,}')]:txt(f,x,yy,value,6.5,ha='center')
 bar(f,54,12,76,'Change from 2000 (percentage points)',DIV,Normalize(-6,6),[-6,-3,0,3,6]);export(f,B,'TemporalB1_city45_annual_resource')

def render_c():
 d=rd(C/'source_data/city45_gap_annual.csv');roster=rd(B/'source_data/city_roster.csv').sort_values('city_number');order=roster.single_city_id.tolist()
 f=fig(180,222);txt(f,7,215,'Changing social gaps in canopy resources',10,weight='bold');txt(f,7,207,'Highest minus lowest baseline quartile; change relative to 2000',7)
 y,h=35,158;rowlabels(f,roster,y,h,x=40,size=6.4)
 for j,(v,label) in enumerate(zip(VARS,LABELS)):
  q=d[d.variable.eq(v)];vals=q.pivot(index='single_city_id',columns='year',values='delta_gap').loc[order,YRS].to_numpy();x=42+j*45
  txt(f,x+20,199,label,8,ha='center',weight='bold');a=matrix_ax(f,x,y,30,h,vals,8);a.set_xticks([0,12,25],['2000','2012','2025']);a.tick_params(labelsize=6.3)
  ends=q.pivot(index='single_city_id',columns='year',values='gap').loc[order]
  for xx,yr in [(x+34,2000),(x+41,2025)]:
   txt(f,xx,195,str(yr),6.0,ha='center',weight='bold')
   for i,val in enumerate(ends[yr]):txt(f,xx,y+h-(i+.5)*h/45,number(val),6.1,ha='center')
 bar(f,51,13,84,'Change in quartile gap (percentage points)',DIV,Normalize(-8,8),[-8,-4,0,4,8]);export(f,C,'TemporalB2_city45_social_gap_overview')
 # Detail pages retain numeric endpoints and group counts at readable size.
 for v,label in zip(VARS,LABELS):
  q=d[d.variable.eq(v)];vals=q.pivot(index='single_city_id',columns='year',values='delta_gap').loc[order,YRS].to_numpy();ends=q.pivot(index='single_city_id',columns='year',values='gap').loc[order];counts=q[q.year.eq(2000)].set_index('single_city_id')
  f=fig(180,218);txt(f,7,211,f'{label}: annual change in the canopy gap',10,weight='bold');txt(f,7,203,'Highest minus lowest baseline quartile',7)
  y,h=34,157;rowlabels(f,roster,y,h,x=42);matrix_ax(f,44,y,77,h,vals,8)
  for xx,head in [(131,'2000'),(144,'2025'),(158,'Low n'),(172,'High n')]:txt(f,xx,195,head,6.5,ha='center',weight='bold')
  txt(f,137.5,200,'Gap (pp)',6.5,ha='center')
  for i,r in enumerate(roster.itertuples()):
   yy=y+h-(i+.5)*h/45
   for xx,s in [(131,number(ends.loc[r.single_city_id,2000])),(144,number(ends.loc[r.single_city_id,2025])),(158,str(int(counts.loc[r.single_city_id,'n_q1']))),(172,str(int(counts.loc[r.single_city_id,'n_q4'])))]:txt(f,xx,yy,s,6.5,ha='center')
  bar(f,53,12,80,'Change in quartile gap (percentage points)',DIV,Normalize(-8,8),[-8,-4,0,4,8]);export(f,C,'TemporalB2_detail_'+v)

def validate():
 q=json.loads((O/'qa/preparation.json').read_text(encoding='utf8'))
 for p,h in q['source_hashes'].items():req(sha(p)==h,'Source changed: '+p)
 files=[];overlaps=[]
 for out in [A,B,C]:
  for p in (out/'exports').glob('*.pdf'):
   with pymupdf.open(p) as d:
    req(len(d)==1 and not d[0].get_images(),'PDF raster or page problem');page=d[0];req(any('Arial' in str(f) for f in page.get_fonts()),'Arial missing');spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans'] if s['text'].strip()]
    for i,s in enumerate(spans):
     a=pymupdf.Rect(s['bbox'])
     for t in spans[i+1:]:
      b=pymupdf.Rect(t['bbox']);z=a&b
      if not z.is_empty and z.get_area()/min(a.get_area(),b.get_area())>.1:overlaps.append({'file':p.name,'a':s['text'],'b':t['text']})
    files.append({'file':str(p),'sha256':sha(p),'minimum_font_pt':min(s['size'] for s in spans)})
  for p in (out/'exports').glob('*.svg'):
   s=p.read_text(encoding='utf8');req('<image' not in s and '<text' in s,'SVG not editable vector');files.append({'file':str(p),'sha256':sha(p)})
 req(len(files)==14,'Expected seven figures in two formats');dump(O/'qa/export_validation.json',{'status':'PASS' if not overlaps else 'LAYOUT_FIX_REQUIRED','files':files,'overlaps':overlaps,'sources_unchanged':True});print('Validation overlaps:',len(overlaps),flush=True)

if __name__=='__main__':
 sys.stdout.reconfigure(encoding='utf8');req(Path(sys.executable).parent.name=='sportpark-ee','Wrong environment')
 for out in [O,A,B,C]:
  for d in ['qa','source_data','exports','schemas']:(out/d).mkdir(parents=True,exist_ok=True)
 font_manager.findfont(font_manager.FontProperties(family='Arial'),fallback_to_default=False)
 plt.rcParams.update({'font.family':'Arial','font.size':7,'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':600,'text.color':INK,'axes.labelcolor':INK,'axes.unicode_minus':True,'path.simplify':False})
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','render','validate','all']);s=p.parse_args().stage
 if s in ['prepare','all']:prepare()
 if s in ['render','all']:render_a();render_b();render_c()
 if s in ['validate','all']:validate()
