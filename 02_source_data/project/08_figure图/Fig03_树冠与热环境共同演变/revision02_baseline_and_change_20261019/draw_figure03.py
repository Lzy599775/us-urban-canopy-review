from nc_paths import project_root as _nc_project_root, external_path as _nc_external_path
from pathlib import Path
import json,hashlib,sys
import numpy as np,pandas as pd,geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib import font_manager
from pyproj import Transformer,Geod
import pymupdf
sys.stdout.reconfigure(encoding='utf-8')
R=_nc_project_root();O=Path(__file__).resolve().parent
# Use the verified Temporal_E source directory explicitly.
E=R/'08_figure图/Temporal_E_树冠与温度共同演变';S=E/'source_data';OUT=O/'exports';QA=O/'qa';ST=R/'08_figure图/Fig04_全国响应与估计对象比较/revision03_atlas45/prepared/locator_states.gpkg'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial'],'font.size':7,'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.5,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'axes.unicode_minus':True,'figure.facecolor':'white','savefig.facecolor':'white'})
CAN=LinearSegmentedColormap.from_list('canopy',['#ECF0D3','#BCDCC0','#8BC5B0','#59A99A','#377F79']);TEMP=LinearSegmentedColormap.from_list('temperature',['#88BACD','#BDDAE2','#F0E7D1','#E6B69E','#C77F73']);QCOL=['#70C1AB','#EAB77D','#AAA5CF','#D98EB7'];MARK=['o','s','^'];INK='#2D3940';EXT=(-2450000,2350000,200000,3270000);CY=[2013,2018,2023];TY=[2014,2019,2024]
CAN_CHANGE=LinearSegmentedColormap.from_list('canopy_change',['#8969A5','#B79CC8','#E3E4E0','#91C2AF','#40977E'],N=257)
TEMP_CHANGE=LinearSegmentedColormap.from_list('temperature_change',['#639EB9','#9DC4D5','#E3E4E0','#E0B09E','#C77F73'],N=257)
def rd(n):
 d=pd.read_csv(S/n,dtype={'single_city_id':str,'tract_id':str,'state':str},float_precision='round_trip')
 for c,z in [('single_city_id',5),('tract_id',11),('state',2)]:
  if c in d:d[c]=pd.to_numeric(d[c],errors='raise').astype('Int64').astype('string').str.zfill(z)
 return d
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def eq(a,b):np.testing.assert_allclose(np.asarray(a,float),np.asarray(b,float),rtol=0,atol=1e-10,equal_nan=True)
def lab(f,x,y,s,fs=7,ha='left',weight='normal',rotation=0):return f.text(x/f.mm[0],y/f.mm[1],s,fontsize=fs,ha=ha,va='center',fontweight=weight,color=INK,rotation=rotation)
def axmm(f,x,y,w,h):return f.add_axes([x/f.mm[0],y/f.mm[1],w/f.mm[0],h/f.mm[1]])
def mapdecor(a):
 proj=Transformer.from_crs(4326,5070,always_xy=True);inv=Transformer.from_crs(5070,4326,always_xy=True);geod=Geod(ellps='GRS80')
 # Scale calibration uses its actual map coordinates, not a relocated reference latitude.
 x0=EXT[0]+.045*(EXT[1]-EXT[0]); y0=EXT[2]+.075*(EXT[3]-EXT[2]); lo,hi=0.,2e6
 def dis(dx):return geod.inv(*inv.transform(x0,y0),*inv.transform(x0+dx,y0))[2]
 for _ in range(65):
  mid=(lo+hi)/2
  if dis(mid)<1e6:lo=mid
  else:hi=mid
 dx=(lo+hi)/2;assert abs(dis(dx)-1e6)<.01
 a.plot([x0,x0+dx],[y0,y0],color=INK,lw=.55,zorder=8)
 cap=.008*(EXT[3]-EXT[2])
 for xx in [x0,x0+dx]:a.plot([xx,xx],[y0-cap,y0+cap],color=INK,lw=.45,zorder=8)
 a.text(x0+dx+.012*(EXT[1]-EXT[0]),y0,'1,000 km',fontsize=7,va='center',color=INK,zorder=8)
 # True north at the central meridian, drawn in northern whitespace.
 nx,ny=proj.transform(-96,48.8);lon2,lat2,_=geod.fwd(-96,48.8,0,1000);xx,yy=proj.transform(lon2,lat2)
 assert abs(xx-nx)<1e-6 and yy>ny
 af=a.transAxes.inverted().transform(a.transData.transform([nx,ny]));x,y=af
 a.plot([x,x],[y-.055,y+.015],transform=a.transAxes,color=INK,lw=.55,zorder=8)
 a.add_patch(Polygon([[x,y+.028],[x-.009,y+.009],[x+.009,y+.009]],transform=a.transAxes,facecolor=INK,edgecolor='none',zorder=8))
 a.text(x,y+.04,'N',transform=a.transAxes,fontsize=7,ha='center',va='bottom',color=INK,zorder=8)
 return {'projected_length_m':float(dx),'endpoints_epsg5070':[[x0,y0],[x0+dx,y0]],'distance_m':dis(dx),'ellipsoid':'GRS80','north_reference_lonlat':[-96,48.8],'true_north_projected_dx_m':float(xx-nx)}
def map_panel(f,x,y,letter,year,field,cmap,norm,states,city,value_col=None,title=None):
 h=54*(EXT[3]-EXT[2])/(EXT[1]-EXT[0])
 if value_col is None:value_col=field
 if title is None:title=f'{year} {"canopy" if field=="canopy" else "temperature"}'
 lab(f,x,y+h+2.2,letter,9,weight='bold');lab(f,x+4.5,y+h+2.2,title,8)
 a=axmm(f,x,y,54,h);a.set_gid('map_'+letter)
 states.plot(ax=a,facecolor='white',edgecolor='#C6CED0',linewidth=.25,aspect=None)
 z=city[city.outcome_year.eq(year if field=='temperature' else year+1)]
 assert len(z)==379 and z.single_city_id.nunique()==379 and z[value_col].notna().all()
 assert z[value_col].between(norm.vmin,norm.vmax).all(),('Colour clipping',letter)
 c=a.scatter(z.x,z.y,c=z[value_col],s=5,cmap=cmap,norm=norm,edgecolors='white',linewidths=.18,zorder=3)
 c.set_gid('city_points_'+letter);eq(c.get_array(),z[value_col]);eq(c.get_offsets(),z[['x','y']])
 a.set_xlim(EXT[:2]);a.set_ylim(EXT[2:]);a.set_aspect('equal');a.set_xticks([]);a.set_yticks([])
 for sp in a.spines.values():sp.set_visible(True);sp.set_color('#D2D8D9');sp.set_linewidth(.35)
 return {'panel':letter,'year':int(year),'field':field,'value_column':value_col,'title':title,'cities':379,'tracts':int(z.n.sum()),'box_mm':[x,y,54,h],'scale_and_north':mapdecor(a),'colour_limits':[float(norm.vmin),float(norm.vmax)],'value_range':[float(z[value_col].min()),float(z[value_col].max())],'clipped_values':0}
def add_colorbar(f,x,y,w,cmap,norm,ticks,title,gid):
 ca=axmm(f,x,y,w,2.2);ca.set_gid(gid)
 cb=f.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),cax=ca,orientation='horizontal',ticks=ticks)
 cb.solids.set_rasterized(False);cb.solids.set_edgecolor('face');cb.outline.set_linewidth(.3)
 cb.ax.tick_params(labelsize=7,length=1.3,width=.35,pad=1.6)
 lab(f,x+w/2,y+4.5,title,7.5,ha='center')
def prepare_city_changes(city):
 assert not city.duplicated(['single_city_id','outcome_year']).any()
 assert city.groupby('single_city_id')[['n','x','y']].nunique().eq(1).all().all()
 baseline=city[city.outcome_year.eq(2014)][['single_city_id','canopy','temperature']].rename(columns={'canopy':'canopy_2013','temperature':'temperature_2014'})
 z=city.merge(baseline,on='single_city_id',how='left',validate='many_to_one',indicator=True)
 assert len(z)==len(city) and z._merge.eq('both').all();z=z.drop(columns='_merge')
 z['canopy_change_pp']=z.canopy-z.canopy_2013;z['temperature_change_c']=z.temperature-z.temperature_2014
 assert z[['canopy_change_pp','temperature_change_c']].notna().all().all()
 assert z.loc[z.outcome_year.eq(2014),['canopy_change_pp','temperature_change_c']].eq(0).all().all()
 z.to_csv(O/'source_data/city_levels_and_cumulative_changes.csv',index=False,encoding='utf-8-sig')
 return z
def clean(a):a.tick_params(labelsize=7,length=2,width=.45,pad=2.8,color='#7E8A8F');a.spines['bottom'].set_color('#8E999D');a.spines['left'].set_color('#8E999D');a.spines['bottom'].set_linewidth(.5);a.spines['left'].set_linewidth(.5)
def path_arrow(a,x,y,color,ms=3.5,ls='-',open_markers=False,gid=''):
 x=np.asarray(x,float);y=np.asarray(y,float)
 assert len(x)==3 and np.isfinite(x).all() and np.isfinite(y).all()
 assert (x>=a.get_xlim()[0]).all() and (x<=a.get_xlim()[1]).all()
 assert (y>=a.get_ylim()[0]).all() and (y<=a.get_ylim()[1]).all()
 line,=a.plot(x,y,color=color,lw=.85,ls=ls,zorder=2);line.set_gid(gid+'_line')
 for i in range(3):
  dot,=a.plot(x[i],y[i],MARK[i],ms=ms,mfc='none' if open_markers else color,mec=color if open_markers else 'white',mew=.65 if open_markers else .45,zorder=4)
  dot.set_gid(gid+'_period_'+str(i))
 # Arrowheads sit on the actual segment, with physical point sizing.
 span=np.array([np.diff(a.get_xlim())[0],np.diff(a.get_ylim())[0]])
 origin=np.array([a.get_xlim()[0],a.get_ylim()[0]])
 dim=a.get_position().size*a.figure.get_size_inches()*72
 lengths=[]
 for i in range(2):
  start=(np.array([x[i],y[i]])-origin)/span*dim;end=(np.array([x[i+1],y[i+1]])-origin)/span*dim
  v=end-start;length=np.linalg.norm(v);assert length>0
  unit=v/length;perp=np.array([-unit[1],unit[0]]);tip=start+.64*v
  head=min(2.6,length*.22);half=min(1.05,length*.10);base=tip-head*unit
  poly=Polygon(np.array([tip,base+half*perp,base-half*perp])/dim,transform=a.transAxes,facecolor=color,edgecolor='none',zorder=3)
  poly.set_gid(gid+'_arrow_'+str(i));a.add_patch(poly);lengths.append(float(length))
 return lengths
def main():
 font_manager.findfont('Arial',fallback_to_default=False)
 verification=json.loads((O.parent/'plan_20261019/qa/source_verification.json').read_text('utf-8'));assert verification['status']=='PASS'
 for record in verification['source_files'].values():assert sha(record['path'])==record['sha256'],('Input changed',record['path'])
 city=rd('city_paired_means.csv');groups=rd('social_group_trajectories.csv');gaps=rd('social_equal_city_gap_trajectories.csv');states=gpd.read_file(ST);assert len(city)==1137 and city.single_city_id.nunique()==379 and len(groups)==36 and len(gaps)==9 and len(states)==49 and states.crs.to_epsg()==5070 and states.is_valid.all();f=plt.figure(figsize=(180/25.4,245/25.4),dpi=180,facecolor='white');f.mm=(180,245);rec={'source_hashes':{record['path']:record['sha256'] for record in verification['source_files'].values()},'maps':[],'group_panels':[],'gap_panels':[]}
 city=prepare_city_changes(city)
 for col,x in enumerate([5,63,121]):
  yr=CY[col];letter='abc'[col]
  rec['maps'].append(map_panel(f,x,203,letter,yr,'canopy',CAN if col==0 else CAN_CHANGE,Normalize(0,100) if col==0 else Normalize(-12,12),states,city,value_col='canopy' if col==0 else 'canopy_change_pp',title='2013 canopy' if col==0 else f'{yr} − 2013'))
  yr=TY[col];letter='def'[col]
  rec['maps'].append(map_panel(f,x,150.5,letter,yr,'temperature',TEMP if col==0 else TEMP_CHANGE,Normalize(10,60) if col==0 else Normalize(-8,8),states,city,value_col='temperature' if col==0 else 'temperature_change_c',title='2014 temperature' if col==0 else f'{yr} − 2014'))
 add_colorbar(f,7,194.5,48,CAN,Normalize(0,100),[0,20,40,60,80,100],'Canopy cover (%)','canopy_level_colourbar')
 add_colorbar(f,72,194.5,96,CAN_CHANGE,Normalize(-12,12),[-12,-6,0,6,12],'Canopy change (pp)','canopy_change_colourbar')
 add_colorbar(f,7,143,48,TEMP,Normalize(10,60),[10,20,30,40,50,60],'Adjusted LST (°C)','temperature_level_colourbar')
 add_colorbar(f,72,143,96,TEMP_CHANGE,Normalize(-8,8),[-8,-4,0,4,8],'LST change (°C)','temperature_change_colourbar')
 lab(f,5,134,'Equal-tract means',8,weight='bold');f.legend(handles=[Line2D([],[],color=c,lw=1.7,ls=('--' if q=='Q2' else ':' if q=='Q3' else '-'),label=q) for c,q in zip(QCOL,['Q1','Q2','Q3','Q4'])],loc='center',bbox_to_anchor=(125/180,134/245),ncol=4,fontsize=7,handlelength=1.2,columnspacing=1)
 for j,(v,tit) in enumerate(zip(['income','poverty','age65plus'],['Income','Poverty','Age 65+'])):
  x=14+60*j;a=axmm(f,x,85,40,38);clean(a);a.set_xlim(9.5,25);a.set_ylim(34,39);a.set_xticks([10,15,20,25]);a.set_yticks([34,35,36,37,38,39]);lab(f,x,127.5,chr(103+j),9,weight='bold');lab(f,x+4.5,127.5,tit,8)
  if j:a.tick_params(labelleft=False)
  for q,c in zip([1,2,3,4],QCOL):
   z=groups[(groups.variable==v)&(groups.quartile==q)].sort_values('outcome_year');assert len(z)==3 and z.n.eq(z.n.iloc[0]).all();ls='--' if q==2 else ':' if q==3 else '-';lengths=path_arrow(a,z.canopy,z.temperature,c,3.5,ls,True,f'group_{v}_Q{q}');rec['group_panels'].append({'panel':chr(103+j),'variable':v,'quartile':q,'segment_lengths_pt':lengths,'n':z.n.tolist(),'canopy':z.canopy.tolist(),'temperature':z.temperature.tolist()})
 lab(f,3.2,104,'Mean adjusted LST (°C)',7.5,ha='center',rotation=90);lab(f,90,76,'Mean canopy cover (%)',7.5,ha='center');lab(f,5,68,'Equal-city gaps (Q4 − Q1)',8,weight='bold')
 for j,(v,tit,xlim,ylim) in enumerate([('income','Income',(7,8.5),(-2.8,-2)),('poverty','Poverty',(-7.5,-6),(1.8,2.6)),('age65plus','Age 65+',(-.5,1),(-.2,.6))]):
  x=14+60*j;a=axmm(f,x,19,40,38);clean(a);a.set_xlim(xlim);a.set_ylim(ylim);a.set_xticks(np.linspace(*xlim,4));a.set_yticks(np.linspace(*ylim,5));a.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'));a.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'));lab(f,x,61.5,chr(106+j),9,weight='bold');lab(f,x+4.5,61.5,tit,8);z=gaps[gaps.variable==v].sort_values('outcome_year');assert len(z)==3 and z.cities.eq(295).all();lengths=path_arrow(a,z.canopy,z.temperature,'#647F8C',4,gid=f'gap_{v}');rec['gap_panels'].append({'panel':chr(106+j),'variable':v,'segment_lengths_pt':lengths,'cities':z.cities.tolist(),'canopy_gap':z.canopy.tolist(),'temperature_gap':z.temperature.tolist()})
  offsets={'income':[(0,8,'center'),(0,-11,'center'),(4,8,'center')],'poverty':[(7,-8,'left'),(0,10,'center'),(-7,-8,'right')],'age65plus':[(0,-11,'center'),(6,9,'left'),(-6,9,'right')]}
  year_labels=[]
  for k,(yr,(dx,dy,ha)) in enumerate(zip(['2013/14','2018/19','2023/24'],offsets[v])):
   ann=a.annotate(yr,xy=(z.canopy.iloc[k],z.temperature.iloc[k]),xytext=(dx,dy),textcoords='offset points',ha=ha,va='bottom' if dy>0 else 'top',fontsize=7,color=INK,zorder=6)
   ann.set_gid(f'gap_{v}_year_{k}');year_labels.append({'text':yr,'x':float(z.canopy.iloc[k]),'y':float(z.temperature.iloc[k]),'offset_pt':[dx,dy]})
  rec['gap_panels'][-1]['year_labels']=year_labels
  if v=='age65plus':a.axvline(0,color='#C9CFD2',lw=.4,ls=(0,(2,2)));a.axhline(0,color='#C9CFD2',lw=.4,ls=(0,(2,2)))
 lab(f,3.2,38,'LST gap (°C)',7.5,ha='center',rotation=90);lab(f,90,10,'Canopy gap (pp)',7.5,ha='center');f.legend(handles=[Line2D([],[],marker=m,color='#647F8C',mfc='none',linestyle='none',ms=3.5,label=t) for m,t in zip(MARK,['2013 / 2014','2018 / 2019','2023 / 2024'])],loc='center',bbox_to_anchor=(.5,3.7/245),ncol=3,fontsize=7,handletextpad=.4,columnspacing=2.5)
 export_figure(f,'Figure03_baseline_and_cumulative_change',rec,verification,'drawing_receipts.json')
 render_absolute_companion(city,states,verification)
def export_figure(f,name,rec,verification,receipt_name):
 f.canvas.draw();ren=f.canvas.get_renderer();texts=[];bad=[]
 for t in f.findobj(matplotlib.text.Text):
  if t.get_visible() and t.get_text():
   b=t.get_window_extent(ren);texts.append((t,b));
   if b.x0<-.5 or b.y0<-.5 or b.x1>f.bbox.x1+.5 or b.y1>f.bbox.y1+.5:bad.append(t.get_text())
 assert not bad,('Text clipped',bad);assert min(t.get_fontsize() for t,b in texts)>=7;over=[]
 for i,(t,b) in enumerate(texts):
  for u,c in texts[i+1:]:
   iw=min(b.x1,c.x1)-max(b.x0,c.x0);ih=min(b.y1,c.y1)-max(b.y0,c.y0)
   if iw>f.dpi/72*.2 and ih>f.dpi/72*.2:over.append((t.get_text(),u.get_text(),float(iw),float(ih)))
 assert not over,over
 for ext in ['pdf','svg']:f.savefig(OUT/f'{name}.{ext}',dpi=600,facecolor='white')
 f.savefig(OUT/f'{name}.png',dpi=600,facecolor='white')
 with pymupdf.open(OUT/f'{name}.pdf') as d:d[0].get_pixmap(matrix=pymupdf.Matrix(1.7,1.7)).save(QA/f'{name}_preview.png')
 plt.close(f)
 for record in verification['source_files'].values():assert sha(record['path'])==record['sha256'],('Input changed during render',record['path'])
 rec.update({'status':'PASS','figure_name':name,'size_mm':list(f.mm),'text_min_pt':float(min(t.get_fontsize() for t,b in texts)),'text_count':len(texts),'text_clipped':bad,'text_overlaps':over,'formal_model_run':False,'new_inference':False,'png_dpi':600});(QA/receipt_name).write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in rec.items() if k not in ['source_hashes','maps','group_panels','gap_panels']},ensure_ascii=False))

def render_absolute_companion(city,states,verification):
 f=plt.figure(figsize=(180/25.4,112/25.4),dpi=180,facecolor='white');f.mm=(180,112)
 rec={'source_hashes':{record['path']:record['sha256'] for record in verification['source_files'].values()},'maps':[],'group_panels':[],'gap_panels':[]}
 for x,yr in zip([5,63,121],CY):rec['maps'].append(map_panel(f,x,70,chr(97+CY.index(yr)),yr,'canopy',CAN,Normalize(0,100),states,city))
 for x,yr in zip([5,63,121],TY):rec['maps'].append(map_panel(f,x,17.5,chr(100+TY.index(yr)),yr,'temperature',TEMP,Normalize(10,60),states,city))
 add_colorbar(f,45,61.5,90,CAN,Normalize(0,100),[0,20,40,60,80,100],'Mean canopy cover (%)','canopy_level_colourbar')
 add_colorbar(f,45,10,90,TEMP,Normalize(10,60),[10,20,30,40,50,60],'Mean adjusted LST (°C)','temperature_level_colourbar')
 export_figure(f,'Figure03_companion_absolute_levels',rec,verification,'companion_drawing_receipts.json')
if __name__=='__main__':main()
