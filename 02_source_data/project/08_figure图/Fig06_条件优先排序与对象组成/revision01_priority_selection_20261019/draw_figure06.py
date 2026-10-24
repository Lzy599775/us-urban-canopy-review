"""Figure 6: render approved frozen candidate-rule outputs, without reranking."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from pyproj import Transformer, Geod
import pymupdf

sys.stdout.reconfigure(encoding='utf-8')
OUT=Path(__file__).resolve().parent
CHECK=OUT.parent/'plan_20261019/qa/plan_source_check.json'
NAME='Figure06_priority_selection_and_composition'
MM=(180,220); EXT=(-2450000,2350000,200000,3270000)
INK='#27383D'; PURPLE='#9181AD'; TIE='#A7745E'; OTHER='#D8DEDB'
RULES=['poverty_priority','income_priority','age65plus_priority','canopy_deficit_priority','population_priority','response_strength']
LABELS=['High poverty','Low income','High age 65+','Low baseline canopy','Large population','Stronger response']
FIELDS=['poverty_share_2000','income_point_usd_2000','age65plus_share_2000','baseline_canopy_percent_fixed_land_2000','population_2000','theta']
ASC=[False,True,False,True,False,True]
plt.rcParams.update({'font.family':'Arial','font.size':7,'pdf.fonttype':42,'ps.fonttype':42,
                    'svg.fonttype':'none','axes.linewidth':.45,'text.color':INK,'axes.labelcolor':INK,
                    'xtick.color':INK,'ytick.color':INK,'axes.unicode_minus':True})

def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rd(p):return pd.read_csv(p,dtype={'state':str,'tract_id':str},float_precision='round_trip')
def eq(a,b):np.testing.assert_allclose(a,b,rtol=0,atol=1e-10)
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),'utf-8')
def lab(f,x,y,t,size=7,ha='left',weight='normal'):
    return f.text(x/180,y/220,t,fontsize=size,ha=ha,va='center',fontweight=weight,color=INK)
def axmm(f,x,y,w,h,gid):
    a=f.add_axes([x/180,y/220,w/180,h/220]);a.set_gid(gid);return a
def title(f,x,y,letter,t):
    lab(f,x,y,letter,10,weight='bold');lab(f,x+5,y,t,8)

def prepare():
    for sub in ['exports','source_data','qa']:(OUT/sub).mkdir(exist_ok=True)
    gate=json.loads(CHECK.read_text('utf-8'));assert gate['status']=='PASS_FOR_PLAN'
    src=gate['source_files']
    for v in src.values():assert sha(v['path'])==v['sha256'],v['path']
    p=rd(src['maps']['path']).sort_values(['state','tract_id']).reset_index(drop=True)
    s=rd(src['summary_original']['path']);c=rd(src['cutoffs']['path'])
    pd.testing.assert_frame_equal(s,rd(src['summary_copy']['path']),check_exact=True)
    assert len(p)==42372 and not p.duplicated(['state','tract_id']).any()
    assert p.candidate_records.sum()==104367 and len(s)==24 and len(c)==24
    xy=p[['display_x_epsg5070_m','display_y_epsg5070_m']]
    assert xy.notna().all().all()
    assert xy.iloc[:,0].between(*EXT[:2]).all() and xy.iloc[:,1].between(*EXT[2:]).all()
    counts=[]
    for rule,field,asc in zip(RULES,FIELDS,ASC):
        z=c[c.rule.eq(rule)&c.fraction.eq(.1)].iloc[0]
        expected=np.where(p[field].eq(z.cutoff),'boundary_tie',
                          np.where(p[field].lt(z.cutoff) if asc else p[field].gt(z.cutoff),'strict','other_candidate'))
        assert (p[rule+'_status']==expected).all(),rule
        n=p.groupby(rule+'_status').candidate_records.agg(['size','sum'])
        assert n.loc['strict','size']==z.strict_tracts and n.loc['boundary_tie','size']==z.tied_tracts
        assert n.loc['strict','sum']==z.strict_rows and n.loc['boundary_tie','sum']==z.tied_rows
        counts.append({'rule':rule,'counts':n.to_dict('index'),'cutoff':float(z.cutoff),
                       'boundary_slots':int(z.available_boundary_slots),'reported_selected_tracts':int(z.reported_selected_tracts)})
    for frac,n in zip([.01,.05,.1,.2],[1044,5219,10437,20874]):
        z=s[s.target_fraction.eq(frac)]
        assert len(z)==6 and z.selected_rows.eq(n).all() and z.candidate_rows.eq(104367).all()
    states=gpd.read_file(src['basemap']['path'])
    assert len(states)==49 and states.crs.to_epsg()==5070 and states.is_valid.all()
    assert not states.geometry.is_empty.any()
    b=states.total_bounds
    assert EXT[0]<=b[0]<b[2]<=EXT[1] and EXT[2]<=b[1]<b[3]<=EXT[3]
    outline=gpd.GeoSeries([states.geometry.union_all()],crs=states.crs).boundary
    p.to_csv(OUT/'source_data/map_states_10percent.csv',index=False,encoding='utf-8-sig')
    s.to_csv(OUT/'source_data/summary_24_combinations.csv',index=False,encoding='utf-8-sig')
    c.to_csv(OUT/'source_data/cutoffs_24_combinations.csv',index=False,encoding='utf-8-sig')
    rec={'source_files':src,'candidate_records':104367,'candidate_tracts':42372,'map_counts':counts,
         'size_mm':list(MM),'extent':list(EXT),'crs':'EPSG:5070','maps':[],'statistics':[],
         'new_ranking':False,'new_inference':False,'model_refit':False,'original_figures_modified':False}
    return p,s,states,outline,rec

def decor(a):
    inv=Transformer.from_crs(5070,4326,always_xy=True)
    fw=Transformer.from_crs(4326,5070,always_xy=True);geod=Geod(ellps='GRS80')
    x0=EXT[0]+.05*(EXT[1]-EXT[0]);y0=EXT[2]+.105*(EXT[3]-EXT[2])
    def distance(dx):return geod.inv(*inv.transform(x0,y0),*inv.transform(x0+dx,y0))[2]
    lo,hi=0.,2000000.
    for _ in range(65):
        mid=(lo+hi)/2
        if distance(mid)<1000000:lo=mid
        else:hi=mid
    dx=(lo+hi)/2;assert abs(distance(dx)-1000000)<.01
    a.plot([x0,x0+dx],[y0,y0],color=INK,lw=.65,zorder=5)
    for xx,txt,align in [(x0,'0','left'),(x0+dx,'1,000 km','left')]:
        a.plot([xx,xx],[y0-18000,y0+18000],color=INK,lw=.55,zorder=5)
        a.annotate(txt,(xx,y0),xytext=(0,-3),textcoords='offset points',ha=align,va='top',fontsize=7,color=INK,zorder=5)
    nx,ny=fw.transform(-96,48.8);lon,lat,_=geod.fwd(-96,48.8,0,1000);xx,yy=fw.transform(lon,lat)
    assert abs(xx-nx)<1e-6 and yy>ny
    fx,fy=a.transAxes.inverted().transform(a.transData.transform([nx,ny]))
    a.plot([fx,fx],[fy-.05,fy+.01],transform=a.transAxes,color=INK,lw=.6,zorder=5)
    a.add_patch(Polygon([[fx,fy+.025],[fx-.009,fy+.006],[fx+.009,fy+.006]],transform=a.transAxes,facecolor=INK,edgecolor='none',zorder=5))
    a.text(fx,fy+.04,'N',transform=a.transAxes,fontsize=7,ha='center',va='bottom',color=INK)
    return {'scale_actual_m':distance(dx),'scale_projected_m':dx,'scale_start':[x0,y0],
            'north_reference_lonlat':[-96,48.8],'north_projected_dx_m':xx-nx,'ellipsoid':'GRS80'}

def map_panel(f,i,p,states,outline,rec):
    x=[4,63,122][i%3];y=[176,131][i//3];w=54;h=w*3070000/4800000
    letter='abcdef'[i];rule=RULES[i]
    title(f,x,[215,170][i//3],letter,LABELS[i])
    a=axmm(f,x,y,w,h,'panel_'+letter)
    states.plot(ax=a,facecolor='white',edgecolor='#CCD2D3',linewidth=.25,aspect=None)
    outline.plot(ax=a,color='#8D999D',linewidth=.45,aspect=None)
    for status,size,color,marker,lw in [('other_candidate',.25,OTHER,'o',0),('strict',.8,PURPLE,'o',0),('boundary_tie',9,TIE,'D',.6)]:
        z=p[p[rule+'_status'].eq(status)]
        art=a.scatter(z.display_x_epsg5070_m,z.display_y_epsg5070_m,s=size,marker=marker,
                      facecolors='none' if status=='boundary_tie' else color,
                      edgecolors=color if status=='boundary_tie' else 'none',linewidths=lw,zorder=3 if status=='other_candidate' else 4)
        art.set_gid('map_'+letter+'_'+status)
        eq(art.get_offsets(),z[['display_x_epsg5070_m','display_y_epsg5070_m']])
    a.set(xlim=EXT[:2],ylim=EXT[2:],xticks=[],yticks=[]);a.set_aspect('equal')
    a.xaxis.offsetText.set_visible(False);a.yaxis.offsetText.set_visible(False)
    for sp in a.spines.values():sp.set_color('#D0D7D9');sp.set_linewidth(.35)
    rec['maps'].append({'panel':letter,'rule':rule,'points':len(p),'box_mm':[x,y,w,h],'decoration':decor(a)})

def stat_axis(f,x,y,h,lim,ticks,gid):
    a=axmm(f,x,y,21,h,gid);a.set(xlim=lim,ylim=(-.5,5.5),xticks=ticks,yticks=[])
    a.spines[['left','right','top']].set_visible(False);a.spines['bottom'].set_color('#A4AFB2')
    a.tick_params(axis='x',labelsize=7,length=2,width=.45,pad=2)
    for yy in range(6):a.axhline(yy,color='#E8ECEC',lw=.4,zorder=0)
    a.xaxis.offsetText.set_visible(False);a.yaxis.offsetText.set_visible(False)
    return a

def draw(p,s,states,outline,rec):
    font_manager.findfont('Arial',fallback_to_default=False)
    f=plt.figure(figsize=(180/25.4,220/25.4),dpi=180)
    for i in range(6):map_panel(f,i,p,states,outline,rec)
    handles=[Line2D([],[],marker='o',ls='',ms=3,color=PURPLE),
             Line2D([],[],marker='D',ls='',ms=3.6,mfc='none',mec=TIE,mew=.6),
             Line2D([],[],marker='o',ls='',ms=2.4,color=OTHER)]
    f.legend(handles,['Strict priority','Cutoff ties','Other candidates'],loc='center',
             bbox_to_anchor=(.5,124/220),ncol=3,frameon=False,handletextpad=.5,columnspacing=2,fontsize=7)
    title(f,4,115,'g','Mean predicted response by selected record share')
    for j,frac in enumerate([.01,.05,.1,.2]):
        x=39+35*j;lab(f,x+16,108,f'{int(frac*100)}%',8,ha='center',weight='bold' if frac==.1 else 'normal')
        if frac==.1:
            f.add_artist(Line2D([(x+10)/180,(x+22)/180],[105.4/220]*2,transform=f.transFigure,color=PURPLE,lw=1.1))
        a=stat_axis(f,x,72,30,(-.40,0),[-.4,-.2,0],'panel_g' if j==0 else f'g_share_{j}')
        a.set_xticklabels(['−0.4','−0.2','0'])
        z=s[s.target_fraction.eq(frac)].set_index('rule').loc[RULES]
        vals=z.mean_theta_c_per_pp.to_numpy();ys=np.arange(5,-1,-1)
        dots=a.scatter(vals,ys,s=16,c=PURPLE,edgecolors='none',zorder=3);dots.set_gid(f'g_data_{j}')
        eq(dots.get_offsets()[:,0],vals)
        if frac==.1:
            ring=a.scatter(vals,ys,s=42,facecolors='none',edgecolors=PURPLE,linewidths=.65,zorder=4);ring.set_gid('g_highlight_10percent')
        for i,v in enumerate(vals):
            yy=72+(5-i+.5)/6*30
            if j==0:lab(f,4,yy,LABELS[i],7)
            lab(f,x+32,yy,f'{v:.3f}'.replace('-','−'),7,ha='right')
            rec['statistics'].append({'panel':'g','rule':RULES[i],'fraction':frac,'value':float(v),'selected_rows':int(z.iloc[i].selected_rows)})
    lab(f,106,63,'Mean predicted response (°C per pp)',7.5,ha='center')
    title(f,4,55,'h','Composition at the 10% record share')
    specs=[('Income','1999 US$','mean_income_usd',1,(0,60000),[0,60000],0),
           ('Poverty','%','mean_poverty_share',100,(0,50),[0,25,50],1),
           ('Age 65+','%','mean_age65plus_share',100,(0,35),[0,15,30],1),
           ('Baseline canopy','%','mean_baseline_canopy',1,(0,25),[0,10,20],1)]
    z=s[s.target_fraction.eq(.1)].set_index('rule').loc[RULES]
    for j,(heading,unit,field,mul,lim,ticks,dec) in enumerate(specs):
        x=39+35*j;lab(f,x+16,48,heading,7.5,ha='center');lab(f,x+16,43.6,f'({unit})',7,ha='center')
        a=stat_axis(f,x,13,27,lim,ticks,'panel_h' if j==0 else f'h_attribute_{j}')
        if j==0:a.set_xticklabels(['0','60,000'])
        vals=z[field].to_numpy()*mul
        dots=a.scatter(vals,np.arange(5,-1,-1),s=16,c=PURPLE,edgecolors='none',zorder=3);dots.set_gid(f'h_data_{j}')
        eq(dots.get_offsets()[:,0],vals)
        for i,v in enumerate(vals):
            yy=13+(5-i+.5)/6*27
            if j==0:lab(f,29,yy,'abcdef'[i],7,ha='center',weight='bold')
            lab(f,x+32,yy,f'{v:,.{dec}f}',7,ha='right')
            rec['statistics'].append({'panel':'h','rule':RULES[i],'source_field':field,'multiplier':mul,'value':float(v)})
    return f

def export(f,rec):
    f.canvas.draw();renderer=f.canvas.get_renderer();texts=[];clipped=[];over=[]
    for t in f.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text():
            box=t.get_window_extent(renderer);texts.append((t,box))
            if box.x0<-.5 or box.y0<-.5 or box.x1>f.bbox.x1+.5 or box.y1>f.bbox.y1+.5:clipped.append(t.get_text())
    for i,(t,b) in enumerate(texts):
        for u,c in texts[i+1:]:
            w=min(b.x1,c.x1)-max(b.x0,c.x0);h=min(b.y1,c.y1)-max(b.y0,c.y0)
            if w>f.dpi/72*.2 and h>f.dpi/72*.2:over.append([t.get_text(),u.get_text(),float(w),float(h)])
    layout={'minimum_font_pt':min(t.get_fontsize() for t,b in texts),'text_count':len(texts),'clipped':clipped,'overlapping':over}
    dump(OUT/'qa/layout_check.json',layout)
    assert not clipped,clipped
    assert not over,over
    assert layout['minimum_font_pt']>=7
    for ext in ['pdf','svg','png']:f.savefig(OUT/'exports'/f'{NAME}.{ext}',dpi=600,facecolor='white')
    with pymupdf.open(OUT/'exports'/f'{NAME}.pdf') as d:d[0].get_pixmap(matrix=pymupdf.Matrix(2,2)).save(OUT/'qa/preview.png')
    plt.close(f)
    for v in rec['source_files'].values():assert sha(v['path'])==v['sha256']
    rec.update(status='PASS',layout=layout,exports={ext:{'path':str(OUT/'exports'/f'{NAME}.{ext}'),'sha256':sha(OUT/'exports'/f'{NAME}.{ext}')} for ext in ['pdf','svg','png']})
    dump(OUT/'qa/drawing_receipt.json',rec)
    print(json.dumps({'status':'PASS','maps':6,'points_per_map':42372,'statistical_points':48,'layout':layout},ensure_ascii=False))

if __name__=='__main__':
    p,s,states,outline,rec=prepare()
    export(draw(p,s,states,outline,rec),rec)
