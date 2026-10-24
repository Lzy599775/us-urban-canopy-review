"""Figure 7 from frozen conditional ranking summaries; no new policy effects.
Maps show strict and boundary-tied sets, never guessed top-N identities.
"""
from nc_paths import project_root as _nc_project_root, external_path as _nc_external_path
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager
from matplotlib.patches import Polygon
from pyproj import Transformer, Geod
import pymupdf

ROOT=_nc_project_root()
OUT=Path(__file__).resolve().parent
BASE=ROOT/'06_logs/state_alignment_policy_holdout_20261012_run01'
F1=ROOT/'08_figure图/Fig01_全国主分析样本空间分布'
F4=ROOT/'08_figure图/Fig04_全国响应与估计对象比较/revision03_atlas45'
SOURCES={'candidate':BASE/'policy_candidate_support_audit.csv','summary':BASE/'conditional_policy_ranking_summary.csv',
 'scores':ROOT/'06_logs/geo_adjusted_national_orthogonal_20261009_run01/crossfit_scores.csv',
 'points':F1/'source_data/figure01_display_points.csv','point_review':F1/'qa/geometry_validation.json',
 'states':F4/'prepared/locator_states.gpkg'}
KEY=['state','tract_id','exposure_year','outcome_year']
RULES=['poverty_priority','income_priority','age65plus_priority','canopy_deficit_priority','population_priority','response_strength']
TITLES=['High poverty','Low income','High share aged 65+','Low canopy','Large population','Stronger predicted response']
SHORT=['High poverty','Low income','High age 65+','Low canopy','Large population','Stronger response']
COLORS=['#B889A2','#829BB6','#BCA575','#83AAA0','#C38F7D','#777CA8']
FRACTIONS=[.01,.05,.1,.2]
MARKERS=['D','s','o','^']
XY=['display_x_epsg5070_m','display_y_epsg5070_m']
EXTENT=[-2450000,2300000,230000,3200000]
INK='#29353B'
CARTOGRAPHIC_KEYS=[]


def cartographic_key(ax):
    """Map footer: true north at central meridian and geodesic scale at 40 degrees N."""
    project=Transformer.from_crs(4326,5070,always_xy=True)
    unproject=Transformer.from_crs(5070,4326,always_xy=True)
    geod=Geod(ellps='WGS84');cx,cy=project.transform(-96,40)
    def distance(dx):
        ll=[unproject.transform(cx-dx/2,cy),unproject.transform(cx+dx/2,cy)]
        return geod.inv(*ll[0],*ll[1])[2]
    lo,hi=0.,2000000.
    for _ in range(60):
        mid=(lo+hi)/2
        if distance(mid)<1000000:lo=mid
        else:hi=mid
    dx=(lo+hi)/2
    require(abs(distance(dx)-1000000)<.01,'Scale calibration failed')
    lon2,lat2,_=geod.fwd(-96,40,0,100)
    xx,yy=project.transform(lon2,lat2)
    direction=np.array([xx-cx,yy-cy]);direction/=np.linalg.norm(direction)
    np.testing.assert_allclose(direction,[0,1],atol=1e-9)
    f=ax.figure;fw,fh=f.mm;p=ax.get_position();width=p.width*fw
    sx=p.x0*fw+1;sy=p.y0*fh-1.6
    length=width*dx/(EXTENT[1]-EXTENT[0])
    f.add_artist(Line2D([sx/fw,(sx+length)/fw],[sy/fh]*2,transform=f.transFigure,color=INK,lw=.65))
    for x in [sx,sx+length]:
        f.add_artist(Line2D([x/fw]*2,[(sy-.3)/fh,(sy+.3)/fh],transform=f.transFigure,color=INK,lw=.5))
    txt(f,sx+length+1,sy,'1,000 km',6.2)
    nx=(p.x0+p.width)*fw-5
    tail=np.array([nx,sy-1]);tip=np.array([nx,sy+1]);base=tip-[0,.7]
    f.add_artist(Line2D([nx/fw]*2,[tail[1]/fh,base[1]/fh],transform=f.transFigure,color=INK,lw=.65))
    f.add_artist(Polygon(np.array([tip,base+[-.35,0],base+[.35,0]])/[fw,fh],
                        transform=f.transFigure,closed=True,facecolor=INK,edgecolor='none'))
    txt(f,nx+1,sy,'N',6.2)
    CARTOGRAPHIC_KEYS.append({'axes_position':list(p.bounds),'scale_paper_mm':length,
                             'reference_lon_lat':[-96,40],'ground_distance_m':1000000,
                             'projected_distance_m':dx,'error_m':distance(dx)-1000000,
                             'north_tail_mm':tail.tolist(),'north_tip_mm':tip.tolist(),
                             'scale_start_mm':[sx,sy],'figure_mm':[fw,fh]})

def require(ok,msg):
    if not ok:raise ValueError(msg)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def rd(p,**kw):return pd.read_csv(p,dtype={'state':str,'tract_id':str},float_precision='round_trip',**kw)
def init():
    sys.stdout.reconfigure(encoding='utf8')
    require(Path(sys.executable).parent.name=='sportpark-ee','Use sportpark-ee')
    for k in ['exports','source_data','qa','schemas']:(OUT/k).mkdir(parents=True,exist_ok=True)
    font_manager.findfont(font_manager.FontProperties(family='Arial'),fallback_to_default=False)
    plt.rcParams.update({'font.family':'Arial','font.size':7,'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':600,'path.simplify':False,'axes.unicode_minus':True,'text.color':INK,'axes.labelcolor':INK})

def prepare():
    hashes={str(v):sha(v) for v in SOURCES.values()}
    geom=load(SOURCES['point_review']);require(geom['status']=='PASS','Figure 1 points unapproved')
    require(sha(SOURCES['points'])==geom['display_point_csv_sha256'],'Figure 1 display point file changed')
    f4=load(F4/'qa/preparation.json');require(sha(SOURCES['states'])==f4['locator_sha256'],'State boundary package changed')
    p=rd(SOURCES['candidate']);summ=rd(SOURCES['summary']);pts=rd(SOURCES['points'])
    # The frozen candidate export used pandas' default CSV parser for model scores.
    # Reproduce that input representation exactly; never substitute or rerank scores.
    s=pd.read_csv(SOURCES['scores'],dtype={'state':str,'tract_id':str})
    # Candidate file originally serialised numeric identifiers; restore widths explicitly.
    p['state']=p.state.str.zfill(2);p['tract_id']=p.tract_id.str.zfill(11)
    require(len(p)==104371 and len(s)==104371 and len(pts)==42373,'National sample changed')
    require(not p.duplicated(KEY).any() and not s.duplicated(KEY).any() and not pts.duplicated(['state','tract_id']).any(),'Duplicate key')
    check=p.merge(s[KEY+['theta']],on=KEY,validate='one_to_one',suffixes=('_p','_s'),indicator=True)
    require(check._merge.eq('both').all() and np.array_equal(check.theta_p,check.theta_s),'Candidate/model mapping differs')
    require(p.common_candidate.isin([True,False]).all(),'Invalid candidate flag')
    excluded=p.loc[~p.common_candidate,KEY+['support_1pp','headroom_1pp']]
    excluded.to_csv(OUT/'source_data/noncandidate_rows.csv',index=False)
    p=p.loc[p.common_candidate].copy()
    require(len(p)==104367 and p.tract_id.nunique()==42372,'Candidate scope changed')
    require(p.support_1pp.all() and p.headroom_1pp.all(),'Candidate eligibility conflict')
    require(len(summ)==24 and set(summ.rule)==set(RULES) and set(summ.target_fraction)==set(FRACTIONS),'Incomplete summary')
    fields={'mean_theta_c_per_pp':'theta','mean_poverty_share':'poverty_share_2000','mean_income_usd':'income_point_usd_2000','mean_age65plus_share':'age65plus_share_2000','mean_baseline_canopy':'baseline_canopy_percent_fixed_land_2000'}
    values=list(fields.values())+['population_2000']
    require(np.isfinite(p[values]).all().all(),'Missing candidate ranking variable')
    require(p.groupby(['state','tract_id'])[values].nunique().eq(1).all().all(),'Ranking varies within tract; tract map requires a different definition')
    t=p.groupby(['state','tract_id'],as_index=False).agg(**{v:(v,'first') for v in values},candidate_records=('theta','size'))
    t=t.merge(pts[['state','tract_id']+XY+['longitude','latitude']],on=['state','tract_id'],how='left',validate='one_to_one',indicator=True)
    require(t._merge.eq('both').all(),'Missing exact Figure 1 display coordinates');t=t.drop(columns='_merge')
    require(t[XY].notna().all().all() and len(t.state.unique())==49,'Coordinate or state coverage problem')
    require(t[XY[0]].between(EXTENT[0],EXTENT[1]).all() and t[XY[1]].between(EXTENT[2],EXTENT[3]).all(),'Map extent crops candidates')
    cutoffs=[];reproduction=[]
    for r in summ.itertuples():
        require(r.candidate_rows==len(p),'Summary candidate denominator differs')
        require(r.selected_rows==int(np.ceil(len(p)*r.target_fraction)),'Selection count convention changed')
        asc=r.rule not in ['poverty_priority','age65plus_priority','population_priority']
        d=p.sort_values(r.ranking_variable,ascending=asc,kind='stable').iloc[:r.selected_rows]
        err={k:float(d[v].mean()-getattr(r,k)) for k,v in fields.items()}
        err['population_weighted_theta_c_per_pp']=float(np.average(d.theta,weights=d.population_2000)-r.population_weighted_theta_c_per_pp)
        require(max(abs(v) for v in err.values())<1e-8 and d.tract_id.nunique()==r.selected_tracts,'Existing summary not reproduced: '+r.rule+str(r.target_fraction))
        cutoff=float(d.iloc[-1][r.ranking_variable]);strict=p[r.ranking_variable].lt(cutoff) if asc else p[r.ranking_variable].gt(cutoff);tied=p[r.ranking_variable].eq(cutoff)
        require(strict.sum()<r.selected_rows<=strict.sum()+tied.sum(),'Cutoff does not bracket target count')
        row=dict(rule=r.rule,fraction=r.target_fraction,ranking_variable=r.ranking_variable,ascending=asc,cutoff=cutoff,candidate_rows=len(p),target_rows=int(r.selected_rows),strict_rows=int(strict.sum()),strict_tracts=int(p.loc[strict,'tract_id'].nunique()),tied_rows=int(tied.sum()),tied_tracts=int(p.loc[tied,'tract_id'].nunique()),available_boundary_slots=int(r.selected_rows-strict.sum()),reported_selected_tracts=int(r.selected_tracts))
        cutoffs.append(row);reproduction.append(dict(rule=r.rule,fraction=r.target_fraction,errors=err,status='PASS'))
        if r.target_fraction==.1:
            strict_t=t[r.ranking_variable].lt(cutoff) if asc else t[r.ranking_variable].gt(cutoff)
            tied_t=t[r.ranking_variable].eq(cutoff)
            t[r.rule+'_status']=np.select([strict_t,tied_t],['strict','boundary_tie'],default='other_candidate')
            require(strict_t.sum()==row['strict_tracts'] and tied_t.sum()==row['tied_tracts'],'Record-to-tract selection inconsistency')
    require(len(cutoffs)==24,'Missing rule/fraction')
    t.to_csv(OUT/'source_data/tract_map_states_10percent.csv',index=False)
    pd.DataFrame(cutoffs).to_csv(OUT/'source_data/cutoffs_all_24_combinations.csv',index=False)
    summ.to_csv(OUT/'source_data/original_summary_24_combinations.csv',index=False)
    dump(OUT/'qa/preparation.json',{'status':'PASS','main_records':104371,'main_tracts':42373,'candidate_records':len(p),'candidate_tracts':len(t),'noncandidate_records':len(excluded),'ten_percent_target_records':10437,'unmatched_coordinates':0,'duplicate_tracts':0,'state_and_dc_count':49,'coordinate_origin':'Figure 1 deterministic interior point of largest fixed urban-intersection component','display_crs':'EPSG:5070','source_hashes':hashes,'summary_reproduction':reproduction,'model_refits':0,'selected_tie_identities_saved':False})
    print('PASS: 24 summaries reproduced; 104,367 candidate records / 42,372 tracts; all coordinates matched.',flush=True)

def fig(w,h):
    f=plt.figure(figsize=(w/25.4,h/25.4),dpi=150,facecolor='white');f.mm=(w,h);return f
def txt(f,x,y,s,size=7,weight='normal',ha='left',**kw):return f.text(x/f.mm[0],y/f.mm[1],s,fontsize=size,fontweight=weight,ha=ha,va='center',**kw)
def axis(f,x,y,w,h):return f.add_axes([x/f.mm[0],y/f.mm[1],w/f.mm[0],h/f.mm[1]])
def cleanax(ax):
    ax.spines[['top','right','left']].set_visible(False);ax.spines['bottom'].set_linewidth(.55)
    ax.tick_params(labelsize=6.5,length=2,width=.55,pad=2)
def mapping(ax,t,states,rule,color,detail=False):
    # Vector state outlines and fixed, unjittered tract symbols.
    states.plot(ax=ax,facecolor='white',edgecolor='#C9CECC',linewidth=.24 if detail else .18,aspect=None,zorder=0)
    other=t.loc[t[rule+'_status'].eq('other_candidate')];strict=t.loc[t[rule+'_status'].eq('strict')];ties=t.loc[t[rule+'_status'].eq('boundary_tie')]
    for d,c,size,zorder in [(other,'#D5DAD8',.35 if detail else .16,1),(strict,color,2.1 if detail else .8,2)]:
        ax.scatter(d[XY[0]],d[XY[1]],s=size,c=c,linewidths=0,alpha=1,zorder=zorder,rasterized=False)
    ax.scatter(ties[XY[0]],ties[XY[1]],s=19 if detail else 10,marker='D',facecolors='none',edgecolors='#444B51',linewidths=.6,zorder=3,rasterized=False)
    ax.set_xlim(*EXTENT[:2]);ax.set_ylim(*EXTENT[2:]);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([]);ax.axis('off')
    require(len(other)+len(strict)+len(ties)==42372,'Map omits candidate tracts')
    return dict(rule=rule,other_tracts=len(other),strict_tracts=len(strict),tied_tracts=len(ties),plotted_total=len(t),extent=EXTENT,axis_position=list(ax.get_position().bounds))
def map_legend(f,y):
    handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor='#8A9FA4',markeredgecolor='none',ms=4,label='Strict priority (panel colour)'),Line2D([0],[0],marker='D',color='none',markerfacecolor='none',markeredgecolor='#444B51',ms=4,label='At cutoff (ties)'),Line2D([0],[0],marker='o',color='none',markerfacecolor='#D5DAD8',markeredgecolor='none',ms=4,label='Other eligible tracts')]
    f.legend(handles=handles,loc='center',bbox_to_anchor=(.51,y/f.mm[1]),ncol=3,fontsize=7,frameon=False,columnspacing=1.5,handletextpad=.5)
def save(f,name):
    f.canvas.draw();renderer=f.canvas.get_renderer();bad=[]
    for a in f.findobj(matplotlib.text.Text):
        if not a.get_visible() or not a.get_text():continue
        b=a.get_window_extent(renderer)
        if b.x0<-.5 or b.y0<-.5 or b.x1>f.bbox.x1+.5 or b.y1>f.bbox.y1+.5:bad.append(a.get_text())
    require(not bad,'Clipped text: '+str(bad))
    for ext in ['pdf','svg']:f.savefig(OUT/f'exports/{name}.{ext}',dpi=600,facecolor='white')
    f.savefig(OUT/f'qa/{name}_preview.png',dpi=180,facecolor='white');plt.close(f)
    with pymupdf.open(OUT/f'exports/{name}.pdf') as d:d[0].get_pixmap(matrix=pymupdf.Matrix(2,2)).save(OUT/f'qa/{name}_pdf_render.png')
    print('Exported '+name,flush=True)

def render():
    q=load(OUT/'qa/preparation.json');require(q['status']=='PASS','Preparation not passed')
    for p,h in q['source_hashes'].items():require(sha(p)==h,'Source changed: '+p)
    t=rd(OUT/'source_data/tract_map_states_10percent.csv');summary=rd(OUT/'source_data/original_summary_24_combinations.csv');cuts=rd(OUT/'source_data/cutoffs_all_24_combinations.csv')
    states=gpd.read_file(SOURCES['states'],layer='states');require(states.crs.to_epsg()==5070 and len(states)==49 and states.is_valid.all(),'State geometry invalid')
    f=fig(180,238);maps=[]
    txt(f,90,236,'10% record share · 10,437 records per rule',6.5,ha='center')
    for i,(rule,title,col) in enumerate(zip(RULES,TITLES,COLORS)):
        x=6+(i%3)*58;top=230-(i//3)*49
        txt(f,x,top,'abcdef'[i],9,'bold');txt(f,x+4,top,title,7.2,'bold')
        ax=axis(f,x,top-37,53,33.3);maps.append(mapping(ax,t,states,rule,col))
        cartographic_key(ax)
        c=cuts.loc[cuts.rule.eq(rule)&cuts.fraction.eq(.1)].iloc[0]
        txt(f,x+26.5,top-42.7,f"{int(c.strict_tracts):,} strict · {int(c.tied_tracts)} at cutoff",6.4,ha='center')
    map_legend(f,132.5)
    txt(f,90,127.8,'Scale at 40°N · North at 96°W, 40°N',6.2,ha='center')
    # Panel g: discrete shares and exact mean responses; no interpolated curve.
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle
    txt(f,6,122,'g',9,'bold');txt(f,11,122,'Predicted response by selection share',8,'bold')
    matrix=np.array([[summary.loc[summary.rule.eq(r)&summary.target_fraction.eq(fr),'mean_theta_c_per_pp'].iloc[0] for fr in FRACTIONS] for r in RULES])
    cmap=LinearSegmentedColormap.from_list('response_lilac',['#F4F2F7','#DCD5E8','#B6ACCE','#8A83B0'])
    norm=Normalize(vmin=-.38,vmax=-.13)
    ax=axis(f,41,78,104,32)
    ax.set_xlim(-.5,3.5);ax.set_ylim(5.5,-.5);ax.axis('off')
    # More negative values use deeper colour. The numeric label is authoritative.
    for i,(rule,col) in enumerate(zip(RULES,COLORS)):
        for j,fr in enumerate(FRACTIONS):
            value=matrix[i,j]
            ax.add_patch(Rectangle((j-.46,i-.43),.92,.86,facecolor=cmap(1-norm(value)),edgecolor='none'))
            ax.text(j,i,f'{value:.3f}'.replace('-', '\N{MINUS SIGN}'),ha='center',va='center',fontsize=7.2,fontweight='bold' if fr==.1 else 'normal',color=INK)
        txt(f,37,110-(i+.5)*32/6,SHORT[i],6.7,ha='right')
    ax.add_patch(Rectangle((1.51,-.49),.98,5.98,facecolor='none',edgecolor='#71648E',lw=1.0,clip_on=False))
    for j,fr in enumerate(FRACTIONS):
        txt(f,41+(j+.5)*26,113.5,f'{fr:.0%}',7.5,'bold' if fr==.1 else 'normal',ha='center')
    # A compact colour key retains actual response units.
    cax=axis(f,153,78,2.7,32)
    import matplotlib as mpl
    cb=mpl.colorbar.ColorbarBase(cax,cmap=cmap.reversed(),norm=norm,orientation='vertical',ticks=[-.35,-.25,-.15])
    cb.solids.set_rasterized(False)
    cb.outline.set_visible(False);cb.ax.tick_params(labelsize=6.5,length=0,pad=3)
    txt(f,99,72,'Mean predicted response (°C per canopy percentage point)',6.7,ha='center')
    # Panel h: lengths measured from zero, original units and direct value labels.
    txt(f,6,62,'h',9,'bold');txt(f,11,62,'Community composition at the 10% share',8,'bold')
    attributes=[('mean_income_usd','Income','1999 US dollars',1,60000,[0,30000,60000]),('mean_poverty_share','Poverty','%',100,50,[0,25,50]),('mean_age65plus_share','Age 65+','%',100,35,[0,15,30]),('mean_baseline_canopy','Canopy','%',1,25,[0,10,20])]
    for j,(field,title,unit,factor,limit,ticks) in enumerate(attributes):
        x=41+j*34
        txt(f,x+14.5,54,title,7.2,'bold',ha='center')
        a=axis(f,x,16,28.5,32);cleanax(a);a.set_xlim(0,limit);a.set_ylim(5.65,-.65);a.set_xticks(ticks);a.set_yticks([])
        a.spines['bottom'].set_color('#A8B0B2');a.tick_params(axis='x',length=0,pad=3,labelsize=6.3)
        a.set_axisbelow(True);a.grid(axis='x',color='#E5E9E8',lw=.4)
        if j==0:a.set_xticklabels(['0','30,000','60,000'])
        for i,(rule,col) in enumerate(zip(RULES,COLORS)):
            r=summary.loc[summary.rule.eq(rule)&summary.target_fraction.eq(.1)].iloc[0];value=r[field]*factor
            a.barh(i,value,height=.43,color=col,edgecolor='none',zorder=3)
            label=f'{value:,.0f}' if j==0 else f'{value:.1f}'
            # Place labels above each bar at its right edge, within the same row.
            a.text(max(value,.09*limit),i-.28,label,ha='right',va='bottom',fontsize=6.5,color=INK,zorder=4)
        if j==0:
            for i in range(6):a.text(-.025,i,SHORT[i],transform=a.get_yaxis_transform(),ha='right',va='center',fontsize=6.7,clip_on=False)
        txt(f,x+14.25,7,unit,6.5,ha='center')
    save(f,'Figure07_spatial_priority_and_composition')
    # Larger pairs improve visibility of the boundary ties without moving symbols.
    for page in range(3):
        f=fig(180,220)
        for k in range(2):
            i=page*2+k;top=211-k*100;rule=RULES[i];txt(f,7,top,'abcdef'[i],10,'bold');txt(f,14,top,TITLES[i],10,'bold')
            ax=axis(f,8,top-85,164,79);maps.append(mapping(ax,t,states,rule,COLORS[i],True))
            cartographic_key(ax)
            c=cuts.loc[cuts.rule.eq(rule)&cuts.fraction.eq(.1)].iloc[0]
            txt(f,90,top-91.5,f"{int(c.strict_tracts):,} strict tracts · {int(c.tied_tracts)} at cutoff",7,ha='center')
        map_legend(f,9)
        txt(f,90,3.5,'Scale at 40°N · North at 96°W, 40°N',6.2,ha='center')
        save(f,f'Figure07_national_maps_{page+1:02d}')
    with pymupdf.open() as doc:
        for p in sorted((OUT/'exports').glob('Figure07_national_maps_[0-9][0-9].pdf')):
            with pymupdf.open(p) as d:doc.insert_pdf(d)
        require(len(doc)==3,'Map atlas incomplete');doc.save(OUT/'exports/Figure07_national_maps_3pages.pdf')
    dump(OUT/'qa/render_receipt.json',{'status':'RENDERED','maps':maps,'scientific_values_transformed_only_for_percentage_units':True,'point_jitter':False,'subsampling':False,'smoothing':False,'fitted_policy_curve':False,'all_vector':True,'save_dpi':600,'main_dimensions_mm':[180,238]})
    dump(OUT/'qa/cartography_redraw/cartographic_keys.json',CARTOGRAPHIC_KEYS)

def validate():
    q=load(OUT/'qa/preparation.json')
    for p,h in q['source_hashes'].items():require(sha(p)==h,'Source changed')
    outputs=[];overlaps=[]
    for p in sorted((OUT/'exports').glob('*.pdf')):
        with pymupdf.open(p) as d:
            require(len(d)==(3 if p.stem.endswith('3pages') else 1),'Page count mismatch');minimum=100
            for page in d:
                require(not page.get_images(),'Expected fully vector figure')
                require(any('Arial' in str(v) for v in page.get_fonts()),'Missing Arial')
                spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans'] if s['text'].strip()]
                minimum=min(minimum,min(s['size'] for s in spans))
                for i,a in enumerate(spans):
                    ra=pymupdf.Rect(a['bbox'])
                    for b in spans[i+1:]:
                        rb=pymupdf.Rect(b['bbox']);inter=ra&rb
                        if not inter.is_empty and inter.get_area()/min(ra.get_area(),rb.get_area())>.08:overlaps.append(dict(file=p.name,page=page.number+1,a=a['text'],b=b['text']))
            outputs.append(dict(file=str(p),pages=len(d),minimum_font_pt=minimum,sha256=sha(p),bytes=p.stat().st_size))
    for p in sorted((OUT/'exports').glob('*.svg')):
        text=p.read_text(encoding='utf8');require('<text' in text and '<image' not in text,'SVG vector/text failure');outputs.append(dict(file=str(p),sha256=sha(p),bytes=p.stat().st_size))
    require(len(outputs)==9,'Expected five PDF / four SVG files')
    dump(OUT/'qa/export_validation.json',{'status':'PASS' if not overlaps else 'REQUIRES_LAYOUT_FIX','outputs':outputs,'overlap_candidates':overlaps,'formal_sources_unchanged':True,'visual_review':'PENDING'})
    print('Validated exports; overlap candidates:',len(overlaps),flush=True)

if __name__=='__main__':
    init();ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','render','validate','all']);args=ap.parse_args()
    for name,fun in [('prepare',prepare),('render',render),('validate',validate)]:
        if args.stage in [name,'all']:fun()
