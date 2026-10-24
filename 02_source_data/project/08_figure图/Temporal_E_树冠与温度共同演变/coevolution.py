"""Paired, descriptive canopy-temperature figures; formal study inputs remain read-only."""
from pathlib import Path
import argparse,json,hashlib,sys,time
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from pyproj import Transformer,Geod
import pymupdf

O=Path(__file__).resolve().parent;FIG=O.parent;R=FIG.parent
D=FIG/'Temporal_CD_小区图集与三期热环境/TemporalD_三期热环境'
F4=FIG/'Fig04_全国响应与估计对象比较/revision03_atlas45'
A=FIG/'Temporal_AB_全国与45城年度轨迹'
PANELS=[O/'01_全国城市配对地图',O/'02_45城二维轨迹',O/'03_社会差距共同演变']
S=O/'source_data';Q=O/'qa'
SRC={'temperature':D/'source_data/common_tract_temperature_panel.csv',
     'canopy':R/'06_logs/canopy_cooling_equity_20261009_run01/annual_canopy_equity_row_manifest.csv.gz',
     'city_points':A/'TemporalA_全国树冠空间演变/source_data/city_display_points.csv',
     'roster':F4/'source_data/city_index.csv','display_tracts':F4/'source_data/figure04_tract_values.csv',
     'states':F4/'prepared/locator_states.gpkg'}
CY=[2013,2018,2023];TY=[2014,2019,2024];VARS=['income','poverty','age65plus']
COL='urban_canopy_percent_fixed_land';TEMP='weather_adjusted_lst_reference_mean'
GROUP=['#638DA5','#96BAB1','#CAA985','#A681A8'];TIME=['#7196AF','#A1B5A0','#CB9983'];MARK=['o','s','^']
INK='#2D3940';EXT=[-2450000,2300000,230000,3200000]
CANOPY=LinearSegmentedColormap.from_list('canopy',['#F0F3E4','#D3E1C0','#9BC8AC','#62AA9B','#367F80'])
HEAT=LinearSegmentedColormap.from_list('heat',['#689EB6','#B6CCD3','#F4EBDF','#E6B29C','#BE796B'])
DRAW=[]
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rd(p):return pd.read_csv(p,dtype={'state':str,'tract_id':str,'single_city_id':str},float_precision='round_trip')
def save(d,n):d.to_csv(S/n,index=False,encoding='utf-8-sig')
def close(a,b):np.testing.assert_allclose(a,b,rtol=0,atol=1e-10,equal_nan=True)

def prepare():
    prev=load(D.parent/'qa/preparation.json');assert prev['status']=='PASS'
    for key in ['canopy','roster','display_tracts','states']:
        assert sha(SRC[key])==prev['source_hashes'][str(SRC[key])],('Source changed',key)
    assert sha(SRC['temperature'])==load(D/'qa/compact_revision/input_hashes.json')[str(SRC['temperature'])]
    hashes={str(p):sha(p) for p in SRC.values()}
    t=rd(SRC['temperature']);a=rd(SRC['canopy'])
    assert len(t)==79431 and t.tract_id.nunique()==26477 and not t.duplicated(['state','tract_id','outcome_year']).any()
    assert t.tract_id.str.fullmatch(r'\d{11}').all() and t.state.str.fullmatch(r'\d{2}').all()
    assert t.exposure_year.eq(t.outcome_year-1).all() and set(t.outcome_year)==set(TY)
    a=a[a.year.isin(CY)].copy();assert not a.duplicated(['state','tract_id','year']).any()
    columns=['state','tract_id','year',COL,'complete_fixed_land_year','balanced_26years','change_from_2000_pp']
    p=t.merge(a[columns],left_on=['state','tract_id','exposure_year'],right_on=['state','tract_id','year'],how='left',validate='one_to_one',indicator=True)
    p['pair_complete']=p._merge.eq('both')&p.complete_fixed_land_year.eq(True)&p[COL].between(0,100)&np.isfinite(p[TEMP])
    complete=p.groupby('tract_id').pair_complete.agg(['all','size']);good=complete.index[complete['all']&complete['size'].eq(3)]
    excluded=p[~p.tract_id.isin(good)].copy();save(excluded,'excluded_pair_records.csv')
    p=p[p.tract_id.isin(good)].drop(columns=['_merge','pair_complete']).sort_values(['state','tract_id','outcome_year'])
    close(p.change_from_2000_pp,p.treatment_pp_change_from_2000)
    for c in ['single_city_id']+[v+'_city_quartile' for v in VARS]:assert p.groupby('tract_id')[c].nunique(dropna=False).eq(1).all()
    save(p,'paired_tract_panel.csv')
    base=p[p.outcome_year.eq(2014)]
    save(base[base.single_city_id.isna()][['state','tract_id']],'unassigned_city_tracts.csv')
    save(base[~base.balanced_26years.eq(True)][['state','tract_id','single_city_id','balanced_26years']],'complete_three_pairs_not_balanced26.csv')
    city=p.dropna(subset=['single_city_id']).groupby(['single_city_id','exposure_year','outcome_year']).agg(canopy=(COL,'mean'),temperature=(TEMP,'mean'),n=('tract_id','size')).reset_index()
    assert city.groupby('single_city_id').n.nunique().eq(1).all()
    pts=rd(SRC['city_points']);assert pts.single_city_id.is_unique
    city=city.merge(pts,on='single_city_id',how='left',validate='many_to_one');assert city[['x','y']].notna().all().all()
    assert city.x.between(*EXT[:2]).all() and city.y.between(*EXT[2:]).all()
    # Check representative points against the original fixed 2000 urban geometry.
    apre=load(A/'qa/preparation.json')
    urbanpath=next(Path(k) for k in apre['source_hashes'] if Path(k).name=='tl_2009_us_uac00.zip')
    assert sha(urbanpath)==apre['source_hashes'][str(urbanpath)]
    u=gpd.read_file('zip://'+urbanpath.as_posix());u=u[u.UACE00.isin(city.single_city_id)].to_crs(5070)
    assert len(u)==city.single_city_id.nunique() and u.is_valid.all()
    for row in u.itertuples():
        parts=list(row.geometry.geoms) if row.geometry.geom_type=='MultiPolygon' else [row.geometry]
        part=max(parts,key=lambda geom:geom.area);pt=part.representative_point()
        expect=pts.set_index('single_city_id').loc[row.UACE00];close([pt.x,pt.y],[expect.x,expect.y]);assert part.contains(pt)
    hashes[str(urbanpath)]=sha(urbanpath);save(city,'city_paired_means.csv')
    roster=rd(SRC['roster']).sort_values('city_number');assert len(roster)==45
    manifest=rd(SRC['display_tracts']);display=p[p.tract_id.isin(manifest.tract_id)]
    all45=p[p.single_city_id.isin(roster.single_city_id)];assert set(display.tract_id)==set(all45.tract_id)
    count=display[display.outcome_year.eq(2014)].groupby('single_city_id').size()
    roster['paired_tracts']=roster.single_city_id.map(count).fillna(0).astype(int)
    save(roster,'city45_support.csv')
    c45=city[city.single_city_id.isin(roster.single_city_id)].merge(roster[['single_city_id','city_number','short_name','division']],on='single_city_id',validate='many_to_one')
    save(c45,'city45_trajectories.csv')
    support=[]
    for var in VARS:
        present=base.dropna(subset=['single_city_id']).groupby('single_city_id')[var+'_city_quartile'].nunique()
        for cid,n in present.items():support.append({'single_city_id':cid,'variable':var,'quartiles':n,'all_four':n==4})
    support=pd.DataFrame(support);eligible=support.groupby('single_city_id').all_four.all();fixed=eligible[eligible].index
    save(support,'social_city_support.csv');save(pd.DataFrame({'single_city_id':fixed}),'social_fixed_cities.csv')
    social=p[p.single_city_id.isin(fixed)].copy();save(social[['state','tract_id','exposure_year','outcome_year','single_city_id']],'social_fixed_panel_keys.csv')
    groups=[];citygroups=[];gaps=[]
    for var in VARS:
        k=var+'_city_quartile';assert social[k].notna().all()
        g=social.groupby(['exposure_year','outcome_year',k]).agg(canopy=(COL,'mean'),temperature=(TEMP,'mean'),n=('tract_id','size')).reset_index().rename(columns={k:'quartile'});g['variable']=var;groups.append(g)
        cg=social.groupby(['single_city_id','exposure_year','outcome_year',k]).agg(canopy=(COL,'mean'),temperature=(TEMP,'mean'),n=('tract_id','size')).reset_index().rename(columns={k:'quartile'});cg['variable']=var;citygroups.append(cg)
        assert cg.groupby(['single_city_id','outcome_year']).quartile.nunique().eq(4).all()
        lo=cg[cg.quartile.eq(1)].set_index(['single_city_id','exposure_year','outcome_year']);hi=cg[cg.quartile.eq(4)].set_index(['single_city_id','exposure_year','outcome_year'])
        delta=(hi[['canopy','temperature']]-lo[['canopy','temperature']]).reset_index();delta['variable']=var;gaps.append(delta)
    groups=pd.concat(groups,ignore_index=True);citygroups=pd.concat(citygroups,ignore_index=True);gaps=pd.concat(gaps,ignore_index=True)
    save(groups,'social_group_trajectories.csv');save(citygroups,'social_city_group_means.csv');save(gaps,'social_city_gap_pairs.csv')
    agg=gaps.groupby(['variable','exposure_year','outcome_year']).agg(canopy=('canopy','mean'),temperature=('temperature','mean'),cities=('single_city_id','size')).reset_index();save(agg,'social_equal_city_gap_trajectories.csv')
    # Independent direct aggregation from paired records, not from city summary tables.
    for row in groups.itertuples():
        d=social[social.outcome_year.eq(row.outcome_year)&social[row.variable+'_city_quartile'].eq(row.quartile)]
        close([d[COL].sum()/len(d),d[TEMP].sum()/len(d)],[row.canopy,row.temperature]);assert len(d)==row.n
    for row in agg.itertuples():
        d=social[social.outcome_year.eq(row.outcome_year)];key=row.variable+'_city_quartile'
        m=d.groupby(['single_city_id',key])[[COL,TEMP]].mean().unstack(key)
        close([(m[COL][4]-m[COL][1]).sum()/len(m),(m[TEMP][4]-m[TEMP][1]).sum()/len(m)],[row.canopy,row.temperature]);assert len(m)==row.cities
    span=c45.groupby('single_city_id')[['canopy','temperature']].agg(lambda z:z.max()-z.min()).max()
    widths={k:float(np.ceil((span[k]+2)/2)*2) for k in span.index}
    audit={'status':'PASS','source_hashes':hashes,'temperature_input_records':len(t),'temperature_input_tracts':t.tract_id.nunique(),
           'paired_records':len(p),'paired_tracts':p.tract_id.nunique(),'excluded_pair_records':len(excluded),
           'city_tracts':base.single_city_id.notna().sum().item(),'unassigned_city_tracts':base.single_city_id.isna().sum().item(),
           'paired_cities':city.single_city_id.nunique(),'complete_pairs_not_balanced26':int((~base.balanced_26years.eq(True)).sum()),
           'city45_supported':int(roster.paired_tracts.gt(0).sum()),'city45_tracts':int(roster.paired_tracts.sum()),
           'city45_no_support':roster.loc[roster.paired_tracts.eq(0),['city_number','short_name']].to_dict('records'),
           'social_cities':len(fixed),'social_tracts':social.tract_id.nunique(),'year_pairs':list(zip(CY,TY)),
           'city45_axis_spans':widths,'source_duplicates':0,'city_point_unmatched':0,'new_confidence_intervals':False,'formal_model_run':False,
           'prepared_hashes':{str(f):sha(f) for f in S.glob('*.csv')}}
    dump(Q/'preparation.json',audit);print(json.dumps({k:v for k,v in audit.items() if k not in ['source_hashes','prepared_hashes']},ensure_ascii=False),flush=True)

def fig(w,h):
    f=plt.figure(figsize=(w/25.4,h/25.4),dpi=150,facecolor='white');f.mm=(w,h);return f
def ax(f,x,y,w,h):return f.add_axes([x/f.mm[0],y/f.mm[1],w/f.mm[0],h/f.mm[1]])
def txt(f,x,y,s,size=7,ha='left',bold=False,**kw):return f.text(x/f.mm[0],y/f.mm[1],s,fontsize=size,ha=ha,va='center',fontweight='bold' if bold else 'normal',color=INK,**kw)
def clean(a,size=6.5):
    a.spines[['top','right']].set_visible(False)
    for key in ['bottom','left']:a.spines[key].set_color('#A8B3B5');a.spines[key].set_linewidth(.45)
    a.tick_params(labelsize=size,length=2,width=.45,pad=1.5,color='#8C9A9E')
def trajectory(a,x,y,color,colors=None,ms=4):
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert ((np.array(x)>=a.get_xlim()[0])&(np.array(x)<=a.get_xlim()[1])).all()
    assert ((np.array(y)>=a.get_ylim()[0])&(np.array(y)<=a.get_ylim()[1])).all()
    a.plot(x,y,color=color,lw=.85,zorder=2)
    for i,(xx,yy) in enumerate(zip(x,y)):a.plot(xx,yy,marker=MARK[i],color=colors[i] if colors else color,ms=ms,mec='white',mew=.4,linestyle='none',zorder=4)
    # Arrowheads are placed on the true straight segment in physical coordinates.
    span=np.array([a.get_xlim()[1]-a.get_xlim()[0],a.get_ylim()[1]-a.get_ylim()[0]])
    origin=np.array([a.get_xlim()[0],a.get_ylim()[0]])
    dims=np.array(a.get_position().size)*a.figure.get_size_inches()*72
    for i in range(2):
        start=(np.array([x[i],y[i]])-origin)/span*dims;end=(np.array([x[i+1],y[i+1]])-origin)/span*dims
        v=end-start;length=np.linalg.norm(v)
        assert length>0,'Exactly coincident periods require an explicit no-change symbol'
        unit=v/length;perp=np.array([-unit[1],unit[0]]);tip=start+.65*v
        head=min(2.5,.20*length);half=min(1.15,.095*length);base=tip-head*unit
        a.add_patch(Polygon(np.array([tip,base+half*perp,base-half*perp])/dims,transform=a.transAxes,closed=True,facecolor=color,edgecolor='none',zorder=3))
    DRAW.append({'type':'trajectory','x':list(map(float,x)),'y':list(map(float,y)),'axis_bounds':[list(a.get_xlim()),list(a.get_ylim())]})
def timelegend(f,y,coloured=False):
    handles=[Line2D([],[],marker=m,color=TIME[i] if coloured else '#7E8B90',linestyle='none',ms=4,label=f'{c} → {t}') for i,(c,t,m) in enumerate(zip(CY,TY,MARK))]
    f.legend(handles=handles,ncol=3,loc='center',bbox_to_anchor=(.52,y/f.mm[1]),frameon=False,fontsize=6.8,handletextpad=.5,columnspacing=2)
def export(f,folder,name):
    f.canvas.draw();renderer=f.canvas.get_renderer()
    for t in f.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text():continue
        b=t.get_window_extent(renderer)
        assert b.x0>=-1 and b.y0>=-1 and b.x1<=f.bbox.x1+1 and b.y1<=f.bbox.y1+1,('Text clipped',name,t.get_text())
    for ext in ['pdf','svg']:f.savefig(folder/f'exports/{name}.{ext}',dpi=600,facecolor='white')
    f.savefig(folder/f'qa/{name}.png',dpi=200,facecolor='white');plt.close(f);print('Exported',name,flush=True)
def map_key(a):
    proj=Transformer.from_crs(4326,5070,always_xy=True);inv=Transformer.from_crs(5070,4326,always_xy=True);g=Geod(ellps='WGS84');cx,cy=proj.transform(-96,40)
    def distance(dx):return g.inv(*inv.transform(cx-dx/2,cy),*inv.transform(cx+dx/2,cy))[2]
    lo,hi=0.,2000000.
    for _ in range(60):
        mid=(lo+hi)/2
        if distance(mid)<1000000:lo=mid
        else:hi=mid
    dx=(lo+hi)/2;assert abs(distance(dx)-1000000)<.01
    lon,lat,_=g.fwd(-96,40,0,100);xx,yy=proj.transform(lon,lat);assert abs(xx-cx)<1e-7 and yy>cy
    frac=dx/(EXT[1]-EXT[0]);sx,sy=.045,.059
    a.plot([sx,sx+frac],[sy,sy],transform=a.transAxes,lw=.65,color=INK)
    for x in [sx,sx+frac]:a.plot([x,x],[sy-.007,sy+.007],transform=a.transAxes,lw=.5,color=INK)
    a.text(sx+frac+.017,sy,'1,000 km',transform=a.transAxes,fontsize=6.2,va='center',color=INK)
    a.plot([.962,.962],[.045,.087],transform=a.transAxes,lw=.65,color=INK)
    a.add_patch(Polygon([[.962,.105],[.956,.087],[.968,.087]],transform=a.transAxes,color=INK,lw=0))
    a.text(.962,.13,'N',transform=a.transAxes,fontsize=6.2,ha='center',va='center',color=INK)
    return dx
def render_maps(audit):
    city=rd(S/'city_paired_means.csv');states=gpd.read_file(SRC['states']);assert states.crs.to_epsg()==5070 and len(states)==49
    f=fig(180,185);w=83.5;h=w*(EXT[3]-EXT[2])/(EXT[1]-EXT[0]);bottoms=[180-h,123.9-h,67.8-h]
    for row,(cy,ty) in enumerate(zip(CY,TY)):
        z=city[city.outcome_year.eq(ty)].sort_values('single_city_id');assert len(z)==audit['paired_cities']
        for col,(field,cmap,norm,year) in enumerate([('canopy',CANOPY,Normalize(0,80),cy),('temperature',HEAT,Normalize(19,57),ty)]):
            left=5+86.5*col;y=bottoms[row];txt(f,left,y+h+1.9,chr(97+row*2+col),8,bold=True);txt(f,left+5.4,y+h+1.9,str(year),8,bold=True)
            a=ax(f,left,y,w,h);states.plot(ax=a,facecolor='white',edgecolor='#CCD3D4',linewidth=.22,aspect=None)
            c=a.scatter(z.x,z.y,c=z[field],s=7.2,edgecolors='white',linewidths=.18,cmap=cmap,norm=norm,zorder=3)
            close(c.get_array(),z[field]);close(c.get_offsets(),z[['x','y']]);assert z[field].between(norm.vmin,norm.vmax).all()
            a.set_xlim(EXT[:2]);a.set_ylim(EXT[2:]);a.set_aspect('equal');a.set_xticks([]);a.set_yticks([])
            for sp in a.spines.values():sp.set_color('#D5DBDC');sp.set_linewidth(.35)
            dx=map_key(a);DRAW.append({'type':'city_map','field':field,'exposure_year':cy,'outcome_year':ty,'cities':len(z),'tracts':int(z.n.sum()),'size_points2':7.2,'bounds':EXT,'scale_projected_m':dx,'axes_position':list(a.get_position().bounds),'colour_limits':[norm.vmin,norm.vmax]})
    for x,cmap,norm,label,ticks in [(5,CANOPY,Normalize(0,80),'Mean canopy cover (%)',[0,20,40,60,80]),(91.5,HEAT,Normalize(19,57),'Mean adjusted surface temperature (°C)',[19,30,40,50,57])]:
        ca=ax(f,x+5,6.4,73.5,2.4);cb=f.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),cax=ca,orientation='horizontal',ticks=ticks)
        cb.solids.set_rasterized(False);cb.solids.set_edgecolor('face');cb.outline.set_linewidth(.35);cb.ax.tick_params(labelsize=6.5,length=1.6,pad=2,width=.4)
        txt(f,x+41.75,11.9,label,7,ha='center')
    export(f,PANELS[0],'TemporalE1_city_paired_maps')

def render_city45(audit):
    city=rd(S/'city45_trajectories.csv');roster=rd(S/'city45_support.csv').sort_values('city_number');f=fig(180,270);records=[]
    wx=audit['city45_axis_spans']['canopy'];wy=audit['city45_axis_spans']['temperature']
    for i,r in enumerate(roster.itertuples()):
        col=i%5;row=i//5;left=8+34*col;top=264-row*27
        if col==0:txt(f,2.0,top-13,r.division,6.2,ha='center',bold=True,rotation=90)
        txt(f,left,top,f'{r.city_number:02d}  {r.short_name}',6,bold=False)
        txt(f,left,top-3.1,f'n = {r.paired_tracts:,}',6)
        a=ax(f,left+3.6,top-21.6,26.5,15.4);clean(a,6)
        z=city[city.single_city_id.eq(r.single_city_id)].sort_values('outcome_year')
        if not r.paired_tracts:
            assert z.empty;a.set_xticks([]);a.set_yticks([])
            a.text(.5,.5,'No common\nobservations',transform=a.transAxes,ha='center',va='center',fontsize=6.1,color='#6D797C')
            records.append({'city_number':r.city_number,'n':0,'trajectory_drawn':False});continue
        assert len(z)==3 and z.outcome_year.tolist()==TY and z.n.eq(r.paired_tracts).all()
        x0=max(0,2*round(float((z.canopy.min()+z.canopy.max())/4))-wx/2)
        y0=2*round(float((z.temperature.min()+z.temperature.max())/4))-wy/2
        assert z.canopy.between(x0,x0+wx).all() and z.temperature.between(y0,y0+wy).all()
        a.set_xlim(x0,x0+wx);a.set_ylim(y0,y0+wy);a.set_xticks([x0,x0+wx/2,x0+wx]);a.set_yticks([y0,y0+wy/2,y0+wy])
        trajectory(a,z.canopy.to_numpy(),z.temperature.to_numpy(),'#819399',TIME,ms=3.2)
        records.append({'city_number':r.city_number,'n':int(r.paired_tracts),'trajectory_drawn':True,'xlim':list(a.get_xlim()),'ylim':list(a.get_ylim()),'x':z.canopy.tolist(),'y':z.temperature.tolist()})
    timelegend(f,14.6,coloured=True);txt(f,90,6,'Horizontal: canopy cover (%)   ·   Vertical: adjusted temperature (°C)',7,ha='center')
    dump(Q/'city45_panel_receipts.json',records);export(f,PANELS[1],'TemporalE2_city45_trajectories')

def render_social(audit):
    groups=rd(S/'social_group_trajectories.csv');gaps=rd(S/'social_equal_city_gap_trajectories.csv');f=fig(180,150)
    txt(f,6,145,'Equal-tract group means',8.5,bold=True);txt(f,6,72,'Equal-city within-city gaps',8.5,bold=True)
    for j,(var,label) in enumerate(zip(VARS,['Income','Poverty','Age 65+'])):
        x=15+56*j;txt(f,x-6,136,chr(97+j),9,bold=True);txt(f,x+22,136,label,8.3,ha='center',bold=True)
        a=ax(f,x,95,44,35);clean(a);a.set_xlim(9,25);a.set_ylim(34,39);a.set_xticks([10,15,20,25]);a.set_yticks([34,35,36,37,38,39])
        if j==0:a.set_ylabel('Temperature (°C)',fontsize=7,labelpad=3)
        else:a.tick_params(labelleft=False)
        for quartile,color in zip(range(1,5),GROUP):
            z=groups[groups.variable.eq(var)&groups.quartile.eq(quartile)].sort_values('outcome_year')
            assert len(z)==3 and z.n.nunique()==1
            trajectory(a,z.canopy.to_numpy(),z.temperature.to_numpy(),color,ms=4)
        a.set_xlabel('Canopy cover (%)',fontsize=7,labelpad=2)
        txt(f,x-6,65,chr(100+j),9,bold=True);b=ax(f,x,22,44,36);clean(b)
        z=gaps[gaps.variable.eq(var)].sort_values('outcome_year')
        # Gap origins differ by dimension; equal spans preserve common movement scale.
        midx=(z.canopy.min()+z.canopy.max())/2;midy=(z.temperature.min()+z.temperature.max())/2
        x0=np.floor(midx*2)/2-.5;y0=np.floor(midy*5)/5-.2
        b.set_xlim(x0,x0+1.5);b.set_ylim(y0,y0+.8)
        b.set_xticks([x0,x0+.5,x0+1,x0+1.5]);b.set_yticks([y0,y0+.2,y0+.4,y0+.6,y0+.8])
        b.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'));b.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
        assert z.canopy.between(*b.get_xlim()).all() and z.temperature.between(*b.get_ylim()).all()
        if b.get_xlim()[0]<=0<=b.get_xlim()[1]:b.axvline(0,color='#D1D8D9',lw=.5,ls=(0,(2,2)))
        if b.get_ylim()[0]<=0<=b.get_ylim()[1]:b.axhline(0,color='#D1D8D9',lw=.5,ls=(0,(2,2)))
        trajectory(b,z.canopy.to_numpy(),z.temperature.to_numpy(),'#6F8791',ms=4.5)
        b.set_xlabel('Canopy gap (percentage points)',fontsize=6.5,labelpad=2)
        if j==0:b.set_ylabel('Temperature gap (°C)',fontsize=7,labelpad=3)
    handles=[Line2D([],[],color=c,lw=1.5,label=q) for c,q in zip(GROUP,['Q1 (lowest)','Q2','Q3','Q4 (highest)'])]
    f.legend(handles=handles,ncol=4,loc='center',bbox_to_anchor=(.53,80/150),frameon=False,fontsize=6.8,handlelength=1.2,columnspacing=1.6)
    timelegend(f,6.5);export(f,PANELS[2],'TemporalE3_social_coevolution')

def render():
    audit=load(Q/'preparation.json');assert audit['status']=='PASS'
    for p,h in {**audit['source_hashes'],**audit['prepared_hashes']}.items():assert sha(p)==h,('Changed input',p)
    render_maps(audit);render_city45(audit);render_social(audit);dump(Q/'drawing_receipts.json',DRAW)

def validate():
    results=[]
    for folder in PANELS:
        for path in (folder/'exports').glob('*.pdf'):
            with pymupdf.open(path) as doc:
                assert len(doc)==1;p=doc[0]
                spans=[s for b in p.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans'] if s['text'].strip()]
                overlaps=[]
                for i,s in enumerate(spans):
                    assert 'Arial' in s['font'] and s['size']>=5.99,('Font',s['text'],s['size'])
                    a=pymupdf.Rect(s['bbox']);assert (a&p.rect).get_area()>=a.get_area()-.1,('Page clip',path.name,s['text'])
                    for t in spans[i+1:]:
                        b=pymupdf.Rect(t['bbox']);area=(a&b).get_area()
                        if area/min(a.get_area(),b.get_area())>.08:overlaps.append([s['text'],t['text']])
                assert not overlaps,('Overlap',path.name,overlaps)
                assert not p.get_images(),'Figures should remain vector'
                if folder==PANELS[0]:assert p.get_text().splitlines().count('N')==6 and p.get_text().count('1,000 km')==6
                if folder==PANELS[1]:assert p.get_text().count('n = ')==45 and p.get_text().count('No common')==1
                p.get_pixmap(matrix=pymupdf.Matrix(2,2)).save(folder/f'qa/{path.stem}_pdf_preview.png')
                svg=path.with_suffix('.svg');assert '<text' in svg.read_text(encoding='utf8')
                results.append({'file':str(path),'size_mm':(np.array([p.rect.width,p.rect.height])*25.4/72).tolist(),'text_overlap':0,'font':'Arial','minimum_font_pt':min(s['size'] for s in spans),'vector':True,'sha256':sha(path),'svg_sha256':sha(svg)})
    assert len(results)==3
    audit=load(Q/'preparation.json')
    for path,h in {**audit['source_hashes'],**audit['prepared_hashes']}.items():assert sha(path)==h
    dump(Q/'export_validation.json',{'status':'PASS','files':results,'original_inputs_unchanged':True});print('PASS exports, text, vector output and input hashes',flush=True)

def main():
    sys.stdout.reconfigure(encoding='utf8');arg=argparse.ArgumentParser();arg.add_argument('command',choices=['prepare','render','validate','all']);cmd=arg.parse_args().command
    assert Path(sys.executable).parent.name=='sportpark-ee';font_manager.findfont(font_manager.FontProperties(family='Arial'),fallback_to_default=False)
    for p in [S,Q,O/'schemas']+[f/d for f in PANELS for d in ['exports','qa']]:p.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Arial','font.size':7,'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':600,'path.simplify':False,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK})
    start=time.time()
    if cmd in ['prepare','all']:prepare()
    if cmd in ['render','all']:render()
    if cmd in ['validate','all']:validate()
    dump(Q/f'execution_{cmd}.json',{'status':'PASS','command':cmd,'seconds':time.time()-start,'script_sha256':sha(__file__),'python':sys.version,'formal_model':'NOT RUN'})
    print('Completed',cmd,'in',round(time.time()-start,1),'seconds',flush=True)
if __name__=='__main__':main()
