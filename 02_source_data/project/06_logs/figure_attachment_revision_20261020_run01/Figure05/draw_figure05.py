"""Approved Figure 5: frozen spatial combinations and descriptive compositions."""
from nc_paths import project_root as _nc_project_root, external_path as _nc_external_path
from pathlib import Path
import hashlib, importlib.util, json, sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from matplotlib import font_manager
import pymupdf

sys.stdout.reconfigure(encoding='utf-8')
ROOT=_nc_project_root()
OUT=Path(__file__).resolve().parent
NAME='Figure05_canopy_change_response_combinations'
CHECK=_nc_project_root() / '08_figure图/Fig05_树冠资源变化与条件响应空间组合/plan_20261019/qa/plan_source_check.json'
F4=ROOT/'08_figure图/Fig04_全国响应与估计对象比较/revision04_national_response_social_groups_20261019/draw_figure04.py'
# Reuse the verified cartographic decoration function without calling Figure 4 main.
spec=importlib.util.spec_from_file_location('figure04_cartography',F4)
cart=importlib.util.module_from_spec(spec)
spec.loader.exec_module(cart)
MM=(180,240);EXT=cart.EXT;INK=cart.INK
DIMS=[('income','Income'),('poverty','Poverty'),('age65plus','Older population')]
CHANGE_CMAP=LinearSegmentedColormap.from_list('net_canopy_change',
    ['#976DB3','#E2D6EA','#E3E4E0','#CEE8E3','#3F9593'],N=1025)
CHANGE_NORM=Normalize(-45,45,clip=False)

def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def eq(a,b):np.testing.assert_allclose(a,b,rtol=0,atol=1e-10,equal_nan=True)
def rd(p):return pd.read_csv(p,dtype={'state':str,'tract_id':str,'single_city_id':str},float_precision='round_trip')
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),'utf-8')
def lab(f,x,y,text,size=7,ha='left',weight='normal'):
    return f.text(x/MM[0],y/MM[1],text,fontsize=size,ha=ha,va='center',color=INK,fontweight=weight)
def axmm(f,x,y,w,h,gid):
    a=f.add_axes([x/MM[0],y/MM[1],w/MM[0],h/MM[1]]);a.set_gid(gid);return a
def title(f,x,y,letter,text):
    lab(f,x,y,letter,10,weight='bold');lab(f,x+5,y,text,8)

def prepare():
    for sub in ['exports','source_data','qa']:(OUT/sub).mkdir(exist_ok=True)
    audit=json.loads(CHECK.read_text('utf-8'));assert audit['status']=='PASS_FOR_PLAN'
    src=audit['source_files']
    for v in src.values():assert sha(v['path'])==v['sha256'],('Source changed',v['path'])
    src['figure04_cartography_script']={'path':str(F4),'sha256':sha(F4)}
    p=rd(src['paired']['path']);loc=rd(src['locations']['path'])
    rules=json.loads(Path(src['rules']['path']).read_text('utf-8'))
    dump(OUT/'source_data/legacy_display_rules.input.json',rules)
    rules=dict(rules)
    rules.update(continuous_change_limits=[-45,45], continuous_response_limits=[-.55,.05],
                 continuous_map_scaling='linear, full national ranges, no clipping',
                 continuous_normalization='linear', continuous_clip=False)
    assert [CHANGE_NORM.vmin,CHANGE_NORM.vmax]==rules['continuous_change_limits']
    assert [cart.NORM.vmin,cart.NORM.vmax]==rules['continuous_response_limits']
    assert not CHANGE_NORM.clip and not cart.NORM.clip
    ns=rd(src['national_shares']['path']);ns=ns[ns.scope.eq('National paired')].copy()
    social=rd(src['social_composition']['path']);social=social[social.scope.eq('National paired')].copy()
    keys=['state','tract_id'];codes=rules['category_order'];colors=rules['colors']
    assert len(p)==42369 and not p.duplicated(keys).any() and not loc.duplicated(keys).any()
    p=p.merge(loc[keys+['display_x_epsg5070_m','display_y_epsg5070_m']],on=keys,
              how='left',validate='one_to_one',indicator=True)
    assert p._merge.eq('both').all();p=p.drop(columns='_merge').sort_values(keys).reset_index(drop=True)
    assert p[['theta','change_pp','display_x_epsg5070_m','display_y_epsg5070_m']].notna().all().all()
    eq(p.canopy_2025-p.canopy_2000,p.change_pp)
    assert p.change_pp.between(-45,45).all() and p.theta.between(-.55,.05).all()
    assert p.display_x_epsg5070_m.between(*EXT[:2]).all() and p.display_y_epsg5070_m.between(*EXT[2:]).all()
    threshold=rules['threshold_negative_median']
    # Validate the frozen labels; do not replace them or choose a new threshold.
    expected=pd.Series(index=p.index,dtype='object')
    for sign,mask in [('loss',p.change_pp.lt(0)),('gain',p.change_pp.gt(0))]:
        expected.loc[mask&p.theta.lt(0)&p.theta.le(threshold)]=sign+'_stronger'
        expected.loc[mask&p.theta.gt(threshold)&p.theta.lt(0)]=sign+'_weaker'
    expected.loc[p.change_pp.eq(0)]='zero_change';expected.loc[p.theta.gt(0)]='positive_response'
    assert expected.notna().all() and expected.eq(p.category).all()
    assert p.category.map(dict(zip(codes,range(6)))).eq(p.category_code).all()
    assert p.theta.eq(0).sum()==0 and (p.theta.gt(0)&p.change_pp.eq(0)).sum()==0
    assert len(ns)==6 and ns.denominator.eq(42369).all()
    ns=ns.set_index('category').loc[codes].reset_index()
    for row in ns.itertuples():
        assert row.tracts==p.category.eq(row.category).sum();eq(row.percent,row.tracts/42369*100)
    eq(ns.percent.sum(),100)
    assert len(social)==90
    for dim,_ in DIMS:
        for q in [1,2,3,4,0]:
            col=dim+'_city_quartile';subset=p[p[col].isna() if q==0 else p[col].eq(q)]
            z=social[social.variable.eq(dim)&social.quartile.eq(q)]
            assert len(z)==6 and z.denominator.eq(len(subset)).all()
            for row in z.itertuples():
                assert row.tracts==subset.category.eq(row.category).sum();eq(row.percent,row.tracts/len(subset)*100)
            eq(z.percent.sum(),100)
        assert p[dim+'_city_quartile'].notna().sum()==42079
    states=gpd.read_file(src['basemap']['path'])
    assert len(states)==49 and states.crs.to_epsg()==5070 and states.is_valid.all()
    assert not states.geometry.is_empty.any()
    bounds=states.total_bounds
    assert EXT[0]<=bounds[0]<bounds[2]<=EXT[1] and EXT[2]<=bounds[1]<bounds[3]<=EXT[3]
    outline=gpd.GeoSeries([states.geometry.union_all()],crs=states.crs).boundary
    for data,name in [(p,'map_paired_tracts.csv'),(ns,'national_category_shares.csv'),(social,'national_social_composition.csv')]:
        data.to_csv(OUT/'source_data'/name,index=False,encoding='utf-8-sig')
    dump(OUT/'source_data/display_rules.json',rules)
    rec=dict(source_files=src,paired_tracts=42369,unmatched_locations=0,duplicate_keys=0,
             classified_tracts=42369,zero_change=27,positive_response=20,
             grouped_tracts_per_dimension=42079,ungrouped_tracts=290,
             threshold=threshold,threshold_ties=int(p.theta.eq(threshold).sum()),
             categories=ns.to_dict('records'),crs='EPSG:5070',geometries_valid=True,
             extent=list(EXT),geometry_repair=False,model_refit=False,new_inference=False,
             original_figures_modified=False,maps=[],bars=[])
    return p,ns,social,rules,states,outline,rec

def map_panel(f,x,y,w,letter,field,cmap,norm,p,states,outline,rec):
    h=w*(EXT[3]-EXT[2])/(EXT[1]-EXT[0])
    a=axmm(f,x,y,w,h,'panel_'+letter)
    states.plot(ax=a,facecolor='white',edgecolor='#CCD2D3',linewidth=.25,aspect=None)
    outline.plot(ax=a,color='#8D999D',linewidth=.45,aspect=None)
    c=a.scatter(p.display_x_epsg5070_m,p.display_y_epsg5070_m,c=p[field],
                s=.55,cmap=cmap,norm=norm,linewidths=0,edgecolors='none',alpha=1,zorder=3)
    c.set_gid('map_42369_'+letter);eq(c.get_array(),p[field])
    eq(c.get_offsets(),p[['display_x_epsg5070_m','display_y_epsg5070_m']])
    a.set(xlim=EXT[:2],ylim=EXT[2:],xticks=[],yticks=[]);a.set_aspect('equal')
    a.xaxis.offsetText.set_visible(False);a.yaxis.offsetText.set_visible(False)
    for s in a.spines.values():s.set_visible(True);s.set_color('#D0D7D9');s.set_linewidth(.35)
    rec['maps'].append(dict(panel=letter,field=field,points=len(p),box_mm=[x,y,w,h],
                            decoration=cart.map_decor(a),values_clipped=0))

def colorbar(f,x,w,cmap,norm,ticks,labels,text,gid):
    ca=axmm(f,x,73,w,2.6,gid)
    cb=f.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap,norm=norm),cax=ca,
                  orientation='horizontal',ticks=ticks)
    cb.solids.set_rasterized(False);cb.solids.set_edgecolor('face');cb.outline.set_linewidth(.35)
    cb.ax.tick_params(labelsize=7,length=2,width=.4,pad=2)
    cb.ax.set_xticklabels(labels);cb.ax.axvline(0,color='#80898B',lw=.5)
    lab(f,x+w/2,79.5,text,7.5,ha='center')

def draw(p,ns,social,rules,states,outline,rec):
    font_manager.findfont('Arial',fallback_to_default=False)
    f=plt.figure(figsize=(180/25.4,240/25.4),dpi=180)
    colors=rules['colors'];codes=rules['category_order']
    title(f,4,236,'a','Canopy change × response')
    map_panel(f,4,161,112,'a','category_code',ListedColormap(colors),
              BoundaryNorm(np.arange(-.5,6.5),6),p,states,outline,rec)
    title(f,124,236,'b','National composition')
    # b is one composition panel containing its shared legend and national shares.
    b=axmm(f,124,153,52,78,'panel_b');b.set_axis_off()
    lab(f,153,232,'Negative response',7,ha='center')
    for col,row,idx in [(0,0,0),(1,0,1),(0,1,2),(1,1,3)]:
        r=Rectangle(((145+9*col)/180,(212+9*row)/240),9/180,9/240,
                     transform=f.transFigure,facecolor=colors[idx],edgecolor='white',linewidth=.5)
        r.set_gid('bivariate_key_'+codes[idx]);f.add_artist(r)
    lab(f,143,216.5,'Weaker',7,ha='right');lab(f,143,225.5,'Stronger',7,ha='right')
    lab(f,149.5,208.7,'Loss',7,ha='center');lab(f,158.5,208.7,'Gain',7,ha='center')
    # Percent bars are all measured against one 0–30% axis, with no minimum width.
    ba=axmm(f,124,154,52,46,'national_share_axis')
    ba.set(xlim=(0,30),ylim=(0,46),yticks=[],xticks=[0,10,20,30])
    ba.set_xticklabels(['0','10','20','30%'])
    ba.spines[['left','right','top']].set_visible(False);ba.spines['bottom'].set_color('#929DA1')
    ba.tick_params(axis='x',labelsize=7,length=2,width=.45,pad=2,colors=INK)
    for i,row in ns.iterrows():
        label_y=198-7.5*i;bar_y=label_y-3
        chip=Rectangle((124/180,(label_y-.7)/240),1.4/180,1.4/240,
                        transform=f.transFigure,facecolor=colors[i],edgecolor='none')
        f.add_artist(chip)
        lab(f,126.8,label_y,row.label,7)
        lab(f,176,label_y,f'{int(row.tracts):,} · {row.percent:.2f}%',7,ha='right')
        patch=ba.barh(bar_y-154,row.percent,height=1.5,color=colors[i],edgecolor='none')[0]
        patch.set_gid('national_share_'+row.category);eq(patch.get_width(),row.percent)
        rec['bars'].append(dict(kind='national',category=row.category,percent=float(row.percent),tracts=int(row.tracts)))

    title(f,4,144,'c','Canopy change, 2000–2025')
    title(f,92,144,'d','Predicted response')
    map_panel(f,4,85,84,'c','change_pp',CHANGE_CMAP,CHANGE_NORM,p,states,outline,rec)
    map_panel(f,92,85,84,'d','theta',cart.CMAP,cart.NORM,p,states,outline,rec)
    colorbar(f,8,76,CHANGE_CMAP,CHANGE_NORM,[-45,-30,-15,0,15,30,45],
             ['−45','−30','−15','0','15','30','45'],'Canopy change (pp)','canopy_change_colourbar')
    colorbar(f,96,76,cart.CMAP,cart.NORM,[-.5,-.4,-.3,-.2,-.1,0,.05],
             ['−0.5','−0.4','−0.3','−0.2','−0.1','0','0.05'],'Response (°C per pp)','response_colourbar')

    lab(f,4,62,'Group composition',8,weight='bold')
    for j,(dim,heading) in enumerate(DIMS):
        x=4+59*j;letter='efg'[j];title(f,x,55,letter,heading)
        a=axmm(f,x+7,15,33,32,'panel_'+letter)
        a.set(xlim=(0,100),ylim=(.5,4.5),xticks=[0,50,100],yticks=[])
        a.spines[['left','right','top']].set_visible(False);a.spines['bottom'].set_color('#929DA1')
        a.tick_params(axis='x',labelsize=7,length=2,width=.45,pad=2,colors=INK)
        lab(f,x+50,49.5,'n',7,ha='center')
        for q in [1,2,3,4]:
            z=social[social.variable.eq(dim)&social.quartile.eq(q)].set_index('category').loc[codes]
            left=0.;y=5-q
            for i,code in enumerate(codes):
                value=float(z.loc[code,'percent'])
                patch=a.barh(y,value,left=left,height=.48,color=colors[i],edgecolor='none')[0]
                patch.set_gid(f'{dim}_Q{q}_{code}');eq(patch.get_width(),value)
                rec['bars'].append(dict(kind='social',variable=dim,quartile=q,category=code,
                                        left=left,percent=value,tracts=int(z.loc[code,'tracts']),
                                        denominator=int(z.loc[code,'denominator'])))
                left+=value
            eq(left,100)
            a.add_patch(Rectangle((0,y-.24),100,.48,facecolor='none',edgecolor='#AAB4B7',linewidth=.3))
            yy=15+(y-.5)/4*32
            lab(f,x+.2,yy,f'Q{q}',7);lab(f,x+50,yy,f'{int(z.denominator.iloc[0]):,}',7,ha='center')
    lab(f,90,6.5,'Share of tracts (%)',7.5,ha='center')
    rec.update(size_mm=list(MM),point_area_pt2=.55,draw_order='state, tract_id',
               category_order=codes,category_colors=colors,net_change_limits=[-45,45],
               response_limits=[-.55,.05],continuous_normalization='linear',
               change_color_nodes=['#976DB3','#E2D6EA','#E3E4E0','#CEE8E3','#3F9593'],
               national_share_axis=[0,30],social_share_axis=[0,100])
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
    layout=dict(minimum_font_pt=min(t.get_fontsize() for t,b in texts),text_count=len(texts),clipped=clipped,overlapping=over)
    dump(OUT/'qa/layout_check.json',layout)
    assert not clipped,clipped
    assert not over,over
    assert layout['minimum_font_pt']>=7
    for ext in ['pdf','svg','png']:f.savefig(OUT/'exports'/f'{NAME}.{ext}',dpi=600,facecolor='white')
    with pymupdf.open(OUT/'exports'/f'{NAME}.pdf') as doc:
        doc[0].get_pixmap(matrix=pymupdf.Matrix(1.8,1.8)).save(OUT/'qa'/f'{NAME}_preview.png')
    plt.close(f)
    for source in rec['source_files'].values():assert sha(source['path'])==source['sha256']
    rec.update(status='PASS',layout=layout,png_dpi=600,
               exports={ext:{'path':str(OUT/'exports'/f'{NAME}.{ext}'),'sha256':sha(OUT/'exports'/f'{NAME}.{ext}')}
                        for ext in ['pdf','svg','png']})
    dump(OUT/'qa/drawing_receipt.json',rec)
    print(json.dumps({'status':'PASS','points_per_map':42369,'maps':3,'social_bars':12,'layout':layout},ensure_ascii=False))

if __name__=='__main__':
    p,ns,social,rules,states,outline,receipt=prepare()
    f=draw(p,ns,social,rules,states,outline,receipt)
    export(f,receipt)
