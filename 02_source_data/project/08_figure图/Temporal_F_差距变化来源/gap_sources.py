"""Descriptive endpoint decomposition; read-only study sources, deterministic labels."""
from pathlib import Path
import json, hashlib, math, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
import pymupdf
from PIL import Image, ImageOps

O=Path(__file__).resolve().parent; F=O.parent
B=F/'Temporal_AB_全国与45城年度轨迹/TemporalB2_45城社会差距/source_data'
SOURCES={'decomposition':B/'endpoint_gap_decomposition.csv',
 'annual':B/'city45_quartile_annual.csv',
 'pathways':F/'Temporal_CD_小区图集与三期热环境/AB_review_and_gap_decomposition/source_data/city45_gap_pathways.csv',
 'roster':F/'Fig04_全国响应与估计对象比较/revision03_atlas45/source_data/city_index.csv'}
VARS=['income','poverty','age65plus']; TITLES=['Income','Poverty','Population aged 65+']
COLORS=['#78AFA1','#BF938E','#8FAAC4','#B7A1C2']
FILLS=['#F0F6F2','#FAF1ED','#EFF4FA','#F5F0F7']
INK='#34414A'; NAME='TemporalF_city45_gap_change_sources'

def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return pd.read_csv(p,dtype={'single_city_id':str},float_precision='round_trip')
def close(a,b): np.testing.assert_allclose(a,b,rtol=0,atol=1e-10)
def pathway(x,y):
    if x>0 and y>0:return 0
    if x<0 and y<0:return 1
    if x<0 and y>0:return 2
    if x>0 and y<0:return 3
    return 4

def prepare():
    for sub in ['source_data','exports','qa','schemas']:(O/sub).mkdir(exist_ok=True,parents=True)
    hashes={str(p):sha(p) for p in SOURCES.values()}
    d=read(SOURCES['decomposition']); a=read(SOURCES['annual']); p=read(SOURCES['pathways']); r=read(SOURCES['roster']).sort_values('city_number')
    key=['single_city_id','variable']
    assert len(d)==135 and not d.duplicated(key).any()
    assert len(r)==45 and r.city_number.tolist()==list(range(1,46)) and r.single_city_id.is_unique
    assert set(d.variable)==set(VARS)
    assert not a.duplicated(key+['year','quartile']).any()
    assert set(a.year)==set(range(2000,2026)) and set(a.quartile)=={1,2,3,4}
    assert len(a)==45*3*26*4
    assert a.groupby(key+['quartile']).n.nunique().eq(1).all()
    counts=a.groupby(['variable','year']).n.sum(); assert counts.eq(24493).all()
    vals=a.pivot(index=key,columns=['year','quartile'],values='mean')
    nvals=a.pivot(index=key,columns=['year','quartile'],values='n')
    check=d.set_index(key).sort_index(); vals=vals.loc[check.index]
    close(check.low_group_change_pp,vals[(2025,1)]-vals[(2000,1)])
    close(check.high_group_change_pp,vals[(2025,4)]-vals[(2000,4)])
    close(check.gap_change_pp,check.high_group_change_pp-check.low_group_change_pp)
    for var in VARS:
        sub=d[d.variable.eq(var)].sort_values('city_number')
        assert sub.city_number.tolist()==r.city_number.tolist()
        assert sub.single_city_id.tolist()==r.single_city_id.tolist()
        assert sub.short_name.tolist()==r.short_name.tolist()
    pc=p.set_index(key).loc[check.index]
    for c in ['low_group_change_pp','high_group_change_pp','gap_change_pp']:close(check[c],pc[c])
    for c,idx in [('low_group_2000',(2000,1)),('low_group_2025',(2025,1)),('high_group_2000',(2000,4)),('high_group_2025',(2025,4))]:close(pc[c],vals[idx])
    out=check.copy()
    for c in ['low_group_2000','low_group_2025','high_group_2000','high_group_2025','gap_2000_pp','gap_2025_pp','absolute_gap_change_pp','gap_sign_reversal']:out[c]=pc[c]
    out['lowest_group_n']=nvals.loc[out.index,(2000,1)].astype(int)
    out['highest_group_n']=nvals.loc[out.index,(2000,4)].astype(int)
    out['quadrant_code']=[pathway(x,y) for x,y in zip(out.low_group_change_pp,out.high_group_change_pp)]
    out=out.reset_index().sort_values(['variable','city_number'])
    assert np.isfinite(out[['low_group_change_pp','high_group_change_pp']]).all().all()
    assert out[['low_group_change_pp','high_group_change_pp']].abs().max().max()<8
    out.to_csv(O/'source_data/city45_gap_change_sources.csv',index=False,encoding='utf-8-sig')
    r[['city_number','single_city_id','short_name','division']].to_csv(O/'source_data/city_key.csv',index=False,encoding='utf-8-sig')
    ct=out.groupby(['variable','quadrant_code']).size().rename('cities').reset_index()
    ct.to_csv(O/'source_data/quadrant_counts.csv',index=False,encoding='utf-8-sig')
    dump(O/'qa/data_validation.json',{'status':'PASS','input_rows':len(d),'output_rows':len(out),'unique_city_variable_pairs':len(out),'cities_per_dimension':45,'fixed_display_tracts':24493,'annual_rows_checked':len(a),'duplicate_pairs':0,'unmatched_cities':0,'missing_coordinates':0,'dropped_rows':0,'endpoint_recomputation_tolerance_pp':1e-10,'maximum_gap_identity_error':float((out.high_group_change_pp-out.low_group_change_pp-out.gap_change_pp).abs().max()),'source_hashes':hashes})
    return out,r

def boxes_overlap(a,b):return a.overlaps(b)
def segment_cross(a,b,c,d):
    def ccw(p,q,r):
        u=q-p;v=r-p;return u[0]*v[1]-u[1]*v[0]
    return ccw(a,b,c)*ccw(a,b,d)<0 and ccw(c,d,a)*ccw(c,d,b)<0

def place_labels(ax,data,fig):
    """Move annotation boxes only; point coordinates never change. No randomness."""
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    xy=ax.transData.transform(data[['low_group_change_pp','high_group_change_pp']].to_numpy())
    pt=fig.dpi/72
    texts=[ax.text(0,0,f'{int(n):02}',fontsize=6,color=INK,ha='center',va='center',zorder=6) for n in data.city_number]
    dims=np.array([[t.get_window_extent(renderer).width+1.0*pt,t.get_window_extent(renderer).height+0.6*pt] for t in texts])
    bounds=ax.get_window_extent(renderer).padded(-1.8*pt)
    dist=np.sqrt(((xy[:,None]-xy[None,:])**2).sum(axis=2)); density=(dist<20*pt).sum(axis=1)
    order=np.lexsort((data.city_number.to_numpy(),-density))
    chosen={}; boxes={}; leaders=[]
    angles=np.deg2rad(np.arange(0,360,15))
    for i in order:
        w,h=dims[i]; best=None
        for radius in np.arange(5.5,73,2.5):
            for angle in angles:
                c=xy[i]+radius*pt*np.array([np.cos(angle),np.sin(angle)])
                box=Bbox.from_bounds(c[0]-w/2,c[1]-h/2,w,h)
                if not (bounds.contains(*box.p0) and bounds.contains(*box.p1)):continue
                if any(box.overlaps(b) for b in boxes.values()):continue
                # No annotation may cover any measured point, including its own.
                if any(box.padded(1.4*pt).contains(*p) for p in xy):continue
                crosses=sum(segment_cross(xy[i],c,u,v) for u,v in leaders)
                # Short leader preferred; crossing penalized without changing data.
                score=radius+8*crosses+0.002*abs(angle-np.pi/4)
                if best is None or score<best[0]:best=(score,c,box)
            if best is not None and radius>best[0]+3:break
        if best is None:raise RuntimeError(f'No collision-free annotation position for city {data.iloc[i].city_number}')
        _,c,box=best; chosen[i]=c; boxes[i]=box
        leaders.append((xy[i],c))
    receipts=[]
    for i,t in enumerate(texts):
        c=chosen[i]; pos=ax.transData.inverted().transform(c); t.set_position(pos)
        delta=c-xy[i]; half=dims[i]/2
        factor=min(half[0]/max(abs(delta[0]),1e-9),half[1]/max(abs(delta[1]),1e-9))
        end=c-delta*min(factor,1)
        start=xy[i]+delta/np.linalg.norm(delta)*1.5*pt
        line=ax.transData.inverted().transform(np.vstack([start,end]))
        ax.plot(line[:,0],line[:,1],lw=.35,color='#929A9F',zorder=3,solid_capstyle='round')
        receipts.append({'variable':data.iloc[i].variable,'city_number':int(data.iloc[i].city_number),'point_x':float(data.iloc[i].low_group_change_pp),'point_y':float(data.iloc[i].high_group_change_pp),'label_x':float(pos[0]),'label_y':float(pos[1]),'leader_length_points':float(np.linalg.norm(delta)/pt)})
    fig.canvas.draw()
    actual=[t.get_window_extent(renderer).padded(.15*pt) for t in texts]
    assert not any(actual[i].overlaps(actual[j]) for i in range(len(actual)) for j in range(i))
    assert all(bounds.contains(*b.p0) and bounds.contains(*b.p1) for b in actual)
    return receipts,texts

def render(d,r):
    font=font_manager.findfont(font_manager.FontProperties(family='Arial'),fallback_to_default=False)
    plt.rcParams.update({'font.family':'Arial','font.size':7,'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.45,'axes.edgecolor':'#9BA3A8','axes.labelcolor':INK,'text.color':INK,'xtick.color':INK,'ytick.color':INK,'savefig.dpi':600,'axes.unicode_minus':True})
    W,H=180,151; fig=plt.figure(figsize=(W/25.4,H/25.4),dpi=150,facecolor='white')
    axes=[]; receipt=[]; all_text=[]
    for j,var in enumerate(VARS):
        x0=13+j*56; ax=fig.add_axes([x0/W,87/H,49/W,49/H]);axes.append(ax)
        ax.set(xlim=(-8,8),ylim=(-8,8),aspect='equal',xticks=[-8,-4,0,4,8],yticks=[-8,-4,0,4,8])
        ax.tick_params(length=2,width=.45,pad=2,labelsize=6)
        for (x,y),fill in zip([(0,0),(-8,-8),(-8,0),(0,-8)],FILLS):ax.add_patch(Rectangle((x,y),8,8,fc=fill,ec='none',zorder=0))
        ax.axhline(0,c='#A1A7AB',lw=.55,zorder=1);ax.axvline(0,c='#A1A7AB',lw=.55,zorder=1)
        ax.plot([-8,8],[-8,8],c='#68747C',lw=.65,ls=(0,(3,2)),zorder=1)
        sub=d[d.variable.eq(var)].sort_values('city_number')
        collection=ax.scatter(sub.low_group_change_pp,sub.high_group_change_pp,s=10,c=[COLORS[int(q)] if q<4 else INK for q in sub.quadrant_code],edgecolors='white',linewidths=.28,zorder=4)
        close(collection.get_offsets(),sub[['low_group_change_pp','high_group_change_pp']].to_numpy())
        rec,txt=place_labels(ax,sub,fig);receipt.extend(rec);all_text.extend(txt)
        fig.text((x0-4)/W,140/H,chr(97+j),fontweight='bold',fontsize=9,va='center')
        fig.text((x0+24.5)/W,140/H,TITLES[j],ha='center',va='center',fontsize=9)
    fig.text(3/W,111.5/H,'Highest quartile: canopy change (percentage points)',ha='center',va='center',rotation=90,fontsize=7)
    fig.text(93.5/W,78/H,'Lowest quartile: canopy change (percentage points)',ha='center',va='center',fontsize=7)
    fig.text(93.5/W,73.5/H,'2000–2025',ha='center',va='center',fontsize=7)
    legend=[Line2D([],[],marker='o',ls='none',color=c,markersize=4,label=l) for c,l in zip(COLORS,['Both gain','Both lose','Lowest loses; highest gains','Lowest gains; highest loses'])]
    fig.legend(handles=legend,loc='center',bbox_to_anchor=(.52,66/H),ncol=2,frameon=False,fontsize=6.5,handletextpad=.5,columnspacing=3,labelspacing=.6)
    fig.text(13/W,55.5/H,'City numbers follow Figures 4 and 6',fontsize=6.5,color='#626E75')
    fig.lines.append(Line2D([13/W,174/W],[53/H,53/H],transform=fig.transFigure,lw=.4,color='#C3C9CC'))
    for i,row in enumerate(r.itertuples()):
        col=i//15; line=i%15
        fig.text((13+col*56)/W,(49.5-line*3.1)/H,f'{int(row.city_number):02}',fontsize=6.2,color=INK,va='center')
        fig.text((19+col*56)/W,(49.5-line*3.1)/H,row.short_name,fontsize=6.2,va='center')
    fig.canvas.draw()
    dump(O/'qa/label_positions.json',receipt)
    pd.DataFrame(receipt).to_csv(O/'source_data/point_and_annotation_coordinates.csv',index=False,encoding='utf-8-sig')
    for suffix in ['pdf','svg']:fig.savefig(O/'exports'/f'{NAME}.{suffix}',dpi=600,facecolor='white')
    fig.savefig(O/'exports'/f'{NAME}.png',dpi=600,facecolor='white')
    fig.savefig(O/'qa/preview.png',dpi=200,facecolor='white')
    fig.savefig(O/'qa/final_size_150dpi.png',dpi=150,facecolor='white')
    plt.close(fig)
    img=Image.open(O/'qa/preview.png').convert('RGB');ImageOps.grayscale(img).save(O/'qa/grayscale.png')
    img.thumbnail((800,800));img.save(O/'qa/thumbnail.png')
    dump(O/'qa/render_receipt.json',{'status':'PASS','panel_points':[45]*3,'point_coordinate_validation':'PASS','number_labels':len(receipt),'label_label_overlaps':0,'label_point_overlaps':0,'labels_inside_axes':True,'label_placement':'deterministic annotation-only placement with leaders; measured points unchanged','axis_limits_all_panels':[-8,8],'aspect_ratio_all_panels':1,'width_mm':W,'height_mm':H,'font_path':font,'minimum_font_pt':6,'dpi':600})

def validate():
    before=json.loads((O/'qa/data_validation.json').read_text(encoding='utf8'))['source_hashes']
    assert all(sha(Path(p))==h for p,h in before.items())
    path=O/'exports'/f'{NAME}.pdf'; doc=pymupdf.open(path);assert len(doc)==1
    # PDF page boxes use finite-precision points; physical size tolerance 0.001 mm.
    # The independent scientific-value tolerance remains 1e-10 percentage points.
    page=doc[0]; np.testing.assert_allclose([page.rect.width*25.4/72,page.rect.height*25.4/72],[180,151],rtol=0,atol=.001)
    spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for line in b['lines'] for s in line['spans'] if s['text'].strip()]
    assert all('Arial' in s['font'] for s in spans)
    assert min(s['size'] for s in spans)>=5.99
    clipped=[s['text'] for s in spans if not page.rect.contains(pymupdf.Rect(s['bbox']))];assert not clipped,clipped
    # Actual exported text boxes, excluding intersections of separate lines that do not share area.
    overlaps=[]
    for i,a in enumerate(spans):
        ra=pymupdf.Rect(a['bbox'])
        for b in spans[:i]:
            rb=pymupdf.Rect(b['bbox']);inter=ra&rb
            if not inter.is_empty and inter.width>.5 and inter.height>.5:overlaps.append([a['text'],b['text']])
    dump(O/'qa/export_text_intersections.json',overlaps)
    assert not overlaps,overlaps
    assert not page.get_images()
    page.get_pixmap(matrix=pymupdf.Matrix(2.5,2.5)).save(O/'qa/pdf_preview.png')
    svg=(O/'exports'/f'{NAME}.svg').read_text(encoding='utf8');assert '<text' in svg and '<image' not in svg
    dump(O/'qa/export_validation.json',{'status':'PASS','source_hashes_unchanged':True,'pdf_pages':1,'pdf_text_intersections':len(overlaps),'clipped_text':len(clipped),'font':'Arial','vector_pdf':True,'live_svg_text':True,'pdf_sha256':sha(path),'svg_sha256':sha(O/'exports'/f'{NAME}.svg')})

if __name__=='__main__':
    data,roster=prepare();render(data,roster);validate();print('PASS: 135 city-dimension points; 45 fixed city numbers; endpoint reconstruction; vector exports.')
