"""Display-only Figure 1, frozen sources with tighter display-only city windows."""
from nc_paths import project_root as _nc_project_root, external_path as _nc_external_path
from pathlib import Path
import json, hashlib, sys, platform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.transforms import Affine2D
import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer, Geod
from shapely.affinity import affine_transform
from shapely.geometry import box
from scipy.optimize import brentq

ROOT=_nc_project_root()
OUT=Path(__file__).resolve().parent
SRC=_nc_project_root() / '08_figure图/Fig01_全国主分析样本空间分布'
F4=ROOT/"08_figure图/Fig04_全国响应与估计对象比较/revision03_atlas45"
E=ROOT/"08_figure图/Temporal_E_树冠与温度共同演变/source_data"
WINDOW=ROOT/"08_figure图/Fig06_资源增量与条件响应对应/revision03_atlas45/qa/cartography_redraw/north_up_windows.json"
MANIFEST=ROOT/"06_logs/geo_adjusted_national_orthogonal_20261009_run01/sample_manifest.csv"
CASES=[(6,"63217","New York–Newark"),(10,"75664","Rochester, NY"),(11,"16264","Chicago"),(20,"23743","Des Moines"),(24,"56602","Miami"),(28,"56116","Memphis"),(36,"69184","Phoenix–Mesa"),(43,"80389","Seattle"),(41,"51445","Los Angeles")]
INK="#354C49"; FRAME="#C3CBC8"; WATER="#EDF4F5"; POINT="#88ADA6"
PAGE=(180,270)
CANOPY_COLORS=["#ECF0D3","#BCDCC0","#8BC5B0","#59A99A","#377F79"]
TEMP_COLORS=["#88BACD","#BDDAE2","#F0E7D1","#E6B69E","#C77F73"]
CMAPS=[LinearSegmentedColormap.from_list("canopy",CANOPY_COLORS,256),LinearSegmentedColormap.from_list("temperature",TEMP_COLORS,256)]
NORMS=[Normalize(0,100),Normalize(10,60)]
TICKS=[[0,20,40,60,80,100],[10,20,30,40,50,60]]
GEO=Transformer.from_crs(5070,4326,always_xy=True)
PROJ=Transformer.from_crs(4326,5070,always_xy=True)
GEOD=Geod(ellps="GRS80")

def require(x,msg):
    if not x: raise ValueError(msg)
def sha(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding="utf-8")
def mm(fig,x,top,w,h):return fig.add_axes([x/180,1-(top+h)/270,w/180,h/270])
def label(fig,x,top,s,**kw):return fig.text(x/180,1-top/270,s,va="top",color=INK,**kw)
def north(ax,xy=(.08,.88),dy=.12):
    ax.annotate("",xy=xy,xytext=(xy[0],xy[1]-dy),xycoords="axes fraction",arrowprops=dict(arrowstyle="-|>",lw=.6,color=INK,mutation_scale=7))
    ax.text(xy[0],xy[1]+.025,"N",transform=ax.transAxes,ha="center",va="bottom",fontsize=6,color=INK)

def scale(fig,ax,meters,matrix=None,footer=True):
    pos=ax.get_position(); xmin,xmax=ax.get_xlim(); ymin,ymax=ax.get_ylim()
    inv=np.linalg.inv(np.eye(3) if matrix is None else matrix)
    # Scale is located along the lower map edge; its real endpoints define the distance.
    x=xmin+.05*(xmax-xmin); y=ymin if footer else ymin+.09*(ymax-ymin)
    def dist(dx):
        a=inv@np.array([x,y,1]); b=inv@np.array([x+dx,y,1])
        return GEOD.inv(*GEO.transform(*a[:2]),*GEO.transform(*b[:2]))[2]
    dx=brentq(lambda d:dist(d)-meters,meters*.5,meters*2,xtol=1e-6)
    require(x+dx<xmax,"Scale exceeds map")
    fx=pos.x0+.05*pos.width; fy=pos.y0-.55/270 if footer else pos.y0+.09*pos.height
    ex=fx+dx/(xmax-xmin)*pos.width
    fig.add_artist(Line2D([fx,ex],[fy,fy],transform=fig.transFigure,lw=.7,color=INK))
    for xx in (fx,ex):fig.add_artist(Line2D([xx,xx],[fy-.30/270,fy+.30/270],transform=fig.transFigure,lw=.5,color=INK))
    fig.text(ex+1/180,fy,f"{meters/1000:g} km",va="center",ha="left",fontsize=6,color=INK)
    return {"ground_m":meters,"verified_m":dist(dx),"display_dx_m":dx,"display_reference_y":y,"inverse_display_transform":inv.tolist()}

def render():
    require(Path(sys.executable).parent.name=="sportpark-ee","Use sportpark-ee")
    for n in ("exports","qa","source_data"):(OUT/n).mkdir(exist_ok=True)
    paths=[MANIFEST,SRC/"source_data/figure01_display_points.csv",E/"paired_tract_panel.csv",E/"city45_support.csv",WINDOW]
    paths += [SRC/"raw_basemap"/n for n in ("cb_2024_us_state_500k.zip","ne_50m_land.zip","ne_10m_lakes.zip")]
    paths += [F4/f"prepared/city_{n:02d}_{cid}.gpkg" for n,cid,_ in CASES]
    # Protect existing original figure exports as well as numerical and geometric sources.
    old=list((SRC/"local_details_4/publication_20261014/exports").glob("*"))
    old += list((SRC/"revision05_national_hero_city_pairs_20261019/exports").glob("*"))
    old += list((SRC/"revision06_city_zoom_20261019/exports").glob("*"))
    before={str(p):sha(p) for p in paths+old if p.is_file()}
    dtypes={"state":str,"tract_id":str,"single_city_id":str}
    manifest=pd.read_csv(MANIFEST,dtype=dtypes)
    points=pd.read_csv(SRC/"source_data/figure01_display_points.csv",dtype=dtypes)
    require(len(manifest)==104371 and manifest.tract_id.nunique()==42373,"Main sample differs")
    require(len(points)==42373 and points.tract_id.is_unique,"Point count differs")
    require(set(points.tract_id)==set(manifest.tract_id),"Point identities differ")
    pair=pd.read_csv(E/"paired_tract_panel.csv",dtype=dtypes)
    require(len(pair)==79431 and pair.tract_id.nunique()==26477,"Three-period sample differs")
    require(not pair.duplicated(["tract_id","outcome_year"]).any(),"Duplicate tract-years")
    require(set(zip(pair.exposure_year,pair.outcome_year))=={(2013,2014),(2018,2019),(2023,2024)},"Year pairing differs")
    p=pair.loc[pair.outcome_year.eq(2024)].copy()
    # Nullable numeric city code -> five-character Census code, without replacing absent IDs.
    codes=pd.to_numeric(p.single_city_id,errors="raise")
    require((codes.dropna()%1==0).all(),"Non-integral city ID")
    p["city_id"]=codes.astype("Int64").astype("string").str.zfill(5)
    require(p.tract_id.is_unique and len(p)==26477,"Final pair duplicates")
    require(p[["urban_canopy_percent_fixed_land","weather_adjusted_lst_reference_mean"]].notna().all().all(),"Missing values")
    support=pd.read_csv(E/"city45_support.csv").set_index("city_number")
    windows={w["city_number"]:w for w in json.loads(WINDOW.read_text(encoding="utf8"))}
    city={}; all_source=[]; rows=[]
    for n,cid,name in CASES:
        print("Prepare",n,name,flush=True)
        f=F4/f"prepared/city_{n:02d}_{cid}.gpkg"; g=gpd.read_file(f,layer="sample_parts")
        require(g.crs.to_epsg()==5070 and g.geometry.is_valid.all(),"Source geometry/CRS invalid")
        z=p.loc[p.city_id.eq(cid)].copy()
        require(len(z)==int(support.loc[n,"paired_tracts"]),f"City {n}: support mismatch")
        require(set(z.tract_id)<=set(g.tract_id),f"City {n}: unmatched pair IDs")
        g=g.merge(z[["tract_id","urban_canopy_percent_fixed_land","weather_adjusted_lst_reference_mean"]],on="tract_id",how="left",validate="many_to_one")
        active=g.urban_canopy_percent_fixed_land.notna()
        require(g.loc[active].tract_id.nunique()==len(z),"Missing geometry after merge")
        w=windows[n]; M=np.asarray(w["display_rotation_matrix"])
        require(np.allclose(M[:2,:2].T@M[:2,:2],np.eye(2),atol=1e-12),"Non-rigid display matrix")
        lon,lat=w["north_reference_lon_lat"]; a=np.array(PROJ.transform(lon,lat)); lo,la,_=GEOD.fwd(lon,lat,0,100)
        v=np.array(PROJ.transform(lo,la))-a; v/=np.linalg.norm(v)
        require(np.allclose(M[:2,:2]@v,[0,1],atol=1e-7),"North alignment failed")
        original_areas=g.geometry.area.to_numpy()
        g.geometry=g.geometry.map(lambda x:affine_transform(x,[M[0,0],M[0,1],M[1,0],M[1,1],M[0,2],M[1,2]]))
        require(g.geometry.is_valid.all(),"Display geometry invalid; no repair applied")
        require(np.allclose(original_areas,g.geometry.area,rtol=1e-9,atol=.01),"Rotation changed area")
        old_bounds=list(w["bounds_epsg5070_m"])
        zoom=1.0
        if n in {6,11,24,36,43,41}:
            # Fit every displayed part, including unpaired context. Preserve 1.6 aspect,
            # at least 3% padding per side, and cap enlargement at 35%.
            gb=g.total_bounds
            old_height=old_bounds[3]-old_bounds[1]
            height=max(max((gb[2]-gb[0])/1.6,gb[3]-gb[1])*1.06,old_height/1.35)
            center=(gb[:2]+gb[2:])/2
            half=np.array([height*1.6,height])/2
            w=dict(w)
            w["bounds_epsg5070_m"]=np.r_[center-half,center+half].tolist()
            zoom=old_height/height
        b=w["bounds_epsg5070_m"]; require(g.geometry.covered_by(box(*b)).all(),f"City {n}: cropped geometry")
        z["city_number"]=n; all_source.append(z)
        city[n]=(g,w,active)
        rows.append({"city_number":n,"city_id":cid,"name":name,"paired_tracts":len(z),"main_tracts":g.tract_id.nunique(),"paired_parts":int(active.sum()),"context_parts":int((~active).sum()),"unmatched":0,"bounds":b,"previous_bounds":old_bounds,"linear_zoom":zoom,"all_parts_within_window":True,"display_rotation_matrix":M.tolist(),"north_reference_lon_lat":[lon,lat],"north_vector_after_rotation":(M[:2,:2]@v).tolist(),"holes_retained":int(sum(len(poly.interiors) for geom in g.geometry for poly in ([geom] if geom.geom_type=="Polygon" else geom.geoms)))})
    local=pd.concat(all_source,ignore_index=True)
    require(len(local)==9242 and local.tract_id.is_unique,"Nine-city count differs")
    require(local.urban_canopy_percent_fixed_land.between(0,100).all(),"Canopy outside colour limits")
    require(local.weather_adjusted_lst_reference_mean.between(10,60).all(),"Temperature outside colour limits")
    local.to_csv(OUT/"source_data/paired_2023_2024_display.csv",index=False)
    points.to_csv(OUT/"source_data/national_display_points.csv",index=False)
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,list)} for r in rows]).to_csv(OUT/"source_data/city_support.csv",index=False)
    states=gpd.read_file("zip://"+(SRC/"raw_basemap/cb_2024_us_state_500k.zip").as_posix())
    states=states.loc[~states.STATEFP.isin(["02","15","60","66","69","72","78"])].to_crs(5070)
    require(len(states)==49 and states.geometry.is_valid.all(),"CONUS state coverage differs")
    lakes=gpd.read_file("zip://"+(SRC/"raw_basemap/ne_10m_lakes.zip").as_posix(),bbox=(-130,20,-60,55)).to_crs(5070)
    font=font_manager.findfont("Arial",fallback_to_default=False)
    plt.rcParams.update({"font.family":"Arial","font.size":7,"pdf.fonttype":42,"svg.fonttype":"none","savefig.dpi":600,"axes.unicode_minus":False})
    fig=plt.figure(figsize=(180/25.4,270/25.4),dpi=150,facecolor="white")
    ax=mm(fig,5,9,170,94)
    ax.set_facecolor("white")
    states.plot(ax=ax,color="#FAF7F0",edgecolor="#AEBAB6",linewidth=.30,zorder=1)
    lakes.plot(ax=ax,color="white",edgecolor="none",linewidth=0,zorder=2)
    sc=ax.scatter(points.display_x_epsg5070_m,points.display_y_epsg5070_m,s=.58,c=POINT,alpha=.75,edgecolors="none",zorder=4)
    sc.set_gid("national_42373_tracts")
    xy=points[["display_x_epsg5070_m","display_y_epsg5070_m"]].to_numpy()
    lo=np.minimum(states.total_bounds[:2],xy.min(axis=0)); hi=np.maximum(states.total_bounds[2:],xy.max(axis=0))
    extent=np.r_[lo-(hi-lo)*.025,hi+(hi-lo)*.025]
    ax.set_xlim(extent[0],extent[2]); ax.set_ylim(extent[1],extent[3]);ax.set_aspect("equal");ax.set_xticks([]);ax.set_yticks([])
    for s in ax.spines.values():s.set_visible(False)
    label(fig,5,3,"a  National temperature-analysis sample",fontsize=9,fontweight="bold")
    # National reference north at the Albers central meridian, where true north is vertical.
    xx,yy=PROJ.transform(-96,47)
    ax.annotate("N",xy=(xx,yy),xytext=(xx,yy-210000),ha="center",fontsize=7,color=INK,arrowprops=dict(arrowstyle="-|>",lw=.65,color=INK))
    scales=[{"national":scale(fig,ax,500000,footer=False)}]
    locators=[]
    offsets={6:(6,-5),10:(-6,7),11:(5,5),20:(-5,-7),24:(6,-3),28:(5,6),36:(-6,-6),43:(-7,6),41:(-8,-6)}
    for n,cid,name in CASES:
        u=gpd.read_file(F4/f"prepared/city_{n:02d}_{cid}.gpkg",layer="urban_2000")
        x,y=PROJ.transform(float(u.iloc[0].INTPTLON00),float(u.iloc[0].INTPTLAT00))
        ax.scatter([x],[y],s=15,facecolors="none",edgecolors=INK,linewidths=.7,zorder=7)
        dx,dy=offsets[n]
        ax.annotate(f"{n:02d}",(x,y),xytext=(dx,dy),textcoords="offset points",fontsize=6.7,fontweight="bold",color=INK,ha="left" if dx>0 else "right",zorder=8)
        locators.append({"city_number":n,"xy":[x,y],"source":"Census urban_2000 INTPTLON00/INTPTLAT00","label_offset_pt":[dx,dy]})
    # Shared column titles precede all city modules. No explanatory subtitle rows.
    for x in (5,92):
        label(fig,x+20.55,105,"Canopy 2023",ha="center",fontsize=7.1)
        label(fig,x+62.45,105,"Adjusted LST 2024",ha="center",fontsize=7.1)
    mapw=41.1; maph=mapw/1.6
    for k,(n,cid,name) in enumerate(CASES):
        col=k%2; row=k//2; x=5+87*col; top=113+31.4*row
        label(fig,x,top-3.3,f"{chr(98+k)}  {n:02d}  {name}",fontsize=7.3,fontweight="bold")
        g,w,active=city[n]
        for j,var in enumerate(["urban_canopy_percent_fixed_land","weather_adjusted_lst_reference_mean"]):
            axc=mm(fig,x+j*(mapw+.8),top,mapw,maph)
            if (~active).any():
                g.loc[~active].plot(ax=axc,color="#E2E6E4",edgecolor="none",zorder=1)
                axc.collections[-1].set_gid(f"context_{n:02d}_{j}")
            g.loc[active].plot(ax=axc,column=var,cmap=CMAPS[j],norm=NORMS[j],edgecolor="none",linewidth=0,zorder=2)
            axc.collections[-1].set_gid(f"paired_{n:02d}_{j}")
            b=w["bounds_epsg5070_m"];axc.set_xlim(b[0],b[2]);axc.set_ylim(b[1],b[3]);axc.set_aspect("equal")
            axc.set_xticks([]);axc.set_yticks([])
            for s in axc.spines.values():s.set_edgecolor(FRAME);s.set_linewidth(.35)
            north(axc)
            if j==0:scales.append({"city_number":n,**scale(fig,axc,w["scale_distance_m"],np.array(w["display_rotation_matrix"]))})
    # Last module on the right is exclusively the common legend.
    for j,(title,top) in enumerate([("Canopy cover (%)",237.6),("Adjusted LST (°C)",251.6)]):
        label(fig,94,top,title,fontsize=7.1)
        cax=mm(fig,94,top+3.6,79,2.5)
        cb=fig.colorbar(matplotlib.cm.ScalarMappable(norm=NORMS[j],cmap=CMAPS[j]),cax=cax,orientation="horizontal",ticks=TICKS[j])
        cb.outline.set_linewidth(.3);cb.outline.set_edgecolor(FRAME);cb.ax.tick_params(labelsize=6.3,length=1.7,width=.4,pad=1)
        cb.solids.set_rasterized(False)
        cb.solids.set_edgecolor("face")
        for val in TICKS[j][1:-1]:cb.ax.axvline(val,color="white",lw=.55)
    legend_ax=mm(fig,94,264.0,80,4);legend_ax.set_axis_off()
    handles=[Line2D([],[],marker="o",ls="none",ms=3,color=POINT,label="Main sample"),Line2D([],[],marker="o",ls="none",ms=3,mfc="none",mec=INK,label="Display case"),Line2D([],[],marker="s",ls="none",ms=3,color="#E2E6E4",label="Outside three-period\nsample")]
    legend=legend_ax.legend(handles=handles,loc="center left",bbox_to_anchor=(0,0.5),ncol=3,frameon=False,fontsize=6,handlelength=.7,handletextpad=.3,columnspacing=.8,borderaxespad=0)
    for text in legend.get_texts():text.set_linespacing(1.4)
    fig.canvas.draw()
    print("Export",flush=True)
    for ext in ("pdf","svg","png"):
        fig.savefig(OUT/f"exports/Figure01_national_coverage_city_pairs.{ext}",dpi=600,facecolor="white")
    fig.savefig(OUT/"qa/Figure01_preview_180mm.png",dpi=190,facecolor="white")
    require(all(sha(Path(p))==h for p,h in before.items()),"Protected input or old export changed")
    receipt={"status":"RENDERED_PENDING_VISUAL_REVIEW","page_mm":PAGE,"dpi":600,"national_unique_tracts":42373,"main_records":104371,"paired_unique_tracts":26477,"local_unique_tracts":9242,"national_extent":extent.tolist(),"states":49,"national_points_within_extent":bool(((xy>=extent[:2])&(xy<=extent[2:])).all()),"state_bounds_within_extent":bool((states.total_bounds[:2]>=extent[:2]).all() and (states.total_bounds[2:]<=extent[2:]).all()),"source_hashes":before,"protected_sources_unchanged":True,"cities":rows,"scales":scales,"locators":locators,"scale_ticks":TICKS,"intervals_per_scale":5,"continuous_mapping":True,"color_anchors":[CANOPY_COLORS,TEMP_COLORS],"font_path":font,"python":sys.executable,"versions":{"python":platform.python_version(),"matplotlib":matplotlib.__version__,"geopandas":gpd.__version__},"script_sha256":sha(Path(__file__)),"outputs":{ext:sha(OUT/f"exports/Figure01_national_coverage_city_pairs.{ext}") for ext in ("pdf","svg","png")}}
    dump(OUT/"qa/render_receipt.json",receipt)
    plt.close(fig)
    print("Rendered; national 42373, paired local 9242",flush=True)

if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:render()
    except Exception:
        import traceback
        (OUT/"qa/failure.log").write_text(traceback.format_exc(),encoding="utf8")
        raise
