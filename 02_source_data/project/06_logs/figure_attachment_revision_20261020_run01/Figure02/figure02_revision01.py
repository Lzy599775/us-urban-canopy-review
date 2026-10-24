"""Figure 2: long-term canopy change and social distribution.

Read-only frozen result inputs. No refitting, smoothing, or reclassification.
"""
from nc_paths import project_root as _nc_project_root, external_path as _nc_external_path
from pathlib import Path
import json, hashlib, sys, platform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import geopandas as gpd
import numpy as np
import pandas as pd
import pymupdf
from pyproj import Transformer, Geod
from scipy.optimize import brentq

ROOT = _nc_project_root()
OUT = Path(__file__).resolve().parent
BASE = ROOT / "06_logs/canopy_cooling_equity_20261009_run01"
TEMP = ROOT / "08_figure图/Temporal_AB_全国与45城年度轨迹/TemporalA_全国树冠空间演变/source_data"
SOC = ROOT / "08_figure图/Fig03_长期树冠资源变化与社会差距/source_data"
STATES = ROOT / "08_figure图/Fig04_全国响应与估计对象比较/revision03_atlas45/prepared/locator_states.gpkg"
VARS = ["income", "poverty", "age65plus"]
TITLES = ["Income", "Poverty", "Age 65+"]
GROUP_COLORS = {1: "#70C1AB", 2: "#EAB77D", 3: "#AAA5CF", 4: "#D98EB7"}
INK = "#354C49"
FRAME = "#AEBAB6"
PAPER = "#FAF7F0"
WATER = "white"
CHANGE = LinearSegmentedColormap.from_list("change", ["#947AAC", "#CEC0DC", "#FAF9F5", "#BEDBD2", "#489C8E"], 256)
GEOM_NORM = Normalize(-12, 12)
GAP_NORM = Normalize(-15, 15)
YEARS = np.arange(2000, 2026)

def req(x, msg):
    if not x:
        raise ValueError(msg)

def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def mm(fig, x, top, w, h):
    return fig.add_axes([x / 180, 1 - (top + h) / 250, w / 180, h / 250])

def text(fig, x, top, s, size=7, weight="normal", ha="left", color=INK):
    return fig.text(x / 180, 1 - top / 250, s, ha=ha, va="top", fontsize=size,
                    fontweight=weight, color=color)

GEO=Transformer.from_crs(5070,4326,always_xy=True)
PROJ=Transformer.from_crs(4326,5070,always_xy=True)
GEOD=Geod(ellps="GRS80")

def clean_axis(ax, grid=False):
    ax.spines[["top","right"]].set_visible(False)
    for name in ["left","bottom"]:
        ax.spines[name].set_color("#A8B3B4");ax.spines[name].set_linewidth(.45)
    ax.tick_params(labelsize=7,length=1.8,width=.45,pad=2,colors=INK)
    if grid:ax.grid(axis="y",color="#E0E7E4",lw=.3,zorder=0)

def map_base(ax, states, extent, points, values, norm, gid, size):
    ax.set_facecolor("white")
    states.plot(ax=ax,color=PAPER,edgecolor="#BCC5C2",linewidth=.22,zorder=1)
    ax.set_xlim(extent[0],extent[2]);ax.set_ylim(extent[1],extent[3]);ax.set_aspect("equal")
    ax.set_xticks([]);ax.set_yticks([])
    for sp in ax.spines.values():sp.set_visible(False)
    sc=ax.scatter(points.x,points.y,s=size,c=values,cmap=CHANGE,norm=norm,
                  edgecolors="#7E908A",linewidths=.15,zorder=4)
    sc.set_gid(gid)
    # At Albers central meridian, true north is vertical. Reference point fixed.
    lon,lat=-96.,47.7
    xx,yy=PROJ.transform(lon,lat)
    lon2,lat2,_=GEOD.fwd(lon,lat,0,100)
    v=np.array(PROJ.transform(lon2,lat2))-np.array([xx,yy]);v/=np.linalg.norm(v)
    req(np.allclose(v,[0,1],atol=1e-7),"National reference north is not vertical")
    ax.annotate("",xy=(xx,yy),xytext=(0,-11),textcoords="offset points",
                arrowprops=dict(arrowstyle="-|>",lw=.6,color=INK,mutation_scale=6),zorder=7)
    ax.annotate("N",xy=(xx,yy),xytext=(0,2),textcoords="offset points",ha="center",fontsize=7,color=INK,zorder=7)
    # Solve actual ground distance at scale reference. No fixed axes-fraction length.
    x=extent[0]+.065*(extent[2]-extent[0]);y=extent[1]+.06*(extent[3]-extent[1]);meters=500000.
    def ground(dx):return GEOD.inv(*GEO.transform(x,y),*GEO.transform(x+dx,y))[2]
    dx=brentq(lambda d:ground(d)-meters,meters*.5,meters*2,xtol=1e-6)
    h=.006*(extent[3]-extent[1])
    ax.plot([x,x+dx],[y,y],color=INK,lw=.65,zorder=7)
    for z in [x,x+dx]:ax.plot([z,z],[y-h,y+h],color=INK,lw=.45,zorder=7)
    ax.annotate("500 km",xy=(x+dx,y),xytext=(2,0),textcoords="offset points",va="center",fontsize=7,color=INK,zorder=7)
    return {"panel":gid,"start_display_xy":[x,y],"end_display_xy":[x+dx,y],"target_ground_m":meters,"verified_ground_m":ground(dx),"north_reference_lon_lat":[lon,lat],"north_vector":v.tolist(),"extent":extent.tolist(),"points":len(points)}

def add_colorbar(fig,x,top,w,label,norm,ticks):
    text(fig,x+w/2,top-4,label,size=7,ha="center")
    cax=mm(fig,x,top,w,2.3)
    cb=fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=CHANGE),cax=cax,orientation="horizontal",ticks=ticks)
    cb.solids.set_rasterized(False);cb.solids.set_edgecolor("face")
    cb.outline.set_visible(False);cb.ax.tick_params(labelsize=7,length=1.4,width=.4,pad=1.5)
    return cb

def label_positions(ends,minimum=.235):
    # Adjust text labels only; the plotted observations are never moved.
    order=sorted(ends,key=lambda q:(ends[q],q));ys=[ends[q] for q in order]
    for i in range(1,len(ys)):ys[i]=max(ys[i],ys[i-1]+minimum)
    offset=np.mean([ys[i]-ends[q] for i,q in enumerate(order)])
    ys=np.array(ys)-offset
    if ys[0]<-.64:ys+=-.64-ys[0]
    if ys[-1]>1.38:ys-=ys[-1]-1.38
    return dict(zip(order,ys))

def load_data():
    city = pd.read_csv(TEMP / "city_annual_448.csv")
    points = pd.read_csv(TEMP / "city_display_points.csv")
    nat = pd.read_csv(TEMP / "national_annual_46424.csv")
    traj = pd.read_csv(SOC / "trajectories_balanced.csv")
    ends = pd.read_csv(SOC / "coverage_endpoints.csv")
    gaps = pd.read_csv(BASE / "annual_same_city_canopy_gaps.csv")
    summary = pd.read_csv(SOC / "city_gaps_balanced.csv")
    req(len(city) == 11648 and city.single_city_id.nunique() == 448, "City annual source differs")
    req(len(points) == 448 and points.single_city_id.is_unique, "City point source differs")
    req(len(nat) == 26 and nat.n.eq(46424).all(), "National source differs")
    traj = traj[traj.panel.eq("balanced_26years")].copy()
    req(len(traj) == 312, "Trajectory source differs")
    gaps = gaps[gaps.panel.eq("balanced_26years")].copy()
    req(len(gaps) == 448 * 26 * 3, "City gap source differs")
    summary = summary[summary.panel.eq("balanced_26years")].copy()
    req(len(summary) == 78, "Gap summary source differs")
    return city, points, nat, traj, ends, gaps, summary

def render():
    require_env = Path(sys.executable).parent.name == "sportpark-ee"
    req(require_env, "Run with sportpark-ee")
    for d in [OUT / "exports", OUT / "qa", OUT / "source_data"]: d.mkdir(parents=True, exist_ok=True)
    source_paths = [TEMP / "city_annual_448.csv", TEMP / "city_display_points.csv", TEMP / "national_annual_46424.csv",
                    SOC / "trajectories_balanced.csv", SOC / "coverage_endpoints.csv", BASE / "annual_same_city_canopy_gaps.csv",
                    SOC / "city_gaps_balanced.csv", STATES]
    before = {str(p): sha(p) for p in source_paths}
    city, points, nat, traj, ends, gaps, summary = load_data()
    states = gpd.read_file(STATES, layer="states")
    req(states.crs.to_epsg() == 5070 and len(states) == 49 and states.geometry.is_valid.all(), "States invalid")
    city_w = city.pivot(index="single_city_id", columns="year", values="canopy")
    net = city_w[2025] - city_w[2000]
    req(np.allclose(net.to_numpy(), city[city.year.eq(2025)].set_index("single_city_id").change.loc[net.index], atol=1e-10), "City change mismatch")
    req(net.between(-10.004, 9.240).all(), "Unexpected city range")
    rows = []
    for v in VARS:
        q = traj[traj.variable.eq(v)].copy()
        req(set(q.quartile) == {1,2,3,4} and q.groupby("quartile").year.nunique().eq(26).all(), f"{v} groups incomplete")
        rows.extend(q.to_dict("records"))
    traj = pd.DataFrame(rows)
    # Unified endpoint and map support tables for reproducible downstream review.
    pd.DataFrame({"single_city_id": net.index, "change_2000_2025_pp": net.to_numpy()}).merge(points, on="single_city_id", validate="one_to_one").to_csv(OUT / "source_data/city_net_change_points.csv", index=False)
    traj.to_csv(OUT / "source_data/group_trajectories_used.csv", index=False)
    summary.to_csv(OUT / "source_data/equal_city_gaps_used.csv", index=False)

    plt.rcParams.update({"font.family":"Arial","font.size":7,"pdf.fonttype":42,"svg.fonttype":"none","axes.unicode_minus":True,"savefig.dpi":600})
    fig=plt.figure(figsize=(180/25.4,250/25.4),dpi=150,facecolor="white")
    xy=points[["x","y"]].to_numpy()
    lo=np.minimum(states.total_bounds[:2],xy.min(axis=0));hi=np.maximum(states.total_bounds[2:],xy.max(axis=0))
    ext=np.r_[lo-(hi-lo)*.025,hi+(hi-lo)*.025]
    req(((xy>=ext[:2])&(xy<=ext[2:])).all(),"City point cropped")
    scales=[]; plotted_lines=[]; endpoint_labels=[]; gapmap_rows=[]
    text(fig,5,4,"a  Canopy change, 2000–2025",size=8.3,weight="bold")
    a=mm(fig,5,12,108,61)
    ap=points.set_index("single_city_id").loc[net.index]
    scales.append(map_base(a,states,ext,ap,net.to_numpy(),GEOM_NORM,"city_net_change_448",7.4))
    add_colorbar(fig,23,78,76,"Canopy change (pp)",GEOM_NORM,[-12,-8,-4,0,4,8,12])
    text(fig,122,4,"b  National canopy",size=8.3,weight="bold")
    top=mm(fig,134,17,40,23);bot=mm(fig,134,54,40,20)
    for ax in [top,bot]:
        clean_axis(ax);ax.set_xlim(1999,2026);ax.set_xticks([2000,2010,2025]);ax.tick_params(axis="x",pad=3)
    top.set_ylim(19.5,20.65);top.set_yticks([19.6,20.0,20.4])
    top.plot(nat.year,nat["mean"],color="#3F8279",lw=1.25,gid="national_mean_26")
    top.scatter([2000,2025],nat.set_index("year").loc[[2000,2025],"mean"],s=8,color="#3F8279",zorder=3)
    top.set_ylabel("Mean (%)",fontsize=7,labelpad=1)
    top.tick_params(axis="x",labelbottom=False)
    for yr in [2000,2025]:
        val=float(nat.loc[nat.year.eq(yr),"mean"].iloc[0])
        top.annotate(f"{val:.3f}",(yr,val),xytext=(1 if yr==2000 else -1,-9),textcoords="offset points",ha="left" if yr==2000 else "right",fontsize=7,color=INK)
    bot.fill_between(nat.year,nat.q25,nat.q75,color="#BEDBD2",lw=0,gid="national_iqr_26")
    bot.plot(nat.year,nat.q25,color="#6AA99D",lw=.55);bot.plot(nat.year,nat.q75,color="#6AA99D",lw=.55)
    bot.set_ylim(-3,43);bot.set_yticks([0,20,40]);bot.set_ylabel("Canopy cover (%)",fontsize=7,labelpad=1)
    # Three columns: consistent group-change axes and fixed endpoint strips.
    for j,(v,title) in enumerate(zip(VARS,TITLES)):
        x=[15,73,131][j];ax=mm(fig,x,98,35,30)
        text(fig,[5,63,121][j],90,f"{chr(99+j)}  {title}",size=8.3,weight="bold")
        sub=traj[traj.variable.eq(v)];endvals={int(q):float(h.loc[h.year.eq(2025),"change_from_2000_pp_mean"].iloc[0]) for q,h in sub.groupby("quartile")}
        positions=label_positions(endvals)
        for q in [1,2,3,4]:
            h=sub[sub.quartile.eq(q)].sort_values("year");ls="-" if q in [1,4] else ("--" if q==2 else ":")
            line,=ax.plot(h.year,h.change_from_2000_pp_mean,color=GROUP_COLORS[q],lw=1.15 if q in [1,4] else .95,ls=ls,zorder=3,gid=f"trajectory_{v}_Q{q}")
            ax.scatter([2025],[endvals[q]],s=7,color=GROUP_COLORS[q],edgecolors="white",linewidths=.2,zorder=4)
            ax.plot([2025,2027],[endvals[q],positions[q]],color=GROUP_COLORS[q],lw=.45,clip_on=False,zorder=5)
            ax.text(2028,positions[q],f"Q{q}",fontsize=7,ha="left",va="center",color=INK,clip_on=False)
            plotted_lines.append({"id":f"trajectory_{v}_Q{q}","x":line.get_xdata().tolist(),"y":line.get_ydata().tolist()})
        ax.set_xlim(1999,2026);ax.set_ylim(-.75,1.5);ax.set_xticks([2000,2010,2025]);ax.set_yticks([-.5,0,.5,1,1.5]);clean_axis(ax)
        ax.axhline(0,color="#B5C1BC",lw=.45,zorder=1)
        if j==0:ax.set_ylabel("Change (pp; tract mean)",fontsize=7,labelpad=1)
        else:ax.tick_params(labelleft=False)
        ep=ends[ends.variable.eq(v)].pivot(index="quartile",columns="year",values="canopy_percent_mean")
        tx=[5,63,121][j]
        text(fig,tx+25,136,"2000 (%)",size=7,ha="center");text(fig,tx+45,136,"2025 (%)",size=7,ha="center")
        for i,q in enumerate([1,2,3,4]):
            yy=140.3+i*3.2
            text(fig,tx+3,yy,f"Q{q}",size=7,color=INK)
            # Line sample repeats the actual curve style, not only colour.
            fig.add_artist(Line2D([(tx+9)/180,(tx+14)/180],[1-(yy+1.2)/250]*2,transform=fig.transFigure,color=GROUP_COLORS[q],lw=1.1,ls="-" if q in [1,4] else ("--" if q==2 else ":")))
            for year,xx in [(2000,tx+25),(2025,tx+45)]:
                label=f"{ep.loc[q,year]:.3f}";text(fig,xx,yy,label,size=7,ha="center");endpoint_labels.append({"variable":v,"quartile":q,"year":year,"value":float(ep.loc[q,year]),"label":label})
    # Three complete national city maps with a shared scale.
    for j,(v,title) in enumerate(zip(VARS,TITLES)):
        x=[5,63,121][j];text(fig,x,157,f"{chr(102+j)}  {title if j<2 else 'Age'} gap change",size=8,weight="bold")
        ax=mm(fig,x,163,54,32)
        g=gaps[gaps.variable.eq(v)].pivot(index="single_city_id",columns="year",values="high_minus_low_canopy_pp")
        delta=g[2025]-g[2000];pp=points.set_index("single_city_id").loc[delta.index]
        req(delta.between(-15,15).all(),"Gap value outside colour limits")
        scales.append(map_base(ax,states,ext,pp,delta.to_numpy(),GAP_NORM,f"city_gap_change_{v}_448",3.9))
        tmp=pp.reset_index();tmp["variable"]=v;tmp["gap_2000_pp"]=g[2000].to_numpy();tmp["gap_2025_pp"]=g[2025].to_numpy();tmp["gap_change_pp"]=delta.to_numpy();gapmap_rows.append(tmp)
        sub=summary[summary.variable.eq(v)].set_index("year")
        req(abs(delta.mean()-(sub.loc[2025,"equal_city_mean_gap_pp"]-sub.loc[2000,"equal_city_mean_gap_pp"]))<1e-10,"Map-summary identity")
    add_colorbar(fig,43,199,94,"Change in Q4−Q1 gap (pp)",GAP_NORM,[-15,-10,-5,0,5,10,15])
    text(fig,90,211,"Within-city gap · Equal-city mean",size=7,ha="center")
    for j,(v,title) in enumerate(zip(VARS,TITLES)):
        x=[15,73,131][j];text(fig,[5,63,121][j],217,f"{chr(105+j)}  {title if j<2 else 'Age'} gap",size=8.3,weight="bold")
        ax=mm(fig,x,224,43,17);sub=summary[summary.variable.eq(v)].sort_values("year");limits=[(8,10),(-9,-7),(-.5,1.5)][j]
        line,=ax.plot(sub.year,sub.equal_city_mean_gap_pp,color=INK,lw=1.15,gid=f"gap_{v}_26")
        ax.scatter([2000,2025],sub[sub.year.isin([2000,2025])].equal_city_mean_gap_pp,s=8,color=INK,zorder=3)
        ax.set_xlim(1999,2026);ax.set_ylim(*limits);ax.set_xticks([2000,2010,2025]);ax.set_yticks(np.linspace(*limits,3));clean_axis(ax)
        if j==0:ax.set_ylabel("Gap (pp)",fontsize=7,labelpad=1)
        if j==2:ax.axhline(0,color="#B5C1BC",lw=.4,zorder=0)
        for yr in [2000,2025]:
            val=float(sub.loc[sub.year.eq(yr),"equal_city_mean_gap_pp"].iloc[0]);offset=-12 if (j==1 and yr==2000) else (10 if yr==2000 else 5)
            ax.annotate(f"{val:.3f}",(yr,val),xytext=(3 if yr==2000 else -3,offset),textcoords="offset points",ha="left" if yr==2000 else "right",fontsize=7,color=INK)
        plotted_lines.append({"id":f"gap_{v}_26","x":line.get_xdata().tolist(),"y":line.get_ydata().tolist()})
    pd.concat(gapmap_rows).to_csv(OUT/"source_data/city_gap_change_points.csv",index=False)
    nat.to_csv(OUT/"source_data/national_annual_used.csv",index=False)
    ends.to_csv(OUT/"source_data/coverage_endpoints_used.csv",index=False)
    (OUT/"qa/plotted_values.json").write_text(json.dumps({"lines":plotted_lines,"endpoint_labels":endpoint_labels,"scales":scales},indent=2),encoding="utf8")
    # Export and protect inputs.
    fig.canvas.draw()
    for extn in ["pdf","svg","png"]:
        fig.savefig(OUT / "exports" / f"Figure02_canopy_change_social_distribution.{extn}", dpi=600, facecolor="white")
    fig.savefig(OUT / "qa" / "Figure02_preview_180x250mm.png", dpi=190, facecolor="white")
    req(all(sha(Path(p)) == h for p,h in before.items()), "Input source changed")
    receipt={"status":"RENDERED_PENDING_VALIDATION","page_mm":[180,250],"dpi":600,"resource_tracts":46424,"cities":448,"city_assigned_tracts":46092,"unassigned_tracts":332,"group_tracts":{"income":45799,"poverty":45889,"age65plus":45944},"city_net_range":[float(net.min()),float(net.max())],"gap_color_range":[-15,15],"trajectory_limits":[-.75,1.5],"map_extent":ext.tolist(),"scales":scales,"source_hashes":before,"outputs":{e:sha(OUT/"exports"/f"Figure02_canopy_change_social_distribution.{e}") for e in ["pdf","svg","png"]},"python":sys.executable,"versions":{"python":platform.python_version(),"matplotlib":matplotlib.__version__,"geopandas":gpd.__version__}}
    (OUT/"qa"/"render_receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding="utf8")
    plt.close(fig)

if __name__ == "__main__":
    try: render()
    except Exception:
        (OUT/"qa"/"failure.log").write_text(__import__("traceback").format_exc(),encoding="utf8")
        raise
