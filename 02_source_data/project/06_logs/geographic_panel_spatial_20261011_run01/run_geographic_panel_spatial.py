"""Geographic-control comparison: tract FE and state-by-outcome-year FE."""
from pathlib import Path
import hashlib,json,platform,sys,time
import numpy as np,pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import norm

ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
SCORES=ROOT/'06_logs/baseline_equity_support_spatial_20261011_run01/group_analysis_row_manifest.csv'
PANEL=ROOT/'06_logs/geographic_measurement_repair_20261009_run01/revised_analysis_sample.csv'
PLAN=OUT/'analysis_plan_zh.md'
ID=['state','tract_id']; KEY=ID+['exposure_year','outcome_year']
SOC=['income_point_usd_2000','poverty_share_2000','age65plus_share_2000']
WEATHER=['tmmx_c_warm_season_mean_available_land','tmmn_c_warm_season_mean_available_land','pr_mm_warm_season_mean_available_land']
COORD=['centroid_epsg5070_x_m','centroid_epsg5070_y_m']; T='treatment_pp_change_from_2000'; Y='weather_adjusted_lst_reference_mean'
RADII=np.array([100000.,200000.,400000.])

def req(x,m):
    if not x: raise RuntimeError(m)
def sha(p):
    h=hashlib.sha256();
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def read(p): return pd.read_csv(p,dtype={'state':str,'tract_id':str},float_precision='round_trip')
def wendland(r): return np.maximum(1-r,0)**4*(1+4*r)
def group_mean(a,c):
    n=np.bincount(c); return np.column_stack([np.bincount(c,weights=a[:,j])/n for j in range(a.shape[1])])
def absorb(a,groups,tol=1e-10):
    r=a.copy()
    for it in range(1,10001):
        for c in groups:r-=group_mean(r,c)[c]
        err=max(float(np.abs(group_mean(r,c)).max()) for c in groups)
        if err<=tol:return r,{'iterations':it,'max_abs_group_mean':err,'tolerance':tol}
    raise RuntimeError('Fixed effect residualization did not converge')
def spatial(xy,q):
    tree=cKDTree(xy); out=np.zeros((len(RADII),q.shape[1])); pairs=np.zeros(len(RADII),int); diag=(q*q).sum(0); t=time.monotonic()
    for st in range(0,len(xy),256):
        en=min(st+256,len(xy)); ds=cKDTree(xy[st:en]).sparse_distance_matrix(tree,float(RADII.max()),output_type='coo_matrix')
        for k,h in enumerate(RADII):
            use=ds.data<=h; rows,cols=ds.row[use],ds.col[use]
            from scipy.sparse import coo_matrix
            m=coo_matrix((wendland(ds.data[use]/h),(rows,cols)),shape=(en-st,len(xy))).tocsr()
            out[k]+=(q[st:en]*(m@q)).sum(0); pairs[k]+=int(np.count_nonzero(st+rows!=cols))
        if st//256%25==0 or en==len(xy): print(f'SPATIAL {en}/{len(xy)} elapsed_seconds={time.monotonic()-t:.1f}',flush=True)
    req((out>=-1e-14).all(),'Negative spatial variance'); req((pairs%2==0).all(),'Kernel symmetry failure')
    return out,diag,pairs//2
def write(name,obj):
    p=OUT/name; req(not p.exists(),f'Output exists: {p}'); obj.to_csv(p,index=False,encoding='utf-8-sig')
def main():
    start=time.perf_counter(); paths=[SCORES,PANEL,PLAN,Path(__file__)]; src={str(p):sha(p) for p in paths}
    s=read(SCORES); p=read(PANEL); req(len(s)==104371 and s.tract_id.nunique()==42373,'Frozen sample mismatch')
    req(not s.duplicated(KEY).any() and not p.duplicated(KEY).any(),'Duplicate join keys')
    req(set(s.spatial_fold)==set(range(5)),'Fold set mismatch')
    cols=KEY+[Y,*WEATHER,'baseline_canopy_percent_fixed_land_2000','urban_fixed_land_m2']
    d=s.merge(p[cols],on=KEY,how='outer',validate='one_to_one',indicator=True); req(d._merge.eq('both').all(),'Lossy join')
    req(np.isfinite(d[[T,Y,*SOC,*WEATHER,'theta','orthogonal_score']].to_numpy(float)).all(),'Nonfinite input')
    uniq=d.drop_duplicates(ID); centers=uniq[SOC].mean(); scales=uniq[SOC].std(ddof=0); req(scales.gt(0).all(),'Degenerate social variable')
    z=((d[SOC]-centers)/scales).to_numpy(float); t=d[T].to_numpy(float); w=d[WEATHER].to_numpy(float)
    x=np.column_stack([t,t[:,None]*z,w]); names=[T,'treatment_x_income_one_baseline_sd','treatment_x_poverty_one_baseline_sd','treatment_x_age65plus_one_baseline_sd',*WEATHER]
    tract=pd.factorize(d.tract_id,sort=True)[0]; stateyear=pd.factorize(d.state+'_'+d.outcome_year.astype(str),sort=True)[0]
    mat=np.column_stack([d[Y].to_numpy(float),x]); scale=np.maximum(mat.std(0),1.); r,tr=absorb(mat/scale,[tract,stateyear]); r*=scale; yr=r[:,0]; xr=r[:,1:]
    xs=xr.std(0); req((xs>0).all(),'No within fixed-effect support'); design=xr/xs; b=np.linalg.lstsq(design,yr,rcond=None)[0]/xs; resid=yr-xr@b; rank=np.linalg.matrix_rank(design)
    req(rank==len(names),'Rank deficiency'); req(np.max(np.abs(xr.T@resid))/len(d)<1e-8,'Normal equation check')
    inv=np.linalg.inv(xr.T@xr); q=(xr@inv)*resid[:,None]; req(np.max(np.abs(q.sum(0))) <= 1e-7*max(1.0,float(np.max(np.abs(b)))), f'Centered influence mismatch: {np.max(np.abs(q.sum(0)))}')
    iq=pd.DataFrame({'state':d.state,'tract_id':d.tract_id,**{n:q[:,j] for j,n in enumerate(names)}})
    coords=d[ID+COORD].drop_duplicates(ID).sort_values(ID); req(len(coords)==42373,'Coordinate identity mismatch')
    agg=iq.groupby(ID,sort=True).sum().reset_index().merge(coords,on=ID,validate='one_to_one'); qagg=agg[names].to_numpy(); var,diag,pairs=spatial(agg[COORD].to_numpy(),qagg)
    rows=[]
    for k,h in enumerate(RADII):
        for j,n in enumerate(names):
            se=float(np.sqrt(max(var[k,j],0))); pv=float(2*norm.sf(abs(b[j]/se)))
            rows.append({'model':'tract_FE_state_year_FE','term':n,'estimate':float(b[j]),'bandwidth_km':h/1000,'variance':float(var[k,j]),'standard_error':se,'lower95':float(b[j]-1.959963984540054*se),'upper95':float(b[j]+1.959963984540054*se),'p_value':pv,'within_tract_variance_component':float(diag[j]),'between_tract_pairs':int(pairs[k]),'status':'CONDITIONAL_GEOGRAPHIC_CONTROL_REFERENCE_INTERVAL'})
    rr=pd.DataFrame(rows)
    for h,g in rr.groupby('bandwidth_km'):
        idx=g[g.term.isin(names[1:4])].index; ps=rr.loc[idx,'p_value'].to_numpy(); order=np.argsort(ps); adj=np.empty(3); adj[order]=np.maximum.accumulate(np.minimum(1,ps[order]*np.arange(3,0,-1))); rr.loc[idx,'holm_p_social_interactions']=adj
    rr.loc[~rr.term.isin(names[1:4]),'holm_p_social_interactions']=np.nan; write('geographic_control_spatial_results_three_scales.csv',rr)
    support=[]
    ct=d.groupby('tract_id').size(); wt=d[T]-d.groupby('tract_id')[T].transform('mean')
    for n,j in zip(names,range(len(names))):
        sh=(qagg[:,j]**2); sh=sh/sh.sum(); order=np.argsort(-sh)
        support.append({'term':n,'max_diagonal_influence_share':float(sh.max()),'top10_diagonal_influence_share':float(sh[order[:10]].sum()),'within_tract_sd_of_treatment':float(wt.std(ddof=0)),'within_treatment_variance_fraction':float(np.var(wt)/np.var(t)),'one_period_tracts':int((ct==1).sum()),'zero_within_change_tracts':int(d.groupby('tract_id')[T].nunique().eq(1).sum()),'rows':len(d),'tracts':d.tract_id.nunique()})
    write('geographic_control_support_and_influence.csv',pd.DataFrame(support))
    write('geographic_control_coefficients.csv',pd.DataFrame([{'term':n,'estimate':float(b[j]),'rank':rank,'condition_number':float(np.linalg.cond(design)),'fe_iterations':tr['iterations'],'max_abs_group_mean':tr['max_abs_group_mean']} for j,n in enumerate(names)]))
    receipt={'status':'PASS_GEOGRAPHIC_CONTROL_SPATIAL_COMPARISON_CONDITIONAL','rows':len(d),'tracts':int(d.tract_id.nunique()),'duplicate_keys':0,'unmatched_join_rows':0,'dropped_rows':0,'model':'tract fixed effects plus state by outcome-year fixed effects','terms':names,'radii_km':[100,200,400],'rank':int(rank),'columns':len(names),'condition_number':float(np.linalg.cond(design)),'fe_trace':tr,'source_sha256':src,'new_forest_fits':0,'formal_forest_outputs_changed':False,'measurement_error_propagation':'NOT RUN','unconditional_causal_validity':'NOT VERIFIED','interpretation':'Same-tract comparison conditional on fixed tract features, state-year shocks, weather covariates and spatial covariance assumptions; not a guarantee against time-varying confounding','elapsed_seconds':time.perf_counter()-start,'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__}
    with (OUT/'execution_record.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
    print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True); print(rr[rr.bandwidth_km==200].to_string(index=False),flush=True)
if __name__=='__main__': sys.stdout.reconfigure(encoding='utf-8');main()
