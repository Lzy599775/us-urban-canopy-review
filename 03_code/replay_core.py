"""Offline deterministic R2 replay. No model imports, fitting or source writes."""
from pathlib import Path
import argparse,csv,json,math,hashlib
from statistics import NormalDist
import numpy as np
import pandas as pd

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    root=a.package/'02_source_data/project'
    model=root/'06_logs/geo_adjusted_national_orthogonal_20261009_run01'
    group=root/'06_logs/group_mean_compatibility_repair_20261014_run01/results_csv'
    def read(p):return pd.read_csv(p,dtype={'state':str,'tract_id':str,'single_city_id':str},float_precision='round_trip')
    checks=[]
    def check(name,actual,expected,tol=0):
        ok=bool(abs(actual-expected)<=tol)
        checks.append(dict(check=name,actual=actual,expected=expected,absolute_tolerance=tol,status='PASS' if ok else 'FAIL'))
        if not ok: raise RuntimeError(checks[-1])
    s=read(model/'crossfit_scores.csv');p=read(group/'crossfit_group_probabilities.csv')
    keys=['state','tract_id','exposure_year','outcome_year']
    check('model_rows',len(s),104371);check('model_tracts',len(s[['state','tract_id']].drop_duplicates()),42373)
    check('row_identity_order',int(s[keys].equals(p[keys])),1)
    check('duplicate_record_keys',int(s.duplicated(keys).sum()),0)
    check('tract_id_width',int(s.tract_id.str.len().eq(11).all()),1)
    check('folds_per_tract',int(s.groupby('tract_id').spatial_fold.nunique().max()),1)
    cor=s.residual_t.to_numpy()*(s.residual_y-s.intercept-s.theta*s.residual_t).to_numpy()/s.conditional_variance.to_numpy()
    check('score_reconstruction_max_error',float(np.max(np.abs(s.theta+cor-s.orthogonal_score))),0,1e-12)
    mu={'national':float((s.theta+cor).mean())}
    saved=read(group/'estimates_and_corrections.csv').set_index('key')
    for d in ['income','poverty','age65plus']:
        check(d+'_unassigned_rows',int(p[d+'_city_quartile'].isna().sum()),715)
        check(d+'_probability_sum_error',float(abs(p[[d+'_p'+str(q) for q in range(5)]].sum(axis=1)-1).max()),0,1e-12)
        labels=p[d+'_city_quartile'].fillna(0).to_numpy(int)
        for q in range(5):
            key=f'{d}_q{q}';g=labels==q
            mu[key]=float((np.sum(g*s.theta)+np.sum(p[f'{d}_p{q}']*cor))/g.sum())
            check(key,mu[key],float(saved.loc[key,'estimate']),1e-12)
        key=d+'_q4_minus_q1';mu[key]=mu[d+'_q4']-mu[d+'_q1']
        check(key,mu[key],float(saved.loc[key,'estimate']),1e-12)
    national=read(model/'spatial_covariance_reference.csv')
    check('national_augmented_mean',mu['national'],float(national.estimate.iloc[0]),1e-12)
    check('prediction_mean_rounded',float(s.theta.mean()),-0.164652,0.0000005)
    z=np.load(group/'joint_spatial_covariance.npz',allow_pickle=False)
    refs=read(group/'all_group_reference_intervals.csv')
    replay=[];normal975=NormalDist().inv_cdf(.975)
    for j,r in enumerate(z['radii_m']):
        covariance=z['covariance'][j];names=z['keys'].tolist();idx={x:i for i,x in enumerate(names)}
        check(f'covariance_symmetry_{r}',float(np.max(abs(covariance-covariance.T))),0,1e-12)
        ps=[]
        for key in names:
            val=mu[key];se=math.sqrt(covariance[idx[key],idx[key]])
            lower=val-normal975*se;upper=val+normal975*se;pv=math.erfc(abs(val/se)/math.sqrt(2))
            old=refs[(refs.key==key)&(refs.bandwidth_km==r/1000)].iloc[0]
            for n,v in [('estimate',val),('standard_error',se),('lower95',lower),('upper95',upper),('reference_p',pv)]:check(f'{key}_{r}_{n}',v,float(old[n]),1e-11)
            replay.append(dict(key=key,bandwidth_km=r/1000,estimate=val,standard_error=se,lower95=lower,upper95=upper,reference_p=pv,holm_p_three_contrasts=None))
            if key.endswith('_q4_minus_q1'):ps.append((key,pv))
        ps.sort(key=lambda x:x[1]);last=0
        for rank,(key,pv) in enumerate(ps):
            adj=min(1,max(last,(3-rank)*pv));last=adj
            old=refs[(refs.key==key)&(refs.bandwidth_km==r/1000)].iloc[0]
            check(f'{key}_{r}_holm',adj,float(old['holm_p_three_contrasts']),1e-11)
            next(x for x in replay if x['key']==key and x['bandwidth_km']==r/1000)['holm_p_three_contrasts']=adj
    j1=read(root/'06_logs/NLCD_J1_descriptive_20261015_run01/J1_city_equal_Q4_Q1.csv')
    check('J1_rows',len(j1),60)
    check('J1_identity_max_error',float(abs(j1.D_delta_pp-(j1.D_2025_pp-j1.D_2000_pp)).max()),0,1e-10)
    for dim,band,expected in [('income',1,2.085),('poverty',1,-1.560),('income',8,-1.335),('poverty',8,1.491)]:
        check(f'J1_{dim}_{band}',float(j1[(j1.dimension==dim)&(j1.end2025_band_id==band)].D_delta_pp.iloc[0]),expected,.0005)
    annual=root/'06_logs/canopy_cooling_equity_20261009_run01/annual_canopy_equity_row_manifest.csv.gz'
    raw=read(annual);check('resource_source_rows_before_existing_eligibility',len(raw),1689974)
    # Exact frozen selection from analyze_canopy_cooling_equity_20261009.py lines 197-208.
    resource=raw.loc[raw.complete_fixed_land_year & raw.balanced_26years]
    check('resource_records',len(resource),1207024);check('resource_tracts',resource.tract_id.nunique(),46424)
    check('resource_26_years',int(resource.groupby('tract_id').year.nunique().eq(26).all()),1)
    means=resource.groupby('year').urban_canopy_percent_fixed_land.mean()
    check('canopy_2000_rounded',float(means.loc[2000]),19.765,.0005)
    check('canopy_2025_rounded',float(means.loc[2025]),19.866,.0005)
    resource.to_csv(a.output/'resource_balanced_26years.csv.gz',index=False,compression='gzip')
    means.rename('canopy_percent').to_csv(a.output/'resource_national_annual_means.csv')
    pd.DataFrame(replay).to_csv(a.output/'recomputed_estimates_intervals.csv',index=False)
    pd.DataFrame(checks).to_csv(a.output/'NUMERIC_REPLAY.csv',index=False)
    (a.output/'receipt.json').write_text(json.dumps({'status':'PASS','checks':len(checks),'models_fitted':0,'new_inference':False,'scope':'R2 means recomputed from frozen predictions/residuals/probabilities; intervals and Holm replayed from SAVED joint spatial covariance, not kernel recalculation; J1 identity; resource counts','unverified':['R1 full geographic redraw','R2 new kernel covariance from tract distances','R3 upstream satellite processing']},indent=2),encoding='utf-8')
    print(json.dumps({'checks':len(checks),'status':'PASS','output':str(a.output)}))
if __name__=='__main__':main()
