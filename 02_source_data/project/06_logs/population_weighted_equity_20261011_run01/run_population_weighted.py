"""Population-size-weighted conditional effects and descriptive canopy paths."""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'05_scripts'))
import run_baseline_equity_support_20261011 as original

OUT = Path(__file__).resolve().parent
YEARS = [2014, 2019, 2024]
VARS = ['income','poverty','age65plus']
ID = ['state','tract_id']
COORD = original.COORD
CANOPY = ROOT/'06_logs/canopy_cooling_equity_20261009_run01/annual_canopy_equity_row_manifest.csv.gz'
BASE = ROOT/'06_logs/national_analysis_baseline_join_20260914_run01/national_analysis_baseline_readiness.csv'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def save_json(name, obj):
    with (OUT/name).open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)


def save_csv(name, rows):
    require(not (OUT/name).exists(),f'Output already exists: {name}')
    pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')


def ratio(y, w, mask):
    require(mask.any() and w[mask].sum()>0,'Empty or zero-weight target')
    weights=np.where(mask,w,0.); total=weights.sum()
    mu=float(weights@y/total)
    a=weights*(y-mu)/total
    require(abs(a.sum())<1e-10,'Weighted influence not centered')
    return mu,a


def self_test():
    # Algebra test only, not study observations.
    y=np.array([2.,5.,1.,4.,7.,3.]);w=np.array([1.,4.,2.,8.,3.,7.]);mask=np.array([1,0,1,0,1,0],bool)
    mu,a=ratio(y,w,mask);mu2,a2=ratio(y,13*w,mask)
    require(np.isclose(mu,mu2) and np.allclose(a,a2),'Weight scale invariance')
    eps=1e-6
    perturbed=w.copy();perturbed[0]*=1+eps
    require(np.isclose((ratio(y,perturbed,mask)[0]-mu)/eps,a[0],rtol=1e-5,atol=1e-8),'Ratio derivative mismatch')
    xy=np.array([[0.,0.],[0.,0.],[200.,0.],[900.,0.]])
    q=np.array([[a[0],-a[0]],[a[2],a[2]],[a[4],-a[4]],[0.,0.]])
    radii=np.array([100.,400.,1200.])
    actual,_,_=original.spatial_diagonal(xy,q,radii)
    dist=np.linalg.norm(xy[:,None,:]-xy[None,:,:],axis=2)
    expected=np.array([np.diag(q.T@original.wendland(dist/h)@q) for h in radii])
    require(np.allclose(actual,expected,atol=1e-13,rtol=1e-12),'Weighted spatial kernel mismatch')
    print('PASS_WEIGHTED_RATIO_DERIVATIVE_AND_DENSE_SPARSE_ALGEBRA',flush=True)


def effects(f):
    psi=f.orthogonal_score.to_numpy(float);pop=f.population_2000.to_numpy(float)
    require(np.isfinite(pop).all() and (pop>0).all(),'Invalid effect weights')
    require(f.groupby(ID).population_2000.nunique().eq(1).all(),'Population drift')
    annual={};annual_meta={};point=[];weight_rows=[]
    for year in YEARS:
        yr=f.outcome_year.eq(year).to_numpy()
        labels=[('national',None,np.ones(len(f),bool))]
        labels += [(v,q,f[v+'_city_quartile'].eq(q).to_numpy()) for v in VARS for q in range(1,5)]
        for v,q,group in labels:
            mask=yr&group;mu,a=ratio(psi,pop,mask)
            annual[(year,v,q)]=(mu,a)
            equal,_=ratio(psi,np.ones(len(f)),mask)
            annual_meta[(year,v,q)]={'rows':int(mask.sum()),'tracts':int(f.loc[mask,'tract_id'].nunique()),'population_weight_sum':float(pop[mask].sum())}
            point.append({'period':str(year),'variable':v,'quartile':q,'population_weighted_effect_c_per_pp':mu,'equal_weighted_effect_c_per_pp':equal,**annual_meta[(year,v,q)]})
            p=pop[mask]/pop[mask].sum()
            weight_rows.append({'period':year,'variable':v,'quartile':q,**annual_meta[(year,v,q)],'weight_concentration_equivalent_tracts_NOT_SPATIAL_ESS':float(1/np.sum(p*p)),'top_one_percent_weight_share':float(np.sort(p)[-max(1,int(np.ceil(len(p)*.01))):].sum()),'maximum_tract_weight':float(p.max()),'unassigned_rows':0 if v=='national' else int((yr&f[v+'_city_quartile'].isna().to_numpy()).sum())})
    columns=[];meta=[]
    def add(period,v,comparison,q,mu,a,**extra):
        columns.append(a)
        meta.append({'period':period,'variable':v,'comparison':comparison,'quartile':q,'estimate_c_per_pp':float(mu),**extra})
    targets=[('national',None)]+[(v,q) for v in VARS for q in range(1,5)]
    for v,q in targets:
        mu=sum(annual[(y,v,q)][0] for y in YEARS)/3
        a=sum((annual[(y,v,q)][1] for y in YEARS),np.zeros(len(f)))/3
        mask=np.ones(len(f),bool) if v=='national' else f[v+'_city_quartile'].eq(q).to_numpy()
        add('three_period_equal_mean',v,'national_mean' if v=='national' else 'quartile_mean',q,mu,a,rows=int(mask.sum()),tracts=int(f.loc[mask,'tract_id'].nunique()))
        point.append({'period':'three_period_equal_mean','variable':v,'quartile':q,'population_weighted_effect_c_per_pp':mu,'equal_weighted_effect_c_per_pp':float(np.mean([p['equal_weighted_effect_c_per_pp'] for p in point if p['period'] in list(map(str,YEARS)) and p['variable']==v and p['quartile']==q])),'rows':int(mask.sum()),'tracts':int(f.loc[mask,'tract_id'].nunique()),'population_weight_sum':None})
    for v in VARS:
        mu=sum(annual[(y,v,4)][0]-annual[(y,v,1)][0] for y in YEARS)/3
        a=sum((annual[(y,v,4)][1]-annual[(y,v,1)][1] for y in YEARS),np.zeros(len(f)))/3
        mask=f[v+'_city_quartile'].isin([1,4])
        add('three_period_equal_mean',v,'q4_minus_q1',None,mu,a,rows=int(mask.sum()),tracts=int(f.loc[mask,'tract_id'].nunique()))
    for y in YEARS:
        mu,a=annual[(y,'national',None)]
        add(str(y),'national','national_mean',None,mu,a,**annual_meta[(y,'national',None)])
        for v in VARS:
            mu=annual[(y,v,4)][0]-annual[(y,v,1)][0]
            a=annual[(y,v,4)][1]-annual[(y,v,1)][1]
            mask=f.outcome_year.eq(y)&f[v+'_city_quartile'].isin([1,4])
            add(str(y),v,'q4_minus_q1',None,mu,a,rows=int(mask.sum()),tracts=int(f.loc[mask,'tract_id'].nunique()))
    # Continuous social projection stays primary, quartiles stay display contrasts.
    x=np.column_stack([np.ones(len(f)),f.outcome_year.eq(2019),f.outcome_year.eq(2024),f.income_point_usd_2000/10000.,f.poverty_share_2000/.1,f.age65plus_share_2000/.1,f.population_2000/10000.]).astype(float)
    w=np.zeros(len(f))
    for y in YEARS:
        mask=f.outcome_year.eq(y).to_numpy();w[mask]=pop[mask]/pop[mask].sum()/3
    gram=x.T@(w[:,None]*x);require(np.linalg.matrix_rank(gram)==7,'Weighted continuous rank')
    beta=np.linalg.solve(gram,x.T@(w*psi));inv=np.linalg.inv(gram);e=psi-x@beta
    for j,v in enumerate(VARS,3):
        a=w*(x@inv[:,j])*e
        add('three_period_equal_mean',v,'joint_continuous_projection',None,beta[j],a,rows=len(f),tracts=f.tract_id.nunique(),social_unit='10000_USD_1999' if v=='income' else '10_PERCENTAGE_POINTS')
    mat=np.column_stack(columns);require(np.max(np.abs(mat.sum(axis=0)))<1e-9,'Influence centering failure')
    save_csv('population_weighted_point_means_and_equal_comparison.csv',point)
    save_csv('population_weights_and_group_accounting.csv',weight_rows)
    return mat,meta


def aggregate(f,mat):
    cols=[f'if_{j}' for j in range(mat.shape[1])]
    temp=f[ID+COORD].copy()
    for j,col in enumerate(cols):temp[col]=mat[:,j]
    require(temp.groupby(ID)[COORD].nunique().le(1).all().all(),'Coordinate drift')
    agg=temp.groupby(ID,sort=True).agg({**{c:'first' for c in COORD},**{c:'sum' for c in cols}}).reset_index()
    return agg,cols


def trajectories(f,baseline):
    d=original.read(CANOPY)
    require(len(d)==1689974 and not d.duplicated([*ID,'year']).any(),'Resource input differs')
    b=original.read(BASE)
    require(len(b)==64999 and not b.duplicated(ID).any(),'Population baseline identity')
    d=d.merge(b[ID+['population_2000_100_percent']],on=ID,how='left',validate='many_to_one',indicator=True)
    require(d._merge.eq('both').all() and len(d)==1689974,'Resource population join loss')
    pop=d.population_2000_100_percent
    require(np.isfinite(pop).all() and pop.ge(0).all(),'Invalid resource population')
    effect_ids=pd.MultiIndex.from_frame(f[ID].drop_duplicates())
    d['in_temperature_sample']=pd.MultiIndex.from_frame(d[ID]).isin(effect_ids)
    rows=[];flow=[]
    for label,mask in [('original_balanced_resource_scope',d.balanced_26years),('temperature_sample_balanced_resource_scope',d.balanced_26years&d.in_temperature_sample)]:
        scope=d.loc[mask&d.complete_fixed_land_year].copy()
        count=scope.groupby(ID).year.nunique();require(count.eq(26).all(),'Resource scope is not balanced')
        for y,g in scope.groupby('year',sort=True):
            labels=[('national',None,g)]+[(v,q,g.loc[g[v+'_city_quartile'].eq(q)]) for v in VARS for q in range(1,5)]
            for v,q,h in labels:
                w=h.population_2000_100_percent.to_numpy(float)
                require(len(h)>0 and w.sum()>0,'Empty resource group')
                rows.append({'scope':label,'year':int(y),'variable':v,'quartile':q,'tracts':len(h),'zero_population_rows_retained':int((w==0).sum()),'population_weight_sum':float(w.sum()),'canopy_percent_population_weighted':float(np.average(h.urban_canopy_percent_fixed_land,weights=w)),'canopy_percent_equal_weighted':float(h.urban_canopy_percent_fixed_land.mean()),'change_pp_population_weighted':float(np.average(h.change_from_2000_pp,weights=w)),'role':'DESCRIPTIVE_RESOURCE_NOT_REALIZED_COOLING'})
            flow.append({'scope':label,'year':int(y),'rows':len(g),'unassigned_rows':int(g.income_city_quartile.isna().sum()),'zero_population_rows_retained':int(g.population_2000_100_percent.eq(0).sum())})
    tab=pd.DataFrame(rows);gaps=[]
    for (scope,y,v),g in tab.loc[tab.variable.ne('national')].groupby(['scope','year','variable'],sort=True):
        lo=g.loc[g.quartile.eq(1)].iloc[0];hi=g.loc[g.quartile.eq(4)].iloc[0]
        gaps.append({'scope':scope,'year':int(y),'variable':v,'high_minus_low_canopy_population_weighted_pp':float(hi.canopy_percent_population_weighted-lo.canopy_percent_population_weighted),'high_minus_low_canopy_equal_weighted_pp':float(hi.canopy_percent_equal_weighted-lo.canopy_percent_equal_weighted),'high_minus_low_change_population_weighted_pp':float(hi.change_pp_population_weighted-lo.change_pp_population_weighted)})
    save_csv('population_weighted_canopy_2000_2025.csv',rows)
    save_csv('population_weighted_resource_gaps_2000_2025.csv',gaps)
    save_csv('resource_population_scope_accounting.csv',flow)
    return {'input_rows':len(d),'unmatched_rows':0,'input_zero_population_rows':int(pop.eq(0).sum()),'scopes':{lab:int(pd.DataFrame(flow).loc[lambda z:z.scope.eq(lab)].rows.iloc[0]) for lab in ['original_balanced_resource_scope','temperature_sample_balanced_resource_scope']}}


def main():
    start=time.perf_counter();require(not (OUT/'audit.json').exists(),'Existing completed run')
    f,b,cuts,hashes=original.verify_and_load()
    input_group=ROOT/'06_logs/baseline_equity_support_spatial_20261011_run01/group_analysis_row_manifest.csv'
    earlier=original.read(input_group)
    require(f[original.KEY].equals(earlier[original.KEY]),'Current score row order differs')
    for col in ['population_2000',*[v+'_city_quartile' for v in VARS],'orthogonal_score']:
        require(f[col].equals(earlier[col]),'Current field differs: '+col)
    paths=[Path(__file__),OUT/'analysis_plan_zh.md',input_group,CANOPY,BASE,ROOT/'05_scripts/run_baseline_equity_support_20261011.py']
    hashes.update({str(p):original.digest(p) for p in paths})
    mat,meta=effects(f);agg,cols=aggregate(f,mat)
    agg.to_parquet(OUT/'population_tract_influence.parquet',index=False)
    save_json('estimate_metadata.json',meta)
    variances,diag,pairs=original.spatial_diagonal(agg[COORD].to_numpy(float),agg[cols].to_numpy(float),original.RADII,progress=True)
    refs=[];concentration=[];q=agg[cols].to_numpy(float)
    for j,m in enumerate(meta):
        sq=q[:,j]**2;require(sq.sum()>0,'Zero influence variance');share=sq/sq.sum();order=np.argsort(-share)
        concentration.append({**m,'max_tract_diagonal_share':float(share.max()),'top10_tract_diagonal_share':float(share[order[:10]].sum()),'highest_influence_tract':agg.iloc[order[0]].tract_id,'diagnostic_not_total_spatial_variance_share':True})
    for k,h in enumerate(original.RADII):
        for j,m in enumerate(meta):
            se=float(np.sqrt(variances[k,j]));require(se>0,'Nonpositive spatial standard error');mu=m['estimate_c_per_pp']
            refs.append({**m,'bandwidth_km':float(h/1000),'variance':float(variances[k,j]),'standard_error':se,'conditional_lower95':mu-1.959963984540054*se,'conditional_upper95':mu+1.959963984540054*se,'conditional_p_value':float(2*norm.sf(abs(mu/se))),'within_tract_variance_component':float(diag[j]),'unique_between_tract_pairs':int(pairs[k]),'status':'CONDITIONAL_POPULATION_WEIGHTED_REFERENCE_NOT_UNCONDITIONAL_CAUSAL_INTERVAL'})
    rr=pd.DataFrame(refs);rr['holm_p_three_social_tests']=np.nan
    for (period,h,comparison),group in rr.loc[rr.comparison.isin(['q4_minus_q1','joint_continuous_projection'])].groupby(['period','bandwidth_km','comparison']):
        require(len(group)==3,'Social testing family differs')
        rr.loc[group.index,'holm_p_three_social_tests']=original.holm(group.conditional_p_value.to_numpy())
    save_csv('population_weighted_spatial_results_three_scales.csv',rr)
    save_csv('population_weighted_influence_concentration.csv',concentration)
    trajectory=trajectories(f,b)
    require(all(original.digest(p)==h for p,h in hashes.items()),'Input changed')
    audit={'status':'COMPLETED_POPULATION_WEIGHTED_CONDITIONAL_ANALYSIS','rows':len(f),'tracts':int(f.tract_id.nunique()),'estimates':len(meta),'spatial_result_rows':len(rr),'population_weight':'FIXED_2000_WHOLE_TRACT_POPULATION','primary_time_target':'THREE_PERIOD_EQUAL_MEAN_OF_WITHIN_YEAR_POPULATION_WEIGHTED_TARGETS','years':YEARS,'spatial_radii_km':[100,200,400],'effect_rows_dropped':0,'new_forest_fits':0,'new_measurement_error_propagation':False,'pixel_population_exposure_estimated':False,'frozen_design_or_authority_changed':False,'interpretation':'CONDITIONAL_CAUSAL_IF_ORIGINAL_IDENTIFICATION_GROUP_COMPATIBILITY_AND_SPATIAL_ASSUMPTIONS_HOLD','resource':trajectory,'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'elapsed_seconds':time.perf_counter()-start,'source_hashes':hashes,'output_hashes':{p.name:original.digest(p) for p in OUT.iterdir() if p.is_file()}}
    save_json('audit.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ['source_hashes','output_hashes']},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:self_test()
    else:main()
