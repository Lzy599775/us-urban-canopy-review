"""Fixed-baseline group support and conditional spatial reference comparisons.

No training, sample trimming, outcome replacement or formal gate release.
City-relative quartiles keep their pre-existing display role. Reference intervals
condition on the existing cross-fitted score and require additional identification
and spatial regularity assumptions; they are not a validated formal causal result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / '06_logs'
EQ = LOG / 'canopy_cooling_equity_20261009_run01'
RUN = LOG / 'geo_adjusted_national_orthogonal_20261009_run01'
KEY = ['state', 'tract_id', 'exposure_year', 'outcome_year']
ID = ['state', 'tract_id']
COORD = ['centroid_epsg5070_x_m', 'centroid_epsg5070_y_m']
VARIABLES = ['income', 'poverty', 'age65plus']
BASE_FIELDS = {'income': 'income_point_usd', 'poverty': 'poverty_share', 'age65plus': 'age65plus_share_sf3_p8'}
RADII = np.array([100000., 200000., 400000.])


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return pd.read_csv(path, dtype={'state': str, 'tract_id': str, 'single_city_id': str}, float_precision='round_trip')


def write_json(path, obj):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def wendland(r):
    return np.maximum(1-r, 0)**4 * (1+4*r)


def spatial_diagonal(xy, q, radii, progress=False):
    """q already contains all estimate-specific denominators; return each variance.

    Accumulate an ordered symmetric sparse kernel including self and cross-state
    neighbors. Repeated years have been summed into their tract first.
    """
    require(np.isfinite(q).all() and np.isfinite(xy).all(), 'Nonfinite spatial inputs')
    tree = cKDTree(xy)
    diag = (q*q).sum(axis=0)
    values = np.zeros((len(radii), q.shape[1]))
    pairs = np.zeros(len(radii), dtype=np.int64)
    started = time.monotonic()
    for start in range(0, len(xy), 256):
        end = min(start+256, len(xy))
        dist = cKDTree(xy[start:end]).sparse_distance_matrix(tree, float(max(radii)), output_type='coo_matrix')
        for k, radius in enumerate(radii):
            use = dist.data <= radius
            rows, cols = dist.row[use], dist.col[use]
            weights = wendland(dist.data[use]/radius)
            from scipy.sparse import coo_matrix
            mat = coo_matrix((weights, (rows, cols)), shape=(end-start, len(xy))).tocsr()
            values[k] += (q[start:end] * (mat @ q)).sum(axis=0)
            pairs[k] += int(np.count_nonzero(start+rows != cols))
        if progress and (start//256 % 20 == 0 or end == len(xy)):
            print(f'SPATIAL_GROUPS {end}/{len(xy)} elapsed_seconds={time.monotonic()-started:.1f}', flush=True)
    require((values >= 0).all(), 'Negative variance: no truncation')
    require((pairs % 2 == 0).all(), 'Ordered kernel neighbor symmetry mismatch')
    return values, diag, pairs//2


def self_test():
    # A meaningful algebra test: 3 repeated-year groups, including coincident
    # coordinates, and dependent high-minus-low influence vectors.
    xy = np.array([[0.,0.], [0.,0.], [200.,0.], [900.,0.]])
    q = np.array([[1.,2.,-1.], [2.,-1.,3.], [-1.,0.,1.], [-2.,-1.,-3.]])/11
    radii = np.array([100.,400.,1200.])
    out, _, _ = spatial_diagonal(xy,q,radii)
    d = np.linalg.norm(xy[:,None,:]-xy[None,:,:],axis=2)
    expected = np.array([np.diag(q.T @ wendland(d/h) @ q) for h in radii])
    require(np.allclose(out,expected,atol=1e-13,rtol=1e-12),'Sparse covariance disagrees with dense algebra')
    s=np.array([2.,5.,1.,4.,7.,3.]); group=np.array([1,4,1,4,1,4])
    a=np.where(group==1,(s-s[group==1].mean())/3,0)
    b=np.where(group==4,(s-s[group==4].mean())/3,0)
    require(np.isclose((b-a).sum(),0),'Contrast influence centering')
    labels=1+(np.array([1.,1.,2.,3.])[:,None]>np.quantile([1.,1.,2.,3.],[.25,.5,.75])).sum(axis=1)
    require(labels[0]==labels[1],'Equal baseline values split')
    print('PASS_DENSE_SPARSE_CONTRAST_AND_TIE_ALGEBRA',flush=True)


def verify_and_load():
    va=LOG/'geo_adjusted_national_orthogonal_validation_20261009_run01/validation.json'
    ev=LOG/'canopy_cooling_equity_validation_20261009_run01/validation.json'
    a=json.loads((RUN/'audit.json').read_text(encoding='utf-8'))
    v=json.loads(va.read_text(encoding='utf-8'))
    ea=json.loads((EQ/'audit.json').read_text(encoding='utf-8'))
    ve=json.loads(ev.read_text(encoding='utf-8'))
    require(v['status']=='PASS_INDEPENDENT_REQUALIFIED_NESTED_MODELS_SCORES_AND_EXECUTION' and v['maximum_recompute_error']==0,'Model reload gate')
    require(ve['fixed_baseline_thresholds_verified'] and ve['group_accounting_verified'],'Existing grouping validation')
    require(ve['input_audit_sha256']==digest(EQ/'audit.json'),'Grouping receipt mismatch')
    files = [RUN/'crossfit_scores.csv', RUN/'sample_manifest.csv', EQ/'effect_equity_row_manifest.csv',
             EQ/'baseline_equity_group_manifest.csv', EQ/'fixed_baseline_city_quartile_thresholds.csv']
    for p in files:
        expected = a['outputs'][p.name] if p.parent==RUN else ea['output_file_sha256'][p.name]
        require(digest(p)==expected,f'Changed input {p}')
    f=read(EQ/'effect_equity_row_manifest.csv'); s=read(RUN/'crossfit_scores.csv'); sample=read(RUN/'sample_manifest.csv')
    require(len(f)==len(s)==len(sample)==104371 and f[KEY].equals(s[KEY]) and f[KEY].equals(sample[KEY]),'Frozen rows/order mismatch')
    require(not f.duplicated(KEY).any() and f.tract_id.nunique()==42373,'Frozen sample identity')
    for col in ['spatial_fold','spatial_block_id','theta','orthogonal_score']:
        require(f[col].equals(s[col]),f'Existing score identity {col}')
    require(f.groupby(ID).spatial_fold.nunique().eq(1).all(),'Tract folds differ')
    require(f.groupby('spatial_block_id').spatial_fold.nunique().eq(1).all(),'Block folds differ')
    score=s.theta+s.residual_t*(s.residual_y-s.intercept-s.theta*s.residual_t)/s.conditional_variance
    require(np.allclose(score,s.orthogonal_score,atol=1e-12,rtol=1e-12),'Orthogonal score mismatch')
    require((s.conditional_variance>0).all() and np.isfinite(s[['conditional_variance','residual_t','orthogonal_score',*COORD]].to_numpy()).all(),'Invalid support or coordinates')
    for col in [*COORD,'residual_t','conditional_variance','standardized_residual_square']:
        f[col]=s[col].to_numpy()
    baseline=read(EQ/'baseline_equity_group_manifest.csv'); cuts=read(EQ/'fixed_baseline_city_quartile_thresholds.csv')
    original_path=LOG/'national_analysis_baseline_join_20260914_run01/national_analysis_baseline_readiness.csv'
    require(digest(original_path)==ea['input_file_sha256'][str(original_path)],'Changed baseline source')
    original=read(original_path)
    require(len(baseline)==len(original)==64999 and baseline[ID].equals(original[ID]),'Baseline ID mismatch')
    z=original.merge(baseline,on=ID,validate='one_to_one',how='left',suffixes=('','_manifest'))
    for variable in VARIABLES:
        field=BASE_FIELDS[variable]
        ok=z.positive_land & z.single_city_id.notna() & z[field].notna()
        if variable=='income': ok &= z.income_status.eq('REPORTED_POINT_ESTIMATE')
        for city,g in z.loc[ok].groupby('single_city_id',sort=True):
            cut=cuts.loc[cuts.variable.eq(variable)&cuts.single_city_id.eq(city)]
            require(len(cut)==1,'Threshold city not unique')
            edges=np.quantile(g[field].to_numpy(float),[.25,.5,.75],method='linear')
            require(np.array_equal(edges,cut[['q25','q50','q75']].to_numpy(float)[0]),'Baseline threshold changed')
            labels=1+(g[field].to_numpy()[:,None]>edges).sum(axis=1)
            require(np.array_equal(labels,g[variable+'_city_quartile'].to_numpy()),'Baseline labels changed')
        require(f.groupby(ID)[variable+'_city_quartile'].nunique(dropna=False).eq(1).all(),'Annual group reassignment')
    sources=files+[va,ev,original_path,EQ/'audit.json',RUN/'audit.json',ROOT/'00_admin/project_state.yaml',Path(__file__)]
    return f,baseline,cuts,{str(p):digest(p) for p in sources}


def support_tables(f):
    rows=[]; overlap=[]; fold_checks=[]
    for var in VARIABLES:
        field=var+'_city_quartile'
        strata=[('ALL','ALL',f)]
        strata += [(str(y),'ALL',g) for y,g in f.groupby('outcome_year')]
        strata += [('ALL',str(int(k)),g) for k,g in f.groupby('spatial_fold')]
        strata += [(str(y),str(int(k)),g) for (y,k),g in f.groupby(['outcome_year','spatial_fold'])]
        for year,fold,g in strata:
            for q,h in g.groupby(field,dropna=False):
                r={'variable':var,'outcome_year':year,'spatial_fold':fold,'quartile':'unassigned' if pd.isna(q) else str(int(q)),
                   'rows':len(h),'tracts':h.tract_id.nunique(),'cities':h.single_city_id.nunique(),'spatial_blocks':h.spatial_block_id.nunique(),
                   'negative_treatment_rows':int(h.treatment_pp_change_from_2000.lt(0).sum()),'zero_treatment_rows':int(h.treatment_pp_change_from_2000.eq(0).sum()),
                   'positive_treatment_rows':int(h.treatment_pp_change_from_2000.gt(0).sum()),'conditional_variance_positive_rows':int(h.conditional_variance.gt(0).sum()),
                   'negative_residual_t_rows':int(h.residual_t.lt(0).sum()),'positive_residual_t_rows':int(h.residual_t.gt(0).sum())}
                for name in ['treatment_pp_change_from_2000','residual_t','conditional_variance','standardized_residual_square']:
                    for quant in [0,.01,.5,.99,1]:r[name+'_q'+str(quant)]=float(h[name].quantile(quant))
                r['support_status']='NUMERICAL_LOCAL_VARIATION_PRESENT' if h.residual_t.lt(0).any() and h.residual_t.gt(0).any() else 'ONE_SIDED_OR_ZERO_LOCAL_VARIATION'
                rows.append(r)
            groups=[g.loc[g[field].eq(q)] for q in [1,2,3,4]]
            if any(h.empty for h in groups):
                overlap.append({'variable':var,'outcome_year':year,'spatial_fold':fold,'status':'AT_LEAST_ONE_EMPTY_QUARTILE'});continue
            for mode in ['observed_full_range','central_1_99_percent_DIAGNOSTIC_ONLY']:
                lo,hi=(0,1) if mode=='observed_full_range' else (.01,.99)
                low=max(float(h.treatment_pp_change_from_2000.quantile(lo)) for h in groups)
                high=min(float(h.treatment_pp_change_from_2000.quantile(hi)) for h in groups)
                for q,h in enumerate(groups,1):
                    within=h.treatment_pp_change_from_2000.between(low,high) if low<=high else pd.Series(False,index=h.index)
                    overlap.append({'variable':var,'outcome_year':year,'spatial_fold':fold,'quartile':q,'range_definition':mode,
                         'shared_low_pp':low,'shared_high_pp':high,'rows':len(h),'inside_shared_range_rows':int(within.sum()),
                         'outside_shared_range_rows':int((~within).sum()),'inside_shared_range_share':float(within.mean()),
                         'status':'DESCRIPTIVE_RANGE_OVERLAP_NOT_CONDITIONAL_DENSITY_PROOF' if low<high else 'NO_NONDEGENERATE_SHARED_RANGE',
                         'rows_dropped':0})
        for outer in range(5):
            for q in [1,2,3,4]:
                train=f.loc[f[field].eq(q)&f.spatial_fold.ne(outer)]; test=f.loc[f[field].eq(q)&f.spatial_fold.eq(outer)]
                require(len(train)>0 and len(test)>0,'Empty group in frozen train/test fold')
                lo=float(train.treatment_pp_change_from_2000.min());hi=float(train.treatment_pp_change_from_2000.max())
                within=test.treatment_pp_change_from_2000.between(lo,hi)
                fold_checks.append({'variable':var,'quartile':q,'outer_fold':outer,'train_rows':len(train),'test_rows':len(test),
                       'train_treatment_min_pp':lo,'train_treatment_max_pp':hi,'test_outside_training_range_rows':int((~within).sum()),
                       'test_positive_variance_rows':int(test.conditional_variance.gt(0).sum()),'rows_dropped':0,
                       'status':'TRAIN_TEST_RANGE_DIAGNOSTIC_NO_REFIT_NO_TRIMMING'})
    return pd.DataFrame(rows),pd.DataFrame(overlap),pd.DataFrame(fold_checks)


def influences(f):
    psi=f.orthogonal_score.to_numpy(float); n=len(f); columns=[]; meta=[]
    for var in VARIABLES:
        field=var+'_city_quartile'; means={}; qs={}
        for q in [1,2,3,4]:
            mask=f[field].eq(q).to_numpy(); count=int(mask.sum()); require(count>0,'Empty quartile')
            mu=float(psi[mask].mean()); influence=np.where(mask,(psi-mu)/count,0.)
            means[q]=mu;qs[q]=influence;columns.append(influence)
            meta.append({'variable':var,'comparison':'quartile_mean','quartile':q,'estimate_c_per_pp':mu,'rows':count,
                         'target':'ROW_WEIGHTED_SCORE_MEAN_OF_CITY_RELATIVE_GROUP_POOLED_CITIES'})
        columns.append(qs[4]-qs[1]); meta.append({'variable':var,'comparison':'q4_minus_q1','quartile':None,
           'estimate_c_per_pp':means[4]-means[1],'rows':int(f[field].isin([1,4]).sum()),
           'target':'POOLED_GROUP_DIFFERENCE_NOT_EQUAL_CITY_OR_MATCHED_CITY_CONTRAST'})
    # Preserve the existing primary continuous heterogeneity design.
    x=np.column_stack([np.ones(n),f.outcome_year.eq(2019),f.outcome_year.eq(2024),f.income_point_usd_2000/10000.,
                       f.poverty_share_2000/.1,f.age65plus_share_2000/.1,f.population_2000/10000.]).astype(float)
    require(np.linalg.matrix_rank(x)==7,'Continuous design rank')
    inv=np.linalg.inv(x.T@x); beta=np.linalg.lstsq(x,psi,rcond=None)[0]; e=psi-x@beta
    for j,var in enumerate(VARIABLES,3):
        columns.append((x@inv[:,j])*e);meta.append({'variable':var,'comparison':'joint_continuous_projection','quartile':None,
         'estimate_c_per_pp':float(beta[j]),'rows':n,'target':'CONTINUOUS_EFFECT_SCORE_GRADIENT_WITH_YEAR_POPULATION_AND_ALL_THREE_SOCIAL_VARIABLES',
         'social_unit':'10000_USD_1999' if var=='income' else '10_PERCENTAGE_POINTS'})
    mat=np.column_stack(columns);require(np.max(np.abs(mat.sum(axis=0)))<1e-9,'Influence not centered')
    t=f[ID+COORD].copy()
    for j in range(mat.shape[1]):t[f'if_{j}']=mat[:,j]
    require(t.groupby(ID)[COORD].nunique().le(1).all().all(),'Within-tract coordinate drift')
    cols=[f'if_{j}' for j in range(mat.shape[1])]
    agg=t.groupby(ID,sort=True).agg({**{c:'first' for c in COORD},**{c:'sum' for c in cols}}).reset_index()
    return agg,cols,meta


def holm(ps):
    ps=np.asarray(ps,float);order=np.argsort(ps);ranked=ps[order];res=np.empty(len(ps));res[order]=np.minimum(1,np.maximum.accumulate(ranked*(len(ps)-np.arange(len(ps)))))
    return res


def execute(args):
    start=time.monotonic(); require(not args.out.exists(),'Refuse existing output overwrite')
    f,b,cuts,hashes=verify_and_load();args.out.mkdir(parents=True)
    b.to_csv(args.out/'baseline_groups_2000_frozen_copy.csv',index=False)
    cuts.to_csv(args.out/'baseline_city_quartile_thresholds_frozen_copy.csv',index=False)
    s,o,fc=support_tables(f)
    s.to_csv(args.out/'group_support_by_year_fold.csv',index=False);o.to_csv(args.out/'group_shared_treatment_range_DIAGNOSTIC.csv',index=False);fc.to_csv(args.out/'group_fold_training_support.csv',index=False)
    f.to_csv(args.out/'group_analysis_row_manifest.csv',index=False)
    write_json(args.out/'support_receipt.json',{'status':'PASS_FIXED_BASELINE_GROUPING_AND_NUMERICAL_SUPPORT',
        'rows':len(f),'tracts':f.tract_id.nunique(),'baseline_tracts':len(b),'baseline_cities':cuts.single_city_id.nunique(),
        'unassigned_rows':{v:int(f[v+'_city_quartile'].isna().sum()) for v in VARIABLES},
        'rows_dropped':0,'folds_changed':False,'thresholds_changed':False,'new_model_fits':0,
        'continuous_density_support_proven':False,'source_hashes':hashes})
    print('SUPPORT_COMPLETED rows=104371 unchanged baseline_groups_verified',flush=True)
    a,cols,meta=influences(f);q=a[cols].to_numpy(float);xy=a[COORD].to_numpy(float)
    a.to_parquet(args.out/'tract_group_influence.parquet',index=False)
    variances,diag,pairs=spatial_diagonal(xy,q,RADII,progress=True)
    concentration=[]
    for j,m in enumerate(meta):
        d=q[:,j]**2;require(d.sum()>0,'No influence variance');share=d/d.sum();order=np.argsort(-share)
        concentration.append({**m,'max_tract_diagonal_share':float(share.max()),'top10_tract_diagonal_share':float(share[order[:10]].sum()),
           'max_influence_tract_id':str(a.iloc[order[0]].tract_id),'diagnostic_not_automatic_gate':True})
    pd.DataFrame(concentration).to_csv(args.out/'group_influence_concentration.csv',index=False)
    refs=[]
    for k,radius in enumerate(RADII):
        for j,m in enumerate(meta):
            se=float(np.sqrt(variances[k,j]));estimate=m['estimate_c_per_pp'];p=float(2*norm.sf(abs(estimate/se))) if se>0 else None
            refs.append({**m,'bandwidth_km':radius/1000,'variance':float(variances[k,j]),'standard_error':se,
               'reference_lower95':estimate-1.959963984540054*se,'reference_upper95':estimate+1.959963984540054*se,
               'reference_p_value':p,'within_tract_variance_component':float(diag[j]),'unique_between_tract_pairs':int(pairs[k]),
               'status':'CONDITIONAL_SPATIAL_REFERENCE_NOT_FORMALLY_RELEASED','frozen_measurement_uncertainty_propagated':False,
               'city_relative_group_score_identification':'REQUIRES_GROUP_COMPATIBLE_CONDITIONAL_CAUSAL_MODEL_ADDITIONAL_ASSUMPTION'})
    rr=pd.DataFrame(refs);rr['holm_reference_p_three_primary_contrasts']=np.nan
    for radius in RADII/1000:
        idx=rr.index[rr.bandwidth_km.eq(radius)&rr.comparison.eq('q4_minus_q1')]
        rr.loc[idx,'holm_reference_p_three_primary_contrasts']=holm(rr.loc[idx,'reference_p_value'].to_numpy())
    rr.to_csv(args.out/'group_spatial_reference_intervals_NOT_FORMAL.csv',index=False)
    require(all(digest(p)==h for p,h in hashes.items()),'Source changed during calculation')
    audit={'status':'PASS_GROUP_SUPPORT_AND_CONDITIONAL_SPATIAL_REFERENCE_COMPUTATION',
       'rows':len(f),'tracts':f.tract_id.nunique(),'baseline_thresholds_fixed':True,'rows_dropped':0,
       'unassigned_rows':{v:int(f[v+'_city_quartile'].isna().sum()) for v in VARIABLES},'numerical_estimates':len(meta),
       'reference_interval_rows':len(rr),'bandwidths_km':list(RADII/1000),'new_model_fits':0,
       'formal_group_inference_gate':'HOLD_MEASUREMENT_MECHANISM_SPATIAL_REGULARITY_AND_GROUP_SCORE_IDENTIFICATION',
       'formal_causal_intervals_released':False,'formal_inputs_or_authority_modified':False,
       'continuous_variables_primary_quartiles_display_role_unchanged':True,
       'source_hashes':hashes,'elapsed_seconds':time.monotonic()-start,
       'outputs':{p.name:digest(p) for p in args.out.iterdir() if p.is_file()}}
    write_json(args.out/'audit.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ['source_hashes','outputs']},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);p.add_argument('--self-test',action='store_true');args=p.parse_args()
    if args.self_test:self_test()
    else:
        require(args.out is not None,'Output directory required');execute(args)
