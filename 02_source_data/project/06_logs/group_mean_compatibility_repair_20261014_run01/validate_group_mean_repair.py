"""Independent reload and direct-pair verification; never calls producing functions."""
from pathlib import Path
import hashlib,json,sys,time
import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import norm

TASK=Path(__file__).resolve().parent
OUT=TASK/'results_csv';CHECKPOINT=TASK/'results/checkpoints';ROOT=TASK.parents[1];LOG=ROOT/'06_logs'
KEY=['state','tract_id','exposure_year','outcome_year'];ID=['state','tract_id']
XY=['centroid_epsg5070_x_m','centroid_epsg5070_y_m'];DIMS=['income','poverty','age65plus']
sys.stdout.reconfigure(encoding='utf8')

def require(ok,msg):
    if not ok:raise RuntimeError(msg)

def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def arrhash(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.shape).encode()+str(a.dtype).encode()+a.tobytes()).hexdigest()

def csv(p):return pd.read_csv(p,dtype={'state':str,'tract_id':str},float_precision='round_trip')

def main():
    start=time.monotonic();audit=load(OUT/'audit.json');cfg=load(TASK/'protocol.json')
    require(not (TASK/'independent_validation.json').exists(),'Refuse validation overwrite')
    require(all(sha(p)==h for p,h in audit['source_hashes'].items()),'Input hash mismatch')
    require(all(sha(OUT/p)==h for p,h in audit['outputs'].items()),'Output hash mismatch')
    s=csv(LOG/'geo_adjusted_national_orthogonal_20261009_run01/crossfit_scores.csv')
    g=csv(LOG/'baseline_equity_support_spatial_20261011_run01/group_analysis_row_manifest.csv')
    probs=csv(OUT/'crossfit_group_probabilities.csv')
    require(s[KEY].equals(g[KEY]) and s[KEY].equals(probs[KEY]),'Row order/identity')
    require(len(s)==104371 and len(g[ID].drop_duplicates())==42373,'Sample size')
    require(not s.duplicated(KEY).any(),'Duplicate keys')
    folds=g.spatial_fold.to_numpy(int);x=g[cfg['x_columns']].to_numpy()
    pred_errors=[];fits=[]
    for dim in DIMS:
        y=g[dim+'_city_quartile'].fillna(0).to_numpy(int)
        for k in range(5):
            name=f'{dim}_fold_{k}';receipt=load(CHECKPOINT/f'{name}.json')
            z=np.load(CHECKPOINT/f'{name}.npz');train=np.flatnonzero(folds!=k);test=np.flatnonzero(folds==k)
            require(np.array_equal(train,z['train_indices']) and np.array_equal(test,z['test_indices']),'Persisted split mismatch')
            require(receipt['identity']['train_indices_sha256']==arrhash(train),'Train index hash')
            require(receipt['identity']['test_indices_sha256']==arrhash(test),'Test index hash')
            require(receipt['identity']['train_x_sha256']==arrhash(x[train]),'Actual training X hash')
            require(receipt['identity']['train_y_sha256']==arrhash(y[train]),'Actual training labels hash')
            require(not set(g.tract_id.iloc[train])&set(g.tract_id.iloc[test]),'Tract leakage')
            require(not set(g.spatial_block_id.iloc[train])&set(g.spatial_block_id.iloc[test]),'Block leakage')
            model=joblib.load(CHECKPOINT/f'{name}.joblib')
            require(np.array_equal(model.classes_,np.arange(5)) and len(model.estimators_)==200,'Model classes or trees')
            for param,value in cfg['probability_model'].items():
                if param!='class':require(model.get_params()[param]==value,'Parameter mismatch '+param)
            actual=model.predict_proba(x[test]);saved=probs.loc[test,[f'{dim}_p{j}' for j in range(5)]].to_numpy()
            err=float(np.max(abs(actual-saved)));pred_errors.append(err)
            require(np.allclose(actual,saved,atol=1e-14,rtol=1e-13),'Reload predictions differ')
            require(np.max(abs(actual.sum(axis=1)-1))<1e-12,'Probability sums')
            fits.append({'fit':name,'train_rows':len(train),'test_rows':len(test),'max_reload_error':err,'trees':200})
            del model
        print('MODEL_RELOAD',dim,'verified',flush=True)
    theta=s.theta.to_numpy();v=s.residual_t.to_numpy();u=s.residual_y.to_numpy();b=s.intercept.to_numpy();vv=s.conditional_variance.to_numpy()
    correction=v*(u-b-theta*v)/vv
    refs=csv(OUT/'all_group_reference_intervals.csv');meta=csv(OUT/'estimates_and_corrections.csv');order=load(OUT/'estimate_order.json')
    influence={};estimates={};estimate_errors=[]
    for dim in DIMS:
        y=g[dim+'_city_quartile'].fillna(0).to_numpy(int)
        require((y==0).sum()==715,'Unassigned rows dropped')
        for j in range(5):
            ind=(y==j);ng=ind.sum();p=probs[f'{dim}_p{j}'].to_numpy()
            mu=float((theta[ind].sum()+np.dot(p,correction))/ng)
            key=f'{dim}_q{j}';estimates[key]=mu
            infl=p*correction/ng;infl[ind]+=(theta[ind]-mu)/ng
            influence[key]=infl
            require(abs(infl.sum())<1e-9,'Influence centering')
            rec=meta.loc[meta.key.eq(key)].iloc[0];estimate_errors.append(abs(mu-rec.estimate))
            require(abs(mu-rec.estimate)<1e-12,'Independent group estimate mismatch')
        key=f'{dim}_q4_minus_q1';estimates[key]=estimates[dim+'_q4']-estimates[dim+'_q1'];influence[key]=influence[dim+'_q4']-influence[dim+'_q1']
        rec=meta.loc[meta.key.eq(key)].iloc[0];estimate_errors.append(abs(estimates[key]-rec.estimate))
        require(abs(estimates[key]-rec.estimate)<1e-12,'Independent contrast estimate mismatch')
        require(abs(sum((y==j).mean()*estimates[f'{dim}_q{j}'] for j in range(5))-s.orthogonal_score.mean())<1e-12,'Partition reconstruction')
    estimates['national']=float(s.orthogonal_score.mean());influence['national']=(s.orthogonal_score.to_numpy()-estimates['national'])/len(s)
    data=g[ID+XY].copy()
    for key in order:data[key]=influence[key]
    agg=data.groupby(ID,sort=True).agg({**{c:'first' for c in XY},**{k:'sum' for k in order}}).reset_index()
    saved=csv(OUT/'tract_joint_influences.csv')
    require(agg[ID].equals(saved[ID]),'Tract aggregation order')
    influence_error=float(abs(agg[order].to_numpy()-saved[order].to_numpy()).max())
    require(influence_error<1e-12,'Tract influences differ')
    q=agg[order].to_numpy();xy=agg[XY].to_numpy();tree=cKDTree(xy);radii=np.array([100000.,200000.,400000.])
    # Direct unordered pair products: independent from the producer's sparse
    # matrix multiplication. Recompute every diagonal and the three Q4/Q1 covariances.
    diag=(q*q).sum(axis=0);values=np.tile(diag,(3,1));paircounts=np.zeros(3,dtype=np.int64)
    cross_indices=[(order.index(d+'_q4'),order.index(d+'_q1')) for d in DIMS]
    cross=np.tile(np.array([q[:,a]@q[:,b] for a,b in cross_indices]),(3,1))
    for begin in range(0,len(xy),64):
        end=min(begin+64,len(xy));neighbors=tree.query_ball_point(xy[begin:end],radii[-1],workers=1)
        lengths=np.array([len(a) for a in neighbors]);left=np.repeat(np.arange(begin,end),lengths);right=np.concatenate(neighbors).astype(np.int64)
        use=right>left;left=left[use];right=right[use]
        distance=np.sqrt(((xy[left]-xy[right])**2).sum(axis=1))
        product=q[left]*q[right]
        paircross=np.column_stack([q[left,a]*q[right,b]+q[left,b]*q[right,a] for a,b in cross_indices])
        for k,h in enumerate(radii):
            use=distance<=h;r=distance[use]/h;w=(1-r)**4*(1+4*r)
            values[k]+=2*np.sum(product[use]*w[:,None],axis=0)
            cross[k]+=np.sum(paircross[use]*w[:,None],axis=0);paircounts[k]+=int(use.sum())
        if begin%6400==0 or end==len(xy):print('DIRECT_PAIR_VERIFY',end,'/',len(xy),'seconds',round(time.monotonic()-start,1),flush=True)
    covariance=np.load(OUT/'joint_spatial_covariance.npz')['covariance']
    err=float(np.max(abs(values-np.diagonal(covariance,axis1=1,axis2=2))))
    require(np.allclose(values,np.diagonal(covariance,axis1=1,axis2=2),atol=1e-12,rtol=1e-9),'Independent spatial variances differ')
    for k,h in enumerate(radii):
        for j,(a,b) in enumerate(cross_indices):
            require(np.isclose(cross[k,j],covariance[k,a,b],atol=1e-12,rtol=1e-9),'Independent cross-group covariance differs')
        for j,key in enumerate(order):
            row=refs.loc[refs.key.eq(key)&refs.bandwidth_km.eq(h/1000)].iloc[0]
            require(np.isclose(row.variance,values[k,j],atol=1e-12,rtol=1e-9),'CSV variance differs')
            se=np.sqrt(values[k,j]);require(np.isclose(row.lower95,estimates[key]-norm.ppf(.975)*se,atol=1e-10),'Lower interval endpoint')
            require(np.isclose(row.upper95,estimates[key]+norm.ppf(.975)*se,atol=1e-10),'Upper interval endpoint')
            require(row.between_tract_pairs==paircounts[k],'Pair count mismatch')
        r=refs.loc[refs.comparison.eq('q4_minus_q1')&refs.bandwidth_km.eq(h/1000)]
        ps=r.reference_p.to_numpy();idx=np.argsort(ps);adjusted=np.minimum(1,np.maximum.accumulate(ps[idx]*np.array([3,2,1])))
        require(np.allclose(adjusted,r.holm_p_three_contrasts.to_numpy()[idx],atol=1e-14),'Holm family mismatch')
    events=[json.loads(x) for x in (TASK/'results/fit_events.jsonl').read_text(encoding='utf8').splitlines()]
    require(sum(e['event']=='STARTED' for e in events)==15 and sum(e['event']=='COMPLETED' for e in events)==15,'Executed model counts')
    failures=[e for e in events if e['event']=='FAILED']
    require(len(failures)==1 and 'predicted.to_parquet' in failures[0]['traceback'] and 'pyarrow' in failures[0]['traceback'],'Unexpected failure')
    reused=[json.loads(x) for x in (OUT/'fit_events.jsonl').read_text(encoding='utf8').splitlines()]
    require(len(reused)==15 and all(x['event']=='REUSED' for x in reused),'Recovery repeated or omitted models')
    require(all(sha(p)==h for p,h in audit['source_hashes'].items()),'Source changed during validation')
    pd.DataFrame(fits).to_csv(TASK/'independent_model_reload.csv',index=False)
    result={'status':'PASS_INDEPENDENT_PROBABILITIES_ESTIMATES_INFLUENCES_AND_SPATIAL_VARIANCES','rows':104371,'tracts':42373,'duplicate_keys':0,'unmatched':0,'dropped':0,'new_probability_fits_in_validator':0,'models_reloaded':15,'maximum_probability_reload_error':max(pred_errors),'maximum_estimate_error':max(estimate_errors),'maximum_tract_influence_error':influence_error,'maximum_spatial_variance_error':err,'spatial_variances_directly_verified':57,'high_low_cross_covariances_verified':9,'primary_interval_rows':45,'input_and_produced_output_hashes_verified':True,'formal_authority_modified':False,'coverage_proven':False,'audit_sha256':sha(OUT/'audit.json'),'validator_sha256':sha(__file__),'elapsed_seconds':time.monotonic()-start}
    with (TASK/'independent_validation.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
