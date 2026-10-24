"""Bounded compatible group-score computation; original research inputs are read-only."""
from pathlib import Path
import argparse, datetime, hashlib, json, sys, time, traceback
import joblib
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from scipy.stats import norm
import sklearn
from sklearn.ensemble import RandomForestClassifier

TASK = Path(__file__).resolve().parent
PREFLIGHT = TASK / 'preflight_run02'
ROOT = TASK.parents[1]
LOG = ROOT / '06_logs'
MODEL = LOG / 'geo_adjusted_national_orthogonal_20261009_run01'
GROUP = LOG / 'baseline_equity_support_spatial_20261011_run01'
KEY = ['state','tract_id','exposure_year','outcome_year']
ID = ['state','tract_id']
XY = ['centroid_epsg5070_x_m','centroid_epsg5070_y_m']
VARS = ['income','poverty','age65plus']
sys.stdout.reconfigure(encoding='utf8')

def req(ok, msg):
    if not ok:
        raise RuntimeError(msg)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def ah(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.shape).encode()+str(a.dtype).encode()+a.tobytes()).hexdigest()

def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def write(p, x):
    with Path(p).open('x',encoding='utf8') as f:
        json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)

def event(out, kind, **kw):
    item={'at':datetime.datetime.now().isoformat(),'event':kind,**kw}
    with (out/'fit_events.jsonl').open('a',encoding='utf8') as f:
        f.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(item,ensure_ascii=False),flush=True)

def read(p):
    return pd.read_csv(p,dtype={'state':str,'tract_id':str,'single_city_id':str},float_precision='round_trip')

def inputs():
    cfg=load(TASK/'protocol.json')
    a=load(MODEL/'audit.json'); ga=load(GROUP/'audit.json')
    valid=LOG/'geo_adjusted_national_orthogonal_validation_20261009_run01/validation.json'
    v=load(valid)
    req(v['input_audit_sha256']==sha(MODEL/'audit.json'),'National model validation receipt identity')
    req(v['status']=='PASS_INDEPENDENT_REQUALIFIED_NESTED_MODELS_SCORES_AND_EXECUTION','National validation status')
    req(load(GROUP/'independent_validation.json')['input_audit_sha256']==sha(GROUP/'audit.json'),'Group validation receipt identity')
    for name in ['crossfit_scores.csv','sample_manifest.csv']:
        req(sha(MODEL/name)==a['outputs'][name],'Frozen national input changed: '+name)
    for name in ['group_analysis_row_manifest.csv','baseline_groups_2000_frozen_copy.csv','baseline_city_quartile_thresholds_frozen_copy.csv']:
        req(sha(GROUP/name)==ga['outputs'][name],'Frozen grouping input changed: '+name)
    s=read(MODEL/'crossfit_scores.csv'); g=read(GROUP/'group_analysis_row_manifest.csv'); m=read(MODEL/'sample_manifest.csv')
    req(len(s)==len(g)==len(m)==cfg['rows'],'Row count')
    req(s[KEY].equals(g[KEY]) and s[KEY].equals(m[KEY]),'Row identity/order')
    req(not s.duplicated(KEY).any() and len(s[ID].drop_duplicates())==cfg['tracts'],'Sample keys')
    req(s.state.str.len().eq(2).all() and s.tract_id.str.len().eq(11).all(),'Leading-zero ID strings')
    for col in ['spatial_fold','spatial_block_id','theta','orthogonal_score',*XY]:
        req(s[col].equals(g[col]),'Group/national field mismatch: '+col)
    req(g.groupby(ID).spatial_fold.nunique().eq(1).all(),'Tract leakage')
    req(g.groupby('spatial_block_id').spatial_fold.nunique().eq(1).all(),'Block leakage')
    req(g.groupby(ID)[XY].nunique().eq(1).all().all(),'Coordinate drift')
    req(set(g.spatial_fold)==set(range(5)),'Fold IDs')
    x=g[cfg['x_columns']].to_numpy(float)
    req(np.isfinite(x).all() and g.groupby(ID)[cfg['x_columns']].nunique().eq(1).all().all(),'X values/baseline stability')
    req(np.isfinite(s[['theta','intercept','residual_t','residual_y','conditional_variance',*XY]].to_numpy()).all(),'Nonfinite score inputs')
    req(s.conditional_variance.gt(0).all(),'Nonpositive variance; no clipping')
    correction=s.residual_t.to_numpy()*(s.residual_y.to_numpy()-s.intercept.to_numpy()-s.theta.to_numpy()*s.residual_t.to_numpy())/s.conditional_variance.to_numpy()
    err=float(np.max(abs(s.theta.to_numpy()+correction-s.orthogonal_score.to_numpy())))
    req(err<1e-12,'Existing score recomputation')
    folds=g.spatial_fold.to_numpy(int)
    labels={}
    support=[]
    for dim in VARS:
        col=dim+'_city_quartile'
        req(g[col].isna().sum()==715,'Unassigned count '+dim)
        req(g.groupby(ID)[col].nunique(dropna=False).eq(1).all(),'Group changes between years')
        y=g[col].fillna(0).to_numpy(int)
        req(set(y)==set(range(5)),'Invalid group labels')
        labels[dim]=y
        for k in range(5):
            tr=np.flatnonzero(folds!=k);te=np.flatnonzero(folds==k)
            req(set(y[tr])==set(range(5)),'Training fold missing group class')
            req(not set(g.tract_id.iloc[tr]) & set(g.tract_id.iloc[te]),'Leaked tract')
            req(not set(g.spatial_block_id.iloc[tr]) & set(g.spatial_block_id.iloc[te]),'Leaked spatial block')
            for q in range(5):
                support.append({'dimension':dim,'fold':k,'group':q,'train_rows':int((y[tr]==q).sum()),'test_rows':int((y[te]==q).sum())})
    paths=[Path(__file__),TASK/'protocol.json',MODEL/'audit.json',MODEL/'executed_config.json',MODEL/'sample_manifest.csv',MODEL/'crossfit_scores.csv',MODEL/'spatial_covariance_reference.csv',valid,GROUP/'audit.json',GROUP/'independent_validation.json',GROUP/'group_analysis_row_manifest.csv',GROUP/'baseline_groups_2000_frozen_copy.csv',GROUP/'baseline_city_quartile_thresholds_frozen_copy.csv',ROOT/'00_admin/project_state.yaml',LOG/'state_alignment_policy_holdout_20261012_run02/state_alignment_record_20261012_run02.json',LOG/'manuscript_baseline_adjudication_20261014_run01/方法裁定_社会组得分与平均响应.md']
    req(cfg['x_columns']==load(MODEL/'executed_config.json')['x_columns'],'X definition differs')
    ident={'source_hashes':{str(p):sha(p) for p in paths},'x_sha256':ah(x),'fold_sha256':ah(folds),'label_sha256':{d:ah(y) for d,y in labels.items()},'rows':len(g),'tracts':cfg['tracts'],'expected_unique_fits':15,'python':sys.version,'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__}
    return cfg,g,s,x,folds,labels,correction,ident,pd.DataFrame(support)

def kernel(r):
    return np.maximum(1-r,0)**4*(1+4*r)

def covariance(xy,q,radii,progress=True):
    tree=cKDTree(xy); result=np.zeros((len(radii),q.shape[1],q.shape[1]));pairs=np.zeros(len(radii),dtype=np.int64)
    for start in range(0,len(xy),128):
        end=min(start+128,len(xy))
        d=cKDTree(xy[start:end]).sparse_distance_matrix(tree,float(max(radii)),output_type='coo_matrix')
        for k,h in enumerate(radii):
            use=d.data<=h;rows=d.row[use];cols=d.col[use]
            mat=coo_matrix((kernel(d.data[use]/h),(rows,cols)),shape=(end-start,len(xy))).tocsr()
            result[k]+=q[start:end].T@(mat@q)
            pairs[k]+=int(np.count_nonzero(start+rows!=cols))
        if progress and (start%6400==0 or end==len(xy)):
            print('SPATIAL',end,'/',len(xy),flush=True)
    req((pairs%2==0).all(),'Pair symmetry')
    req(np.max(abs(result-result.transpose(0,2,1)))<1e-12,'Covariance asymmetric')
    req(np.diagonal(result,axis1=1,axis2=2).min()>=0,'Negative variance; no truncation')
    req(min(np.linalg.eigvalsh(c).min() for c in result)>-1e-12,'Joint covariance not PSD')
    return result,pairs//2

def algebra_test():
    # Exact finite-support population: X changes group probability; group also
    # changes treatment variance and true response conditional on X.
    xx=np.repeat([0,1],4);gg=np.tile(np.repeat([0,1],2),2)
    p1=np.where(xx==0,.2,.8); ww=.5*np.where(gg==1,p1,1-p1)/2
    variance=np.where(gg==0,1.,4.);v=np.sqrt(variance)*np.tile([-1,1],4)
    tau=np.where(gg==0,-.1,-.3)+xx*.04;u=v*tau
    vx=(1-p1)*1+p1*4
    th=((1-p1)*(-.1+xx*.04)+p1*4*(-.3+xx*.04))/vx
    ind=(gg==1).astype(float);pi=float(ww@ind);target=float(ww@(ind*th)/pi)
    def est(h=0,ve=0,be=0,pe=0,ee=0,mm=0):
        vv=v-ee*(1+gg+xx);uu=u-mm*(2+gg+xx)
        return float(ww@(ind*(th+h)+(p1+pe)*vv*(uu-be-(th+h)*vv)/(vx+ve))/pi)
    req(abs(est()-target)<1e-12,'Compatible score target algebra')
    derivatives={}
    for param in ['h','ve','be','pe','ee','mm']:
        eps=1e-5;d=(est(**{param:eps})-est(**{param:-eps}))/(2*eps)
        derivatives[param]=d;req(abs(d)<1e-8,'Nonorthogonal direction '+param)
    old=float(ww@(ind*(th+v*(u-th*v)/vx))/pi)
    req(abs(old-target)>1e-3,'Counterexample not informative')
    xy=np.array([[0.,0.],[0.,0.],[250.,0.],[900.,0.]])
    two=np.array([[1,2],[2,-1],[-1,0],[-2,-1]],float)/11
    q=np.column_stack([two,two[:,0]-two[:,1]])
    radii=[100.,400.,1200.]; got,_=covariance(xy,q,radii,False)
    d=np.linalg.norm(xy[:,None]-xy[None,:],axis=2)
    expected=np.array([q.T@kernel(d/h)@q for h in radii])
    req(np.allclose(got,expected,rtol=1e-12,atol=1e-14),'Dense/sparse full covariance test')
    req(np.allclose(got[:,2,2],got[:,0,0]+got[:,1,1]-2*got[:,0,1]),'Joint contrast cross-covariance test')
    return {'status':'PASS_EXACT_SYNTHETIC_ALGEBRA_AND_JOINT_SPATIAL_COVARIANCE','target':target,'compatible_value':est(),'old_group_score':old,'central_derivatives':derivatives,'epsilon':1e-5,'empirical_research_result':False}

def preflight():
    cfg,g,s,x,folds,labels,c,identity,support=inputs()
    out=PREFLIGHT;req(not out.exists(),'Preflight already exists');out.mkdir()
    write(out/'identity.json',identity);write(out/'algebra_test.json',algebra_test())
    support.to_csv(out/'fold_class_counts.csv',index=False)
    write(out/'receipt.json',{'status':'PASS_INPUTS_SPLITS_AND_ALGEBRA','rows':len(g),'tracts':len(g[ID].drop_duplicates()),'duplicates':0,'unmatched':0,'dropped':0,'unassigned_per_dimension':715,'probability_fits':0,'source_model_fits':0,'identity_sha256':sha(out/'identity.json'),'protocol_sha256':sha(TASK/'protocol.json')})
    print('PREFLIGHT PASS rows=104371 tracts=42373 expected_probability_fits=15',flush=True)

def fit_probabilities(out,cfg,x,folds,labels,identity,resume):
    allp={};diagnostics=[];calibration=[];checkpoint=out/'checkpoints'
    for dim in VARS:
        y=labels[dim];p=np.full((len(x),5),np.nan)
        for k in range(5):
            tr=np.flatnonzero(folds!=k);te=np.flatnonzero(folds==k)
            name=f'{dim}_fold_{k}';mp=checkpoint/(name+'.joblib');npz=checkpoint/(name+'.npz');rp=checkpoint/(name+'.json')
            fitid={'dimension':dim,'outer_fold':k,'train_rows':len(tr),'test_rows':len(te),'train_indices_sha256':ah(tr),'test_indices_sha256':ah(te),'train_x_sha256':ah(x[tr]),'train_y_sha256':ah(y[tr]),'parameters':cfg['probability_model']}
            if rp.exists():
                req(resume,'Existing checkpoint requires --resume')
                rec=load(rp);req(rec['identity']==fitid,'Checkpoint identity differs')
                req(sha(mp)==rec['model_sha256'] and sha(npz)==rec['predictions_sha256'],'Checkpoint changed')
                z=np.load(npz);req(np.array_equal(z['test_indices'],te),'Checkpoint test index')
                pr=z['probabilities'];event(out,'REUSED',fit=name)
            else:
                req(not mp.exists() and not npz.exists(),'Incomplete checkpoint: stop for explicit recovery, no silent refit')
                event(out,'STARTED',fit=name,**fitid)
                start=time.monotonic();params={a:b for a,b in cfg['probability_model'].items() if a!='class'}
                model=RandomForestClassifier(**params);model.fit(x[tr],y[tr])
                req(np.array_equal(model.classes_,np.arange(5)),'Class order/missing class')
                pr=model.predict_proba(x[te]);req(np.isfinite(pr).all(),'Nonfinite predictions')
                joblib.dump(model,mp,compress=3)
                np.savez_compressed(npz,train_indices=tr,test_indices=te,probabilities=pr,classes=model.classes_)
                rec={'identity':fitid,'model_sha256':sha(mp),'predictions_sha256':sha(npz),'elapsed_seconds':time.monotonic()-start,'trees_executed':len(model.estimators_)}
                write(rp,rec);event(out,'COMPLETED',fit=name,elapsed_seconds=rec['elapsed_seconds'],trees=200)
                del model
            req(pr.shape==(len(te),5) and np.isfinite(pr).all() and (pr>=0).all() and (pr<=1).all(),'Probability bounds')
            req(np.max(abs(pr.sum(axis=1)-1))<1e-12,'Probability simplex')
            p[te]=pr
            one=np.eye(5)[y[te]];prevalence=np.bincount(y[tr],minlength=5)/len(tr)
            brier=float(np.mean(np.sum((pr-one)**2,axis=1)));baseline=float(np.mean(np.sum((one-prevalence)**2,axis=1)))
            diagnostics.append({'dimension':dim,'fold':k,'train_rows':len(tr),'test_rows':len(te),'multiclass_brier':brier,'training_prevalence_brier':baseline,'brier_improvement':baseline-brier,'accuracy':float((pr.argmax(axis=1)==y[te]).mean()),'role':'HELD_OUT_DIAGNOSTIC_NO_POSTHOC_MODEL_SELECTION'})
            for j in range(5):
                bins=np.minimum((pr[:,j]*10).astype(int),9)
                for b in range(10):
                    mask=bins==b
                    if mask.any():
                        calibration.append({'dimension':dim,'fold':k,'group':j,'probability_bin_low':b/10,'probability_bin_high':(b+1)/10,'n':int(mask.sum()),'predicted_mean':float(pr[mask,j].mean()),'observed_fraction':float(one[mask,j].mean())})
        req(np.isfinite(p).all(),'Missing out-of-fold probabilities');allp[dim]=p
    return allp,pd.DataFrame(diagnostics),pd.DataFrame(calibration)

def score_outputs(g,s,labels,probs,c):
    n=len(g);theta=s.theta.to_numpy();meta=[];columns=[];component=[]
    for dim in VARS:
        mus={};ifs={};y=labels[dim];p=probs[dim]
        for j in range(5):
            ind=(y==j);count=int(ind.sum());prediction=float(theta[ind].mean())
            adjustment=float(p[:,j]@c/count);mu=prediction+adjustment
            influence=(ind*(theta-mu)+p[:,j]*c)/count
            req(abs(influence.sum())<1e-9,'Group influence not centered')
            key=dim+'_q'+str(j);columns.append(influence);mus[j]=mu;ifs[j]=influence
            meta.append({'key':key,'dimension':dim,'comparison':'unassigned_mean' if j==0 else 'group_mean','group':j,'rows':count,'tracts':len(g.loc[ind,ID].drop_duplicates()),'prediction_mean':prediction,'compatible_correction':adjustment,'estimate':mu,'old_group_score':float(s.orthogonal_score[ind].mean()),'unassigned_rows_correction':float(p[y==0,j]@c[y==0]/count),'theta_direction_empirical_moment':float((ind-p[:,j]*s.standardized_residual_square.to_numpy()).sum()/count)})
            for k in range(5):
                m=g.spatial_fold.eq(k).to_numpy();ng=int((ind&m).sum())
                req(ng>0,'Empty test group')
                component.append({'dimension':dim,'group':j,'fold':k,'group_rows':ng,'all_test_rows':int(m.sum()),'prediction':float(theta[ind&m].mean()),'correction':float(p[m,j]@c[m]/ng),'estimate':float((theta[ind&m].sum()+p[m,j]@c[m])/ng),'contribution_to_pooled_estimate':float((theta[ind&m].sum()+p[m,j]@c[m])/count)})
        req(abs(sum((y==j).mean()*mus[j] for j in range(5))-s.orthogonal_score.mean())<1e-12,'Five-group partition fails to reconstruct national point estimate')
        columns.append(ifs[4]-ifs[1]);a=meta[-1];b=meta[-4]
        req(a['group']==4 and b['group']==1,'Contrast indexing')
        meta.append({'key':dim+'_q4_minus_q1','dimension':dim,'comparison':'q4_minus_q1','group':None,'rows':int(np.isin(y,[1,4]).sum()),'tracts':len(g.loc[np.isin(y,[1,4]),ID].drop_duplicates()),'prediction_mean':a['prediction_mean']-b['prediction_mean'],'compatible_correction':a['compatible_correction']-b['compatible_correction'],'estimate':mus[4]-mus[1],'old_group_score':a['old_group_score']-b['old_group_score'],'unassigned_rows_correction':a['unassigned_rows_correction']-b['unassigned_rows_correction']})
    mu=float(s.orthogonal_score.mean());columns.append((s.orthogonal_score.to_numpy()-mu)/n)
    meta.append({'key':'national','dimension':'national','comparison':'national_reference','group':None,'rows':n,'tracts':42373,'prediction_mean':float(theta.mean()),'estimate':mu,'compatible_correction':float(c.mean()),'old_group_score':mu})
    row=np.column_stack(columns)
    agg=g[ID+XY].copy()
    for j,m in enumerate(meta):agg[m['key']]=row[:,j]
    keys=[m['key'] for m in meta]
    agg=agg.groupby(ID,sort=True).agg({**{v:'first' for v in XY},**{v:'sum' for v in keys}}).reset_index()
    req(len(agg)==42373 and abs(row.sum(axis=0)).max()<1e-9,'Tract aggregation')
    return row,agg,meta,pd.DataFrame(component)

def holm(p):
    p=np.asarray(p);order=np.argsort(p);out=np.empty(len(p));out[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return out

def execute(resume):
    start=time.monotonic();cfg,g,s,x,folds,labels,c,identity,support=inputs()
    req(load(PREFLIGHT/'identity.json')==identity,'Preflight source/code/environment changed')
    req(load(PREFLIGHT/'receipt.json')['status']=='PASS_INPUTS_SPLITS_AND_ALGEBRA','Preflight required')
    out=TASK/'results'
    if resume:
        req(load(out/'run_identity.json')==identity,'Resume identity differs')
        if (out/'audit.json').exists():
            audit=load(out/'audit.json');req(all(sha(out/p)==h for p,h in audit['outputs'].items()),'Completed output changed')
            print('COMPLETED_RUN_VERIFIED_NO_FITS_REPEATED');return
    else:
        req(not out.exists(),'Refuse output overwrite');out.mkdir();(out/'checkpoints').mkdir();write(out/'run_identity.json',identity)
    probabilities,diagnostics,calibration=fit_probabilities(out,cfg,x,folds,labels,identity,resume)
    predicted=g[KEY+['spatial_fold','spatial_block_id']+cfg['x_columns']+[v+'_city_quartile' for v in VARS]].copy()
    for dim in VARS:
        for j in range(5):predicted[f'{dim}_p{j}']=probabilities[dim][:,j]
    predicted.to_parquet(out/'crossfit_group_probabilities.parquet',index=False)
    diagnostics.to_csv(out/'probability_diagnostics.csv',index=False);calibration.to_csv(out/'probability_calibration.csv',index=False)
    support.to_csv(out/'fold_class_counts.csv',index=False)
    row,agg,meta,component=score_outputs(g,s,labels,probabilities,c)
    component.to_csv(out/'fold_estimate_components.csv',index=False)
    agg.to_parquet(out/'tract_joint_influences.parquet',index=False)
    pd.DataFrame(meta).to_csv(out/'estimates_and_corrections.csv',index=False)
    write(out/'estimate_order.json',[m['key'] for m in meta])
    radii=np.array(cfg['bandwidths_km'])*1000.;q=agg[[m['key'] for m in meta]].to_numpy()
    cov,pairs=covariance(agg[XY].to_numpy(),q,radii)
    np.savez_compressed(out/'joint_spatial_covariance.npz',covariance=cov,radii_m=radii,keys=np.array([m['key'] for m in meta]))
    refs=[]
    for k,h in enumerate(radii):
        for j,m in enumerate(meta):
            se=float(np.sqrt(cov[k,j,j]));req(se>0,'Zero standard error')
            refs.append({**m,'bandwidth_km':h/1000,'variance':float(cov[k,j,j]),'standard_error':se,'lower95':m['estimate']-norm.ppf(.975)*se,'upper95':m['estimate']+norm.ppf(.975)*se,'reference_p':float(2*norm.sf(abs(m['estimate']/se))),'interval_scope':'CONDITIONAL_SPATIAL_REFERENCE','between_tract_pairs':int(pairs[k])})
    result=pd.DataFrame(refs);result['holm_p_three_contrasts']=np.nan
    for h in cfg['bandwidths_km']:
        ix=result.index[result.bandwidth_km.eq(h)&result.comparison.eq('q4_minus_q1')]
        req(len(ix)==3,'Contrast family');result.loc[ix,'holm_p_three_contrasts']=holm(result.loc[ix,'reference_p'])
    original=read(MODEL/'spatial_covariance_reference.csv')
    for k,h in enumerate(radii):
        old=original.loc[original.bandwidth_m==h].iloc[0]
        req(np.isclose(cov[k,-1,-1],old.variance,rtol=1e-9,atol=1e-12),'National covariance regression')
        req(pairs[k]==old.unique_between_tract_pairs,'Spatial pair count differs')
        for dim in VARS:
            keys=[m['key'] for m in meta];i4=keys.index(dim+'_q4');i1=keys.index(dim+'_q1');ig=keys.index(dim+'_q4_minus_q1')
            req(abs(cov[k,ig,ig]-(cov[k,i4,i4]+cov[k,i1,i1]-2*cov[k,i4,i1]))<1e-12,'Contrast covariance identity')
    result.to_csv(out/'all_group_reference_intervals.csv',index=False)
    result[result.comparison.isin(['group_mean','q4_minus_q1'])].to_csv(out/'main_group_reference_intervals.csv',index=False)
    concentration=[]
    for j,m in enumerate(meta):
        z=q[:,j]**2;share=z/z.sum();idx=np.argsort(-share)
        concentration.append({'key':m['key'],'max_tract_share':float(share[idx[0]]),'top10_share':float(share[idx[:10]].sum()),'top1pct_share':float(share[idx[:int(np.ceil(.01*len(q)))]].sum()),'largest_state':agg.iloc[idx[0]].state,'largest_tract_id':agg.iloc[idx[0]].tract_id,'role':'DIAGONAL_INFLUENCE_CONCENTRATION_NOT_AUTOMATIC_INTERVAL_GATE'})
    pd.DataFrame(concentration).to_csv(out/'influence_concentration.csv',index=False)
    events=[json.loads(line) for line in (out/'fit_events.jsonl').read_text(encoding='utf8').splitlines()]
    completed=[e['fit'] for e in events if e['event']=='COMPLETED'];started=[e['fit'] for e in events if e['event']=='STARTED']
    req(len(completed)==len(set(completed))==15 and len(started)==15,'Fit execution counts differ')
    req(all(sha(p)==h for p,h in identity['source_hashes'].items()),'Input changed during run')
    outputs={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    write(out/'audit.json',{'status':'PASS_COMPATIBLE_GROUP_SCORE_AND_JOINT_REFERENCE_CALCULATION','source_hashes':identity['source_hashes'],'outputs':outputs,'rows':len(g),'tracts':len(agg),'unassigned_rows_per_dimension':715,'sample_dropped':0,'expected_unique_fits':15,'started_fits':len(started),'completed_unique_fits':len(completed),'reused_fits':sum(e['event']=='REUSED' for e in events),'failed_fits':0,'source_response_model_fits':0,'groups':12,'primary_contrasts':3,'main_interval_rows':45,'national_estimate_unchanged':True,'probability_clipping':False,'conditional_variance_clipping':False,'old_authority_or_results_modified':False,'intervals':'95% conditional spatial reference intervals','statistical_coverage_proven':False,'independent_validation':'PENDING','elapsed_seconds':time.monotonic()-start})
    print(result.loc[result.bandwidth_km.eq(200)&result.comparison.eq('q4_minus_q1'),['dimension','prediction_mean','estimate','lower95','upper95','holm_p_three_contrasts']].to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['preflight','run']);p.add_argument('--resume',action='store_true');args=p.parse_args()
    try:
        if args.mode=='preflight':preflight()
        else:execute(args.resume)
    except Exception:
        out=TASK/'results'
        if out.exists():event(out,'FAILED',traceback=traceback.format_exc())
        raise
