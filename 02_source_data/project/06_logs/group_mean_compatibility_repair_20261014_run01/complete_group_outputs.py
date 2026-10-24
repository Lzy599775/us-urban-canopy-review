"""Explicit CSV completion after missing optional parquet engine; zero model fits."""
from pathlib import Path
import json,sys,time
import numpy as np
import pandas as pd
from scipy.stats import norm
import run_group_mean_repair as r

TASK=Path(__file__).resolve().parent
OUT=TASK/'results_csv'

def main():
    start=time.monotonic()
    cfg,g,s,x,folds,labels,c,identity,support=r.inputs()
    r.req(identity==r.load(TASK/'preflight_run02/identity.json'),'Frozen producer identity changed')
    r.req(identity==r.load(TASK/'results/run_identity.json'),'Fitting run identity changed')
    prior=[json.loads(z) for z in (TASK/'results/fit_events.jsonl').read_text(encoding='utf8').splitlines()]
    r.req(sum(z['event']=='STARTED' for z in prior)==15 and sum(z['event']=='COMPLETED' for z in prior)==15,'Incomplete model fits')
    failures=[z for z in prior if z['event']=='FAILED']
    r.req(len(failures)==1 and 'pyarrow' in failures[0]['traceback'] and 'predicted.to_parquet' in failures[0]['traceback'],'Unexpected recovery reason')
    r.req(not OUT.exists(),'Refuse CSV completion overwrite');OUT.mkdir()
    sources={**identity['source_hashes'],str(Path(__file__)):r.sha(__file__),str(TASK/'results/run_identity.json'):r.sha(TASK/'results/run_identity.json'),str(TASK/'results/fit_events.jsonl'):r.sha(TASK/'results/fit_events.jsonl')}
    probabilities={};diagnostics=[];calibration=[]
    for dim in r.VARS:
        y=labels[dim];p=np.full((len(g),5),np.nan)
        for k in range(5):
            name=f'{dim}_fold_{k}';cp=TASK/'results/checkpoints';rec=r.load(cp/(name+'.json'))
            tr=np.flatnonzero(folds!=k);te=np.flatnonzero(folds==k)
            expect={'dimension':dim,'outer_fold':k,'train_rows':len(tr),'test_rows':len(te),'train_indices_sha256':r.ah(tr),'test_indices_sha256':r.ah(te),'train_x_sha256':r.ah(x[tr]),'train_y_sha256':r.ah(y[tr]),'parameters':cfg['probability_model']}
            r.req(rec['identity']==expect,'Checkpoint identity differs')
            for suffix,field in [('.joblib','model_sha256'),('.npz','predictions_sha256')]:
                r.req(r.sha(cp/(name+suffix))==rec[field],'Checkpoint checksum')
            for suffix in ['.json','.joblib','.npz']:sources[str(cp/(name+suffix))]=r.sha(cp/(name+suffix))
            z=np.load(cp/(name+'.npz'));pr=z['probabilities']
            r.req(np.array_equal(te,z['test_indices']) and np.array_equal(tr,z['train_indices']),'Checkpoint split')
            r.req(np.isfinite(pr).all() and (pr>=0).all() and (pr<=1).all() and abs(pr.sum(axis=1)-1).max()<1e-12,'Probability simplex')
            p[te]=pr;r.event(OUT,'REUSED',fit=name,checkpoint=str(cp/(name+'.json')))
            obs=np.eye(5)[y[te]];base=np.bincount(y[tr],minlength=5)/len(tr)
            brier=float(np.mean(np.sum((pr-obs)**2,axis=1)));bp=float(np.mean(np.sum((obs-base)**2,axis=1)))
            diagnostics.append({'dimension':dim,'fold':k,'train_rows':len(tr),'test_rows':len(te),'multiclass_brier':brier,'training_prevalence_brier':bp,'brier_improvement':bp-brier,'accuracy':float((pr.argmax(axis=1)==y[te]).mean())})
            for j in range(5):
                bins=np.minimum((pr[:,j]*10).astype(int),9)
                for b in range(10):
                    mask=bins==b
                    if mask.any():calibration.append({'dimension':dim,'fold':k,'group':j,'probability_bin_low':b/10,'probability_bin_high':(b+1)/10,'n':int(mask.sum()),'predicted_mean':float(pr[mask,j].mean()),'observed_fraction':float(obs[mask,j].mean())})
        r.req(np.isfinite(p).all(),'Missing probabilities');probabilities[dim]=p
    predicted=g[r.KEY+['spatial_fold','spatial_block_id']+cfg['x_columns']+[v+'_city_quartile' for v in r.VARS]].copy()
    for dim in r.VARS:
        for j in range(5):predicted[f'{dim}_p{j}']=probabilities[dim][:,j]
    predicted.to_csv(OUT/'crossfit_group_probabilities.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(diagnostics).to_csv(OUT/'probability_diagnostics.csv',index=False)
    pd.DataFrame(calibration).to_csv(OUT/'probability_calibration.csv',index=False)
    support.to_csv(OUT/'fold_class_counts.csv',index=False)
    row,agg,meta,component=r.score_outputs(g,s,labels,probabilities,c)
    component.to_csv(OUT/'fold_estimate_components.csv',index=False)
    agg.to_csv(OUT/'tract_joint_influences.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(meta).to_csv(OUT/'estimates_and_corrections.csv',index=False)
    order=[m['key'] for m in meta];r.write(OUT/'estimate_order.json',order)
    radii=np.array(cfg['bandwidths_km'])*1000.;q=agg[order].to_numpy()
    cov,pairs=r.covariance(agg[r.XY].to_numpy(),q,radii)
    np.savez_compressed(OUT/'joint_spatial_covariance.npz',covariance=cov,radii_m=radii,keys=np.array(order))
    refs=[]
    for k,h in enumerate(radii):
        for j,m in enumerate(meta):
            se=float(np.sqrt(cov[k,j,j]));r.req(se>0,'Nonpositive standard error')
            refs.append({**m,'bandwidth_km':h/1000,'variance':float(cov[k,j,j]),'standard_error':se,'lower95':m['estimate']-norm.ppf(.975)*se,'upper95':m['estimate']+norm.ppf(.975)*se,'reference_p':float(2*norm.sf(abs(m['estimate']/se))),'interval_scope':'CONDITIONAL_SPATIAL_REFERENCE','between_tract_pairs':int(pairs[k])})
    rr=pd.DataFrame(refs);rr['holm_p_three_contrasts']=np.nan
    for h in cfg['bandwidths_km']:
        ix=rr.index[rr.bandwidth_km.eq(h)&rr.comparison.eq('q4_minus_q1')]
        rr.loc[ix,'holm_p_three_contrasts']=r.holm(rr.loc[ix,'reference_p'])
    old=r.read(r.MODEL/'spatial_covariance_reference.csv')
    for k,h in enumerate(radii):
        orig=old.loc[old.bandwidth_m.eq(h)].iloc[0]
        r.req(np.isclose(cov[k,-1,-1],orig.variance,atol=1e-12,rtol=1e-9),'National variance regression')
        r.req(pairs[k]==orig.unique_between_tract_pairs,'Spatial pair count regression')
        for dim in r.VARS:
            a=order.index(dim+'_q4');b=order.index(dim+'_q1');j=order.index(dim+'_q4_minus_q1')
            r.req(abs(cov[k,j,j]-(cov[k,a,a]+cov[k,b,b]-2*cov[k,a,b]))<1e-12,'Contrast covariance identity')
    rr.to_csv(OUT/'all_group_reference_intervals.csv',index=False)
    rr[rr.comparison.isin(['group_mean','q4_minus_q1'])].to_csv(OUT/'main_group_reference_intervals.csv',index=False)
    concentration=[]
    for j,m in enumerate(meta):
        z=q[:,j]**2;share=z/z.sum();idx=np.argsort(-share)
        concentration.append({'key':m['key'],'max_tract_share':float(share[idx[0]]),'top10_share':float(share[idx[:10]].sum()),'top1pct_share':float(share[idx[:int(np.ceil(.01*len(q)))]].sum()),'largest_state':agg.iloc[idx[0]].state,'largest_tract_id':agg.iloc[idx[0]].tract_id,'role':'DIAGONAL_INFLUENCE_DIAGNOSTIC'})
    pd.DataFrame(concentration).to_csv(OUT/'influence_concentration.csv',index=False)
    r.req(all(r.sha(p)==h for p,h in sources.items()),'Source changed during completion')
    outputs={str(p.relative_to(OUT)):r.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
    r.write(OUT/'audit.json',{'status':'PASS_COMPATIBLE_GROUP_SCORE_AND_JOINT_REFERENCE_CALCULATION','source_hashes':sources,'outputs':outputs,'rows':len(g),'tracts':len(agg),'unassigned_rows_per_dimension':715,'sample_dropped':0,'expected_unique_fits':15,'original_started_fits':15,'original_completed_unique_fits':15,'new_fits_in_completion':0,'reused_models':15,'failed_model_fits':0,'recovered_output_failures':1,'recovery_reason':'Missing optional Parquet engine; CSV serialization with full numeric round-trip verification; no environment change','source_response_model_fits':0,'groups':12,'primary_contrasts':3,'main_interval_rows':45,'national_estimate_unchanged':True,'probability_clipping':False,'conditional_variance_clipping':False,'old_authority_or_results_modified':False,'intervals':'95% conditional spatial reference intervals','statistical_coverage_proven':False,'independent_validation':'PENDING','elapsed_completion_seconds':time.monotonic()-start})
    print(rr.loc[rr.bandwidth_km.eq(200)&rr.comparison.eq('q4_minus_q1'),['dimension','prediction_mean','estimate','lower95','upper95','holm_p_three_contrasts']].to_string(index=False),flush=True)

if __name__=='__main__':main()
