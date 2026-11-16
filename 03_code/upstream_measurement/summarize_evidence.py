"""Descriptive comparison on frozen matching units; no model fitting or p values."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyogrio,shapely
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
M=ROOT/'06_logs/canopy_measurement_consistency_20261025_run01'
S=ROOT/'06_logs/canopy_multicity_reference_20261026_run01'
GROUPS=['income_city_quartile','poverty_city_quartile','age65plus_city_quartile']
KEYS=['boston','providence','philadelphia_code_verified','cuyahoga_valid_domain','charlotte','louisville','washington_alexandria','washington_montgomery','seattle','richmond','detroit','indianapolis','phoenix','minneapolis_2009','albuquerque_2020']
def jr(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def read(p):
    d=pd.read_csv(p,dtype={'tract_id':str,'state':str},float_precision='round_trip')
    if 'tract_id' in d:d['tract_id']=d.tract_id.str.zfill(11)
    if 'state' in d:d['state']=d.state.str.zfill(2)
    return d
def save(d,name):d.to_csv(P/(name+'.csv'),index=False,encoding='utf-8-sig')
def metric(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);assert np.isfinite(a).all() and np.isfinite(b).all()
    return dict(n=len(a),reference=float(a.mean()),nlcd=float(b.mean()),nlcd_minus_reference=float((b-a).mean()),mae=float(abs(b-a).mean()),rmse=float(np.sqrt(np.mean((b-a)**2))),correlation=float(np.corrcoef(a,b)[0,1]) if len(a)>1 and a.std()>0 and b.std()>0 else None,sign_agreement_n=int((np.sign(a)==np.sign(b)).sum()))
def national():
    d=read(M/'product_compare_national_2000_2025_20261026/tract_product_comparison.csv')
    assert len(d)==46424 and not d.tract_id.duplicated().any() and d.common_4way_valid_fraction.eq(1).all()
    rows=[];cityrows=[]
    for prod in ['nlcd','science']:
        a=prod+'_2000_fourway_cover';b=prod+'_2025_fourway_cover'
        rows.append(dict(product=prod,target='national_tract_equal',group='all',n=len(d),early=d[a].mean(),late=d[b].mean(),change=(d[b]-d[a]).mean()))
        for g in GROUPS:
            for q in [1,4]:
                x=d.loc[d[g].eq(q)];rows.append(dict(product=prod,target='national_tract_equal',group=g+'_Q'+str(q),n=len(x),early=x[a].mean(),late=x[b].mean(),change=(x[b]-x[a]).mean()))
            city=d.groupby(['single_city_id',g])[[a,b]].mean().unstack()
            assert len(city)==448
            delta=(city[b][4]-city[b][1])-(city[a][4]-city[a][1])
            rows.append(dict(product=prod,target='equal_city_Q4_minus_Q1',group=g,n=448,early=(city[a][4]-city[a][1]).mean(),late=(city[b][4]-city[b][1]).mean(),change=delta.mean()))
            for q in [1,4]:rows.append(dict(product=prod,target='equal_city_group_level',group=g+'_Q'+str(q),n=448,early=city[a][q].mean(),late=city[b][q].mean(),change=(city[b][q]-city[a][q]).mean()))
            for cid,v in delta.items():cityrows.append(dict(product=prod,single_city_id=cid,group=g,gap_change=v,q4_change=city.loc[cid,(b,4)]-city.loc[cid,(a,4)]))
    save(pd.DataFrame(rows),'national_core_results');c=pd.DataFrame(cityrows);save(c,'national_city_gap_changes')
    checks=[]
    for g in GROUPS:
        w=c[c.group.eq(g)].pivot(index='single_city_id',columns='product',values='gap_change')
        checks.append(dict(group=g,n=len(w),same_direction=int((np.sign(w.nlcd)==np.sign(w.science)).sum()),both_positive=int(((w.nlcd>0)&(w.science>0)).sum()),both_negative=int(((w.nlcd<0)&(w.science<0)).sum())))
    save(pd.DataFrame(checks),'national_direction_agreement')
    return d
def main():
    national_table=national()
    missing=[k for k in KEYS if not (P/'comparisons_exact_v2'/k/'status.json').exists() or jr(P/'comparisons_exact_v2'/k/'status.json')['status']!='COMPLETE']
    if missing:raise RuntimeError('Comparisons not complete: '+','.join(missing))
    frames=[];source_status=[];cfgs={}
    for k in KEYS:
        folder=P/'comparisons_exact_v2'/k;cfg=jr(folder/'configuration.json');cfgs[k]=cfg
        d=read(folder/'tract_comparison.csv');st=jr(folder/'status.json')
        assert not d.tract_id.duplicated().any() and len(d)==st['paired_tracts'] and st['paired_tracts']+st['outside']==st['input_city_tracts']
        assert d.common_area_m2.gt(0).all()
        y0,y1=cfg['years'][0],cfg['years'][-1]
        d=d.rename(columns={f'reference_{y0}':'reference_early',f'nlcd_{y0}':'nlcd_early'})
        if y0==y1:d['reference_late']=d.reference_early;d['nlcd_late']=d.nlcd_early
        else:d=d.rename(columns={f'reference_{y1}':'reference_late',f'nlcd_{y1}':'nlcd_late'})
        d['year_early']=y0;d['year_late']=y1;d['source_key']=k;d['execution']='new_exact_area_comparison'
        frames.append(d);source_status.append(dict(key=k,**st))
    # Two disjoint jurisdictions in the same frozen urbanized area: merge areas BEFORE tract means.
    ca=pyogrio.read_dataframe(cfgs['washington_alexandria']['aoi']).to_crs(5070)
    cb=pyogrio.read_dataframe(cfgs['washington_montgomery']['aoi']).to_crs(5070)
    overlap=shapely.intersection(shapely.union_all(ca.geometry.to_numpy()),shapely.union_all(cb.geometry.to_numpy())).area
    assert overlap<.01,overlap
    raw=pd.concat(frames,ignore_index=True);save(raw,'local_component_tracts')
    merged=[]
    for (city,tid),x in raw.groupby(['city_number','tract_id'],sort=True):
        r=x.iloc[0].to_dict()
        for col in ['year_early','year_late','fixed_land_m2']+GROUPS:assert x[col].nunique(dropna=False)==1,(city,tid,col)
        if len(x)>1:
            assert city==21 and len(x)==2
            for col in ['reference_early','reference_late','nlcd_early','nlcd_late']:r[col]=np.average(x[col],weights=x.common_area_m2)
            r['common_area_m2']=x.common_area_m2.sum();r['support_fraction']=r['common_area_m2']/r['fixed_land_m2']
            r['source_key']=';'.join(sorted(x.source_key))
        paired=r['year_early']!=r['year_late']
        r['reference_change']=r['reference_late']-r['reference_early'] if paired else np.nan
        r['nlcd_change']=r['nlcd_late']-r['nlcd_early'] if paired else np.nan
        r['comparison_kind']='paired' if paired else 'single_epoch'
        merged.append(r)
    d=pd.DataFrame(merged)
    ny=read(M/'nyc_reference_level_check/nyc_reference_levels.csv')
    assert len(ny)==2214 and not ny.tract_id.duplicated().any()
    ny=ny.rename(columns={'full_fixed_land_m2':'fixed_land_m2','common_support_m2':'common_area_m2','common_support_fraction':'support_fraction','reference_2010_level_pp':'reference_early','reference_2017_level_pp':'reference_late','nlcd_2010_common_pp':'nlcd_early','nlcd_2017_common_pp':'nlcd_late','reference_change_pp':'reference_change','nlcd_change_common_pp':'nlcd_change'})
    ny['city_number']=6;ny['year_early']=2010;ny['year_late']=2017;ny['source_key']='nyc_inherited';ny['execution']='inherited_completed_exact_cell_overlay';ny['comparison_kind']='paired'
    d=pd.concat([d,ny],ignore_index=True)
    assert not d.tract_id.duplicated().any()
    # Existing frozen annual table: identify support/time effects without rerunning national rasters.
    cache=P/'candidate_annual_frozen.csv'
    if cache.exists():ann=read(cache)
    else:
        pieces=[];ids=set(d.tract_id);years=set(d.year_early)|set(d.year_late)|{2000,2025}
        path=ROOT/'06_logs/canopy_cooling_equity_20261009_run01/annual_canopy_equity_row_manifest.csv.gz'
        cols=['tract_id','year','balanced_26years','urban_canopy_percent_fixed_land']
        for chunk in pd.read_csv(path,usecols=cols,dtype={'tract_id':str},chunksize=200000,float_precision='round_trip'):
            chunk['tract_id']=chunk.tract_id.str.zfill(11);pieces.append(chunk.loc[chunk.tract_id.isin(ids)&chunk.year.isin(years)])
        ann=pd.concat(pieces,ignore_index=True);save(ann,'candidate_annual_frozen')
    assert not ann.duplicated(['tract_id','year']).any() and ann.balanced_26years.all()
    lookup=ann.set_index(['tract_id','year']).urban_canopy_percent_fixed_land
    for name,col in [('nlcd_full_early','year_early'),('nlcd_full_late','year_late')]:d[name]=[lookup.loc[(r.tract_id,int(getattr(r,col)))] for r in d.itertuples()]
    d['nlcd_full_change']=d.nlcd_full_late-d.nlcd_full_early
    d.loc[d.year_early.eq(d.year_late),'nlcd_full_change']=np.nan
    nat=national_table.set_index('tract_id')
    d['nlcd_original_2000_2025_change']=[nat.loc[t,'nlcd_change_4way_pp'] for t in d.tract_id]
    assert d[['reference_early','reference_late','nlcd_early','nlcd_late']].ge(-1e-7).all().all() and d[['reference_early','reference_late','nlcd_early','nlcd_late']].le(100+1e-7).all().all()
    output_columns=['state','tract_id','city_number','source_key','execution','comparison_kind','year_early','year_late']+GROUPS+['fixed_land_m2','common_area_m2','support_fraction','reference_early','reference_late','reference_change','nlcd_early','nlcd_late','nlcd_change','nlcd_full_early','nlcd_full_late','nlcd_full_change','nlcd_original_2000_2025_change']
    save(d[output_columns],'all_matched_tract_results');save(pd.DataFrame(source_status),'source_execution_summary')
    reg=read(S/'city45_reference_registry.csv').set_index('city_number');rows=[];groups=[];group_levels=[]
    for city,x in d.groupby('city_number'):
        base=dict(city_number=int(city),city=reg.loc[city,'NAME00'],year_early=int(x.year_early.iloc[0]),year_late=int(x.year_late.iloc[0]),tracts=len(x),common_area_m2=x.common_area_m2.sum(),covered_tract_fixed_area_m2=x.fixed_land_m2.sum(),median_tract_support=x.support_fraction.median(),min_tract_support=x.support_fraction.min())
        frozen=read(S/'city45_frozen_product_tracts.csv');f=frozen[frozen.city_number.eq(city)]
        base.update(frozen_ua_tracts=len(f),ua_fixed_land_m2=f.fixed_land_m2.sum(),fraction_ua_land=x.common_area_m2.sum()/f.fixed_land_m2.sum())
        targets=['late'] if base['year_early']==base['year_late'] else ['early','late','change']
        for target in targets:rows.append(dict(**base,target=target,**metric(x['reference_'+target],x['nlcd_'+target]),nlcd_full=float(x['nlcd_full_'+target].mean()),nlcd_original_2000_2025_change=float(x.nlcd_original_2000_2025_change.mean())))
        for g in GROUPS:
            for target in targets:
                q1=x[x[g].eq(1)];q4=x[x[g].eq(4)]
                rec=dict(**base,group=g,target=target,q1_n=len(q1),q4_n=len(q4),missing_group_n=int(x[g].isna().sum()))
                for prod in ['reference','nlcd','nlcd_full']:
                    rec[prod+'_q4_minus_q1']=float(q4[prod+'_'+target].mean()-q1[prod+'_'+target].mean())
                rec['same_direction']=bool(np.sign(rec['reference_q4_minus_q1'])==np.sign(rec['nlcd_q4_minus_q1'])) if len(q1) and len(q4) else None
                groups.append(rec)
                for q in [1,2,3,4]:
                    z=x[x[g].eq(q)];group_levels.append(dict(city_number=city,group=g,quartile=q,target=target,n=len(z),reference=z['reference_'+target].mean(),nlcd=z['nlcd_'+target].mean()))
    save(pd.DataFrame(rows),'local_city_comparison');save(pd.DataFrame(groups),'local_social_gap_comparison');save(pd.DataFrame(group_levels),'local_social_group_levels')
    audit=dict(status='PASS_DESCRIPTIVE_COMPARISON',new_reference_components=len(KEYS),total_unique_tracts=len(d),new_unique_tracts=len(d)-2214,paired_cities=int(d.loc[d.year_early.ne(d.year_late),'city_number'].nunique()),single_epoch_cities=int(d.loc[d.year_early.eq(d.year_late),'city_number'].nunique()),washington_aoi_overlap_m2=overlap,component_duplicate_tracts=int(raw.tract_id.duplicated().sum()),final_duplicate_tracts=int(d.tract_id.duplicated().sum()),missing_annual_values=0,formal_outputs_modified=False,new_models_or_p_values=False)
    (P/'summary_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(audit,ensure_ascii=False))
if __name__=='__main__':main()
