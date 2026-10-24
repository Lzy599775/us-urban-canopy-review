from pathlib import Path
import argparse,json
import pandas as pd
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
P=a.package/'02_source_data/project';checks=[]
def check(k,v,e,t=0):
 ok=abs(v-e)<=t;checks.append(dict(check=k,actual=v,expected=e,tolerance=t,status='PASS' if ok else 'FAIL'))
 if not ok:raise ValueError(checks[-1])
def read(p):return pd.read_csv(p,dtype={'state':str,'tract_id':str,'single_city_id':str},float_precision='round_trip')
f5=read(P/'06_logs/figure_attachment_revision_20261020_run01/Figure05/source_data/map_paired_tracts.csv')
check('fig5_tracts',len(f5),42369);check('fig5_duplicate',int(f5.tract_id.duplicated().sum()),0)
check('fig5_endpoint_identity',float(abs(f5.canopy_2025-f5.canopy_2000-f5.change_pp).max()),0,1e-10)
th=f5.loc[f5.theta.lt(0),'theta'].median();check('fig5_threshold',th,-.1630660244,5e-11)
cat=pd.Series(index=f5.index,dtype='object')
for sign,mask in [('loss',f5.change_pp.lt(0)),('gain',f5.change_pp.gt(0))]:
 cat.loc[mask & f5.theta.lt(0)&f5.theta.le(th)]=sign+'_stronger'
 cat.loc[mask & f5.theta.lt(0)&f5.theta.gt(th)]=sign+'_weaker'
cat.loc[f5.change_pp.eq(0)]='zero_change';cat.loc[f5.theta.gt(0)]='positive_response'
check('fig5_category_exact',int(cat.eq(f5.category).sum()),len(f5))
for c,n,pc in [('loss_stronger',11237,26.522),('gain_stronger',9929,23.435)]:
 check(c+'_n',int(cat.eq(c).sum()),n);check(c+'_percent',cat.eq(c).mean()*100,pc,.0005)
check('fig5_assigned',int(f5.income_city_quartile.notna().sum()),42079)
f5.groupby('category').size().rename('tracts').to_csv(a.output/'fig5_categories.csv')
p=read(P/'08_figure图/Temporal_E_树冠与温度共同演变/source_data/paired_tract_panel.csv')
check('fig3_common_records',len(p),79431);check('fig3_common_tracts',p.tract_id.nunique(),26477)
check('fig3_unique_records',p.duplicated(['tract_id','outcome_year']).sum(),0)
cities=read(P/'08_figure图/Fig03_树冠与热环境共同演变/revision02_baseline_and_change_20261019/source_data/city_paired_means.csv')
check('fig3_map_cities',cities.single_city_id.nunique(),379);check('fig3_map_tracts',cities.loc[cities.outcome_year.eq(2014),'n'].sum(),26307)
# Preserve original parser representation and stable ordering from original figure07.py.
b=P/'06_logs/state_alignment_policy_holdout_20261012_run01'
p=read(b/'policy_candidate_support_audit.csv');p['state']=p.state.str.zfill(2);p['tract_id']=p.tract_id.str.zfill(11)
p=p.loc[p.common_candidate].copy();check('fig6_candidate_rows',len(p),104367);check('fig6_candidate_tracts',p.tract_id.nunique(),42372)
s=read(b/'conditional_policy_ranking_summary.csv');check('fig6_rule_fraction_combinations',len(s),24)
fields={'mean_theta_c_per_pp':'theta','mean_poverty_share':'poverty_share_2000','mean_income_usd':'income_point_usd_2000','mean_age65plus_share':'age65plus_share_2000','mean_baseline_canopy':'baseline_canopy_percent_fixed_land_2000'}
cut=[]
for r in s.itertuples():
 key=r.rule+'_'+str(r.target_fraction);asc=r.rule not in ['poverty_priority','age65plus_priority','population_priority']
 d=p.sort_values(r.ranking_variable,ascending=asc,kind='stable').iloc[:r.selected_rows]
 check(key+'_n',len(d),int(np.ceil(len(p)*r.target_fraction)))
 check(key+'_tracts',d.tract_id.nunique(),r.selected_tracts)
 for name,col in fields.items():check(key+'_'+name,float(d[col].mean()),getattr(r,name),1e-8)
 check(key+'_pop_weighted',float(np.average(d.theta,weights=d.population_2000)),r.population_weighted_theta_c_per_pp,1e-8)
 cutoff=d.iloc[-1][r.ranking_variable];strict=p[r.ranking_variable].lt(cutoff) if asc else p[r.ranking_variable].gt(cutoff);ties=p[r.ranking_variable].eq(cutoff)
 check(key+'_cutoff_brackets',int(strict.sum()<len(d)<=strict.sum()+ties.sum()),1)
 cut.append(dict(rule=r.rule,fraction=r.target_fraction,cutoff=cutoff,strict_rows=int(strict.sum()),tied_rows=int(ties.sum()),selected_rows=len(d)))
pd.DataFrame(cut).to_csv(a.output/'fig6_cutoffs_replayed.csv',index=False)
pd.DataFrame(checks).to_csv(a.output/'FIGURE_NUMERIC_REPLAY.csv',index=False)
(a.output/'receipt.json').write_text(json.dumps({'status':'PASS','checks':len(checks),'new_models':0,'new_inference':False},indent=2))
print('PASS',len(checks))
