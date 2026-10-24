from pathlib import Path
import argparse,pandas as pd,json
ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False);P=a.package/'02_source_data/project';checks=[]
def check(k,v,e,t=0):
 ok=abs(v-e)<=t;checks.append(dict(check=k,actual=v,expected=e,tolerance=t,status='PASS' if ok else 'FAIL'))
 if not ok:raise ValueError(checks[-1])
x=pd.read_csv(P/'06_logs/canopy_cooling_equity_20261009_run01/annual_canopy_equity_row_manifest.csv.gz',dtype={'tract_id':str,'single_city_id':str})
x=x.loc[x.complete_fixed_land_year & x.balanced_26years];out=[]
expected={'income':[(12.706,12.797,.091),(26.572,27.147,.575)],'poverty':[(25.692,26.124,.433),(12.916,12.981,.065)],'age65plus':[(17.749,18.742,.992),(20.685,20.091,-.594)]}
for dim in expected:
 col=dim+'_city_quartile';m=x.groupby(['year',col]).urban_canopy_percent_fixed_land.mean().unstack()
 for g,e in zip([1,4],expected[dim]):
  vals=[m.loc[2000,g],m.loc[2025,g],m.loc[2025,g]-m.loc[2000,g]]
  for name,v,z in zip(['2000','2025','net_change'],vals,e):check(f'Table1_{dim}_Q{g}_{name}',v,z,.0005)
  out.append(dict(dimension=dim,group=g,cover2000=vals[0],cover2025=vals[1],change=vals[2]))
 city=x.groupby(['year','single_city_id',col]).urban_canopy_percent_fixed_land.mean().unstack()
 gap=(city[4]-city[1]).groupby('year').mean()
 if dim=='income':check('income_gap_2000',gap.loc[2000],8.376,.0005);check('income_gap_2025',gap.loc[2025],9.440,.0005)
 if dim=='poverty':check('poverty_gap_2000',gap.loc[2000],-7.302,.0005);check('poverty_gap_2025',gap.loc[2025],-8.147,.0005)
d=P/'08_figure图/Temporal_E_树冠与温度共同演变/source_data'
k=pd.read_csv(d/'social_fixed_panel_keys.csv',dtype=str);cities=pd.read_csv(d/'social_fixed_cities.csv',dtype=str)
check('three_period_social_records',len(k),77280);check('three_period_social_tracts',k.tract_id.nunique(),25760);check('three_period_social_cities',len(cities),295)
check('social_city_membership',int(k.single_city_id.isin(cities.single_city_id).all()),1)
pd.DataFrame(out).to_csv(a.output/'Table1_recomputed.csv',index=False);pd.DataFrame(checks).to_csv(a.output/'TABLE_NUMERIC_REPLAY.csv',index=False)
(a.output/'receipt.json').write_text(json.dumps({'status':'PASS','checks':len(checks),'new_inference':False},indent=2));print('PASS',len(checks))
