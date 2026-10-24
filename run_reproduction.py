"""Reproduce study summaries from the supplied processed inputs."""
from pathlib import Path
import argparse,subprocess,sys,json
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 root=Path(__file__).resolve().parent;out=a.output.resolve()
 if out.exists():raise FileExistsError('Choose a new output directory: '+str(out))
 out.mkdir(parents=True)
 for name in ['replay_core','replay_figure_values','replay_tables']:
  subprocess.run([sys.executable,'-I','-X','utf8',str(root/'03_code'/(name+'.py')),'--package',str(root),'--output',str(out/name)],check=True)
 receipts=[json.loads((out/n/'receipt.json').read_text()) for n in ['replay_core','replay_figure_values','replay_tables']]
 if [r['checks'] for r in receipts]!=[342,234,26]:raise ValueError('Unexpected check count')
 print('Completed 602 numerical checks. Results: '+str(out))
if __name__=='__main__':main()
