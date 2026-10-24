from pathlib import Path
import os,json
def configuration():
 p=Path(os.environ['NC_REVIEW_CONFIG']).resolve()
 data=json.loads(p.read_text(encoding='utf8'))
 if data.get('schema')!=1:raise ValueError('Unsupported config schema')
 return p,data
def project_root():
 p,d=configuration();return (p.parent/d['project_root']).resolve()
def external_path(key):
 p,d=configuration();return (p.parent/d['external_paths'][key]).resolve()
