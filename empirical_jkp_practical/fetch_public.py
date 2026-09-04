"""Official JKP public portfolios only; never downloads licensed stock-level data.
Data: Jensen, Kelly, Pedersen (2023), CC BY-NC 4.0; research use.
Endpoint templates are taken from https://jkpfactors.com/data (2026-09-05).
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = 'https://jkpfactors-data.s3.amazonaws.com'
META = 'https://raw.githubusercontent.com/bkelly-lab/jkp-data/main/src/jkp/data/resources/factor_details.xlsx'

def fetch(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429,500,502,503,504])))
    records = []
    def get(url: str, path: Path) -> bytes:
        r=session.get(url, timeout=(15,120));r.raise_for_status()
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(r.content)
        records.append({'url':url,'path':str(path.relative_to(out)),'sha256':hashlib.sha256(r.content).hexdigest(),'bytes':len(r.content),'accessed_utc':datetime.now(timezone.utc).isoformat(),'etag':r.headers.get('ETag'),'last_modified':r.headers.get('Last-Modified')})
        return r.content
    manifest=json.loads(get(BASE+'/public/availability.json',out/'availability.json'))
    print('MANIFEST KEYS',list(manifest));print('REGIONS',json.dumps(manifest.get('dropdown_options',{}).get('regions',[])))
    meta=get(META,out/'factor_details.xlsx')
    pd.read_excel(io.BytesIO(meta)).to_csv(out/'factor_details.csv',index=False)
    requests_list=[('usa','all_factors','vw_cap'),('usa','all_factors','vw'),('usa','all_factors','ew'),('usa','all_themes','vw_cap'),('usa','market','vw_cap'),('gbr','all_factors','vw_cap'),('jpn','all_factors','vw_cap')]
    for country,kind,weight in requests_list:
        slug=f'[{country}]_[{kind}]_[monthly]_[{weight}]'
        url=BASE+'/public/'+slug.replace('[','%5B').replace(']','%5D')+'.zip'
        try:
            content=get(url,out/(slug+'.zip'))
        except requests.HTTPError as exc:
            if (country,kind,weight)==('usa','all_factors','vw_cap'): raise
            print('OPTIONAL UNAVAILABLE',slug,str(exc));continue
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            for info in z.infolist():
                if info.is_dir():continue
                name=Path(info.filename).name
                path=out/(country+'_'+kind+'_'+weight)/name
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(z.read(info))
                print('DOWNLOADED',str(path),info.file_size)
                if name.endswith('.csv'):
                    d=pd.read_csv(path); print('SCHEMA',d.shape,d.columns.tolist());print(d.head(3).to_string(index=False))
        time.sleep(.3)
    (out/'provenance.json').write_text(json.dumps({'source':'Jensen, Kelly, Pedersen (2023), official jkpfactors.com public data','license':'CC BY-NC 4.0; research use only','stock_level_data':False,'files':records},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('jkp_public_raw'));a=p.parse_args();fetch(a.out)
