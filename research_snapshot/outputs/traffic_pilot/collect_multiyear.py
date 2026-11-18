"""Fetch one or two sensor-days per matched crash with bounded concurrency."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request,urlopen
import json,time
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
data=pd.read_csv(ROOT/'outputs/traffic_pilot/multiyear_candidates.csv',dtype={'collision_index':str})
OUT=ROOT/'work/traffic_pilot/readings'
OUT.mkdir(exist_ok=True,parents=True)
jobs=set()
for row in data.itertuples(index=False):
    dt=pd.to_datetime(row.date+' '+row.time,format='%d/%m/%Y %H:%M',errors='coerce')
    if pd.isna(dt): continue
    if '--previous-only' not in sys.argv:
        jobs.add((int(row.site_id),dt.strftime('%d%m%Y')))
    if dt.hour<2:
        jobs.add((int(row.site_id),(dt-pd.Timedelta(days=1)).strftime('%d%m%Y')))
jobs=sorted(jobs)
print('sensor-days',len(jobs),flush=True)
def fetch(job):
    site,day=job; path=OUT/f'{site}_{day}.json'
    if path.exists(): return 'cached'
    params=urlencode({'sites':site,'start_date':day,'end_date':day,'page':1,'page_size':100})
    url='https://apptris-a-uks.azurewebsites.net/api/v1.0/reports/daily?'+params
    for attempt in range(3):
        try:
            with urlopen(Request(url,headers={'User-Agent':'academic-traffic-pilot/1.0'}),timeout=25) as r:
                raw=r.read()
            if not isinstance(json.loads(raw).get('Rows'),list): raise ValueError('invalid response')
            path.write_bytes(raw); return 'downloaded'
        except Exception:
            time.sleep(2**attempt)
    return 'failed'
counts={}
with ThreadPoolExecutor(max_workers=4) as pool:
    for n,f in enumerate(as_completed([pool.submit(fetch,j) for j in jobs]),1):
        status=f.result(); counts[status]=counts.get(status,0)+1
        if n%50==0 or n==len(jobs): print(n,counts,flush=True)
