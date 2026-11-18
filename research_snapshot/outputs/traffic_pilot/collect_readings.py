"""Fetch candidate site/day records, leaving existing crash labels out of retrieval."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json
import time
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'work/traffic_pilot/readings'
OUT.mkdir(parents=True, exist_ok=True)
candidate = pd.read_csv(ROOT/'work/traffic_pilot/near_site2.csv', dtype={'collision_index':str})
base = pd.read_csv(ROOT/'outputs/additional_years/data_processing/processed_crashes.csv.gz',
                   usecols=['collision_index','year'],dtype={'collision_index':str})
candidate = candidate[candidate.collision_index.isin(base.loc[base.year.eq(2024),'collision_index'])].copy()
candidate['day'] = pd.to_datetime(candidate.date, format='%d/%m/%Y').dt.strftime('%d%m%Y')
jobs = list({(str(row.site_id), row.day) for row in candidate.itertuples()})
jobs.sort()
print(f'{len(candidate)} candidate crashes, {len(jobs)} unique sensor-days',flush=True)

def get(job):
    site, day = job
    dest = OUT/f'{site}_{day}.json'
    if dest.exists():
        return 'cached'
    params = urlencode({'sites':site,'start_date':day,'end_date':day,'page':1,'page_size':100})
    url = 'https://apptris-a-uks.azurewebsites.net/api/v1.0/reports/daily?'+params
    for attempt in range(3):
        try:
            with urlopen(Request(url,headers={'User-Agent':'academic-traffic-pilot/1.0'}),timeout=25) as response:
                raw = response.read()
            obj = json.loads(raw)
            if not isinstance(obj.get('Rows'), list):
                raise ValueError('unexpected response')
            dest.write_bytes(raw)
            return 'downloaded'
        except Exception:
            time.sleep(2**attempt)
    return 'failed'

counts={}
with ThreadPoolExecutor(max_workers=4) as pool:
    for n,future in enumerate(as_completed([pool.submit(get,j) for j in jobs]),1):
        status=future.result()
        counts[status]=counts.get(status,0)+1
        if n%25==0 or n==len(jobs):
            print(n,counts,flush=True)
