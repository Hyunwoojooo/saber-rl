from pathlib import Path
import csv,hashlib,json
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parent
STUDY=Path('/home/mimiclab-gigabyte-gen5/ICML2027/draweropen_ddpg_assignment_extension_20261020')
OUT=ROOT/'outputs'
ARMS=('support','random','td_topk')
SEEDS=tuple(range(11,21))
INPUTS={}

def now():return datetime.now(timezone.utc).isoformat()

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def record(path):
 path=Path(path);INPUTS.setdefault(str(path),sha(path));return path

def read(path):return json.loads(record(path).read_text())

def rows(path):
 with record(path).open() as f:return list(csv.DictReader(f))

def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def table(path,data):
 assert data
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)

def attempt(arm,seed):return STUDY/f'outputs/production/{arm}/seed_{seed:02d}/attempt_01'

def require(ok,msg):
 if not ok:raise ValueError(msg)
