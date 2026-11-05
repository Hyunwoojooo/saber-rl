"""Isolated import/config probe; never calls a trainer or edits the source tree."""
import argparse,hashlib,importlib,json,sys
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--label',required=True);ap.add_argument('--output',type=Path,required=True)
args=ap.parse_args();source=args.source.resolve();sys.path.insert(0,str(source))
import protocol

origins={}
for name in ('config','agent','networks','trainer','replay','weights','assignment_comparison','task_drawer_open','envs','numba_support_scan_drawer_open'):
 m=importlib.import_module('metacog_lite_v0.'+name);path=Path(m.__file__).resolve()
 assert path.is_relative_to(source/'code/src'),(name,path)
 origins[name]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

defaults=protocol.make_config('support',1,'/tmp/source_parity_only').as_dict()
same_seed=[]
for device in ('cpu','cuda'):
 for arm in ('support','random','td_topk'):
  for seed in range(1,11):
   same_seed.append(dict(arm=arm,seed=seed,device=device,config=protocol.make_config(arm,seed,'/tmp/source_parity_only',device=device).as_dict()))

# Permit the extension range in this isolated audit process only. This bypasses
# only the old whitelist; the archived make_config function body is unchanged.
protocol.SEEDS=tuple(range(11,21))
extension=[]
for arm in ('support','random','td_topk'):
 for seed in range(11,21):
  extension.append(dict(arm=arm,seed=seed,config=protocol.make_config(arm,seed,'/tmp/source_parity_only',device='cpu').as_dict()))

output=dict(label=args.label,source_root=str(source),probe_only=True,training_called=False,
            historical_executed_config=False,source_default_device=defaults['device'],
            origins=origins,same_seed_configs=same_seed,extension_configs=extension)
args.output.parent.mkdir(parents=True,exist_ok=True)
with args.output.open('x') as f:json.dump(output,f,indent=2,sort_keys=True);f.write('\n')
print(args.label,'imports verified:',len(origins),'same-seed configs:',len(same_seed),'extension configs:',len(extension))
