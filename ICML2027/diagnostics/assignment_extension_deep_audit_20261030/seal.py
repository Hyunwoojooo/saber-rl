"""Seal final derived artifacts after stage status/input records are written."""
from common import *


def seal():
 inputs=json.loads((OUT/'inputs.json').read_text())
 for path,expected in inputs.items():require(sha(path)==expected,'Input checksum changed: '+path)
 outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and
          not {'numba_cache','mpl_cache'}&set(p.relative_to(OUT).parts) and p.name!='manifest.json'}
 write(OUT/'manifest.json',dict(created_utc=now(),inputs_sha256=inputs,
      source_sha256={p.name:sha(p) for p in list(ROOT.glob('*.py'))+list(ROOT.glob('*.md'))},
      outputs_sha256=outputs,new_training_updates=0,read_only_original_inputs=True))
 print('SEALED',len(inputs),'input entries;',len(outputs),'derived artifacts',flush=True)


if __name__=='__main__':seal()
