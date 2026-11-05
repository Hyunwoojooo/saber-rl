"""Read-only artifact, source and input-integrity verification."""
import csv,json,xml.etree.ElementTree as ET
from PIL import Image
from common import *


def main():
 m=json.loads((OUT/'manifest.json').read_text())
 for name,expected in m['source_sha256'].items():require(sha(ROOT/name)==expected,'Analysis source changed: '+name)
 for name,expected in m['inputs_sha256'].items():require(sha(name)==expected,'Input changed: '+name)
 for name,expected in m['outputs_sha256'].items():require(sha(OUT/name)==expected,'Derived output changed: '+name)
 for stage in ('logs','checkpoints','gradients','saturation','geometry','finish'):
  require(read(OUT/f'{stage}_status.json')['status']=='complete','Incomplete stage: '+stage)
 for path in (OUT/'figures').glob('*.png'):
  with Image.open(path) as im:
   require(im.width>=1000 and im.height>=500,'Unexpected figure dimensions');im.verify()
 for path in (OUT/'figures').glob('*.svg'):ET.parse(path)
 g=rows(OUT/'gradient_summary.csv');extra=rows(OUT/'saturation_followup.csv')
 for seed in (13,14,15,16,18,20):
  selected=[r for r in g if r['arm']=='support' and int(r['seed'])==seed and r['checkpoint']=='final']
  require(len(selected)==2 and all(float(r['actor_gradient_l2'])==0 and float(r['tanh_derivative_zero_fraction'])==1 and float(r['action_gradient_l2'])>0 for r in selected),'Terminal saturation finding not reproduced')
  selected=[r for r in extra if int(r['seed'])==seed]
  require(len(selected)==8 and all(float(r['actor_gradient_l2'])==0 and float(r['exact_boundary_fraction'])==1 and float(r['positive_z_fraction'])==1 for r in selected),'Independent batch finding not reproduced')
 primary=read(STUDY/'release_manifest.json')
 for name,expected in primary['files'].items():require(sha(STUDY/name)==expected,'Original study release changed')
 print('VERIFIED:',len(m['inputs_sha256']),'input entries,',len(m['outputs_sha256']),'derived artifacts, six PNG/SVG figures, all six audit stages; original frozen study unchanged.')


if __name__=='__main__':main()
