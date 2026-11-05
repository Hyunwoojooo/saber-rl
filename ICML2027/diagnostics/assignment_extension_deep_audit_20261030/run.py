"""Run independent audit stages, retaining failures and input checksums."""
import argparse,os,traceback
from pathlib import Path

os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',MUJOCO_GL='egl')
from common import *
os.environ['NUMBA_CACHE_DIR']=str(OUT/'numba_cache')
os.environ['MPLCONFIGDIR']=str(OUT/'mpl_cache')


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('logs','checkpoints','gradients','saturation','geometry','finish'),required=True);args=ap.parse_args()
 previous=OUT/'inputs.json'
 if previous.exists():INPUTS.update(json.loads(previous.read_text()))
 try:
  if args.stage=='logs':
   from log_analysis import analyze
   analyze()
  elif args.stage=='checkpoints':
   from checkpoint_analysis import analyze
   analyze()
  elif args.stage=='gradients':
   from gradient_audit import analyze
   analyze()
  elif args.stage=='saturation':
   from saturation_followup import analyze
   analyze()
  elif args.stage=='geometry':
   from geometry_analysis import analyze
   analyze()
  else:
   from presentation import finish
   finish()
  for path,expected in INPUTS.items():require(sha(path)==expected,'Input changed during read-only audit: '+path)
  write(previous,INPUTS)
  write(OUT/f'{args.stage}_status.json',dict(status='complete',finished_utc=now(),source_sha256={p.name:sha(p) for p in ROOT.glob('*.py')},new_training_updates=0))
  if args.stage=='finish':
   from seal import seal
   seal()
 except BaseException:
  OUT.mkdir(parents=True,exist_ok=True)
  (OUT/f'{args.stage}_failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
