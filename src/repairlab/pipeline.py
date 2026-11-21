import subprocess
import sys
from pathlib import Path
from .io import read,write,freeze,lock,digest


def worker(cfg,root,stage,arguments):
    label='initialize' if '--initialize' in arguments else 'worker-'+str(len(list(root.glob('worker-*.log'))))
    with (root/(label+'.log')).open('a') as log:
        subprocess.run([sys.executable,'-m','repairlab.backends.worker','--config',cfg['_config_file'],
                        '--run',str(root),'--stage',stage,*arguments],check=True,stdout=log,stderr=subprocess.STDOUT)


def initialize(cfg,root,stage):
    if not (root/'latest.json').exists():worker(cfg,root,stage,['--initialize'])
    return read(root/'latest.json')


def baseline(cfg,root):
    from .evaluation.run import evaluate
    initial=read(root/'initial.json')
    return evaluate(cfg,initial,root/'evaluation'/'initial',k=1)
