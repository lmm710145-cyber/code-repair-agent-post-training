import math
from ..config import workspace
from ..io import lock,freeze,read,write
from ..pipeline import initialize,worker,baseline
from ..tasks.schema import catalog
from ..evaluation.run import evaluate


def train(cfg,run_name):
    root=workspace(cfg,'runs',run_name);c=cfg['sft']
    with lock(root):
        data=read(workspace(cfg,'datasets/sft/manifest.json'));freeze(root,dict(config=cfg,data=data,tasks=catalog(cfg),stage='sft'))
        meta=initialize(cfg,root,'sft')
        if cfg['evaluation']['during_training']:baseline(cfg,root)
        count=data['train_samples'];targets=sorted(set(math.ceil(count*e/c['batch_size']) for e in c['milestone_epochs']))
        targets.append(math.ceil(count*c['epochs']/c['batch_size']));targets=sorted(set(targets))
        for target in targets:
            if target>meta['updates']:
                worker(cfg,root,'sft',['--target',str(target)]);meta=read(root/'latest.json')
            if meta['updates']==target and cfg['evaluation']['during_training']:
                evaluate(cfg,meta,root/'evaluation'/f'step-{target}',k=1,baseline=root/'evaluation/initial/metrics.json')
        write(root/'status.json',dict(status='complete',updates=meta['updates'],latest=meta));return {'updates':meta['updates'],'run':str(root)}
