import concurrent.futures
from pathlib import Path
from ..io import read,write,digest
from ..tasks.schema import catalog
from ..agent.context import tokenizer
from ..agent.rollout import run
from ..backends.sampling import engine,Sampler
from .metrics import summarize,compare


def evaluate(cfg,meta,folder,*,split='dev',k=None,task_ids=None,baseline=None):
    if split not in ('dev','test'):raise ValueError('Evaluation split must be dev/test')
    tasks=[t for t in catalog(cfg) if t['split']==split and (task_ids is None or t['task_id'] in task_ids)]
    if not tasks:raise ValueError('No evaluation tasks')
    k=k or cfg['evaluation']['k'];folder=Path(folder);path=folder/'metrics.json'
    signature=digest(dict(config=cfg,adapter=meta['adapter_sha256'],tasks=tasks,k=k,split=split))
    if path.exists():
        result=read(path)
        if result['signature']!=signature:raise ValueError('Evaluation inputs changed')
        return result
    tok=tokenizer(cfg);by_task={t['task_id']:[] for t in tasks}
    with engine(cfg,meta,folder/'engine') as version:
        backend=Sampler(cfg,tok,version,meta,evaluation=True)
        for attempt in range(k):
            pending=[t for t in tasks if not any(r['resolved'] for r in by_task[t['task_id']])]
            with concurrent.futures.ThreadPoolExecutor(max_workers=cfg['vllm']['concurrency']) as pool:
                futures={pool.submit(run,t,cfg,backend,tok,cfg['evaluation']['seed']+attempt,
                         folder/'trajectories'/t['task_id']/str(attempt),phase='evaluation'):t for t in pending}
                for future in concurrent.futures.as_completed(futures):
                    r=future.result();by_task[r['task_id']].append(r)
                    write(folder/'progress.json',dict(attempt=attempt+1,finished=sum(len(v) for v in by_task.values()),tasks=len(tasks)))
    result=dict(summarize(by_task,k),signature=signature,split=split,sampling=cfg['evaluation']['sampling'],budget=cfg['budget'],
                checkpoint=meta['adapter_sha256'],taskset_digest=digest(tasks),seed=cfg['evaluation']['seed'],metric_note='Observed success within first k attempts; stop after success. Behavioral/token metrics describe executed attempts only.')
    if baseline:
        base=read(baseline)
        if base['sampling']!=result['sampling'] or base['budget']!=result['budget'] or base['k']!=k:raise ValueError('Baseline budget/sampling/k mismatch')
        result['training_start_comparison']=compare(base,result)
    write(path,result);return result
