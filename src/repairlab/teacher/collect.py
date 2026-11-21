import concurrent.futures
from ..agent.context import tokenizer
from ..agent.rollout import run
from ..tasks.schema import catalog
from ..config import workspace
from ..io import freeze,write,lock
from .api import Teacher


def collect(cfg,run_name,task_ids=None):
    root=workspace(cfg,'runs',run_name)
    with lock(root):
        tasks=[t for t in catalog(cfg) if t['split']=='train' and (task_ids is None or t['task_id'] in task_ids)]
        if task_ids and set(task_ids)!={t['task_id'] for t in tasks}:raise ValueError('Unknown or non-train task requested')
        freeze(root,dict(config=cfg,tasks=tasks,stage='teacher'))
        backend=Teacher(cfg);tok=tokenizer(cfg);results=[]
        jobs=[(t,i) for t in tasks for i in range(cfg['teacher']['attempts_per_task'])]
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg['teacher']['concurrency']) as pool:
            futures=[pool.submit(run,t,cfg,backend,tok,cfg['seed']+i,root/'trajectories'/t['task_id']/str(i),phase='teacher') for t,i in jobs]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result());write(root/'progress.json',{'completed':len(results),'total':len(jobs)})
        return {'completed':len(results),'review_required':True}
