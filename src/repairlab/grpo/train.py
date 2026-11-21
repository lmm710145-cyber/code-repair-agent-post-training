"""On-policy G-way groups, dynamic task selection, atomic optimizer/curriculum commits."""
import concurrent.futures
import shutil
import time
from ..config import workspace
from ..io import lock,freeze,read,write,digest,verify_run
from ..pipeline import initialize,worker,baseline
from ..tasks.schema import catalog
from ..agent.context import tokenizer
from ..agent.rollout import run
from ..backends.sampling import engine,Sampler
from ..evaluation.run import evaluate
from .curriculum import Curriculum


def train(cfg,run_name):
    c=cfg['grpo'];root=workspace(cfg,'runs',run_name)
    if c['sampling']!={'temperature':1.0,'top_p':1.0}:raise ValueError('This classic replay implementation requires training temperature=1 and top_p=1')
    with lock(root):
        all_tasks=catalog(cfg);tasks={t['task_id']:t for t in all_tasks if t['split']=='train'}
        freeze(root,dict(config=cfg,tasks=all_tasks,stage='grpo'))
        meta=initialize(cfg,root,'grpo');tok=tokenizer(cfg)
        if cfg['evaluation']['during_training']:baseline(cfg,root)
        if meta['round']:
            state=read(root/'batches'/f"round-{meta['round']:06d}.json")['curriculum_after']
        else:state=None
        curriculum=Curriculum(list(tasks.values()),c['curriculum'],state)
        no_signal=0
        for h in reversed(meta['history']):
            if h.get('updated'):break
            no_signal+=1
        try:
            for round_id in range(meta['round']+1,c['max_sampling_rounds']+1):
                if meta['updates']>=c['updates']:break
                verify_run(root)
                if shutil.disk_usage(root).free<cfg['environment']['minimum_free_gib']*2**30:raise RuntimeError('Low disk space; checkpoint retained')
                batch_path=root/'batches'/f'round-{round_id:06d}.json'
                write(root/'status.json',dict(status='sampling',round=round_id,updates=meta['updates'],target=c['updates']))
                if batch_path.exists():
                    batch=read(batch_path)
                    if batch['adapter_sha256']!=meta['adapter_sha256']:raise ValueError('Cached batch has stale policy')
                    curriculum=Curriculum(list(tasks.values()),c['curriculum'],batch['curriculum_after'])
                else:
                    chosen=curriculum.select(round_id,cfg['seed'],c['task_groups_per_update'],c['group_size']);groups=[]
                    with engine(cfg,meta,root/'sampling'/f'round-{round_id:06d}') as version:
                        backend=Sampler(cfg,tok,version,meta)
                        jobs=[];results={iid:[] for iid in chosen}
                        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg['vllm']['concurrency']) as pool:
                            for gi,iid in enumerate(chosen):
                                for i in range(c['group_size']):
                                    seed=cfg['seed']+round_id*c['task_groups_per_update']*c['group_size']+gi*c['group_size']+i
                                    folder=root/'trajectories'/f'round-{round_id:06d}'/iid/str(i)
                                    jobs.append(pool.submit(run,tasks[iid],cfg,backend,tok,seed,folder,phase='grpo'))
                            for f in concurrent.futures.as_completed(jobs):
                                r=f.result();results[r['task_id']].append(r)
                                write(root/'sampling-progress.json',dict(round=round_id,completed=sum(len(v) for v in results.values()),total=len(jobs)))
                    for iid in chosen:
                        records=sorted(results[iid],key=lambda x:x['seed']);curriculum.record(iid,round_id,records)
                        groups.append(dict(task_id=iid,results=[r['trajectory_id']+'/result.json' for r in records]))
                    batch=dict(groups=groups,round=round_id,adapter_sha256=meta['adapter_sha256'],curriculum_after=curriculum.state())
                    write(batch_path,batch)
                previous=meta['updates'];worker(cfg,root,'grpo',['--batch',str(batch_path),'--round',str(round_id)])
                meta=read(root/'latest.json');no_signal=no_signal+1 if meta['updates']==previous else 0
                write(root/'status.json',dict(status='running',round=round_id,updates=meta['updates'],target=c['updates']))
                if meta['updates']>previous and meta['updates'] in c['eval_updates'] and cfg['evaluation']['during_training']:
                    evaluate(cfg,meta,root/'evaluation'/f"update-{meta['updates']}",k=1,baseline=root/'evaluation/initial/metrics.json')
                if no_signal>=c['max_consecutive_no_signal']:raise RuntimeError('No gradient signal for too many rounds; not complete')
            complete=meta['updates']==c['updates'];write(root/'status.json',dict(status='complete' if complete else 'round_limit_reached',updates=meta['updates']))
            if not complete:raise RuntimeError('Sampling guard reached before requested updates')
            return {'updates':meta['updates'],'run':str(root)}
        except Exception as exc:
            write(root/'status.json',dict(status='stopped_requires_reconciliation',updates=meta['updates'],error=str(exc)));raise
