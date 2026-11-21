import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from .config import load,workspace
from .io import read


def main():
    p=argparse.ArgumentParser(description='Repository repair agent post-training pipeline')
    p.add_argument('--config',default='configs/pipeline.example.json')
    groups=p.add_subparsers(dest='stage',required=True)
    groups.add_parser('doctor')
    t=groups.add_parser('tasks').add_subparsers(dest='action',required=True)
    b=t.add_parser('build');b.add_argument('--spec',required=True)
    c=t.add_parser('certify');c.add_argument('--task-id')
    t.add_parser('inspect')
    f=t.add_parser('freeze');f.add_argument('--human-reviews',required=True)
    t=groups.add_parser('teacher').add_subparsers(dest='action',required=True)
    c=t.add_parser('collect');c.add_argument('--run',required=True);c.add_argument('--task-id',action='append')
    a=t.add_parser('audit');a.add_argument('--trajectories',required=True)
    s=groups.add_parser('sft').add_subparsers(dest='action',required=True)
    a=s.add_parser('prepare');a.add_argument('--reviews',required=True)
    a=s.add_parser('train');a.add_argument('--run',required=True)
    g=groups.add_parser('grpo');g.add_argument('--run',required=True)
    e=groups.add_parser('evaluate');e.add_argument('--checkpoint',required=True,help='checkpoint metadata.json');e.add_argument('--output',required=True);e.add_argument('--baseline');e.add_argument('--split',choices=['dev','test'],default='dev');e.add_argument('--k',type=int,choices=[1,5],default=1)
    x=groups.add_parser('export-swebench');x.add_argument('--trajectories',required=True);x.add_argument('--mapping',required=True);x.add_argument('--output',required=True);x.add_argument('--model-name',required=True)
    a=groups.add_parser('compare');a.add_argument('baseline');a.add_argument('current')
    d=groups.add_parser('smoke');d.add_argument('--output',required=True);d.add_argument('--image',default='repairlab/tasks:py311')
    args=p.parse_args();cfg=load(args.config)
    if args.stage=='doctor':
        result={'python':os.sys.version.split()[0],'workspace':cfg['workspace'],'docker_command':shutil.which('docker'),
                'model_path_exists':Path(cfg['model']['path']).exists(),'vllm_python':shutil.which(cfg['vllm']['python']),
                'teacher_enabled':cfg['teacher']['enabled'],'api_key_present':bool(os.environ.get(cfg['teacher']['api_key_env'])),
                'gpu_allocation':False}
        if result['docker_command']:
            q=subprocess.run(['docker','info','--format','{{.ServerVersion}}'],capture_output=True,text=True);result['docker_access']=q.returncode==0
    elif args.stage=='tasks':
        from .tasks.schema import catalog
        if args.action=='build':
            from .tasks.build import build
            result=build(cfg,args.spec)
        elif args.action=='certify':
            from .tasks.grader import certify
            selected=[t for t in catalog(cfg) if not args.task_id or t['task_id']==args.task_id]
            if not selected:raise ValueError('Task not found')
            result={t['task_id']:certify(t,cfg) for t in selected}
        elif args.action=='freeze':
            from .tasks.review import freeze_tasks
            result=freeze_tasks(cfg,args.human_reviews)
        else:
            import collections
            tasks=catalog(cfg);result=dict(tasks=len(tasks),repos=len({t['repo'] for t in tasks}),splits=dict(collections.Counter(t['split'] for t in tasks)),defects=dict(collections.Counter(t['defect_type'] for t in tasks)))
    elif args.stage=='teacher':
        if args.action=='collect':
            from .teacher.collect import collect
            result=collect(cfg,args.run,args.task_id)
        else:
            from .teacher.review import audit
            result=audit(args.trajectories,cfg)
    elif args.stage=='sft':
        if args.action=='prepare':
            from .sft.data import prepare
            result=prepare(cfg,args.reviews)
        else:
            from .sft.train import train
            result=train(cfg,args.run)
    elif args.stage=='grpo':
        from .grpo.train import train
        result=train(cfg,args.run)
    elif args.stage=='evaluate':
        from .evaluation.run import evaluate
        result=evaluate(cfg,read(args.checkpoint),args.output,split=args.split,k=args.k,baseline=args.baseline)
    elif args.stage=='export-swebench':
        from .evaluation.export import export_predictions
        result=export_predictions(args.trajectories,args.mapping,args.output,args.model_name)
    elif args.stage=='compare':
        from .evaluation.metrics import compare
        result=compare(read(args.baseline),read(args.current))
    else:
        from .demo import smoke
        result=smoke(cfg,args.output,args.image)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
