"""Exact-context supervised samples and defect -> trajectory -> step sampling."""
import collections
import random
from ..io import read,write,sha,write_jsonl,jsonl,digest
from ..config import workspace
from ..tasks.schema import catalog
from ..teacher.review import validate_reviews
from ..agent.context import tokenizer,Context
from ..grpo.reward import behavior


def prepare(cfg,review_file):
    rows=validate_reviews(review_file);by_path=collections.defaultdict(list)
    tasks={t['task_id']:t for t in catalog(cfg)};tok=tokenizer(cfg);context=Context(tok,cfg['budget']);kept=[];excluded=[]
    for row in rows:by_path[row['trajectory']].append(row)
    for path,reviews in by_path.items():
        r=read(path);task=tasks[r['task_id']]
        if task['split']!='train' or r['phase']!='teacher':raise ValueError('SFT only accepts train teacher trajectories')
        if {x['turn'] for x in reviews}!=set(range(len(r['turns']))):raise ValueError('Every action must be reviewed')
        if not r['resolved'] or behavior(r,cfg)['events']:
            excluded.append({'path':path,'reason':'final_failure_or_behavior_events'});continue
        if cfg['sft']['whole_trajectory_only'] and any(x['decision']!='keep' for x in reviews):
            excluded.append({'path':path,'reason':'trajectory_has_rejected_actions'});continue
        for row in reviews:
            if row['decision']!='keep':continue
            t=r['turns'][row['turn']]
            if context.encode(t['messages'])!=t['input_ids']:raise ValueError('Recorded teacher input/tokenizer mismatch; do not reconstruct expanded history')
            if len(t['input_ids'])>cfg['budget']['input_tokens'] or len(t['output_ids'])>cfg['budget']['output_tokens']:raise ValueError('Sample exceeds collection budget')
            kept.append(dict(task_id=r['task_id'],repo=task['repo'],defect_type=task['defect_type'],trajectory_id=r['trajectory_id'],
                             turn=t['turn_id'],input_ids=t['input_ids'],output_ids=t['output_ids'],
                             source_sha256=row['trajectory_sha256'],reviewer=row['reviewer'],reviewer_kind=row['reviewer_kind']))
    if not kept:raise ValueError('No eligible SFT actions')
    task_ids=sorted({r['task_id'] for r in kept});rng=random.Random(cfg['seed']);rng.shuffle(task_ids)
    fraction=cfg['sft']['holdout_fraction'];n=max(1,round(len(task_ids)*fraction)) if fraction else 0
    if n>=len(task_ids):raise ValueError('Too few tasks for disjoint train/holdout; explicitly use holdout_fraction=0 only for smoke tests')
    hold=set(task_ids[:n]);train=[r for r in kept if r['task_id'] not in hold];held=[r for r in kept if r['task_id'] in hold]
    root=workspace(cfg,'datasets','sft');write_jsonl(root/'train.jsonl',train);write_jsonl(root/'holdout.jsonl',held)
    info=dict(train_samples=len(train),holdout_samples=len(held),tasks=len(task_ids),trajectories=len({r['trajectory_id'] for r in kept}),
              model=cfg['model'],budget=cfg['budget'],review_sha256=sha(review_file),excluded=excluded,
              train_sha256=sha(root/'train.jsonl'),holdout_sha256=sha(root/'holdout.jsonl'),private_reasoning_included=False)
    write(root/'manifest.json',info);return info


def hierarchical_indices(rows,count,seed):
    tree={}
    for i,row in enumerate(rows):tree.setdefault(row['defect_type'],{}).setdefault(row['trajectory_id'],[]).append(i)
    if not tree:raise ValueError('Empty SFT dataset')
    rng=random.Random(seed);result=[]
    for _ in range(count):
        category=rng.choice(sorted(tree));trajectory=rng.choice(sorted(tree[category]));result.append(rng.choice(tree[category][trajectory]))
    return result
