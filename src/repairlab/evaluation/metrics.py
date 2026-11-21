import collections


def summarize(by_task,k):
    records=[r for attempts in by_task.values() for r in attempts]
    if any(not r['infrastructure_valid'] for r in records):raise ValueError('Infrastructure-invalid evaluation is incomplete')
    n=len(by_task);turns=sum(len(r['turns']) for r in records);errors=sum('format_error' in t for r in records for t in r['turns'])
    repeat=0;pairs=0
    for r in records:
        commands=[t['actions'][0]['command'] if t.get('actions') else None for t in r['turns']]
        pairs+=sum(bool(a) and bool(b) for a,b in zip(commands,commands[1:]))
        repeat+=sum(bool(a) and a==b for a,b in zip(commands,commands[1:]))
    return dict(tasks=n,attempts=len(records),pass_at_1=sum(v[0]['resolved'] for v in by_task.values())/n if n else 0,
                pass_at_k=sum(any(r['resolved'] for r in v[:k]) for v in by_task.values())/n if n else 0,k=k,
                format_compliance=1-errors/turns if turns else None,format_error_turns=errors,turns=turns,
                exact_adjacent_repeat_count=repeat,eligible_adjacent_pairs=pairs,exact_adjacent_repeat_rate=repeat/pairs if pairs else 0,
                regression_rate=sum(r['regression'] for r in records)/len(records) if records else 0,
                input_tokens=sum(len(t['input_ids']) for r in records for t in r['turns']),
                output_tokens=sum(len(t['output_ids']) for r in records for t in r['turns']),
                termination_counts=dict(collections.Counter(r['termination'] for r in records)),
                per_task={iid:{'first_success':v[0]['resolved'],'any_success':any(r['resolved'] for r in v),'attempts':len(v)} for iid,v in by_task.items()})


def compare(baseline,current):
    for field in ('sampling','budget','k','seed','taskset_digest'):
        if (field in baseline or field in current) and baseline.get(field)!=current.get(field):
            raise ValueError('Comparison differs in '+field)
    if set(baseline['per_task'])!=set(current['per_task']):raise ValueError('Comparison needs identical task sets')
    keys=['pass_at_1','pass_at_k','format_compliance','exact_adjacent_repeat_rate','regression_rate']
    lost=[k for k,v in baseline['per_task'].items() if v['first_success'] and not current['per_task'][k]['first_success']]
    gained=[k for k,v in baseline['per_task'].items() if not v['first_success'] and current['per_task'][k]['first_success']]
    return {'baseline':{k:baseline[k] for k in keys},'current':{k:current[k] for k in keys},
            'delta':{k:current[k]-baseline[k] for k in keys if current[k] is not None and baseline[k] is not None},
            'success_to_failure':lost,'failure_to_success':gained}
