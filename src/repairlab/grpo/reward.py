"""Outcome plus capped behavioral costs; no claim of causal step credit assignment."""
import math
from ..agent.protocol import parse_response,ActionFormatError
from ..agent.environment import SUBMIT
import shlex
import re


def repeat_candidate(command):
    if any(c in command for c in '\n;|&><`$'):return False
    try:w=shlex.split(command)
    except ValueError:return False
    if not w:return False
    if w[0] in ('cat','head','tail','pwd','ls','rg','grep','pytest'):return True
    if w[:3] in (['python','-m','pytest'],['python3','-m','pytest']):return True
    if w[0]=='git':return len(w)>1 and w[1] in ('diff','status','show','log')
    return w[0]=='sed' and len(w)>=4 and w[1]=='-n' and bool(re.fullmatch(r'\d+(?:,\d+)?p',w[2]))


def behavior(record,cfg):
    units=cfg['reward']['units'];events=[];previous=None;streak=0
    for t in record['turns']:
        if 'format_error' in t:
            kind='format'
            if t.get('finish_reason')=='length':
                try:parse_response(t['content'].rstrip()+'\n```');kind='output_truncation'
                except ActionFormatError:pass
            events.append(dict(turn=t['turn_id'],kind=kind,cost=units[kind]))
            previous=None;streak=0;continue
        e=t.get('execution');command=t['actions'][0]['command']
        if e and repeat_candidate(command) and e['before_state']==e['after_state'] and not e.get('resource_reason'):
            key=(command.strip(),e['before_state'],e['output_sha256'],e['returncode'])
            streak=streak+1 if key==previous else 1;previous=key
            if streak>=3:events.append(dict(turn=t['turn_id'],kind='repetition',cost=units['repetition']))
        else:previous=None;streak=0
    costs={k:math.fsum(e['cost'] for e in events if e['kind']==k) for k in units}
    return {'events':events,'costs':costs,'raw':math.fsum(costs.values())}


def reward(record,cfg):
    r=cfg['reward'];b=behavior(record,cfg)
    exhausted=(len(record['turns'])>=cfg['budget']['max_turns'] or record['generated_tokens']>=cfg['budget']['episode_output_tokens'])
    accepted=record['termination']=='Submitted' and any(t.get('execution',{}).get('command','').strip()==SUBMIT and t['execution']['returncode']==0 for t in record['turns'])
    termination=r['exhausted_without_submission'] if exhausted and not accepted else 0
    capped=min(b['raw'],r['behavior_cap'])
    if not record['infrastructure_valid']:base=None;value=None
    elif not record['scope_clean'] or record['tampered']:base=r['compliance_violation'];value=base
    else:
        base=r['regression'] if record['regression'] else r['success'] if record['grader']['resolved'] else r['failure']
        value=base-capped-termination
    return dict(reward=value,base_reward=base,behavior_cost=capped,behavior=b,termination_cost=termination,
                accepted_submission=accepted,budget_exhausted=exhausted)


def advantages(rewards,epsilon=1e-6):
    if not rewards or any(x is None or not math.isfinite(x) for x in rewards):raise ValueError('Invalid group rewards')
    mean=sum(rewards)/len(rewards);std=(sum((x-mean)**2 for x in rewards)/len(rewards))**.5
    return [0.]*len(rewards) if std<=1e-8 else [(x-mean)/(std+epsilon) for x in rewards]
