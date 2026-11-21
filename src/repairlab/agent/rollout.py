import time
from pathlib import Path
from .environment import TaskEnvironment
from .model import AgentModel
from .prompt import SYSTEM_REACT
from ..io import write,read,digest
from ..tasks.grader import grade


def run(task,cfg,backend,tokenizer,seed,folder,*,phase):
    from minisweagent.agents.default import DefaultAgent
    from ..grpo.reward import reward
    folder=Path(folder);identity=dict(task=digest(task),config=digest(cfg),policy=backend.identity,seed=seed,phase=phase)
    if (folder/'result.json').exists():
        r=read(folder/'result.json')
        if r['identity']!=identity:raise ValueError('Existing rollout belongs to different policy/config')
        return r
    if (folder/'started.json').exists():raise RuntimeError('Interrupted episode requires a fresh attempt directory; partial trajectories are not reusable')
    write(folder/'started.json',dict(identity=identity,started=time.time()))
    started=time.time();local=AgentModel(backend,tokenizer,cfg,seed,folder)
    try:
        with TaskEnvironment(task,cfg) as env:
            prompt=SYSTEM_REACT.replace('1024 tokens',str(cfg['budget']['output_tokens'])+' tokens').replace('1536 tokens',str(cfg['budget']['feedback_tokens'])+' tokens').replace('120-second',str(cfg['budget']['command_seconds'])+'-second')
            agent=DefaultAgent(local,env,system_template=prompt,instance_template='{{task}}',cost_limit=0,
                 step_limit=cfg['budget']['max_turns'],wall_time_limit_seconds=cfg['budget']['episode_seconds'],
                 max_consecutive_format_errors=cfg['budget']['max_format_errors'],output_path=folder/'mini.json')
            terminal=agent.run(task['problem_statement'])
            patch=env.patch();full=env.patch(full=True);executions=env.records
        # Bind actual executions to generated actions, including accepted submission.
        index=0
        for turn in local.records:
            if turn['actions']:
                if index>=len(executions) or executions[index]['command']!=turn['actions'][0]['command']:
                    raise RuntimeError('Action execution records do not align')
                turn['execution']=executions[index];index+=1
        if index!=len(executions):raise RuntimeError('Unmatched execution records')
        r=dict(identity=identity,task_id=task['task_id'],repo=task['repo'],split=task['split'],defect_type=task['defect_type'],
               trajectory_id=str(folder),seed=seed,phase=phase,turns=local.records,termination=terminal.get('exit_status'),
               generated_tokens=local.generated,patch=patch,full_patch=full,scope_clean=patch==full,
               tampered=any(e['observer_tampered'] for e in executions),infrastructure_valid=True)
        write(folder/'candidate.json',r)
        grading=grade(task,cfg,patch,folder/'validation')
        r.update(grader=grading,resolved=grading['resolved'] and r['scope_clean'] and not r['tampered'],regression=grading['regression'])
        r['reward_breakdown']=reward(r,cfg);r['reward']=r['reward_breakdown']['reward'];r['seconds']=time.time()-started
        write(folder/'result.json',r)
        return r
    except Exception as exc:
        write(folder/'infrastructure-error.json',dict(type=type(exc).__name__,message=str(exc),seconds=time.time()-started))
        # Never train on API/environment failures as if they were failed repairs.
        raise
