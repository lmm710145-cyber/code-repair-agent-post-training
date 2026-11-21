"""Train-only repo-balanced sampling, with decayed success statistics and exploration."""
import random
from ..io import read,write


class Curriculum:
    def __init__(self,tasks,cfg,state=None):
        if not tasks or any(t['split']!='train' for t in tasks):raise ValueError('Curriculum only accepts train tasks')
        self.tasks={t['task_id']:t for t in tasks};self.cfg=cfg;self.stats={k:{'success':0.,'failure':0.,'round':0} for k in self.tasks}
        if state:
            if state['task_ids']!=sorted(self.tasks):raise ValueError('Curriculum task set changed')
            self.stats=state['stats']
    def select(self,round_id,seed,count,G):
        rng=random.Random(seed+round_id);repos={}
        for task in self.tasks.values():repos.setdefault(task['repo'],[]).append(task['task_id'])
        selected_repos=rng.sample(sorted(repos),min(count,len(repos)))
        if len(selected_repos)!=count:raise ValueError('Need at least one distinct repository per task group')
        result=[]
        for repo in selected_repos:
            ids=sorted(repos[repo]);weights=[]
            for iid in ids:
                s=self.stats[iid];decay=2**(-(round_id-s['round'])/self.cfg['half_life_rounds'])
                p=(s['success']*decay+1)/(2+(s['success']+s['failure'])*decay)
                weights.append(self.cfg['score_floor']+1-p**G-(1-p)**G)
            total=sum(weights);e=self.cfg['exploration_fraction']
            weights=[e/len(ids)+(1-e)*w/total for w in weights] if self.cfg['enabled'] and round_id>self.cfg['cold_start_rounds'] else None
            result.append(rng.choices(ids,weights=weights,k=1)[0])
        return result
    def record(self,iid,round_id,records):
        if any(not r['infrastructure_valid'] for r in records):return
        s=self.stats[iid];decay=2**(-(round_id-s['round'])/self.cfg['half_life_rounds'])
        success=sum(r['resolved'] for r in records)
        self.stats[iid]={'success':s['success']*decay+success,'failure':s['failure']*decay+len(records)-success,'round':round_id}
    def state(self):return {'task_ids':sorted(self.tasks),'stats':self.stats}
