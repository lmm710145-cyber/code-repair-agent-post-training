"""Short-lived Transformers worker, so sampling and updates never share GPU residency."""
import argparse
import math
from pathlib import Path
from ..config import load
from ..io import read,write,sha,digest,jsonl,verify_run


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--run',required=True);p.add_argument('--stage',choices=['sft','grpo'],required=True);p.add_argument('--initialize',action='store_true');p.add_argument('--target',type=int);p.add_argument('--batch');p.add_argument('--round',type=int,default=0);a=p.parse_args()
    cfg=load(a.config);root=Path(a.run);c=cfg[a.stage];verify_run(root)
    from .training import load as load_model,restore,save
    model,tok=load_model(cfg,c.get('initial_adapter') if a.initialize else None)
    import torch
    from .logprobs import selected_logprobs
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=c['learning_rate'],weight_decay=c.get('weight_decay',0),foreach=False)
    latest=root/'latest.json';identity=digest(cfg);step=0;history=[]
    if not a.initialize:
        meta=read(latest);state=restore(model,opt,meta['state'])
        if state['config_digest']!=identity:raise ValueError('Training config changed')
        step=state['updates'];history=state['history']
    elif latest.exists():raise FileExistsError('Initialization already committed')
    def checkpoint(label):
        meta=save(root/'checkpoints'/label,model,tok,opt,dict(updates=step,round=a.round,history=history,config_digest=identity,stage=a.stage,model=cfg['model'],lora=cfg['lora']))
        write(latest,meta);return meta
    if a.initialize:
        meta=checkpoint('initial');write(root/'initial.json',meta);return
    torch.cuda.reset_peak_memory_stats()
    if a.stage=='sft':
        from ..sft.data import hierarchical_indices
        data=Path(cfg['workspace'])/'datasets/sft';manifest=read(data/'manifest.json')
        if sha(data/'train.jsonl')!=manifest['train_sha256'] or manifest['model']!=cfg['model'] or manifest['budget']!=cfg['budget']:raise ValueError('SFT data/model/budget mismatch')
        rows=jsonl(data/'train.jsonl');indices=hierarchical_indices(rows,a.target*c['batch_size'],cfg['seed'])
        while step<a.target:
            opt.zero_grad(set_to_none=True);loss_sum=0.
            for index in indices[step*c['batch_size']:(step+1)*c['batch_size']]:
                row=rows[index]
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=-selected_logprobs(model,row['input_ids'],row['output_ids']).mean()/c['batch_size']
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite SFT loss')
                loss.backward();loss_sum+=loss.item()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['max_grad_norm'])
            if not torch.isfinite(norm):raise RuntimeError('Nonfinite SFT gradient')
            opt.step();step+=1;history.append({'step':step,'loss':loss_sum,'learning_rate':c['learning_rate']})
            write(root/'training-progress.json',history[-1])
        checkpoint(f'step-{step:06d}')
    else:
        from ..grpo.reward import advantages
        from ..grpo.objective import clipped_loss
        batch=read(a.batch);valid=[];discarded=[]
        for group in batch['groups']:
            records=[read(p) for p in group['results']]
            if len(records)!=c['group_size'] or len({r['seed'] for r in records})!=c['group_size']:raise ValueError('Incomplete/duplicate GRPO group')
            if len({r['task_id'] for r in records})!=1 or any(r['split']!='train' or r['phase']!='grpo' for r in records):raise ValueError('Cross-task or non-train GRPO group')
            expected=str(step)+':'+read(latest)['adapter_sha256']
            if any(r['identity']['config']!=identity for r in records):raise ValueError('Rollout config mismatch')
            if any(r['identity']['policy']!=expected for r in records):raise ValueError('Stale rollout policy')
            adv=advantages([r['reward'] for r in records])
            if not any(adv):discarded.append(group['task_id']);continue
            valid.append((records,adv))
        denominator=sum(len(x[0]) for x in valid);ratio_error=0.;loss_sum=0.;audit=[]
        opt.zero_grad(set_to_none=True)
        lr=c['learning_rate']*min((step+1)/max(1,c['warmup_steps']),1.)
        for pg in opt.param_groups:pg['lr']=lr
        for records,advs in valid:
            for r,adv in zip(records,advs):
                total=sum(len(t['output_ids']) for t in r['turns']);row=[]
                if total<=0:raise ValueError('Empty policy trajectory')
                for t in r['turns']:
                    with torch.autocast('cuda',dtype=torch.bfloat16):old=selected_logprobs(model,t['input_ids'],t['output_ids']).detach()
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        new=selected_logprobs(model,t['input_ids'],t['output_ids'])
                        loss,ratio=clipped_loss(new,old,adv,denominator,total,c['clip_epsilon'])
                        error=(ratio.detach()-1).abs().max().item();ratio_error=max(ratio_error,error)
                        if error>c['replay_ratio_tolerance']:raise RuntimeError('Old/new replay differs before optimizer update')
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite GRPO loss')
                    loss.backward();loss_sum+=loss.item()
                    q=torch.tensor(t['backend_metadata']['sampler_logprobs'],device=old.device)
                    row.append(dict(turn=t['turn_id'],training_old_logprobs=old.cpu().tolist(),sampler_logp_mae=(old-q).abs().mean().item(),ratio_error=error))
                    del loss,new,old,ratio,q
                audit.append(dict(trajectory=r['trajectory_id'],advantage=adv,turns=row))
        if valid:
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['max_grad_norm'])
            if not torch.isfinite(norm) or norm==0:raise RuntimeError('Invalid policy gradient')
            opt.step();step+=1
        history.append(dict(round=a.round,step=step,updated=bool(valid),signal_groups=len(valid),same_reward_groups=discarded,
                            loss=loss_sum,learning_rate=lr,replay_ratio_error=ratio_error))
        write(root/'replay'/f'round-{a.round:06d}.json',audit)
        checkpoint(f'round-{a.round:06d}')
    write(root/'training.json',dict(updates=step,history=history,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))

if __name__=='__main__':main()
