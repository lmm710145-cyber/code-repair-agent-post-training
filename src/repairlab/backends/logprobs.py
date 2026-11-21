import torch
from torch.utils.checkpoint import checkpoint

def selected_logprobs(model,input_ids,output_ids,chunk_size=32):
    """Full-context text forward; checkpoint vocabulary chunks to avoid retaining full logits."""
    seq=torch.tensor([input_ids+output_ids],device='cuda',dtype=torch.long)
    mask=torch.ones_like(seq)
    fixed=getattr(model,'_repairlab_fixed_length',None)
    if fixed:
        if seq.shape[1]>fixed:raise ValueError('fixed context overflow')
        extra=fixed-seq.shape[1]
        seq=torch.nn.functional.pad(seq,(0,extra),value=model._repairlab_pad_token_id)
        mask=torch.nn.functional.pad(mask,(0,extra),value=0)
    base=model.get_base_model()
    hidden=base.model(input_ids=seq,attention_mask=mask,use_cache=False,return_dict=True).last_hidden_state
    h=hidden[0,len(input_ids)-1:len(input_ids)+len(output_ids)-1]
    target=torch.tensor(output_ids,device='cuda',dtype=torch.long)
    def select(x,y):
        logits=base.lm_head(x).float()
        return logits.gather(1,y[:,None]).squeeze(1)-torch.logsumexp(logits,dim=-1)
    out=[]
    for begin in range(0,len(output_ids),chunk_size):
        x=h[begin:begin+chunk_size];y=target[begin:begin+chunk_size]
        out.append(checkpoint(select,x,y,use_reentrant=False) if torch.is_grad_enabled() else select(x,y))
    return torch.cat(out)
