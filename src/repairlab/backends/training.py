"""BF16 language-only LoRA; fixed-shape reference forwards for Qwen3.5."""
import json
import re
import types
from pathlib import Path


def load(cfg,adapter=None):
    # Import Unsloth before transformers; this module is loaded only in GPU workers.
    from unsloth import FastLanguageModel
    import torch
    torch.manual_seed(cfg['seed'])
    model,tok=FastLanguageModel.from_pretrained(model_name=cfg['model']['path'],revision=cfg['model'].get('revision'),
         max_seq_length=cfg['budget']['context_tokens'],dtype=torch.bfloat16,load_in_4bit=False,load_in_16bit=True,
         full_finetuning=False,fast_inference=False,device_map={'':'cuda:0'},local_files_only=cfg['model']['local_files_only'])
    if hasattr(tok,'tokenizer'):tok=tok.tokenizer
    targets=[n for n,m in model.named_modules() if '.language_model.' in n and n.rsplit('.',1)[-1] in cfg['lora']['target_modules'] and isinstance(m,torch.nn.Linear)]
    if not targets:raise ValueError('No Qwen3.5 language-only projection modules matched')
    model=FastLanguageModel.get_peft_model(model,r=cfg['lora']['r'],lora_alpha=cfg['lora']['alpha'],lora_dropout=0,bias='none',
         target_modules='(?:'+'|'.join(re.escape(n) for n in targets)+')',use_gradient_checkpointing='unsloth',random_state=cfg['seed'])
    for n,p in model.named_parameters():
        if p.requires_grad:
            if 'lora_' not in n or '.visual.' in n:raise ValueError('Unexpected trainable parameter')
            p.data=p.data.float()
    if adapter:
        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict,get_peft_model_state_dict
        state=load_file(str(Path(adapter)/'adapter_model.safetensors'))
        current=get_peft_model_state_dict(model)
        if state.keys()!=current.keys() or any(state[k].shape!=v.shape for k,v in current.items()):raise ValueError('Initializer architecture/LoRA mismatch')
        set_peft_model_state_dict(model,state)
    else:
        if any(torch.count_nonzero(p).item() for n,p in model.named_parameters() if 'lora_B' in n):raise ValueError('Fresh LoRA is not zero-effect')
    FastLanguageModel.for_training(model);model.train();model.config.use_cache=False
    for name,param in model.named_parameters():
        trainable='lora_' in name and '.visual.' not in name
        param.requires_grad_(trainable)
        if trainable:param.data=param.data.float()
    model._repairlab_fixed_length=cfg['budget']['context_tokens'];model._repairlab_pad_token_id=tok.pad_token_id
    head=model.get_base_model().lm_head
    def forward(self,x):
        with torch.autocast('cuda',enabled=False):return torch.nn.functional.linear(x.float(),self.weight.float())
    head.forward=types.MethodType(forward,head)
    return model,tok


def cpu(value):
    import torch
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu(v) for v in value)
    return value


def save(folder,model,tok,optimizer,metadata):
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    from ..io import write,sha
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(folder/'adapter');tok.save_pretrained(folder/'adapter')
    state=cpu(get_peft_model_state_dict(model));stored=load_file(str(folder/'adapter/adapter_model.safetensors'))
    if state.keys()!=stored.keys() or any(not torch.equal(v,stored[k]) for k,v in state.items()):raise ValueError('Adapter serialization mismatch')
    checkpoint=dict(adapter=state,optimizer=cpu(optimizer.state_dict()),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),metadata=metadata)
    temp=folder/'state.tmp';torch.save(checkpoint,temp);temp.replace(folder/'state.pt')
    meta=dict(metadata,adapter=str((folder/'adapter').resolve()),state=str((folder/'state.pt').resolve()),adapter_sha256=sha(folder/'adapter/adapter_model.safetensors'))
    write(folder/'metadata.json',meta);return meta


def restore(model,opt,path):
    import torch
    from peft import set_peft_model_state_dict
    state=torch.load(path,map_location='cpu',weights_only=True);set_peft_model_state_dict(model,state['adapter']);opt.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state(state['cuda_rng']);return state['metadata']
