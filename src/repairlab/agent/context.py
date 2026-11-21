"""Crop only before generation; retain the exact teacher/student-visible context."""
class ContextOverflow(ValueError):pass


class Context:
    def __init__(self,tokenizer,budget):self.tokenizer=tokenizer;self.budget=budget
    def encode(self,messages):
        ids=self.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        return ids['input_ids'] if hasattr(ids,'keys') else ids
    def prepare(self,messages):
        visible=[{'role':m['role'],'content':m['content']} for m in messages if m['role']!='exit']
        truncated=0;dropped=0
        for i,m in enumerate(visible):
            if i>1 and m['role']=='user':
                ids=self.tokenizer.encode(m['content'],add_special_tokens=False)
                if len(ids)>self.budget['feedback_tokens']:
                    k=(self.budget['feedback_tokens']-48)//2
                    text=self.tokenizer.decode(ids[:k])+'\n[OUTPUT TRUNCATED; use bounded queries]\n'+self.tokenizer.decode(ids[-k:])
                    while len(self.tokenizer.encode(text,add_special_tokens=False))>self.budget['feedback_tokens']:
                        k-=8;text=self.tokenizer.decode(ids[:k])+'\n[OUTPUT TRUNCATED]\n'+self.tokenizer.decode(ids[-k:])
                    m['content']=text;truncated+=1
        if len(self.encode(visible[:2]))>self.budget['input_tokens']:raise ContextOverflow('Complete issue plus system prompt exceeds input budget')
        while len(self.encode(visible))>self.budget['input_tokens']:
            end=3
            while end<len(visible) and visible[end]['role']!='assistant':end+=1
            if end>=len(visible):raise ContextOverflow('Newest complete interaction does not fit')
            del visible[2:end];dropped+=1
        return visible,self.encode(visible),{'truncated_feedback_messages':truncated,'dropped_interactions':dropped}


def tokenizer(cfg):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(cfg['model']['path'],revision=cfg['model'].get('revision'),local_files_only=cfg['model']['local_files_only'])
