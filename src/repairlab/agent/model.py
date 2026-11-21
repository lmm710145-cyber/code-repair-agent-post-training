from .context import Context
from .protocol import parse_response, ActionFormatError, FORMAT_HINT
from ..io import write


class AgentModel:
    def __init__(self,backend,tokenizer,cfg,seed,folder):
        self.backend=backend;self.tokenizer=tokenizer;self.cfg=cfg;self.seed=seed;self.folder=folder
        self.context=Context(tokenizer,cfg['budget']);self.records=[];self.generated=0
    def get_template_vars(self):return {'model_name':self.backend.name}
    def format_message(self,**kwargs):return kwargs
    def serialize(self):return {'turns':self.records}
    @staticmethod
    def format_observation_messages(message,outputs,template_vars):
        return [{'role':'user','content':out['output']} for out in outputs]
    def query(self,messages):
        from minisweagent.exceptions import LimitsExceeded,FormatError
        budget=self.cfg['budget'];remaining=budget['episode_output_tokens']-self.generated
        if remaining<=0:raise LimitsExceeded({'role':'exit','content':'Token budget exhausted','extra':{'exit_status':'LimitsExceeded'}})
        visible,ids,stats=self.context.prepare(messages)
        reply=self.backend.generate(visible,ids,min(budget['output_tokens'],remaining),self.seed+len(self.records)*100000)
        content=reply['content'];output_ids=reply.get('output_ids')
        if output_ids is None:
            output_ids=self.tokenizer.encode(content,add_special_tokens=False)+[self.tokenizer.convert_tokens_to_ids('<|im_end|>')]
        self.generated+=len(output_ids)
        rec=dict(turn_id=len(self.records),messages=visible,input_ids=ids,output_ids=output_ids,
                 content=content,context_stats=stats,finish_reason=reply.get('finish_reason'),
                 backend_metadata={k:v for k,v in reply.items() if k not in ('content','output_ids','reasoning_content')},
                 reasoning_content=reply.get('reasoning_content'))
        self.records.append(rec)
        message={'role':'assistant','content':content,'extra':{'cost':0.,'turn_id':rec['turn_id']}}
        try:
            if len(output_ids)>min(budget['output_tokens'],remaining):raise ActionFormatError('Visible completion exceeds token budget')
            parsed=parse_response(content)
            action={'command':parsed['command'],'response':content}
            rec.update(thought=parsed['thought'],actions=[action])
            message['extra']['actions']=[action]
        except ActionFormatError as exc:
            rec.update(format_error=str(exc),actions=[])
            raise FormatError(message,{'role':'user','content':'Action was not executed: '+str(exc)+'. '+FORMAT_HINT})
        finally:write(self.folder/'partial.json',{'turns':self.records,'generated_tokens':self.generated})
        return message
