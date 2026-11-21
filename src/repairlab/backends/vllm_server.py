"""Isolated GPU owner. Batched token-ID requests with explicit policy identity."""
import argparse
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from ..config import load
from ..io import sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--adapter',required=True);ap.add_argument('--version',required=True);ap.add_argument('--port',type=int,required=True);args=ap.parse_args()
    cfg=load(args.config);v=cfg['vllm']
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    llm=LLM(model=cfg['model']['path'],revision=cfg['model'].get('revision'),dtype='bfloat16',
            max_model_len=cfg['budget']['context_tokens'],max_num_seqs=v['concurrency'],gpu_memory_utilization=v['memory_fraction'],
            enforce_eager=True,enable_prefix_caching=False,enable_lora=True,max_lora_rank=cfg['lora']['r'],max_loras=1,
            limit_mm_per_prompt={'image':0,'video':0},seed=cfg['seed'])
    adapter=LoRARequest('policy',1,args.adapter);adapter_sha=sha(Path(args.adapter)/'adapter_model.safetensors');pending=queue.Queue()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,data,status=200):
            body=json.dumps(data).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):self.reply(dict(ready=True,policy=args.version,adapter_sha256=adapter_sha))
        def do_POST(self):
            try:
                data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if data['policy']!=args.version:raise ValueError('Stale policy')
                event=threading.Event();item=dict(data=data,event=event);pending.put(item)
                if not event.wait(1800):raise TimeoutError('Generation timeout')
                self.reply(item['result'],500 if 'error' in item['result'] else 200)
            except Exception as exc:self.reply({'error':type(exc).__name__},500)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
    while True:
        items=[pending.get()];deadline=time.monotonic()+.03
        while len(items)<v['concurrency']:
            try:items.append(pending.get(timeout=max(0,deadline-time.monotonic())))
            except queue.Empty:break
        try:
            params=[SamplingParams(temperature=x['data']['temperature'],top_p=x['data']['top_p'],top_k=-1,
                      repetition_penalty=1.,max_tokens=x['data']['max_tokens'],seed=x['data']['seed'],logprobs=0,
                      stop_token_ids=[x['data']['eos_token_id']],detokenize=False) for x in items]
            result=llm.generate([{'prompt_token_ids':x['data']['input_ids']} for x in items],params,lora_request=adapter,use_tqdm=False)
            for item,out in zip(items,result):
                o=out.outputs[0];ids=list(o.token_ids)
                if list(out.prompt_token_ids)!=item['data']['input_ids']:raise ValueError('Input IDs changed')
                logp=[float(lp[t].logprob) for t,lp in zip(ids,o.logprobs)]
                if not ids or len(logp)!=len(ids):raise ValueError('Missing log probabilities')
                item['result']=dict(output_ids=ids,sampler_logprobs=logp,finish_reason=o.finish_reason,policy=args.version,adapter_sha256=adapter_sha)
        except Exception as exc:
            for item in items:item['result']={'error':type(exc).__name__+': '+str(exc)}
        finally:
            for item in items:item['event'].set()

if __name__=='__main__':main()
