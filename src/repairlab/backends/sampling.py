import contextlib
import json
import os
import shutil
import signal
import subprocess
import time
import urllib.request
import urllib.error
from pathlib import Path
from ..io import read,write,sha


def request(port,payload=None,timeout=1800):
    req=urllib.request.Request(f'http://127.0.0.1:{port}/generate' if payload else f'http://127.0.0.1:{port}/health',
         data=json.dumps(payload).encode() if payload else None,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as response:return json.load(response)


@contextlib.contextmanager
def engine(cfg,meta,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);target=folder/'sampling-adapter'
    if not target.exists():
        shutil.copytree(meta['adapter'],target)
        path=target/'adapter_config.json';c=read(path);c['target_modules']=cfg['lora']['target_modules'];write(path,c)
    if sha(target/'adapter_model.safetensors')!=meta['adapter_sha256']:raise ValueError('Sampling adapter mismatch')
    version=str(meta['updates'])+':'+meta['adapter_sha256'];port=cfg['vllm']['port']
    env=os.environ.copy();env.update(TOKENIZERS_PARALLELISM='false',VLLM_WORKER_MULTIPROC_METHOD='spawn')
    with (folder/'vllm.log').open('a') as log:
        process=subprocess.Popen([cfg['vllm']['python'],'-m','repairlab.backends.vllm_server','--config',cfg['_config_file'],
              '--adapter',str(target),'--version',version,'--port',str(port)],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try:
            for _ in range(600):
                if process.poll() is not None:raise RuntimeError('vLLM exited; inspect '+str(folder/'vllm.log'))
                try:
                    status=request(port,timeout=1)
                    if status.get('ready'):
                        if status['policy']!=version or status['adapter_sha256']!=meta['adapter_sha256']:raise RuntimeError('Serving endpoint belongs to another policy')
                        break
                except (urllib.error.URLError,TimeoutError):pass
                time.sleep(1)
            else:raise TimeoutError('vLLM startup timeout')
            yield version
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass


class Sampler:
    name='Qwen3.5-vLLM'
    def __init__(self,cfg,tokenizer,version,meta,*,evaluation=False):
        self.cfg=cfg;self.tok=tokenizer;self.version=version;self.identity=version
        self.sampling=cfg['evaluation' if evaluation else 'grpo']['sampling'];self.meta=meta
    def generate(self,messages,input_ids,max_tokens,seed):
        payload=dict(input_ids=input_ids,max_tokens=max_tokens,seed=seed,policy=self.version,
              eos_token_id=self.tok.convert_tokens_to_ids('<|im_end|>'),**self.sampling)
        raw=request(self.cfg['vllm']['port'],payload)
        if raw['policy']!=self.version or raw['adapter_sha256']!=self.meta['adapter_sha256']:raise ValueError('Stale sampler response')
        return dict(raw,content=self.tok.decode(raw['output_ids'],skip_special_tokens=True))
