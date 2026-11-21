"""OpenAI-compatible Kimi transport. Private reasoning is audit-only, never an SFT label."""
import json
import os
import threading
import urllib.request
import urllib.error
from ..io import digest


class Teacher:
    def __init__(self,cfg):
        self.cfg=cfg['teacher'];self.name=self.cfg['model'];self.identity=digest(self.cfg)
        self.key=os.environ.get(self.cfg['api_key_env'])
        if not self.key:raise ValueError('Missing teacher key environment variable: '+self.cfg['api_key_env'])
        if not self.cfg['enabled']:raise ValueError('Teacher calls are disabled in config')
        self.calls=0;self.lock=threading.Lock()
    def generate(self,messages,input_ids,max_tokens,seed):
        with self.lock:
            if self.calls>=self.cfg['max_requests']:raise RuntimeError('Teacher request cap reached')
            self.calls+=1
        payload=dict(model=self.name,messages=messages,stream=False,max_tokens=self.cfg['completion_tokens'],
                     reasoning_effort=self.cfg['reasoning_effort'])
        payload.update(self.cfg.get('parameters',{}))
        req=urllib.request.Request(self.cfg['base_url'].rstrip('/')+'/chat/completions',
              data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req,timeout=self.cfg['timeout_seconds']) as response:
                body=response.read().decode().replace(self.key,'[REDACTED]');data=json.loads(body)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f'Teacher HTTP {exc.code}; no automatic paid retry') from None
        except (urllib.error.URLError,TimeoutError):raise RuntimeError('Teacher transport failure; no automatic paid retry') from None
        choice=data['choices'][0];m=choice['message']
        if not isinstance(m.get('content'),str):raise ValueError('Teacher returned no visible completion')
        return dict(content=m['content'],reasoning_content=m.get('reasoning_content'),finish_reason=choice.get('finish_reason'),
                    usage=data.get('usage'),requested_reasoning_effort=self.cfg['reasoning_effort'],
                    provider_honored_reasoning_effort='not_independently_verified')
