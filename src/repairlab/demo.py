"""Network/API/GPU-free synthetic smoke task; Docker mode exercises the actual scaffold."""
import copy
import subprocess
from pathlib import Path
from .io import write,read,write_jsonl
from .tasks.build import build
from .tasks.grader import certify
from .agent.rollout import run
from .teacher.review import audit


class ByteTokenizer:
    """Test double only; never use for model training or real inference."""
    def encode(self,text,**kwargs):return list(text.encode())
    def decode(self,ids,**kwargs):return bytes(i for i in ids if i<256).decode(errors='replace')
    def convert_tokens_to_ids(self,token):return 256
    def apply_chat_template(self,messages,**kwargs):
        return self.encode(''.join(m['role']+':'+m['content']+'\n' for m in messages)+'assistant:')


class ScriptedBackend:
    name='scripted-smoke-fixture';identity='scripted-smoke-v1'
    def __init__(self):self.index=0
    def generate(self,messages,input_ids,max_tokens,seed):
        commands=["python -c \"from pathlib import Path; p=Path('src/adder.py'); p.write_text(p.read_text().replace('a - b', 'a + b'))\"",
                  'python -m pytest -q tests','echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT']
        command=commands[self.index];self.index+=1
        return {'content':'Thought:\nApply or verify the small repair.\nAction:\n```mswea_bash_command\n'+command+'\n```','finish_reason':'stop'}


def smoke(cfg,output,image):
    cfg=copy.deepcopy(cfg);root=Path(output).resolve()
    if root.exists() and any(root.iterdir()):raise FileExistsError('Smoke output must be empty')
    root.mkdir(parents=True,exist_ok=True);cfg['workspace']=str(root/'workspace');repo=root/'checkout';repo.mkdir()
    (repo/'src').mkdir();(repo/'tests').mkdir();(repo/'src/adder.py').write_text('def add(a, b):\n    return a - b\n')
    (repo/'tests/test_adder.py').write_text('from adder import add\n\ndef test_zero():\n    assert add(0, 0) == 0\n')
    def git(*args):return subprocess.run(['git','-C',str(repo),*args],check=True,capture_output=True,text=True).stdout.strip()
    git('init','-q');git('config','user.name','Smoke Test');git('config','user.email','smoke@example.invalid');git('add','.');git('commit','-qm','buggy fixture');base=git('rev-parse','HEAD')
    (repo/'src/adder.py').write_text('def add(a, b):\n    return a + b\n')
    with (repo/'tests/test_adder.py').open('a') as f:f.write('\ndef test_positive():\n    assert add(3, 2) == 5\n')
    git('add','.');git('commit','-qm','fix fixture');fix=git('rev-parse','HEAD')
    task=dict(task_id='synthetic-addition',repo='synthetic/adder',checkout=str(repo),base_commit=base,fix_commit=fix,
              split='train',defect_type='boundary_defaults',problem_statement='add(3,2) returns 1; it should return 5.',
              image=image,production_paths=['src/'],test_paths=['tests/'],regression_tests=['tests'],python_path='/workspace/src',
              source_kind='synthetic_smoke',source_review=dict(status='approved',reviewer='fixture-author',evidence=['synthetic smoke, not benchmark data']))
    write(root/'spec.json',{'tasks':[task]});build(cfg,root/'spec.json');task=read(Path(cfg['workspace'])/'tasks/catalog.json')['tasks'][0]
    certificate=certify(task,cfg);folder=root/'teacher/trajectory'
    r=run(task,cfg,ScriptedBackend(),ByteTokenizer(),42,folder,phase='teacher')
    if not r['resolved'] or r['reward']!=1:raise AssertionError('Smoke patch did not pass fresh hidden grading')
    packet=audit(root/'teacher',cfg)
    write(root/'smoke-summary.json',dict(passed=True,certification=certificate,rollout_turns=len(r['turns']),reward=r['reward'],
              review_packet=packet['review_packet'],gpu_used=False,teacher_api_calls=0,scope='Real Docker + real mini-SWE-agent + scripted response fixture; no model quality measurement'))
    return read(root/'smoke-summary.json')
