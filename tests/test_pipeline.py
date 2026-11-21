import copy
import json
import tempfile
from pathlib import Path
import pytest
from repairlab.config import load,expand
from repairlab.agent.protocol import parse_response,ActionFormatError
from repairlab.agent.context import Context,ContextOverflow
from repairlab.demo import ByteTokenizer
from repairlab.grpo.reward import reward,advantages,behavior
from repairlab.grpo.curriculum import Curriculum
from repairlab.tasks.schema import validate_catalog
from repairlab.tasks.grader import junit
from repairlab.sft.data import hierarchical_indices
from repairlab.evaluation.metrics import summarize,compare

ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture
def cfg():return load(ROOT/'configs/pipeline.example.json')

def turn(i=0,command='cat src/a.py'):
    content=f'Thought:\nInspect the code.\nAction:\n```mswea_bash_command\n{command}\n```'
    return dict(turn_id=i,content=content,input_ids=[1],output_ids=[2],actions=[dict(command=command,response=content)],execution=dict(command=command,before_state='a',after_state='a',output_sha256='b',returncode=0,resource_reason=None))

def record(n=1):
    return dict(turns=[turn(i) for i in range(n)],infrastructure_valid=True,scope_clean=True,tampered=False,
                regression=False,grader={'resolved':True},resolved=True,termination='RepeatedFormatError',generated_tokens=10)

def test_action_only():
    text='Thought:\nExample:\n```python\nprint(1)\n```\nNow inspect.\nAction:\n```mswea_bash_command\npwd\n```'
    assert parse_response(text)['command']=='pwd'
    for bad in ['Action:\n```mswea_bash_command\npwd\n```',turn()['content']+'\nextra',turn()['content'].replace('Inspect the code.','')]:
        with pytest.raises(ActionFormatError):parse_response(bad)

def test_context_preserves_issue_and_latest(cfg):
    cfg['budget'].update(input_tokens=300,feedback_tokens=100)
    c=Context(ByteTokenizer(),cfg['budget']);messages=[{'role':'system','content':'system'},{'role':'user','content':'FULL ISSUE'}]
    for i in range(5):messages.extend([{'role':'assistant','content':f'action{i}'},{'role':'user','content':'x'*500}])
    original=copy.deepcopy(messages);visible,ids,stats=c.prepare(messages)
    assert messages==original and visible[1]['content']=='FULL ISSUE' and len(ids)<=300
    assert stats['dropped_interactions']>0 and visible[-2]['content']=='action4'
    assert len(ByteTokenizer().encode(visible[-1]['content']))<=100
    with pytest.raises(ContextOverflow):c.prepare([{'role':'system','content':'x'*500},{'role':'user','content':'issue'}])

def test_reward_termination_independent_and_cap(cfg):
    assert reward(record(),cfg)['reward']==1
    r=record(40);x=reward(r,cfg)
    assert x['reward']==.25 and x['behavior_cost']==.5 and x['termination_cost']==.25
    r=record();r['generated_tokens']=16384
    assert reward(r,cfg)['reward']==.75
    r['grader']['resolved']=False
    assert reward(r,cfg)['reward']==-.25
    r=record();r['termination']='TimeExceeded'
    assert reward(r,cfg)['reward']==1
    r['scope_clean']=False
    assert reward(r,cfg)['reward']==-1
    r['infrastructure_valid']=False
    assert reward(r,cfg)['reward'] is None

def test_actual_submit_marker(cfg):
    r=record();r['generated_tokens']=16384;r['termination']='Submitted'
    assert reward(r,cfg)['reward']==.75
    r['turns']=[turn(command='echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT')]
    assert reward(r,cfg)['reward']==1

def test_repeat_requires_same_state_and_output(cfg):
    assert behavior(record(2),cfg)['raw']==0
    assert behavior(record(4),cfg)['raw']==.04
    r=record(4);r['turns'][1]['execution']['after_state']='changed'
    assert behavior(r,cfg)['raw']==0
    r=record(3);r['turns'][2]['execution']['output_sha256']='new'
    assert behavior(r,cfg)['raw']==0

def test_truncation_not_double_charged(cfg):
    r=record();r['turns'][0].update(content=turn()['content'][:-4],format_error='unclosed',finish_reason='length',actions=[])
    assert behavior(r,cfg)['costs']=={'format':0,'repetition':0,'output_truncation':.01}

def test_group_advantages_and_same_reward():
    assert advantages([1]*16)==[0]*16
    a=advantages([1]+[0]*15)
    assert len(a)==16 and abs(sum(a))<1e-5 and a[0]>0
    with pytest.raises(ValueError):advantages([1,None])

def test_curriculum_reproducible_and_train_only(cfg):
    tasks=[dict(task_id=f'{r}-{i}',repo=r,split='train') for r in ['a','b'] for i in range(2)]
    c=Curriculum(tasks,cfg['grpo']['curriculum']);chosen=c.select(3,42,2,16)
    c.record(chosen[0],3,[{'infrastructure_valid':True,'resolved':False}]*16)
    restored=Curriculum(tasks,cfg['grpo']['curriculum'],c.state())
    assert c.select(8,42,2,16)==restored.select(8,42,2,16)
    with pytest.raises(ValueError):Curriculum([dict(tasks[0],split='dev')],cfg['grpo']['curriculum'])

def test_junit_empty_duplicate_error_and_skip():
    for text in ['<testsuite/>','<testsuite><testcase name="a"/><testcase name="a"/></testsuite>']:
        with pytest.raises(ValueError):junit(text)
    assert junit('<testsuite><testcase name="a"><skipped/></testcase></testsuite>')['::a']=='skipped'

def test_hierarchy_balances_types_not_long_trajectories():
    rows=[dict(defect_type='large',trajectory_id='long') for _ in range(100)]+[dict(defect_type='small',trajectory_id='short')]
    indices=hierarchical_indices(rows,1000,42)
    assert 400<sum(i==100 for i in indices)<600
    assert indices==hierarchical_indices(rows,1000,42)

def test_metrics_success_at_k_and_baseline():
    fail=record();fail['resolved']=False
    data={'a':[fail,record()],'b':[record()]};m=summarize(data,5)
    assert m['pass_at_1']==.5 and m['pass_at_k']==1 and m['attempts']==3
    assert compare(m,m)['delta']['pass_at_1']==0

def test_config_no_key_material(cfg):
    assert cfg['teacher']['enabled'] is False and cfg['grpo']['group_size']==16
    assert cfg['model']['path'] and 'api_key' not in cfg['teacher']

def test_catalog_repo_isolation():
    base=dict(task_id='a',repo='owner/repo',split='train',defect_type='api_contract',problem_statement='a defect',image='image',snapshot='task.tar',snapshot_sha256='hash',production_paths=['src/'],test_paths=['tests/'],source_review={'status':'approved'})
    assert validate_catalog([base])==[base]
    with pytest.raises(ValueError):validate_catalog([base,dict(base,task_id='b',split='test')])
    with pytest.raises(ValueError):validate_catalog([dict(base,production_paths=['../hidden'])])

def test_freeze_detects_changed_inputs(tmp_path):
    from repairlab.io import freeze,verify_run
    freeze(tmp_path,{'seed':1});verify_run(tmp_path)
    with pytest.raises(ValueError):freeze(tmp_path,{'seed':2})

def test_review_hash_and_attribution(tmp_path):
    from repairlab.io import write,sha,write_jsonl
    from repairlab.teacher.review import validate_reviews
    source=tmp_path/'result.json';write(source,{'turns':[]})
    review=tmp_path/'review.jsonl'
    row=dict(trajectory=str(source),trajectory_sha256=sha(source),turn=0,decision='keep',reviewer='tester',reviewer_kind='assistant',rationale='fixture review')
    write_jsonl(review,[row]);assert len(validate_reviews(review))==1
    write(source,{'turns':[1]})
    with pytest.raises(ValueError):validate_reviews(review)

def test_task_patch_split_build(tmp_path,cfg):
    import subprocess
    from repairlab.tasks.build import build
    from repairlab.io import write,read
    repo=tmp_path/'repo';repo.mkdir();(repo/'src').mkdir();(repo/'tests').mkdir()
    def git(*args):return subprocess.run(['git','-C',str(repo),*args],check=True,capture_output=True,text=True).stdout.strip()
    git('init','-q');git('config','user.name','Test');git('config','user.email','test@example.invalid')
    (repo/'src/a.py').write_text('value=1\n');(repo/'tests/test_a.py').write_text('assert True\n');git('add','.');git('commit','-qm','base');base=git('rev-parse','HEAD')
    (repo/'src/a.py').write_text('value=2\n');(repo/'tests/test_a.py').write_text('assert 2==2\n');git('add','.');git('commit','-qm','fix');fix=git('rev-parse','HEAD')
    cfg=copy.deepcopy(cfg);cfg['workspace']=str(tmp_path/'workspace')
    task=dict(task_id='fixture',repo='test/repo',checkout=str(repo),base_commit=base,fix_commit=fix,split='train',defect_type='api_contract',problem_statement='wrong value',image='fixture',production_paths=['src/'],test_paths=['tests/'],regression_tests=['tests/'],source_review={'status':'approved','reviewer':'test','evidence':['synthetic']})
    spec=tmp_path/'spec.json';write(spec,{'tasks':[task]});assert build(cfg,spec)['constructed']==1
    private=Path(cfg['workspace'])/'tasks/private/fixture'
    assert 'tests/test_a.py' not in (private/'gold.patch').read_text()
    assert 'src/a.py' not in (private/'tests.patch').read_text()

def test_sft_exact_teacher_context_and_no_private_reasoning(tmp_path,cfg,monkeypatch):
    from repairlab.sft import data
    from repairlab.io import write,write_jsonl,sha,jsonl
    tok=ByteTokenizer();monkeypatch.setattr(data,'tokenizer',lambda cfg:tok)
    cfg=copy.deepcopy(cfg);cfg['workspace']=str(tmp_path/'workspace');cfg['sft']['holdout_fraction']=0
    task=dict(task_id='fixture',repo='fixture/repo',split='train',defect_type='api_contract',problem_statement='wrong result',image='fixture',snapshot='snapshot.tar',snapshot_sha256='hash',production_paths=['src/'],test_paths=['tests/'],source_review={'status':'approved'})
    write(Path(cfg['workspace'])/'tasks/catalog.json',{'tasks':[task]})
    r=record();r.update(task_id='fixture',trajectory_id='fixture-traj',phase='teacher')
    t=r['turns'][0];t['messages']=[{'role':'system','content':'system'},{'role':'user','content':'issue'}]
    t['input_ids']=Context(tok,cfg['budget']).encode(t['messages']);t['output_ids']=tok.encode(t['content'])+[256];t['reasoning_content']='PRIVATE_ONLY'
    source=tmp_path/'result.json';write(source,r);reviews=tmp_path/'reviews.jsonl'
    row=dict(trajectory=str(source),trajectory_sha256=sha(source),turn=0,decision='keep',reviewer='test',reviewer_kind='human',rationale='synthetic test')
    write_jsonl(reviews,[row]);result=data.prepare(cfg,reviews)
    assert result['train_samples']==1 and result['private_reasoning_included'] is False
    out=Path(cfg['workspace'])/'datasets/sft/train.jsonl';assert 'PRIVATE_ONLY' not in out.read_text()
    assert jsonl(out)[0]['input_ids']==t['input_ids']
    t['input_ids'].append(9);write(source,r);row['trajectory_sha256']=sha(source);write_jsonl(reviews,[row])
    with pytest.raises(ValueError,match='tokenizer mismatch'):data.prepare(cfg,reviews)
