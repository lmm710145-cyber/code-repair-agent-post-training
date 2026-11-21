"""v2h candidate observer: preserve every occurrence of repeated pytest node IDs."""
import json, os, pathlib, time, uuid, subprocess, hashlib, sys
_path=None
_state={}
_last_write=0
_occurrences={}
_active={}
def _revision():
    try:
        paths=json.loads(os.environ['REPAIRLAB_PRODUCTION_PATHSPEC'])
        r=subprocess.run(['git','diff','--binary','--',*paths],cwd='/workspace',capture_output=True,text=True,timeout=10,check=True)
        return hashlib.sha256(r.stdout.encode()).hexdigest()
    except Exception:return None
def _save(force=False):
    global _last_write
    if _path and (force or time.monotonic()-_last_write>=0.25):
        _last_write=time.monotonic()
        temp=_path.with_suffix('.tmp');temp.write_text(json.dumps(_state,ensure_ascii=False));temp.replace(_path)
def pytest_sessionstart(session):
    global _path,_state,_occurrences,_active
    root=os.environ.get('REPAIRLAB_TEST_RECORD_DIR')
    if not root:return
    pathlib.Path(root).mkdir(parents=True,exist_ok=True)
    _occurrences={}
    _active={}
    _path=pathlib.Path(root)/(uuid.uuid4().hex+'.json')
    _state={'session_id':_path.stem,'action_id':os.environ.get('REPAIRLAB_ACTION_ID'),'started':time.time(),'argv':list(session.config.invocation_params.args),'collected':0,'executed':[],'results':{},'collection_errors':[],'collection_skips':[],'complete':False,'source_patch_sha256_start':_revision()}
    _save(force=True)
def pytest_collection_finish(session):
    if _path:_state['collected']=len(session.items);_save(force=True)
def pytest_collectreport(report):
    if not _path:return
    if report.failed:_state['collection_errors'].append({'nodeid':report.nodeid,'message':str(report.longrepr)[-2500:]});_save()
    elif report.skipped:_state['collection_skips'].append({'nodeid':report.nodeid,'message':str(report.longrepr)[-2500:]});_save()
def pytest_runtest_logstart(nodeid, location):
    if not _path:return
    occurrence=_occurrences.get(nodeid,0)+1
    _occurrences[nodeid]=occurrence
    key=nodeid if occurrence==1 else nodeid+"::@repairlab_occurrence="+str(occurrence)
    _active[nodeid]=key
    _state['results'][key]={'nodeid':nodeid,'occurrence':occurrence,'phases':{}}
    _save()
def pytest_runtest_logreport(report):
    if not _path:return
    if report.nodeid not in _active:pytest_runtest_logstart(report.nodeid,None)
    key=_active[report.nodeid]
    result=_state['results'][key]
    result['phases'][report.when]={'outcome':report.outcome,'wasxfail':getattr(report,'wasxfail',None)}
    if report.failed:result['failure']=str(report.longrepr)[-2500:]
    if report.when=='call' and key not in _state['executed']:_state['executed'].append(key)
    _save()
def summarize_outcomes(state):
    counts=dict(passed=0,failed=0,errors=len(state['collection_errors']),skipped=len(state.get('collection_skips',[])),xfailed=0,xpassed=0)
    for item in state['results'].values():
        for phase, report in item['phases'].items():
            outcome=report['outcome']
            if report.get('wasxfail') is not None and outcome=='skipped':counts['xfailed']+=1
            elif report.get('wasxfail') is not None and outcome=='passed':counts['xpassed']+=1
            elif outcome=='failed':counts['failed' if phase=='call' else 'errors']+=1
            elif outcome=='skipped':counts['skipped']+=1
            elif phase=='call' and outcome=='passed':counts['passed']+=1
    return counts

def pytest_sessionfinish(session,exitstatus):
    if not _path:return
    counts=summarize_outcomes(_state)
    packages=[sys.modules.get(name) for name in os.environ.get('REPAIRLAB_IMPORT_PACKAGE','').split(',')]
    package=next((p for p in packages if str(getattr(p,'__file__','')).startswith('/workspace/')),None)
    _state.update(counts=counts,exitstatus=int(exitstatus),complete=True,finished=time.time(),source_patch_sha256_end=_revision(),task_import_file=getattr(package,'__file__',None));_save(force=True)
