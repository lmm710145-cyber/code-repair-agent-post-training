"""Bounded command runner executed inside each task container."""
RUNNER = r'''
import json,os,selectors,signal,subprocess,sys,time
cfg=json.loads(sys.argv[1])
p=subprocess.Popen(['bash','-lc',cfg['command']],stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
sel=selectors.DefaultSelector()
for name,pipe in [('stdout',p.stdout),('stderr',p.stderr)]:sel.register(pipe,selectors.EVENT_READ,name)
buf={'stdout':bytearray(),'stderr':bytearray()};total=0;reason=None;start=time.monotonic()
try:
    while sel.get_map():
        if time.monotonic()-start>=cfg['timeout']:reason='timeout';break
        for key,_ in sel.select(.1):
            chunk=os.read(key.fileobj.fileno(),65536)
            if not chunk:sel.unregister(key.fileobj);continue
            keep=min(len(chunk),cfg['limit']-total)
            buf[key.data].extend(chunk[:keep]);total+=keep
            if len(chunk)>keep:reason='raw_output_limit';break
        if reason:break
finally:
    if reason:
        try:os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError:pass
    try:code=p.wait(timeout=max(.01,cfg['timeout']-(time.monotonic()-start)))
    except subprocess.TimeoutExpired:
        reason='timeout'
        try:os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        code=p.wait()
    sel.close()
print(json.dumps({'stdout':buf['stdout'].decode('utf-8','replace'),'stderr':buf['stderr'].decode('utf-8','replace'),'returncode':125 if reason=='raw_output_limit' else 124 if reason=='timeout' else code,'resource_reason':reason,'captured_bytes':total,'limit_bytes':cfg['limit']}))
'''
