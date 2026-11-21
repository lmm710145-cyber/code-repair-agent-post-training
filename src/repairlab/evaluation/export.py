"""Prediction-file interface only; SWE-bench harness/dataset integration is external."""
from pathlib import Path
from ..io import read,write_jsonl


def export_predictions(trajectory_root,mapping_path,output,model_name):
    mapping=read(mapping_path);rows=[];seen=set()
    for path in sorted(Path(trajectory_root).rglob('result.json')):
        r=read(path);iid=mapping.get(r['task_id'])
        if not iid:raise ValueError('Missing verified SWE-bench instance mapping for '+r['task_id'])
        if iid in seen:raise ValueError('Choose exactly one attempt per instance before export')
        seen.add(iid);rows.append({'instance_id':iid,'model_name_or_path':model_name,'model_patch':r['patch']})
    if not rows:raise ValueError('No predictions')
    write_jsonl(output,rows);return {'predictions':len(rows),'output':str(output),'official_evaluation_executed':False}
