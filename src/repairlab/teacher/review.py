"""Automated triage is not semantic or human review. Approvals bind to exact bytes."""
import re
from pathlib import Path
from ..io import read,write,sha,jsonl,write_jsonl
from ..grpo.reward import behavior


def audit(root,cfg):
    rows=[]
    for path in sorted(Path(root).rglob('result.json')):
        r=read(path)
        if r.get('phase')!='teacher':continue
        events=behavior(r,cfg)['events'];event_turns={e['turn'] for e in events}
        for t in r['turns']:
            flags=[]
            if not r['resolved']:flags.append('final_patch_failed')
            if t['turn_id'] in event_turns:flags.append('behavior_event')
            if not t.get('execution'):flags.append('action_not_executed')
            elif t['execution']['returncode']!=0:flags.append('failed_command_requires_review')
            if re.search(r'\b(pass(?:ed)?|verified|fixed|ready|no regressions?)\b',t.get('thought',''),re.I):flags.append('verification_claim_requires_evidence_review')
            rows.append(dict(trajectory=str(path.resolve()),trajectory_sha256=sha(path),task_id=r['task_id'],turn=t['turn_id'],
                             flags=flags,decision='pending',reviewer='',reviewer_kind='',rationale=''))
    output=Path(root)/'action-review.jsonl';write_jsonl(output,rows)
    return {'review_packet':str(output),'actions':len(rows),'automatic_approval':False}


def validate_reviews(path):
    rows=jsonl(path);seen=set()
    for row in rows:
        key=(row['trajectory'],row['turn'])
        if key in seen:raise ValueError('Duplicate review decision')
        seen.add(key)
        if row['decision'] not in ('keep','reject') or not row['reviewer'] or not row['rationale']:
            raise ValueError('Every action needs an attributed, reasoned decision')
        if row['reviewer_kind'] not in ('human','assistant'):raise ValueError('Invalid reviewer_kind')
        if sha(row['trajectory'])!=row['trajectory_sha256']:raise ValueError('Reviewed trajectory changed')
    return rows
