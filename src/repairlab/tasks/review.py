"""Record an independent human review threshold without pretending to perform it."""
import math
from ..io import read,write,digest
from ..config import workspace
from .schema import catalog


def freeze_tasks(cfg,review_path,fraction=.1):
    tasks=catalog(cfg);ids={t['task_id'] for t in tasks};reviews=read(review_path)['reviews'];seen=set()
    for row in reviews:
        if row['task_id'] not in ids or row['task_id'] in seen:raise ValueError('Invalid or duplicate reviewed task')
        if row.get('reviewer_kind')!='human' or not row.get('reviewer') or not row.get('notes') or row.get('decision')!='pass':
            raise ValueError('Independent human pass record is required')
        seen.add(row['task_id'])
    if len(seen)<math.ceil(len(tasks)*fraction):raise ValueError('Human review coverage below threshold')
    certs={}
    for task in tasks:
        cert=read(workspace(cfg,'tasks/private',task['task_id'],'certification.json'))
        if not cert['certified'] or cert['task_digest']!=digest(task):raise ValueError('Missing/current certification required')
        certs[task['task_id']]=digest(cert)
    result={'catalog_digest':digest(tasks),'certifications':certs,'human_review_digest':digest(reviews),'reviewed_tasks':len(seen),'total_tasks':len(tasks),'frozen':True}
    write(workspace(cfg,'tasks/freeze.json'),result);return result
