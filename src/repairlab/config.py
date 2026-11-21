import os
import re
from pathlib import Path
from .io import read


def expand(value):
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v) for v in value]
    if isinstance(value, str):
        def replace(m):
            name, default = m.group(1), m.group(2)
            if name not in os.environ and default is None:
                raise ValueError(f'Missing environment variable: {name}')
            return os.environ.get(name, default)
        return re.sub(r'\$\{([A-Z_][A-Z_0-9]*)(?::-([^}]*))?\}', replace, value)
    return value


def load(path):
    path = Path(path).resolve()
    cfg = expand(read(path))
    cfg['_config_file'] = str(path)
    cfg['workspace'] = str((path.parent / cfg.get('workspace', '../workspace')).resolve())
    b = cfg['budget']
    if b['input_tokens'] + b['output_tokens'] != b['context_tokens']:
        raise ValueError('Context must equal input plus output budget')
    if not 64 <= b['feedback_tokens'] < b['input_tokens']:
        raise ValueError('Invalid tool feedback budget')
    if any(b[k] <= 0 for k in ('input_tokens','output_tokens','max_turns','episode_output_tokens','command_seconds','episode_seconds')):
        raise ValueError('Budgets must be positive')
    if cfg['sft']['epochs'] <= 0 or any(e <= 0 or e > cfg['sft']['epochs'] for e in cfg['sft']['milestone_epochs']):
        raise ValueError('SFT milestones must be within the requested epochs')
    if cfg['grpo']['updates'] <= 0 or cfg['grpo']['updates'] > cfg['grpo']['max_sampling_rounds']:
        raise ValueError('Invalid GRPO update/round budget')
    if not 0 <= cfg['sft']['holdout_fraction'] < 1:
        raise ValueError('Invalid holdout fraction')
    if cfg['grpo']['group_size'] < 2:
        raise ValueError('GRPO group_size must be >= 2')
    return cfg


def workspace(cfg, *parts):
    return Path(cfg['workspace']).joinpath(*parts)
