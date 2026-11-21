"""Parse only the Action section; never scan Thought for executable commands."""
import re
from .shell_parser import ActionFormatError, parse_action as parse_shell_action

FORMAT_HINT = 'Return Thought: with a nonempty concise explanation, then Action: with exactly one complete mswea_bash_command block and no trailing text.'

def parse_response(content):
    if not isinstance(content, str):
        raise ActionFormatError('Response must be text')
    text=content.strip()
    if '<think>' in text or '</think>' in text:
        raise ActionFormatError('Use visible Thought, not native think tags')
    # Ignore heading-looking text inside a fenced example in Thought or Action.
    headings=[];offset=0;fence=None
    for line in text.splitlines(keepends=True):
        bare=line.rstrip('\r\n')
        marker=re.match(r'^\s*(`{3,}|~{3,})',bare)
        if fence:
            if re.fullmatch(r'\s*'+re.escape(fence[0])+'{'+str(len(fence))+r',}\s*',bare):fence=None
        elif marker:
            fence=marker.group(1)
        elif re.match(r'^(Thought|Action):',bare):
            headings.append((bare.split(':',1)[0],offset,bare))
        offset+=len(line)
    if len(headings)!=2 or [h[0] for h in headings]!=['Thought','Action'] or headings[0][1]!=0:
        raise ActionFormatError('Exactly one Thought section followed by one Action section is required')
    if headings[1][2].strip()!='Action:':
        raise ActionFormatError('Place the action block on the line after Action:')
    split=headings[1][1];thought=text[len('Thought:'):split].strip();body=text[split+len('Action:'):].strip()
    if not thought:
        raise ActionFormatError('Thought must not be empty')
    if not re.fullmatch(r'```mswea_bash_command\r?\n[\s\S]+\r?\n```',body):
        raise ActionFormatError('Action must contain only one complete mswea_bash_command block')
    action=parse_shell_action(body)
    return {'thought':thought,'command':action['command']}

def execution_action(content):
    parsed=parse_response(content)
    return {'command':parsed['command'],'response':content}
