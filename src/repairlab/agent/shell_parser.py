"""Strict versioned action grammar and evidence-based trace checks."""
import collections, hashlib, re, json

class ActionFormatError(ValueError): pass

def parse_action(content):
    # Opening fences may immediately follow prose. This does not make a
    # uniquely delimited command ambiguous and was supported by the old wire
    # grammar. Every additional/malformed fence is still accounted for.
    blocks=[];cursor=0
    marker=re.compile(r'`{3,}|~{3,}')
    allowed={'mswea_bash_command','python','json','text','diff'}
    while True:
        opening=marker.search(content,cursor)
        if opening is None:break
        newline=content.find('\n',opening.end())
        if newline<0:raise ActionFormatError('Unmatched fence without a labelled block')
        raw_label=content[opening.end():newline].rstrip('\r')
        label=raw_label.strip()
        if label not in allowed:raise ActionFormatError('Unexpected executable or misspelled fence: '+label[:80])
        if label=='mswea_bash_command' and raw_label!=label:raise ActionFormatError('Use the exact literal action fence label')
        fence=opening.group();closing=re.compile(r'(?m)^[ \t]*'+re.escape(fence[0])+'{'+str(len(fence))+r',}').search(content,newline+1)
        if closing is None:raise ActionFormatError('Unclosed fenced block')
        body=content[newline+1:closing.start()]
        # A malformed extra opening fence inside a block must not be hidden.
        if re.search(r'(?m)^[ \t]*(?:`{3,}|~{3,})',body):raise ActionFormatError('Nested fenced block')
        endline=content.find('\n',closing.end())
        suffix=content[closing.end():endline if endline>=0 else len(content)].strip()
        if suffix in allowed or re.fullmatch(r'msw\w*[_-]bash[_-]command',suffix):raise ActionFormatError('Closing fence contains an extra action label')
        blocks.append((label,body.strip()));cursor=closing.end()
    actions=[body for label,body in blocks if label=='mswea_bash_command']
    if len(actions)!=1 or not actions[0]:raise ActionFormatError('Exactly one nonempty mswea_bash_command block required')
    return {'command':actions[0]}
