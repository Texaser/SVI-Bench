"""Shared utilities for parsing OpenAI judge output.

Kept tiny on purpose: both analyze_results.py and merge_metrics.py
need identical verdict parsing, so it lives here.
"""

import json
from typing import Optional, Dict


def parse_judge_verdict(content: str) -> Optional[Dict[str, str]]:
    """Parse judge output. Returns {'verdict': 'Right'|'Wrong', 'reason': str}
    on success, or None if the verdict is ambiguous / unparseable.

    Parsing order:
      1. Strict JSON with a 'verdict' field that is exactly 'Right' or 'Wrong'.
      2. Leading-token fallback: the first non-whitespace word of the response
         is the verdict (case-insensitive). Reason is the rest.
      3. Otherwise None — caller should treat as a parse failure, NOT as Wrong.
    """
    if not content:
        return None
    content = content.strip()

    # 1) Strict JSON
    try:
        parsed = json.loads(content)
        v = parsed.get('verdict')
        if v in ('Right', 'Wrong'):
            return {'verdict': v, 'reason': parsed.get('reason', '')}
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass

    # 2) Leading-token fallback (split on any whitespace, not just space)
    parts = content.split(None, 1)
    head = parts[0] if parts else ''
    tail = parts[1] if len(parts) > 1 else ''
    head_clean = head.strip(' \t\n\r.,:;"\'`').lower()
    if head_clean == 'right':
        return {'verdict': 'Right', 'reason': tail.strip()}
    if head_clean == 'wrong':
        return {'verdict': 'Wrong', 'reason': tail.strip()}

    # 3) Ambiguous → caller decides
    return None
