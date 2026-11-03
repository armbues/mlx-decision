# Copyright H2O.ai (the H2O-Lightning authors).
#
# Copied from H2O-Lightning-4B (https://huggingface.co/h2oai/h2o-lightning-4b,
# revision 839c5c9523764f0b68c2785944f5541ee66a92b8, tag v1.4.2), file
# selector_v2.py, under the Apache License 2.0. See
# LICENSES/h2o-lightning-Apache-2.0.txt.
#
# Unchanged below this header, so it can be diffed against the release; the
# shim calls select_state(raw_state, instructions, criteria, budget, encode)
# for a record too long for a question's prompt (models/h2o/fit.py).
class EvidenceSelectionUnsupported(ValueError):
    """Relevant evidence cannot be represented within the configured token budget."""

'Query-aware state selection for lightning_31b_shim (structure-aware BM25 with exact-match bonuses).'
import collections
import json
import math
import re
UNIT_TOKENS = 400
GAP_TOKENS = 40
OPTION_WEIGHT = 0.5
PRF_TERMS = 30
PRF_WEIGHT = 0.3
PRF_UNITS = 4
EXACT_BONUS = 2.0
PACK_TOKENS = 150
SAMPLE_CHARS = 16000
PHASE_A = 0.75
USE_STOP = True
_SIZE_DIV = 6
NEIGHBOURS = True
MARK = '\n[…]\n'
_WORD = re.compile('[^\\W_]+', re.U)
_CJK = re.compile('[\u3040-ヿ㐀-䶿一-鿿가-\ud7af\u0e00-\u0e7f豈-\ufaff]')
_EXACT = re.compile('\\b(?:[A-Za-z]{1,6}-?\\d[\\w-]*|\\d{4}-\\d{2}-\\d{2}|\\d+(?:[.,]\\d+)*)\\b')
_SENT = re.compile('(?<=[.!?。！？])\\s+(?=[A-Z0-9\\"\'(\\[\u3040-鿿])')
_HEADING = re.compile("^\\s*(#{1,6}\\s+\\S.*|={2,}.*={2,}|[A-Z][A-Za-z0-9 ,&()/'-]{0,60}:|(?:section|part|chapter|article)\\s+\\S+.*|\\d+(?:\\.\\d+)*[.)]\\s+[A-Z].{0,80})\\s*$", re.I)
_STOP = frozenset('a an the of to in on for and or is are be was were by with as at from that this it its what which should would will can could do does did how why when who whom whose'.split())

def _terms(text):
    out = []
    for w in _WORD.findall(text.lower()):
        if _CJK.search(w):
            chars = [c for c in w]
            run = []
            for c in chars + [' ']:
                if _CJK.match(c):
                    run.append(c)
                    continue
                if run:
                    out.extend((run[i] + run[i + 1] for i in range(len(run) - 1))) if len(run) > 1 else out.append(run[0])
                    run = []
                if c != ' ':
                    out.append(c)
        elif not USE_STOP or w not in _STOP:
            out.append(w)
    return out

class _Unit:
    __slots__ = ('text', 'head', 'terms', 'ntok', 'a', 'b', 'src')

    def __init__(self, text, head, a=None, b=None, src=None):
        self.text, self.head, self.terms, self.ntok, self.a, self.b, self.src = (text, head, None, 0, a, b, src)
_SPLITTERS = (re.compile('\\n[ \\t]*\\n'), re.compile('\\n'), _SENT)

def _spans(text, a, b, size, ntok, level=0):
    if b - a <= 0:
        return []
    if ntok(text[a:b]) <= size or level >= len(_SPLITTERS):
        if level >= len(_SPLITTERS) and ntok(text[a:b]) > size:
            step = max(64, size * 3)
            return [(i, min(i + step, b)) for i in range(a, b, step)]
        return [(a, b)]
    cuts = [m.end() for m in _SPLITTERS[level].finditer(text, a, b) if a < m.end() < b]
    if not cuts:
        return _spans(text, a, b, size, ntok, level + 1)
    pieces, prev = ([], a)
    for c in cuts + [b]:
        pieces.append((prev, c))
        prev = c
    packed, cur = ([], None)
    for pa, pb in pieces:
        if cur and ntok(text[cur[0]:pb]) <= min(size, PACK_TOKENS):
            cur = (cur[0], pb)
        else:
            if cur:
                packed.append(cur)
            cur = (pa, pb)
    packed.append(cur)
    out = []
    for pa, pb in packed:
        out.extend(_spans(text, pa, pb, size, ntok, level + 1) if ntok(text[pa:pb]) > size else [(pa, pb)])
    return out

def _text_units(text, ntok, head='', size=UNIT_TOKENS):
    units, cur_head = ([], head)
    for a, b in _spans(text, 0, len(text), size, ntok):
        piece = text[a:b]
        if not piece.strip():
            continue
        first = piece.strip().split('\n', 1)[0]
        if len(first) < 120 and _HEADING.match(first):  # length first: the pattern backtracks cubically on long lines
            cur_head = (head + ' > ' if head else '') + first.strip()
        units.append(_Unit(piece, cur_head, a, b, text))
    return units

def _json_units(obj, ntok, path='', size=UNIT_TOKENS):

    def render(v):
        return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, separators=(',', ':'))
    units = []
    if isinstance(obj, dict):
        items = [(f'{path}.{k}' if path else str(k), v) for k, v in obj.items()]
    elif isinstance(obj, list):
        items = [(f'{path}[{i}]', v) for i, v in enumerate(obj)]
    else:
        return _text_units(render(obj), ntok, path, size)
    group = []
    for p, v in items:
        line = f'{p}: {render(v)}'
        if ntok(line) <= size:
            if group and ntok('\n'.join(group + [line])) > size:
                units.append(_Unit('\n'.join(group), path))
                group = []
            group.append(line)
            continue
        if group:
            units.append(_Unit('\n'.join(group), path))
            group = []
        if isinstance(v, (dict, list)):
            units.extend(_json_units(v, ntok, p, size))
        else:
            units.extend(_text_units(render(v), ntok, p, size))
    if group:
        units.append(_Unit('\n'.join(group), path))
    return units

def _bm25(units, qw, k1=1.5, b=0.75):
    N = len(units)
    avg = sum((len(u.terms) for u in units)) / max(N, 1) or 1.0
    df = collections.Counter((t for u in units for t in set(u.terms)))
    idf = {t: math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5)) for t in qw if df.get(t)}
    out = []
    for u in units:
        tf = collections.Counter(u.terms)
        s, L = (0.0, len(u.terms))
        for t, w in qw.items():
            f = tf.get(t)
            if f:
                s += w * idf[t] * f * (k1 + 1) / (f + k1 * (1 - b + b * L / avg))
        out.append(s)
    return (out, df)

def _query(instructions, options):
    qw = collections.Counter()
    for t in _terms(instructions):
        qw[t] = 1.0
    if isinstance(options, dict):
        texts = [f'{k} {v}' for k, v in options.items()]
    else:
        texts = [str(o) for o in options or []]
    for t in _terms(' '.join(texts)):
        qw[t] = max(qw[t], OPTION_WEIGHT)
    exact = {m.group(0).lower() for m in _EXACT.finditer(instructions)}
    return (dict(qw), exact)

def select_state(raw_state, instructions, options, evidence_budget, encode, decode=None):
    text = raw_state if isinstance(raw_state, str) else None
    if text is not None and len(text.encode('utf-8', 'ignore')) <= evidence_budget:
        return text
    full = text if text is not None else json.dumps(raw_state, ensure_ascii=False, separators=(',', ':'))
    if not evidence_budget or evidence_budget <= 0:
        return full
    nbytes = len(full.encode('utf-8', 'ignore'))
    sample = full[:SAMPLE_CHARS]
    ratio = len(encode(sample)) / max(1, len(sample.encode('utf-8', 'ignore')))
    if nbytes * ratio <= 1.25 * evidence_budget and len(encode(full)) <= evidence_budget:
        return full

    def ntok(s):
        return int(len(s.encode('utf-8', 'ignore')) * ratio * 1.05) + 1
    size = max(48, min(UNIT_TOKENS, evidence_budget // _SIZE_DIV))
    units = _text_units(text, ntok, size=size) if text is not None else _json_units(raw_state, ntok, size=size)
    if len(units) <= 2:
        return _budget_unsupported_or_full(full, evidence_budget, encode)
    for u in units:
        u.terms = _terms(u.text)
        u.ntok = ntok(u.text) + 1
    n = len(units)
    qw, exact = _query(instructions, options)
    s1, df = _bm25(units, qw)
    for i, u in enumerate(units):
        low = u.text.lower()
        s1[i] += EXACT_BONUS * sum((1 for e in exact if e in low))
    top = sorted(range(n), key=lambda i: (-s1[i], i))[:PRF_UNITS]
    cand = collections.Counter()
    for i in top:
        if s1[i] <= 0:
            continue
        for t in set(units[i].terms):
            if t not in qw:
                cand[t] = math.log(1 + n / (df[t] + 0.5))
    qw2 = dict(qw)
    for t, _ in sorted(cand.items(), key=lambda kv: (-kv[1], kv[0]))[:PRF_TERMS]:
        qw2[t] = PRF_WEIGHT
    s2, _ = _bm25(units, qw2)
    score = [s2[i] + EXACT_BONUS * sum((1 for e in exact if e in units[i].text.lower())) for i in range(n)]
    if not any(value > 0 for value in score):
        raise EvidenceSelectionUnsupported("No query-relevant evidence can be selected within the evidence budget")
    sep = ntok(MARK) + 2
    keep = {0, n - 1}

    def cost(sel):
        runs, used = (_runs(sel), 0)
        for a, b in runs:
            used += sum((units[i].ntok for i in range(a, b + 1)))
            h = units[a].head
            if h and (not units[a].text.startswith(h)):
                used += ntok(h) + 1
        return used + sep * (len(runs) - 1)
    if cost(keep) > evidence_budget:
        keep = {n - 1} if cost({n - 1}) <= evidence_budget else set()
        if not keep:
            return _budget_unsupported_or_full(full, evidence_budget, encode)
    order = [i for i in sorted(range(n), key=lambda i: (-score[i], i)) if score[i] > 0]
    parent = []
    for u in units:
        if u.src is not None and u.a is not None:
            ls = u.src.rfind('\n', 0, u.a) + 1
            le = u.src.find('\n', u.a)
            le = len(u.src) if le < 0 else le
            parent.append((id(u.src), ls) if u.b <= le + 1 else (id(u.src), u.a, 'own'))
        else:
            parent.append(('unit', id(u)))
    members = collections.defaultdict(list)
    for i, p in enumerate(parent):
        members[p].append(i)
    min_unit = min((u.ntok for u in units))

    def fill(limit):
        nonlocal keep
        used = cost(keep)
        for i in order:
            if limit - used < min_unit:
                break
            if i in keep:
                continue
            for cand in (set(members[parent[i]]) - keep, {i}):
                if not cand or (len(cand) > 1 and sum((units[j].ntok for j in cand)) > max(size, evidence_budget // 3)):
                    continue
                lower = sum((units[j].ntok for j in cand))
                if used + lower > limit:
                    continue
                new_cost = cost(keep | cand)
                if new_cost <= limit:
                    keep, used = (keep | cand, new_cost)
                    break
    fill(int(evidence_budget * PHASE_A))
    for i in [i for i in order if i in keep and NEIGHBOURS]:
        for j in sorted((i - 1, i + 1), key=lambda j: -score[j] if 0 <= j < n else 0):
            if 0 <= j < n and j not in keep and (cost(keep | {j}) <= evidence_budget):
                keep = keep | {j}
    runs = _runs(keep)
    for (a1, b1), (a2, b2) in zip(runs, runs[1:]):
        gap = set(range(b1 + 1, a2))
        if gap and (len(gap) == 1 or sum((units[j].ntok for j in gap)) <= GAP_TOKENS) and (cost(keep | gap) <= evidence_budget):
            keep = keep | gap
    fill(evidence_budget)
    out = []
    for a1, b1 in _runs(keep):
        segs, cur, last_head = ([], None, None)
        for i in range(a1, b1 + 1):
            u = units[i]
            if u.head and u.head != last_head and (not u.text.lstrip().startswith(u.head)):
                if cur:
                    segs.append(cur)
                    cur = None
                segs.append(u.head if u.src is not text else u.head)
            last_head = u.head
            if cur and u.src is not None and (cur[0] is u.src) and (cur[2] == u.a):
                cur = (cur[0], cur[1], u.b)
            else:
                if cur:
                    segs.append(cur)
                cur = (u.src, u.a, u.b) if u.src is not None else (None, u.text, None)
        if cur:
            segs.append(cur)
        body = '\n'.join((x if isinstance(x, str) else x[0][x[1]:x[2]].strip('\n') if x[0] is not None else x[1] for x in segs))
        out.append(body)
    result = MARK.join(out)
    while len(encode(result)) > evidence_budget:
        removable = [i for i in keep if i not in (0, n - 1)]
        if not removable:
            return _budget_unsupported_or_full(full, evidence_budget, encode)
        keep = keep - {min(removable, key=lambda i: (score[i], -i))}
        result = _assemble(units, keep, text)
    return result

def _assemble(units, keep, text):
    out = []
    for a1, b1 in _runs(keep):
        segs, cur, last_head = ([], None, None)
        for i in range(a1, b1 + 1):
            u = units[i]
            if u.head and u.head != last_head and (not u.text.lstrip().startswith(u.head)):
                if cur:
                    segs.append(cur)
                    cur = None
                segs.append(u.head)
            last_head = u.head
            if cur and u.src is not None and (cur[0] is u.src) and (cur[2] == u.a):
                cur = (cur[0], cur[1], u.b)
            else:
                if cur:
                    segs.append(cur)
                cur = (u.src, u.a, u.b) if u.src is not None else (None, u.text, None)
        if cur:
            segs.append(cur)
        out.append('\n'.join((x if isinstance(x, str) else x[0][x[1]:x[2]].strip('\n') if x[0] is not None else x[1] for x in segs)))
    return MARK.join(out)


def _runs(sel):
    runs, prev = ([], None)
    for i in sorted(sel):
        if prev is not None and i == prev + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
        prev = i
    return [tuple(r) for r in runs]

def _budget_unsupported_or_full(full, evidence_budget, encode):
    """Verify genuine token-budget overflow before reporting a capacity refusal."""
    input_tokens = len(encode(full))
    if input_tokens <= evidence_budget:
        return full
    raise EvidenceSelectionUnsupported(
        f"too many tokens to preserve the full input within the configured evidence budget "
        f"({input_tokens} > {evidence_budget}); relevant whole-unit evidence cannot be represented; "
        "boundary-only selection is test-only")
