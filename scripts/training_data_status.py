#!/usr/bin/env python3
"""학습 데이터 수집 현황표(HTML)를 만든다 — 제품코드별 학습·검증·검수 대기·현장 수집 누적 수와 60장 상한 대비 부족분, ★ 대표 목록 (2026-10-12).

  python scripts/training_data_status.py [--out ~/jhw/SL_Inspection_Automation/docs/training_data_status.html]

단일 HTML(검색·상태 필터·열 정렬은 페이지 안의 작은 스크립트). 브라우저로 열면 된다.
★ 대표 프레임은 라벨 윤곽선을 그린 썸네일(JPEG, 긴 변 --thumb px)을 base64 로 내장해 갤러리로 보여 준다 (클릭하면 확대). 300장 ≈ 5 MB.

기준: 학습 = train_current(train_class<N>, 검수 재지정 코드 반영), 검증 = valid_class<N>, 검수 대기 = field_new_<최신> 의 보류 프레임(exemplar_ref 제외),
현장 수집 누적 = SL_under_predict_auto 의 비제외 프레임(코드 대응표 반영). 제품코드 목록은 codes_none_20260929.yaml(none 제외 324, dropped CAP 4 는 뺀다).
"""
import argparse
import collections
import csv
import datetime as dt
import glob
import json
import os
import re
import sys

import base64

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_holes_review import OBJ_DIR  # noqa: E402
from codes_to_sort5 import major_map  # noqa: E402

H = os.path.expanduser('~')
CAP = 60
SORT_KO = {'b_cvr': 'B/CVR', 'h_cvr': 'H/CVR', 'hsg': 'HSG', 'scalp': 'SCALP'}


def load_ds(root):
    """(stem → code) with review_state code override; exemplar set."""
    st = json.load(open(os.path.join(root, 'review_state.json'), encoding='utf-8')) if os.path.exists(os.path.join(root, 'review_state.json')) else {}
    rows = list(csv.DictReader(open(os.path.join(root, 'manifest.csv'), encoding='utf-8')))
    codes = {r['stem']: (st.get(r['stem'], {}).get('code') or r['code']) for r in rows}
    ex = {s for s, v in st.items() if v.get('exemplar')}
    return rows, st, codes, ex


def thumb_b64(img_path, label_path, size, names=None):
    """라벨 폴리곤 윤곽선을 그린 썸네일 JPEG 의 data URI."""
    im = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if im is None:
        return ''
    h, w = im.shape[:2]
    if os.path.exists(label_path):
        for l in open(label_path, encoding='utf-8'):
            v = l.split()
            if len(v) < 7:
                continue
            pts = (np.array([float(x) for x in v[1:]], dtype=np.float32).reshape(-1, 2) * [w, h]).astype(np.int32)
            cv2.polylines(im, [pts], True, (60, 220, 60), 2, cv2.LINE_AA)
    sc = size / max(h, w)
    if sc < 1:
        im = cv2.resize(im, (round(w * sc), round(h * sc)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode('ascii') if ok else ''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=f'{H}/jhw/SL_Inspection_Automation/docs/training_data_status.html')
    ap.add_argument('--train', default=f'{H}/jhw/data/SL/yolo26_dataset/train_current')
    ap.add_argument('--valid', default=sorted(glob.glob(f'{H}/jhw/data/SL/yolo26_dataset/valid_class*'))[-1])
    ap.add_argument('--pending', default=sorted(d for d in glob.glob(f'{H}/jhw/data/SL/field_new_[0-9]*')
                                               if os.path.isdir(d) and re.fullmatch(r'field_new_\d{8}', os.path.basename(d)))[-1],
                    help='검수 대기 데이터셋 (기본 최신 field_new_<날짜> 폴더; _excess 사이드카·백업 tar 는 제외)')
    ap.add_argument('--auto', default=f'{H}/jhw/data/SL_under_predict_auto')
    ap.add_argument('--thumb', type=int, default=384, help='대표 썸네일 긴 변 px (0 = 이미지 없이 stem 만)')
    ap.add_argument('--codes', default=f'{H}/jhw/data/SL/yolo26_dataset/codes_20260921/codes_none_20260929.yaml')
    a = ap.parse_args()

    y = yaml.safe_load(open(a.codes, encoding='utf-8'))
    codes = [n for n in y['names'].values() if re.match(r'^[0-9A-Z]{12}$', n)]
    major = major_map()
    pname = {}
    for fn in os.listdir(OBJ_DIR):
        if fn.lower().endswith('.obj'):
            pname.setdefault(fn[:12], fn[13:-4])

    tr_rows, tr_st, tr_code, tr_ex = load_ds(os.path.realpath(a.train))
    train = collections.Counter(tr_code.values())
    exem = collections.Counter(tr_code[s] for s in tr_ex if s in tr_code)
    exem_stems = collections.defaultdict(list)
    for s in sorted(tr_ex):
        if s in tr_code:
            exem_stems[tr_code[s]].append(s)
    va_rows, va_st, va_code, _ = load_ds(a.valid)
    valid = collections.Counter(va_code.values())
    pd_rows, pd_st, pd_code, _ = load_ds(a.pending)
    pending = collections.Counter(pd_code[r['stem']] for r in pd_rows if r.get('reason') != 'exemplar_ref' and pd_st.get(r['stem'], {}).get('status', 'pending') != 'reject')
    sel = collections.defaultdict(lambda: [0, 0])   # code → [후보, 정제 통과]
    sp = os.path.join(a.pending, 'select_report.csv')
    if os.path.exists(sp):
        for r in csv.DictReader(open(sp, encoding='utf-8')):
            sel[r['code']][0] += 1
            sel[r['code']][1] += r['refine_keep'] == '1'
    alias = json.load(open(f'{H}/jhw/data/SL/autolabel/code_alias.json', encoding='utf-8')) if os.path.exists(f'{H}/jhw/data/SL/autolabel/code_alias.json') else {}
    au_rows, au_st, au_code, _ = load_ds(a.auto)
    auto = collections.Counter(alias.get(c, c) for s, c in au_code.items() if au_st.get(s, {}).get('status') != 'reject')
    auto_dates = (min(r['stem'][:8] for r in au_rows), max(r['stem'][:8] for r in au_rows))

    rows = []
    for c in codes:
        t = train.get(c, 0)
        rows.append(dict(code=c, sort=SORT_KO.get(major.get(c), 'CAP' if c in pname else '?'), name=pname.get(c, ''), train=t, ex=exem.get(c, 0), valid=valid.get(c, 0),
                         pending=pending.get(c, 0), cand=sel[c][0], refined=sel[c][1], auto=auto.get(c, 0), need=max(0, CAP - t),
                         after=min(CAP, t + pending.get(c, 0))))
    def status(r):
        if r['train'] >= CAP: return '충족'
        if r['train'] == 0: return '미수집' if r['auto'] == 0 else '수집만'
        return '부족'
    for r in rows: r['status'] = status(r)
    n = collections.Counter(r['status'] for r in rows)
    by_sort = collections.defaultdict(collections.Counter)
    for r in rows: by_sort[r['sort']][r['status']] += 1
    tag = lambda p: os.path.basename(os.path.realpath(p))
    today = dt.date.today().isoformat()
    esc = lambda x: str(x).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    def tbl(head, body_rows, cls='', tid=''):
        h = ''.join(f'<th>{esc(c)}</th>' for c in head)
        b = '\n'.join('<tr' + (f' class="{r[0]}"' if isinstance(r, tuple) else '') + '>' + ''.join(f'<td>{c}</td>' for c in (r[1] if isinstance(r, tuple) else r)) + '</tr>' for r in body_rows)
        return f'<table class="{cls}"{" id=" + chr(34) + tid + chr(34) if tid else ""}><thead><tr>{h}</tr></thead><tbody>\n{b}\n</tbody></table>'
    sum_rows = [
        ('제품코드 (CAP 4종 제외)', len(rows)), (f'학습 {CAP}장 충족', n['충족']), (f'학습 1~{CAP - 1}장 (부족)', n['부족']),
        ('학습 0장, 현장 수집은 있음', n['수집만']), ('학습 0장, 현장 수집도 0', n['미수집']),
        ('학습 프레임 합계 (none 제외 / none)', f'{sum(train[c] for c in codes):,} / {train.get("none", 0)}'),
        ('부족분 합계 (상한 − 학습)', f'{sum(r["need"] for r in rows):,}'),
        (f'검수 대기 프레임 ({tag(a.pending)})', f'{sum(pending.values()):,} ({len(pending)}코드)'),
        ('검수 반영 후 예상 충족 코드', sum(1 for r in rows if r['after'] >= CAP)),
        ('검증 프레임', f'{sum(valid[c] for c in codes):,} ({sum(1 for c in codes if valid.get(c))}코드) + none {valid.get("none", 0)}'),
        ('★ 대표가 있는 코드', f'{sum(1 for r in rows if r["ex"])} ({sum(r["ex"] for r in rows)}장)')]
    sort_rows = []
    for so in ('B/CVR', 'H/CVR', 'HSG', 'SCALP'):
        rs = [r for r in rows if r['sort'] == so]; c = by_sort[so]
        sort_rows.append([so, len(rs), c['충족'], c['부족'], c['수집만'], c['미수집'], f'{sum(r["train"] for r in rs):,}', f'{sum(r["pending"] for r in rs):,}'])
    STATUS_CLS = {'충족': 'ok', '부족': 'short', '수집만': 'coll', '미수집': 'none'}
    code_rows = [(STATUS_CLS[r['status']], [r['code'], r['sort'], esc(r['name']), r['status'], r['train'], r['need'], r['auto'], r['valid'], r['pending'],
                  f'{r["cand"]}/{r["refined"]}', r['after'], r['ex']]) for r in sorted(rows, key=lambda r: r['code'])]
    troot = os.path.realpath(a.train)
    tman = {r['stem']: r for r in tr_rows}
    def fig(stem):
        sp = tman.get(stem, {}).get('split', 'train')
        src = thumb_b64(f'{troot}/images/{sp}/{stem}.png', f'{troot}/labels/{sp}/{stem}.txt', a.thumb) if a.thumb else ''
        img = f'<img src="{src}" loading="lazy" alt="{esc(stem)}">' if src else ''
        return f'<figure>{img}<figcaption title="{esc(stem)}">{esc(stem.split("__")[-1])}</figcaption></figure>'
    gal = []
    for r in sorted(rows, key=lambda r: r['code']):
        if r['ex']:
            gal.append(f'<section class="exgrp" data-code="{r["code"]}"><h3>{r["code"]} <small>{r["sort"]} · {esc(r["name"])} · ★ {r["ex"]} · 학습 {r["train"]}</small></h3>'
                       f'<div class="gal">{"".join(fig(s) for s in exem_stems[r["code"]])}</div></section>')
    if exem_stems.get('none'):
        gal.append(f'<section class="exgrp" data-code="none"><h3>none <small>제품 아님(게이트 조각 등) · ★ {len(exem_stems["none"])}</small></h3>'
                   f'<div class="gal">{"".join(fig(s) for s in exem_stems["none"])}</div></section>')
    gallery = '\n'.join(gal)
    html = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>학습 데이터 수집 현황 {today}</title>
<style>
:root{{--bg:#fff;--fg:#1b1f23;--mute:#5f6b7a;--line:#d9dee5;--head:#f1f4f8;--ok:#dcf3e3;--short:#fff2cc;--coll:#e3ecfb;--none:#f8e0e0;--hl:#fff7a8}}
@media (prefers-color-scheme:dark){{:root{{--bg:#15181c;--fg:#e6e9ee;--mute:#9aa4b2;--line:#333a44;--head:#1f252d;--ok:#1f4a2e;--short:#4a3f12;--coll:#1e3350;--none:#4d2222;--hl:#5a5410}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans KR",sans-serif}}
main{{max-width:1400px;margin:0 auto}} h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:17px;margin:28px 0 8px}} p.meta{{color:var(--mute);margin:0 0 12px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:6px 0 10px}} th,td{{border:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top;white-space:nowrap}}
th{{background:var(--head);position:sticky;top:0;cursor:pointer;user-select:none}} th.s-asc::after{{content:" ▲"}} th.s-desc::after{{content:" ▼"}}
td:nth-child(n+5){{text-align:right}} table.kv td,table.kv th{{text-align:left}} table.ex td{{white-space:normal}} table.ex td:nth-child(n+5){{text-align:left}} table.ex td:nth-child(3){{text-align:right}}
tr.ok td:nth-child(4){{background:var(--ok)}} tr.short td:nth-child(4){{background:var(--short)}} tr.coll td:nth-child(4){{background:var(--coll)}} tr.none td:nth-child(4){{background:var(--none)}}
code{{font-size:12px;background:var(--head);padding:1px 4px;border-radius:3px}} .wrap{{overflow-x:auto}}
.exgrp{{margin:14px 0 6px}} .exgrp h3{{font-size:15px;margin:6px 0}} .exgrp h3 small{{color:var(--mute);font-weight:normal}} .gal{{display:flex;flex-wrap:wrap;gap:8px}}
figure{{margin:0;width:192px}} figure img{{width:192px;height:192px;object-fit:contain;background:#000;border:1px solid var(--line);border-radius:4px;cursor:zoom-in;display:block}}
figcaption{{font-size:11px;color:var(--mute);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
#lb{{position:fixed;inset:0;background:rgba(0,0,0,.85);display:flex;flex-direction:column;align-items:center;justify-content:center;z-index:9;cursor:zoom-out}} #lb[hidden]{{display:none}} #lb img{{max-width:92vw;max-height:88vh;image-rendering:auto}} .lbcap{{color:#eee;font-size:13px;margin-top:8px}}
.bar{{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:8px 0}} input,select{{font:inherit;padding:4px 8px;border:1px solid var(--line);border-radius:4px;background:var(--bg);color:var(--fg)}}
.grp{{display:inline-flex;gap:6px;align-items:center;padding:2px 8px;border:1px solid var(--line);border-radius:4px}} .grp label{{cursor:pointer}} .grp label span{{padding:0 6px;border-radius:3px}} button{{font:inherit;padding:4px 10px;border:1px solid var(--line);border-radius:4px;background:var(--head);color:var(--fg);cursor:pointer}} .cnt{{color:var(--mute)}} #shown b{{color:var(--fg)}}
</style></head><body><main>
<h1>학습 데이터 수집 현황 <small class="cnt">{today}</small></h1>
<p class="meta">yolo_mask_reviewer/scripts/training_data_status.py 로 생성. 학습 <code>{tag(a.train)}</code> · 검증 <code>{tag(a.valid)}</code> · 검수 대기 <code>{tag(a.pending)}</code> · 현장 수집 누적 <code>SL_under_predict_auto</code> ({auto_dates[0]}~{auto_dates[1]}, 비제외). 제품코드당 학습 상한 <b>{CAP}장</b>.</p>
<h2>요약</h2>
<div class="wrap">{tbl(['항목', '값'], [[esc(k), v] for k, v in sum_rows], 'kv')}</div>
<h2>대분류별</h2>
<div class="wrap">{tbl(['대분류', '코드', '충족', '부족', '수집만', '미수집', '학습 프레임', '검수 대기'], sort_rows)}</div>
<h2>제품코드별 현황 <small class="cnt">({len(rows)}코드)</small></h2>
<p class="meta">학습 = 학습 폴더 프레임 수(검수 재지정 코드 기준) · 부족 = 상한 − 학습 · 수집 누적 = 현장 자동 수집 비제외 프레임 누적 · 검증 = 고정 검증셋 · 대기 = 검수 대기 데이터셋의 보류 프레임 · 후보/정제 = 그 데이터셋을 만들 때 후보 수 / 정제 통과 수 · 반영 후 = 대기분이 전부 확정될 때의 학습 수(상한 적용) · ★ = 대표 라벨 수. 열 머리글을 누르면 정렬되고, 아래 필터(검색·상태·대분류·학습 범위·유무 토글)는 대표 라벨링 표에도 같이 적용된다.</p>
<div class="bar" id="filters">
<input id="q" placeholder="제품코드·제품명 검색 (공백으로 여러 단어)" size="34">
<span class="grp">상태 <label><input type="checkbox" class="st" value="충족" checked> <span style="background:var(--ok)">충족</span></label>
<label><input type="checkbox" class="st" value="부족" checked> <span style="background:var(--short)">부족</span></label>
<label><input type="checkbox" class="st" value="수집만" checked> <span style="background:var(--coll)">수집만</span></label>
<label><input type="checkbox" class="st" value="미수집" checked> <span style="background:var(--none)">미수집</span></label></span>
<span class="grp">대분류 <label><input type="checkbox" class="so" value="B/CVR" checked> B/CVR</label> <label><input type="checkbox" class="so" value="H/CVR" checked> H/CVR</label>
<label><input type="checkbox" class="so" value="HSG" checked> HSG</label> <label><input type="checkbox" class="so" value="SCALP" checked> SCALP</label></span>
<span class="grp">학습 <input id="tmin" type="number" min="0" max="{CAP}" placeholder="min" size="4" style="width:5em"> ~ <input id="tmax" type="number" min="0" max="{CAP}" placeholder="max" style="width:5em"></span>
<span class="grp"><label><input type="checkbox" id="fex"> ★ 대표 있음</label> <label><input type="checkbox" id="fpend"> 검수 대기 있음</label> <label><input type="checkbox" id="fval"> 검증 있음</label> <label><input type="checkbox" id="fauto"> 수집 누적 있음</label> <label><input type="checkbox" id="fneed"> 부족분 있음</label></span>
<button id="reset" type="button">초기화</button>
<span class="cnt" id="shown"></span></div>
<div class="wrap">{tbl(['제품코드', '대분류', '제품명', '상태', '학습', '부족', '수집 누적', '검증', '대기', '후보/정제', '반영 후', '★'], code_rows, 'codes', 'codes')}</div>
<h2>제품코드별 ★ 대표 라벨링 <small class="cnt">({sum(1 for r in rows if r['ex'])}코드, {sum(r['ex'] for r in rows)}장)</small></h2>
<p class="meta">학습 폴더 <code>{tag(a.train)}</code> 의 review_state.json 에서 exemplar 로 표시된 프레임. 초록 선 = 라벨 폴리곤(구멍 포함). 검수 PC 에서 ★ 대표로 표시한 것이며 구멍 전파·SAM2 전파·검증셋 보충의 기준이다. 대표가 없는 코드는 여기 없다. 위 필터가 같이 적용되고, 이미지를 누르면 크게 본다 (캡션은 원본 stem, 전체 이름은 마우스를 올리면 보인다).</p>
<div id="ex">{gallery}</div>
<div id="lb" hidden><img alt=""><div class="lbcap"></div></div>
<script>
(function(){{
const $=id=>document.getElementById(id);
const q=$('q'),tmin=$('tmin'),tmax=$('tmax'),shown=$('shown');
const codes=document.querySelector('#codes tbody'),exg=document.querySelectorAll('#ex .exgrp');
const num=t=>parseFloat(String(t).replace(/,/g,''))||0;const SORTS=new Set(['B/CVR','H/CVR','HSG','SCALP']);
function filt(){{
 const words=q.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
 const st=new Set([...document.querySelectorAll('.st:checked')].map(e=>e.value)),so=new Set([...document.querySelectorAll('.so:checked')].map(e=>e.value));
 const lo=tmin.value===''?-1:num(tmin.value),hi=tmax.value===''?1e9:num(tmax.value);
 const fex=$('fex').checked,fpend=$('fpend').checked,fval=$('fval').checked,fauto=$('fauto').checked,fneed=$('fneed').checked;
 const pass=new Set();let n=0,sTrain=0,sNeed=0,sPend=0,sEx=0;
 for(const tr of codes.rows){{const c=tr.cells;const txt=(c[0].textContent+' '+c[2].textContent).toLowerCase();
  const train=num(c[4].textContent),need=num(c[5].textContent),auto=num(c[6].textContent),val=num(c[7].textContent),pend=num(c[8].textContent),exn=num(c[11].textContent);
  const ok=words.every(w=>txt.includes(w))&&st.has(c[3].textContent)&&(so.has(c[1].textContent)||!SORTS.has(c[1].textContent))&&train>=lo&&train<=hi
   &&(!fex||exn>0)&&(!fpend||pend>0)&&(!fval||val>0)&&(!fauto||auto>0)&&(!fneed||need>0);
  tr.style.display=ok?'':'none';if(ok){{n++;pass.add(c[0].textContent);sTrain+=train;sNeed+=need;sPend+=pend;sEx+=exn;}}}}
 let ne=0;for(const g of exg){{const c=g.dataset.code;const ok=c==='none'?(words.length===0||'none'.includes(words.join(' '))):pass.has(c);g.style.display=ok?'':'none';if(ok)ne++;}}
 shown.innerHTML='<b>'+n+'</b> / '+codes.rows.length+' 코드 · 학습 '+sTrain.toLocaleString()+' · 부족 '+sNeed.toLocaleString()+' · 대기 '+sPend.toLocaleString()+' · ★ '+sEx+' (갤러리 '+ne+'코드)';}}
for(const el of document.querySelectorAll('#filters input'))el.addEventListener(el.type==='checkbox'?'change':'input',filt);
$('reset').onclick=()=>{{q.value='';tmin.value='';tmax.value='';for(const el of document.querySelectorAll('#filters input[type=checkbox]'))el.checked=el.classList.contains('st')||el.classList.contains('so');filt();}};
filt();
const lb=$('lb');document.querySelectorAll('#ex figure img').forEach(im=>im.onclick=()=>{{lb.querySelector('img').src=im.src;lb.querySelector('.lbcap').textContent=im.alt;lb.hidden=false;}});lb.onclick=()=>lb.hidden=true;document.addEventListener('keydown',e=>{{if(e.key==='Escape')lb.hidden=true;}});
for(const t of document.querySelectorAll('table.codes')){{const ths=t.tHead.rows[0].cells;for(let i=0;i<ths.length;i++)ths[i].onclick=()=>{{
 const asc=!ths[i].classList.contains('s-asc');for(const h of ths)h.classList.remove('s-asc','s-desc');ths[i].classList.add(asc?'s-asc':'s-desc');
 const rows=[...t.tBodies[0].rows];const isnum=rows.every(r=>/^-?[\d,]+(\/[\d,]+)?$/.test(r.cells[i].textContent.trim())||r.cells[i].textContent.trim()==='');
 rows.sort((a,b)=>{{let x=a.cells[i].textContent.trim(),y=b.cells[i].textContent.trim();if(isnum){{x=num(x);y=num(y);return asc?x-y:y-x;}}return asc?x.localeCompare(y):y.localeCompare(x);}});
 for(const r of rows)t.tBodies[0].appendChild(r);}};}}
}})();
</script>
</main></body></html>
"""
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, 'w', encoding='utf-8').write(html)
    print(f'→ {a.out}: 코드 {len(rows)}, {dict(n)}, 학습 {sum(train[c] for c in codes)}, 대기 {sum(pending.values())}, 반영 후 충족 {sum(1 for r in rows if r["after"] >= CAP)}')


if __name__ == '__main__':
    main()
