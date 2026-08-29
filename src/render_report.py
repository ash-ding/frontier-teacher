"""Render curves.json as a self-contained HTML report - 3x3 small multiples.

No plotting dependency: the panels are hand-written SVG. matplotlib is
deliberately not installed in the training environment, and adding it to render
a figure would mean touching an environment that nine runs depend on.

  python src/render_report.py --curves outputs/curves.json --out report.html
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIGS = [
    ("llama32-3b",       "Llama-3.2-3B",        "the student"),
    ("qwen3-4b-nothink", "Qwen3-4B non-think",  "reference"),
    ("qwen3-4b-think",   "Qwen3-4B thinking",   "reference"),
]
TASKS = [
    ("math500", "MATH-500",  "pass@1", "in-domain"),
    ("aime",    "AIME 20–24", "pass@4", "transfer"),
    ("hmmt",    "HMMT",       "pass@4", "transfer"),
]
# Bands are ordered by difficulty, and each configuration slices at its own
# measured pass@1, so the slugs differ per model while the roles do not.
BANDS = [
    (["pass1_05-15pct", "pass1_12-25pct"], "hard",   "1"),
    (["pass1_40-60pct", "pass1_37-62pct"], "medium", "2"),
    (["pass1_85-95pct", "pass1_75-87pct"], "easy",   "3"),
]
BAND_LABEL = {"hard": "hard (low pass@1)", "medium": "medium (≈50%)",
              "easy": "easy (high pass@1)"}
W, H = 300, 190
PAD_L, PAD_R, PAD_T, PAD_B = 46, 16, 14, 30


def role_of(band):
    for slugs, role, _ in BANDS:
        if band in slugs:
            return role
    return None


def slot_of(role):
    return {"hard": "1", "medium": "2", "easy": "3"}[role]


def panel(series, task, cfg):
    """One SVG panel: every band of one configuration on one benchmark."""
    rows = [s for s in series if s["config"] == cfg and s["task"] == task
            and role_of(s["band"])]
    if not rows:
        return f'<div class="panel empty">no data</div>'

    ys = [p["score"] * 100 for s in rows for p in s["points"] if p["score"] is not None]
    if not ys:
        return f'<div class="panel empty">no data</div>'
    lo, hi = min(ys), max(ys)
    span = max(hi - lo, 2.0)
    lo, hi = lo - span * 0.22, hi + span * 0.22
    xmax = 10240

    def X(v):
        return PAD_L + (v / xmax) * (W - PAD_L - PAD_R)

    def Y(v):
        return PAD_T + (1 - (v - lo) / (hi - lo)) * (H - PAD_T - PAD_B)

    base = next((p["score"] * 100 for s in rows for p in s["points"]
                 if p["rollouts"] == 0 and p["score"] is not None), None)

    out = [f'<svg viewBox="0 0 {W} {H}" role="img" '
           f'aria-label="{cfg} on {task}" preserveAspectRatio="xMidYMid meet">']
    # y grid, 4 ticks
    for i in range(4):
        v = lo + (hi - lo) * i / 3
        out.append(f'<line class="grid" x1="{PAD_L}" y1="{Y(v):.1f}" '
                   f'x2="{W-PAD_R}" y2="{Y(v):.1f}"/>')
        out.append(f'<text class="tick" x="{PAD_L-6}" y="{Y(v)+3:.1f}" '
                   f'text-anchor="end">{v:.0f}</text>')
    if base is not None:
        out.append(f'<line class="baseline" x1="{PAD_L}" y1="{Y(base):.1f}" '
                   f'x2="{W-PAD_R}" y2="{Y(base):.1f}"/>')
    for xv in (0, 5120, 10240):
        out.append(f'<text class="tick" x="{X(xv):.1f}" y="{H-10}" '
                   f'text-anchor="middle">{xv//1000 if xv else 0}{"k" if xv else ""}</text>')

    for s in rows:
        role = role_of(s["band"])
        slot = slot_of(role)
        dash = ' stroke-dasharray="5 3"' if s.get("variant") else ""
        pts = [(p["rollouts"], p["score"] * 100) for p in s["points"]
               if p["score"] is not None]
        if len(pts) < 2:
            continue
        d = " ".join(f'{"M" if i == 0 else "L"}{X(x):.1f},{Y(y):.1f}'
                     for i, (x, y) in enumerate(pts))
        out.append(f'<path class="line s{slot}" d="{d}"{dash}/>')
        for x, y in pts:
            out.append(f'<circle class="dot s{slot}" cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3.2">'
                       f'<title>{BAND_LABEL[role]} · {x:,} rollouts · {y:.1f}%</title></circle>')
        ex, ey = pts[-1]
        delta = ey - base if base is not None else None
        if delta is not None:
            sign = "+" if delta >= 0 else "−"
            anchor = "end" if ex >= xmax else "start"
            dx = -4 if anchor == "end" else 4
            out.append(f'<text class="endlab s{slot}" x="{X(ex)+dx:.1f}" '
                       f'y="{Y(ey)-6:.1f}" text-anchor="{anchor}">'
                       f'{sign}{abs(delta):.1f}</text>')
        if not s.get("reaches_10240", True):
            out.append(f'<circle class="stop" cx="{X(ex):.1f}" cy="{Y(ey):.1f}" r="6.5"/>')
    out.append("</svg>")
    return '<div class="panel">' + "".join(out) + "</div>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", default=str(ROOT / "outputs" / "curves.json"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "report.html"))
    ap.add_argument("--stats", default="", help="optional JSON of paired results")
    a = ap.parse_args()

    doc = json.loads(Path(a.curves).read_text())
    series = doc["series"]
    body = Path(str(ROOT / "src" / "report_template.html")).read_text()

    grid = []
    for tkey, tname, metric, kind in TASKS:
        grid.append(f'<div class="rowlab"><span class="tname">{tname}</span>'
                    f'<span class="tmetric">{metric}</span>'
                    f'<span class="tkind">{kind}</span></div>')
        for ckey, cname, _ in CONFIGS:
            grid.append(panel(series, tkey, ckey))
    body = body.replace("<!--GRID-->", "\n".join(grid))

    heads = "".join(f'<div class="colhead"><span>{n}</span><em>{r}</em></div>'
                    for _, n, r in CONFIGS)
    body = body.replace("<!--COLHEADS-->", heads)

    # the table is not decoration: the aqua series fails 3:1 contrast against the
    # light surface, so a non-colour reading of every value has to exist
    trs = []
    for ckey, cname, _ in CONFIGS:
        for slugs, role, _ in BANDS:
            for tkey, tname, metric, _k in TASKS:
                s = next((x for x in series if x["config"] == ckey
                          and x["task"] == tkey and x["band"] in slugs
                          and not x.get("variant")), None)
                if not s:
                    continue
                vals = {p["rollouts"]: p["score"] for p in s["points"]}
                cells = "".join(
                    f'<td>{100*vals[r]:.1f}</td>' if vals.get(r) is not None else '<td class="na">—</td>'
                    for r in (0, 2560, 5120, 7680, 10240))
                b0, b1 = vals.get(0), vals.get(10240)
                d = f'{100*(b1-b0):+.1f}' if (b0 is not None and b1 is not None) else "—"
                trs.append(f'<tr><td>{cname}</td><td>{BAND_LABEL[role]}</td>'
                           f'<td>{tname} {metric}</td>{cells}'
                           f'<td class="delta">{d}</td></tr>')
    body = body.replace("<!--TABLE-->", "\n".join(trs))
    body = body.replace("<!--FINDINGS-->", Path(ROOT / "src" / "report_findings.html").read_text())
    body = body.replace("<!--CAVEATS-->", Path(ROOT / "src" / "report_caveats.html").read_text())
    Path(a.out).write_text(body)
    print(f"wrote {a.out}  ({len(body):,} bytes, {len(series)} series)")


if __name__ == "__main__":
    main()
