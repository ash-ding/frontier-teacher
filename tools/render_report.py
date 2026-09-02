"""Render curves.json as a self-contained HTML report - 3x3 small multiples.

No plotting dependency: the panels are hand-written SVG. matplotlib is
deliberately not installed in the training environment, and adding it to render
a figure would mean touching an environment that nine runs depend on.

  python tools/render_report.py --curves outputs/analysis/curves.json --out report.html
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
# Ordered by difficulty, hardest first. Five hues that are all mutually
# separable under protan/deutan/tritan do not exist in this palette - a search
# over every 5-subset of its eight slots returns nothing that passes in both
# light and dark. So the three trainable bands take validated hues and the two
# degenerate ones take neutrals with a dotted stroke: composite encoding, and it
# matches the semantics, since p = 0 and p = 1 are boundaries rather than points
# in the difficulty comparison.
BANDS = [
    (["pass1_eq_0"],                       "p0",     "0"),
    (["pass1_05-15pct", "pass1_12-25pct"], "hard",   "1"),
    (["pass1_40-60pct", "pass1_37-62pct"], "medium", "2"),
    (["pass1_85-95pct", "pass1_75-87pct"], "easy",   "3"),
    (["pass1_eq_1"],                       "p1",     "4"),
    # Not a band: the curriculum a frontier model wrote, step by step, after
    # evaluating the student itself. It is drawn on the same axes because it
    # spent the same 512 rollouts per step for the same 20 steps -- the whole
    # point is that the problems are the only thing that differs.
    (["teacher"],                          "teacher", "T"),
]
BAND_LABEL = {"p0": "p = 0 (never solved when profiled)",
              "hard": "hard (low pass@1)", "medium": "medium (≈50%)",
              "easy": "easy (high pass@1)",
              "p1": "p = 1 (always solved when profiled)",
              "teacher": "teacher-written curriculum"}
W, H = 300, 190
PAD_L, PAD_R, PAD_T, PAD_B = 46, 16, 14, 30


def leak_panel(series, task, cfg):
    """Two lines for one teacher run: the fifth it could see, and the rest.

    Both come from the SAME four full-benchmark evaluations, split by the
    manifest's id list -- same checkpoints, same sample counts, same grader, so
    the only difference between the lines is which problems they are computed
    over. The per-step reference tests run at a finer cadence but cover only the
    seen fifth, so they cannot make this comparison.

    A curriculum that exploited what the teacher read there would push the seen
    line above the held-out one and keep it there.
    """
    row = next((s for s in series if s["config"] == cfg and s["task"] == task
                and s["band"] == "teacher"), None)
    if not row:
        return '<div class="panel empty">no teacher run yet</div>'
    lines = {w: [(p["rollouts"], p[w] * 100) for p in row["points"]
                 if p.get(w) is not None] for w in ("held_out", "seen")}
    ys = [y for v in lines.values() for _, y in v]
    if not ys:
        return '<div class="panel empty">no teacher run yet</div>'
    lo, hi = min(ys), max(ys)
    span = max(hi - lo, 2.0)
    lo, hi = lo - span * 0.25, hi + span * 0.25
    X = lambda v: PAD_L + (v / 10240) * (W - PAD_L - PAD_R)
    Y = lambda v: PAD_T + (1 - (v - lo) / (hi - lo)) * (H - PAD_T - PAD_B)

    out = [f'<svg viewBox="0 0 {W} {H}" role="img" '
           f'aria-label="{cfg} on {task}: seen fifth against held-out">']
    for i in range(4):
        v = lo + (hi - lo) * i / 3
        out.append(f'<line class="grid" x1="{PAD_L}" y1="{Y(v):.1f}" '
                   f'x2="{W-PAD_R}" y2="{Y(v):.1f}"/>')
        out.append(f'<text class="tick" x="{PAD_L-6}" y="{Y(v)+3:.1f}" '
                   f'text-anchor="end">{v:.0f}</text>')
    for xv in (0, 5120, 10240):
        out.append(f'<text class="tick" x="{X(xv):.1f}" y="{H-10}" '
                   f'text-anchor="middle">{xv//1000 if xv else 0}{"k" if xv else ""}</text>')
    for which, cls in (("held_out", "sH"), ("seen", "sS")):
        pts = lines[which]
        if len(pts) < 2:
            continue
        d = " ".join(f'{"M" if i == 0 else "L"}{X(x):.1f},{Y(y):.1f}'
                     for i, (x, y) in enumerate(pts))
        out.append(f'<path class="line {cls}" d="{d}"/>')
        for x, y in pts:
            out.append(f'<circle class="dot {cls}" cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3.2">'
                       f'<title>{"held-out 4/5" if which=="held_out" else "seen 1/5"}'
                       f' · {x:,} rollouts · {y:.1f}%</title></circle>')
    out.append("</svg>")
    return '<div class="panel">' + "".join(out) + "</div>"


def role_of(band):
    for slugs, role, _ in BANDS:
        if band in slugs:
            return role
    return None


def slot_of(role):
    return {"p0": "0", "hard": "1", "medium": "2", "easy": "3", "p1": "4",
            "teacher": "T"}[role]


def panel(series, task, cfg, teacher=False, field="score"):
    """One SVG panel: every band of one configuration on one benchmark.

    Every band is a solid line. p=0 and p=1 used to be dotted, marking them as
    degenerate ends rather than points on the difficulty axis -- and the runs
    disproved that: the p=0 bands train (reward 1.4% -> 6.3% on Llama, 4.7% ->
    9.6% on non-thinking, about half their groups carrying gradient), and p=1
    gains +3.7 on MATH-500. Five bands, one axis, hardest to easiest. A stroke
    style that contradicts the result is a claim, not a decoration.

    With `teacher=True` that reverses, and for a different reason: the bands are
    no longer the subject but the field the teacher is being read against, so
    all five recede together into dashed, muted context and the teacher alone is
    drawn solid. Uniformly all five -- singling any of them out would be the
    claim the paragraph above rejects.
    """
    # The matched-group-size controls are in the table, not here. Drawn as a
    # dashed twin of the band they re-run, they doubled the number of lines in
    # every middle panel to carry one fact - that group size moved nothing
    # consistently - which a table row states better than a line the eye has to
    # separate from its own default.
    rows = [s for s in series if s["config"] == cfg and s["task"] == task
            and role_of(s["band"]) and not s.get("variant")
            and (teacher or role_of(s["band"]) != "teacher")]
    if not rows:
        return f'<div class="panel empty">no data</div>'

    ys = [p[field] * 100 for s in rows for p in s["points"]
          if p.get(field) is not None]
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

    base = next((p[field] * 100 for s in rows for p in s["points"]
                 if p["rollouts"] == 0 and p.get(field) is not None), None)

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

    labels = []
    for s in rows:
        role = role_of(s["band"])
        slot = slot_of(role)
        pts = [(p["rollouts"], p[field] * 100) for p in s["points"]
               if p.get(field) is not None]
        if len(pts) < 2:
            continue
        d = " ".join(f'{"M" if i == 0 else "L"}{X(x):.1f},{Y(y):.1f}'
                     for i, (x, y) in enumerate(pts))
        ctx = teacher and role != "teacher"
        cls = f"line s{slot}" + (" ctx" if ctx else "")
        dash = ' stroke-dasharray="4 3.5"' if ctx else ""
        out.append(f'<path class="{cls}" d="{d}"{dash}/>')
        for x, y in pts:
            out.append(f'<circle class="dot s{slot}{" ctx" if ctx else ""}" '
                       f'cx="{X(x):.1f}" cy="{Y(y):.1f}" r="{2.4 if ctx else 3.2}">'
                       f'<title>{BAND_LABEL[role]} · {x:,} rollouts · {y:.1f}%</title></circle>')
        ex, ey = pts[-1]
        if not s.get("reaches_10240", True):
            out.append(f'<circle class="stop" cx="{X(ex):.1f}" cy="{Y(ey):.1f}" r="6.5"/>')
        if base is not None:
            delta = ey - base
            sign = "+" if delta >= 0 else "−"
            anchor = "end" if ex >= xmax else "start"
            labels.append({"x": X(ex) + (-4 if anchor == "end" else 4),
                           "y": Y(ey) - 6, "anchor": anchor, "slot": slot,
                           "ctx": ctx, "text": f"{sign}{abs(delta):.1f}"})

    # Three lines that end at similar scores put their labels on top of each
    # other - 10 collisions across 5 panels before this. Dodge vertically:
    # order by position, force a minimum gap, then shift the group back inside
    # the panel if the last one has been pushed past the axis.
    # With six lines in a 146 px plot area a fixed gap can need more room than
    # exists; the down-dodge then hits the bottom, the corrective shift hits the
    # top, and the two cancel leaving overlaps. Shrink the gap to fit instead.
    top, bottom = PAD_T + 8, H - PAD_B - 2
    GAP = 12.0        # one box height plus a hair, so labels never touch
    if len(labels) > 1:
        GAP = min(GAP, (bottom - top) / (len(labels) - 1))
    labels.sort(key=lambda l: l["y"])
    for i in range(1, len(labels)):
        if labels[i]["y"] - labels[i - 1]["y"] < GAP:
            labels[i]["y"] = labels[i - 1]["y"] + GAP
    if labels:
        overflow = labels[-1]["y"] - bottom
        if overflow > 0:
            for l in labels:
                l["y"] -= overflow
        under = top - labels[0]["y"]
        if under > 0:
            for l in labels:
                l["y"] += under
    for l in labels:
        out.append(f'<text class="endlab s{l["slot"]}{" ctx" if l.get("ctx") else ""}" x="{l["x"]:.1f}" '
                   f'y="{l["y"]:.1f}" text-anchor="{l["anchor"]}">{l["text"]}</text>')
    out.append("</svg>")
    return '<div class="panel">' + "".join(out) + "</div>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", default=str(ROOT / "outputs" / "analysis" / "curves.json"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "report.html"))
    ap.add_argument("--stats", default="", help="optional JSON of paired results")
    ap.add_argument("--holdout", default=None,
                    help="the held-out / seen split from tools/holdout_split.py")
    a = ap.parse_args()

    doc = json.loads(Path(a.curves).read_text())
    series = doc["series"]
    # Never print a delta without its interval. paired_stats.py resamples both
    # problems and generations; a table showing only the point estimate would
    # imply a precision the five-point curves do not have.
    stats_path = Path(a.stats) if a.stats else ROOT / "outputs" / "analysis" / "paired_stats.json"
    stats = {}
    if stats_path.exists():
        for r in json.loads(stats_path.read_text()):
            stats[(r["config"], r["band"], r["variant"], r["task"], r["metric"])] = r
    # The reference fifth is visible to the teacher and to nothing else, so
    # section 2 has to read every curve on the four fifths it could not see.
    # holdout_split.py recomputes that from the per-problem records.
    hpath = Path(a.holdout) if a.holdout else ROOT / "outputs" / "analysis" / "holdout.json"
    hold = {}
    if hpath.exists():
        for model, rows in json.loads(hpath.read_text()).items():
            for r in rows:
                band = ("teacher" if r["curriculum"] == "TEACHER"
                        else r["curriculum"].split("__")[0])
                var = "g32" if r["curriculum"].endswith("__g32") else ""
                for which in ("held_out", "seen"):
                    if r[which]:
                        hold[(model, band, var, r["task"], r["rollouts"], which)] = \
                            r[which]["score"]
    for x in series:
        for pt in x["points"]:
            for which in ("held_out", "seen"):
                k = (x["config"], x["band"], x.get("variant") or "",
                     x["task"], pt["rollouts"], which)
                pt[which] = hold.get(k)

    body = Path(str(ROOT / "tools" / "report_template.html")).read_text()

    def build_grid(with_teacher, field="score"):
        g = []
        for tkey, tname, metric, kind in TASKS:
            g.append(f'<div class="rowlab"><span class="tname">{tname}</span>'
                     f'<span class="tmetric">{metric}</span>'
                     f'<span class="tkind">{kind}</span></div>')
            for ckey, cname, _ in CONFIGS:
                g.append(panel(series, tkey, ckey, teacher=with_teacher, field=field))
        return "\n".join(g)

    body = body.replace("<!--GRID-->", build_grid(False))
    body = body.replace("<!--GRID_TEACHER-->", build_grid(True, field="held_out"))

    leak = []
    for tkey, tname, metric, kind in TASKS:
        leak.append(f'<div class="rowlab"><span class="tname">{tname}</span>'
                    f'<span class="tmetric">{metric}</span>'
                    f'<span class="tkind">{kind}</span></div>')
        for ckey, _, _ in CONFIGS:
            leak.append(leak_panel(series, tkey, ckey))
    body = body.replace("<!--GRID_LEAK-->", "\n".join(leak))
    # Say which configurations have a teacher run rather than letting a missing
    # line read as a flat one.
    have = sorted({x["config"] for x in series if x["band"] == "teacher"})
    missing = [n for k, n, _ in CONFIGS if k not in have]
    body = body.replace("<!--TEACHER_STATUS-->",
                        ("Still running: " + ", ".join(missing) +
                         " &mdash; those panels show the bands only."
                         if missing else "All three configurations have a teacher run."))

    heads = "".join(f'<div class="colhead"><span>{n}</span><em>{r}</em></div>'
                    for _, n, r in CONFIGS)
    body = body.replace("<!--COLHEADS-->", heads)
    body = body.replace("<!--COLHEADS_2-->", heads)
    body = body.replace("<!--COLHEADS_3-->", heads)

    # the table is not decoration: the aqua series fails 3:1 contrast against the
    # light surface, so a non-colour reading of every value has to exist
    trs = []
    for ckey, cname, _ in CONFIGS:
        for slugs, role, _ in BANDS:
            for tkey, tname, metric, _k in TASKS:
              # default first, then its matched-group-size re-run if there is one
              for variant in ("", "g32"):
                s = next((x for x in series if x["config"] == ckey
                          and x["task"] == tkey and x["band"] in slugs
                          and (x.get("variant") or "") == variant), None)
                if not s:
                    continue
                vals = {p["rollouts"]: p["score"] for p in s["points"]}
                cells = "".join(
                    f'<td>{100*vals[r]:.1f}</td>' if vals.get(r) is not None else '<td class="na">—</td>'
                    for r in (0, 2560, 5120, 7680, 10240))
                def fmt(key):
                    r = stats.get((ckey, s["band"], s.get("variant") or "", tkey, key))
                    if not r:
                        return '<td class="na">—</td>'
                    sig = ' sig' if (r["lo"] > 0 or r["hi"] < 0) else ''
                    return (f'<td class="delta{sig}">{100*r["delta"]:+.1f}'
                            f'<em>[{100*r["lo"]:+.1f}, {100*r["hi"]:+.1f}]</em></td>')
                # MATH-500's headline already IS pass@1; showing it twice invites
                # the reader to think two different things were measured.
                p1 = '<td class="na"></td>' if metric == "pass@1" else fmt("pass@1")
                label = BAND_LABEL[role] + (" &mdash; matched G=32" if variant else "")
                trs.append(f'<tr{" class=variant" if variant else ""}>'
                           f'<td>{cname}</td><td>{label}</td>'
                           f'<td>{tname} {metric}</td>{cells}'
                           f'{fmt(metric)}{p1}</tr>')
    body = body.replace("<!--TABLE-->", "\n".join(trs))
    # The setup strip was hand-typed and went stale as controls landed. Derive it.
    runs = {(x["config"], x["band"], x.get("variant") or "") for x in series}
    ctrls = sum(1 for r in runs if r[2])
    evals = sum(len([q for q in x["points"] if q["rollouts"] > 0]) for x in series)
    body = body.replace("<!--NRUNS-->", f"{len(runs) - ctrls}+{ctrls}")
    body = body.replace("<!--NEVALS-->", str(evals))
    body = body.replace("<!--FINDINGS-->", Path(ROOT / "tools" / "report_findings.html").read_text())
    body = body.replace("<!--CAVEATS-->", Path(ROOT / "tools" / "report_caveats.html").read_text())
    Path(a.out).write_text(body)
    print(f"wrote {a.out}  ({len(body):,} bytes, {len(series)} series)")


if __name__ == "__main__":
    main()
