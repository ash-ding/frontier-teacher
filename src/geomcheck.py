"""Geometry self-check for the rendered report - the step the dataviz procedure
calls "render it and look at it", done programmatically because this session
cannot see the page. It catches exactly what an eyeball catches: marks outside
the plot area, labels off the panel, and labels sitting on top of each other."""
import re, sys, pathlib
h = pathlib.Path(sys.argv[1]).read_text()
W,H,PL,PR,PT,PB = 300,190,46,16,14,30
panels = re.findall(r'<div class="panel">(.*?)</div>', h, re.S)
bad_pts = bad_lab = collisions = 0
for i, p in enumerate(panels):
    for cx, cy in re.findall(r'<circle class="dot [^"]*" cx="([\d.]+)" cy="([\d.]+)"', p):
        x, y = float(cx), float(cy)
        if not (PL-1 <= x <= W-PR+1) or not (PT-1 <= y <= H-PB+1):
            bad_pts += 1; print("  panel %d point out of bounds (%.1f,%.1f)" % (i,x,y))
    labs = []
    for m in re.finditer(r'<text class="endlab s(\d)" x="([\d.]+)" y="([\d.-]+)" text-anchor="(\w+)">([^<]+)</text>', p):
        slot, x, y, anch, txt = m.group(1), float(m.group(2)), float(m.group(3)), m.group(4), m.group(5)
        w = len(txt)*5.2
        x0 = x-w if anch == "end" else x
        labs.append((x0, x0+w, y-9, y+2, slot, txt))
        if y < 2 or y > H-PB+6 or x0 < 0 or x0+w > W:
            bad_lab += 1; print("  panel %d label out of bounds %r" % (i,txt))
    # Require a real overlap, not a touching edge. Labels dodged to exactly one
    # box-height apart come out of float arithmetic 1e-14 apart in the wrong
    # direction (135.2-9 < 124.2+2), which a strict test reports as a collision
    # that does not exist on screen.
    EPS = 0.5
    for a in range(len(labs)):
        for b in range(a+1, len(labs)):
            A, B = labs[a], labs[b]
            if (min(A[1],B[1]) - max(A[0],B[0]) > EPS
                    and min(A[3],B[3]) - max(A[2],B[2]) > EPS):
                collisions += 1
                print("  panel %d labels overlap: %r x %r" % (i, A[5], B[5]))
tokens = set(re.findall(r'--([a-z0-9-]+)\s*:', h))
root = set(re.findall(r'--([a-z0-9-]+)\s*:', re.search(r':root\{(.*?)\}', h, re.S).group(1)))
missing = sorted(t for t in tokens if t not in root)
bg = bool(re.search(r'body\{[^}]*background:var\(--bg\)', h))
print("panels %d | points out %d | labels out %d | overlaps %d" % (len(panels), bad_pts, bad_lab, collisions))
print("tokens missing from bare :root:", missing or "none")
print("body paints an explicit background:", bg)
sys.exit(1 if (bad_pts or bad_lab or collisions or missing or not bg) else 0)
