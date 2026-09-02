"""Validate a categorical palette mechanically. Colour separation is computable,
so compute it - a palette that looks fine to the author can be two identical
greys to a deuteranope, and that is not visible by inspection.

Six checks, following the visualisation guidance this project renders under:
  lightness band   every slot inside the readable band for the surface
  chroma floor     every slot saturated enough to read as a hue, not a grey
  CVD separation   adjacent pairs stay apart under protan/deutan/tritan
  normal vision    adjacent pairs are far apart for everyone else
  contrast         each slot against the surface it sits on
  ordering         (informational) whether lightness is monotone, which matters
                   only if the dimension being encoded is itself ordered

  python tools/palette_check.py "#2a78d6,#eb6834,..." --mode light
"""
import argparse, itertools, math, sys

# Machado et al. 2009, severity 1.0, applied to LINEAR rgb
CVD = {
    "protan": ((0.152286, 1.052583, -0.204868),
               (0.114503, 0.786281, 0.099216),
               (-0.003882, -0.048116, 1.051998)),
    "deutan": ((0.367322, 0.860646, -0.227968),
               (0.280085, 0.672501, 0.047413),
               (-0.011820, 0.042940, 0.968881)),
    "tritan": ((1.255528, -0.076749, -0.178779),
               (-0.078411, 0.930809, 0.147602),
               (0.004733, 0.691367, 0.303900)),
}


def hex2srgb(h):
    h = h.strip().lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def oklab(rgb):
    r, g, b = (to_linear(c) for c in rgb)
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = (math.copysign(abs(v) ** (1 / 3), v) for v in (l, m, s))
    return (0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_)


def simulate(rgb, kind):
    lin = [to_linear(c) for c in rgb]
    M = CVD[kind]
    out = [max(0.0, min(1.0, sum(M[i][j] * lin[j] for j in range(3)))) for i in range(3)]
    # back to sRGB gamma so oklab() re-linearises consistently
    return tuple(12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
                 for c in out)


def de(a, b):
    return 100 * math.dist(oklab(a), oklab(b))


def luminance(rgb):
    r, g, b = (to_linear(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("palette")
    ap.add_argument("--mode", choices=["light", "dark"], default="light")
    ap.add_argument("--surface", default="")
    ap.add_argument("--pairs", choices=["adjacent", "all"], default="adjacent")
    a = ap.parse_args()

    hexes = [h.strip() for h in a.palette.split(",") if h.strip()]
    cols = [hex2srgb(h) for h in hexes]
    surface = hex2srgb(a.surface) if a.surface else hex2srgb(
        "#fcfcfd" if a.mode == "light" else "#1a1a19")
    band = (0.43, 0.77) if a.mode == "light" else (0.48, 0.72)

    print(f"Palette ({a.mode}, surface {a.surface or ('#fcfcfd' if a.mode=='light' else '#1a1a19')}, "
          f"categorical): {len(cols)} slots")
    fails = 0

    Ls = [oklab(c)[0] for c in cols]
    bad = [f"{h} L={L:.2f}" for h, L in zip(hexes, Ls) if not (band[0] <= L <= band[1])]
    print(f"  [{'FAIL' if bad else 'PASS'}] lightness band     "
          f"{'all' if not bad else ''} inside L {band[0]}-{band[1]}"
          + ("   " + ", ".join(bad) if bad else ""))
    fails += bool(bad)

    Cs = [math.hypot(*oklab(c)[1:]) for c in cols]
    bad = [f"{h} C={C:.3f}" for h, C in zip(hexes, Cs) if C < 0.1]
    print(f"  [{'FAIL' if bad else 'PASS'}] chroma floor       "
          f"{'all' if not bad else ''} >= 0.1" + ("   " + ", ".join(bad) if bad else ""))
    fails += bool(bad)

    pairs = (list(zip(range(len(cols) - 1), range(1, len(cols))))
             if a.pairs == "adjacent" else list(itertools.combinations(range(len(cols)), 2)))
    for kind in ("protan", "deutan", "tritan"):
        worst = min(((de(simulate(cols[i], kind), simulate(cols[j], kind)), i, j)
                     for i, j in pairs), key=lambda t: t[0])
        d, i, j = worst
        tag = "PASS" if d >= 8 else ("WARN" if d >= 6 else "FAIL")
        print(f"  [{tag}] CVD {kind:<7}        worst pair {hexes[i]}<->{hexes[j]} dE {d:.1f}")
        fails += tag == "FAIL"

    d, i, j = min(((de(cols[i], cols[j]), i, j) for i, j in pairs), key=lambda t: t[0])
    tag = "PASS" if d >= 15 else "FAIL"
    print(f"  [{tag}] normal vision      worst pair {hexes[i]}<->{hexes[j]} dE {d:.1f}")
    fails += tag == "FAIL"

    low = [(h, contrast(c, surface)) for h, c in zip(hexes, cols) if contrast(c, surface) < 3]
    print(f"  [{'WARN' if low else 'PASS'}] contrast vs surface "
          f"{'all >= 3:1' if not low else str([(h, round(r,2)) for h, r in low]) + '  -> needs labels or a table'}")

    print(f"\n  -> {'ALL CHECKS PASS' if not fails else str(fails) + ' CHECK(S) FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
