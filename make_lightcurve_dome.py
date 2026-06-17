#!/usr/bin/env python3
"""
make_lightcurve_dome.py — Dome master (square fisheye disc) filled with many synthetic
transit light curves (flat baseline -> dip -> recovery), scattered across the dome.

By default each curve is placed at a DIRECTION on the dome (azimuth + elevation) with an
angular size in degrees, then mapped into the master through the true azimuthal-
equidistant fisheye projection. So curves are pre-distorted in the flat PNG (they bow
near the rim) but appear as clean, undistorted little plots WHEN PROJECTED on the dome.

Use --flat to instead draw curves directly in the flat disc (no projection) for
comparison — those look tidy in the PNG but get squished toward the rim on the dome.

Usage:
  python make_lightcurve_dome.py [OUT.png] [--size 2160] [--count 200] [--seed 3]
         [--ang-size 8] [--max-tilt 22] [--colorful] [--axes] [--flat]
         [--background IMAGE.jpg]

  --background composites the curves over an azimuthal-equidistant (dome/fisheye)
  image instead of black — the photo is fit to the disc and the curves glow on top.

Bare output filenames are written to outputs/masters/. If OUT is omitted, a name
is generated there.
"""
#Example usage:
# python make_lightcurve_dome.py lightcurve_v2_seed7.png --count 300 --seed 7 --colorful  

import argparse
import numpy as np
import cv2
from flat_to_dome import disc_fit
from project_paths import display_path, existing_path, output_path

COLORS = np.array([
    [255, 255, 255],   # white
    [120, 200, 255],   # amber
    [255, 230, 150],   # cyan-blue
    [140, 255, 170],   # green
    [255, 190, 120],   # blue
], dtype=np.float32)
COLOR_W = np.array([0.40, 0.25, 0.18, 0.10, 0.07])


def light_curve(rng, n=140):
    """flux(t), t in [0,1]: flat baseline with a trapezoidal/V transit dip + scatter."""
    t = np.linspace(0.0, 1.0, n)
    tc = rng.uniform(0.35, 0.65)
    w = rng.uniform(0.12, 0.42)
    depth = rng.uniform(0.06, 0.55)
    ingress = rng.uniform(0.02, 0.5) * w
    prof = np.clip((w / 2 - np.abs(t - tc)) / max(ingress, 1e-3), 0.0, 1.0)
    flux = 1.0 - depth * prof
    flux += rng.normal(0.0, rng.uniform(0.004, 0.022), n)
    return t, flux


def project_to_master(dirs, cx, cy, R):
    """Unit dome directions (X=right,Y=front,Z=up) -> master pixel coords (azimuthal eq.)."""
    dz = np.clip(dirs[:, 2], -1.0, 1.0)
    phi = np.arccos(dz)
    r = phi / (np.pi / 2.0)
    al = np.arctan2(dirs[:, 0], dirs[:, 1])
    X = cx + (r * np.sin(al)) * R
    Y = cy - (r * np.cos(al)) * R
    return np.stack([X, Y], axis=-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output", nargs="?")
    ap.add_argument("--size", type=int, default=2160)
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--ang-size", type=float, default=8.0, help="mean curve width in degrees on the dome")
    ap.add_argument("--max-tilt", type=float, default=22.0, help="max random tilt (deg)")
    ap.add_argument("--colorful", action="store_true")
    ap.add_argument("--axes", action="store_true", help="faint baseline under each curve")
    ap.add_argument("--flat", action="store_true", help="draw in flat disc (no projection) for comparison")
    ap.add_argument("--background", help="composite the curves over this azimuthal-equidistant "
                                         "(fisheye/dome) image instead of black; it is fit to the disc")
    args = ap.parse_args()

    S = args.size
    output_file = output_path(args.output or f"master_lightcurves_{S}.png", "masters")
    rng = np.random.default_rng(args.seed)
    base = np.zeros((S, S, 3), np.uint8)
    cx = cy = (S - 1) / 2.0
    R = S / 2.0
    Z = np.array([0.0, 0.0, 1.0])

    for _ in range(args.count):
        t, flux = light_curve(rng)
        theta = np.radians(rng.uniform(-args.max_tilt, args.max_tilt))
        ct, st = np.cos(theta), np.sin(theta)
        col = COLORS[rng.choice(len(COLORS), p=COLOR_W)] if args.colorful else \
            COLORS[rng.choice([0, 1], p=[0.6, 0.4])]
        col = (col * rng.uniform(0.7, 1.0)).tolist()

        if args.flat:
            # --- naive: straight in pixel space ---
            bw = rng.uniform(0.07, 0.12) * S
            bh = bw * rng.uniform(0.45, 0.65)
            diag = np.hypot(bw, bh) / 2
            rr = np.sqrt(rng.random()) * (R - diag - 6)
            a0 = rng.random() * 2 * np.pi
            px, py = cx + rr * np.cos(a0), cy + rr * np.sin(a0)
            lx = (t - 0.5) * bw
            ly = (flux - 1.0) * bh * 1.6
            X = px + lx * ct - ly * st
            Y = py - (lx * st + ly * ct)
            pts = np.stack([X, Y], axis=-1)
        else:
            # --- correct: place on dome at (zenith r0, az a0), project through fisheye ---
            r0 = np.sqrt(rng.random()) * 0.95
            a0 = rng.random() * 2 * np.pi
            phi0 = r0 * (np.pi / 2.0)
            c = np.array([np.sin(phi0) * np.sin(a0),
                          np.sin(phi0) * np.cos(a0),
                          np.cos(phi0)])
            e1 = np.cross(Z, c)
            nrm = np.linalg.norm(e1)
            e1 = e1 / nrm if nrm > 1e-6 else np.array([1.0, 0.0, 0.0])
            e2 = np.cross(c, e1)
            aw = np.radians(args.ang_size * rng.uniform(0.7, 1.3))   # width (rad)
            ah = aw * rng.uniform(0.42, 0.6)
            lx = (t - 0.5) * aw
            ly = (flux - 1.0) * ah * 1.6
            rx = lx * ct - ly * st
            ry = lx * st + ly * ct
            d = c[None, :] + np.tan(rx)[:, None] * e1[None, :] + np.tan(ry)[:, None] * e2[None, :]
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            pts = project_to_master(d, cx, cy, R)

        if args.axes:
            base_flux = np.ones_like(t)
            if args.flat:
                ax_pts = np.stack([px + (t - 0.5) * bw * ct, py - ((t - 0.5) * bw * st)], axis=-1)
            else:
                lxa = (t - 0.5) * aw
                da = c[None, :] + np.tan(lxa * ct)[:, None] * e1[None, :] + np.tan(lxa * st)[:, None] * e2[None, :]
                da /= np.linalg.norm(da, axis=1, keepdims=True)
                ax_pts = project_to_master(da, cx, cy, R)
            cv2.polylines(base, [ax_pts.astype(np.int32).reshape(-1, 1, 2)], False,
                          (45, 45, 45), 1, cv2.LINE_AA)

        cv2.polylines(base, [pts.astype(np.int32).reshape(-1, 1, 2)], False,
                      col, thickness=2, lineType=cv2.LINE_AA)

    glow = cv2.GaussianBlur(base, (0, 0), 3.0)
    curves = np.clip(base.astype(np.float32) + 0.7 * glow.astype(np.float32), 0, 255)

    if args.background:
        bg_file = existing_path(args.background)
        bg_img = cv2.imread(str(bg_file), cv2.IMREAD_COLOR)
        if bg_img is None:
            raise SystemExit(f"Could not read background image: {display_path(bg_file)}")
        bg_fit, _ = disc_fit(bg_img, S)                       # fit azimuthal-equidistant bg to the disc
        # premultiplied "over": curves (bright on black) sit on the sky; their glow blends softly.
        alpha = curves.max(axis=2, keepdims=True) / 255.0
        out = np.clip(curves + bg_fit.astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8)
    else:
        out = curves.astype(np.uint8)

    j, i = np.meshgrid(np.arange(S), np.arange(S), indexing="xy")
    out[np.hypot(j - cx, i - cy) > R] = 0
    cv2.imwrite(str(output_file), out)
    mode = "flat (no projection)" if args.flat else "dome-projected"
    bg_note = f", over {existing_path(args.background).name}" if args.background else ""
    print(f"Wrote {display_path(output_file)}  ({S}x{S}, {args.count} light curves, {mode}{bg_note})")


if __name__ == "__main__":
    main()
