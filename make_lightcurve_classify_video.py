#!/usr/bin/env python3
"""
make_lightcurve_classify_video.py — Animated dome master (square fisheye disc) showing a
field of synthetic transit light curves over a sky background, then a "classification"
sequence: one curve is highlighted by a box, the verdict appears (a big green tick +
PLANET, or a big red cross + NOT A PLANET), then the next curve, and so on (~1s each).

This is the moving-picture companion to make_lightcurve_dome.py (which it imports from and
leaves untouched). Two differences from the still:
  - curves are placed ONLY where the background image actually has sky coverage (the
    TESS mosaic footprint), so none float in the black tile gaps / empty corners;
  - the result is a video (square fisheye, dome-ready), not a single PNG.

The box is yellow while "examining", then recolors to the verdict (green=planet,
red=not). A little mascot robot glides to each curve and its eyes/antenna glow the
verdict colour. Decided curves keep a small mark for the rest of the clip.

NOTE: video only (no audio).

Usage:
  python make_lightcurve_classify_video.py [OUT.mp4] [--size 1440] [--count 200]
         [--seed 7] [--examples 8] [--planet-prob 0.4] [--sec-per-example 1.0]
         [--fps 30] [--intro 1.0] [--outro 1.5] [--ang-size 8] [--max-tilt 22]
         [--background resources/tess/TESS_north_hires_azeq_no_labels_4K.jpg]

Bare filenames resolve through resources/ + outputs/; bare output is written to
outputs/videos/. If OUT is omitted, a name is generated there.
"""
import argparse
import numpy as np
import cv2
from flat_to_dome import disc_fit
from make_lightcurve_dome import light_curve, project_to_master, COLORS, COLOR_W
from project_paths import display_path, existing_path, output_path

YELLOW = (0, 235, 255)   # BGR — "examining" box
GREEN = (60, 210, 90)    # BGR — planet
RED = (40, 40, 255)      # BGR — not a planet


def coverage_mask(S, bg_fit, pad_px):
    """Bool (S,S): where curves may be centered (sky coverage, shrunk by pad_px)."""
    cx = cy = (S - 1) / 2.0
    R = S / 2.0
    j, i = np.meshgrid(np.arange(S), np.arange(S), indexing="xy")
    disc = np.hypot(j - cx, i - cy) <= 0.97 * R
    if bg_fit is None:
        cov = disc
    else:
        gray = cv2.cvtColor(bg_fit, cv2.COLOR_BGR2GRAY)
        cov = (gray > 10) & disc          # non-black = covered sky
    if pad_px > 0:                         # keep whole curve inside the footprint
        k = np.ones((2 * pad_px + 1, 2 * pad_px + 1), np.uint8)
        cov = cv2.erode(cov.astype(np.uint8), k).astype(bool)
    return cov


def curve_points(rng, S, ang_size, max_tilt, center_px):
    """Projected pixel polyline for one light curve centered at pixel center_px."""
    cx = cy = (S - 1) / 2.0
    R = S / 2.0
    Z = np.array([0.0, 0.0, 1.0])
    t, flux = light_curve(rng)
    theta = np.radians(rng.uniform(-max_tilt, max_tilt))
    ct, st = np.cos(theta), np.sin(theta)

    dx = (center_px[0] - cx) / R
    dy = (cy - center_px[1]) / R
    r0 = min(np.hypot(dx, dy), 0.95)
    a0 = np.arctan2(dx, dy)
    phi0 = r0 * (np.pi / 2.0)
    c = np.array([np.sin(phi0) * np.sin(a0),
                  np.sin(phi0) * np.cos(a0),
                  np.cos(phi0)])
    e1 = np.cross(Z, c)
    nrm = np.linalg.norm(e1)
    e1 = e1 / nrm if nrm > 1e-6 else np.array([1.0, 0.0, 0.0])
    e2 = np.cross(c, e1)
    aw = np.radians(ang_size * rng.uniform(0.7, 1.3))
    ah = aw * rng.uniform(0.42, 0.6)
    lx = (t - 0.5) * aw
    ly = (flux - 1.0) * ah * 1.6
    rx = lx * ct - ly * st
    ry = lx * st + ly * ct
    d = c[None, :] + np.tan(rx)[:, None] * e1[None, :] + np.tan(ry)[:, None] * e2[None, :]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return project_to_master(d, cx, cy, R)


def build_field(S, count, rng, args, bg_fit, cov):
    """Static base image (curves over background) + per-curve geometry for highlighting."""
    layer = np.zeros((S, S, 3), np.uint8)
    ys, xs = np.where(cov)
    if len(xs) == 0:
        raise SystemExit("No sky coverage to place curves in (check --background).")
    curves = []
    for _ in range(count):
        s = int(rng.integers(0, len(xs)))
        pts = curve_points(rng, S, args.ang_size, args.max_tilt, (xs[s], ys[s]))
        col = COLORS[rng.choice(len(COLORS), p=COLOR_W)]
        col = (col * rng.uniform(0.75, 1.0)).tolist()
        cv2.polylines(layer, [pts.astype(np.int32).reshape(-1, 1, 2)], False,
                      col, thickness=2, lineType=cv2.LINE_AA)
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
        curves.append({"pts": pts.astype(np.int32).reshape(-1, 1, 2),
                       "bbox": (float(x0), float(y0), float(x1), float(y1)),
                       "center": (float((x0 + x1) / 2), float((y0 + y1) / 2))})

    glow = cv2.GaussianBlur(layer, (0, 0), 3.0)
    field = np.clip(layer.astype(np.float32) + 0.7 * glow.astype(np.float32), 0, 255)
    if bg_fit is not None:
        alpha = field.max(axis=2, keepdims=True) / 255.0
        field = np.clip(field + bg_fit.astype(np.float32) * (1.0 - alpha), 0, 255)
    return field.astype(np.uint8), curves


def pick_examples(curves, n, rng):
    """Choose n spread-out curves (farthest-point sampling on their centers)."""
    n = min(n, len(curves))
    centers = np.array([c["center"] for c in curves])
    picked = [int(rng.integers(0, len(curves)))]
    while len(picked) < n:
        d = np.min(np.linalg.norm(centers[:, None] - centers[picked][None], axis=2), axis=1)
        d[picked] = -1
        picked.append(int(np.argmax(d)))
    return picked


def draw_cross(img, center, size, color, thick):
    x, y = int(center[0]), int(center[1])
    h = int(size / 2)
    for c, tk in (((0, 0, 0), thick + 4), (color, thick)):
        cv2.line(img, (x - h, y - h), (x + h, y + h), c, tk, cv2.LINE_AA)
        cv2.line(img, (x - h, y + h), (x + h, y - h), c, tk, cv2.LINE_AA)


def draw_tick(img, center, size, color, thick):
    x, y = center[0], center[1]
    h = size / 2.0
    pts = np.array([[x - 0.55 * h, y + 0.05 * h],
                    [x - 0.12 * h, y + 0.45 * h],
                    [x + 0.55 * h, y - 0.5 * h]], np.float32)
    poly = [pts.astype(np.int32).reshape(-1, 1, 2)]
    for c, tk in (((0, 0, 0), thick + 4), (color, thick)):
        cv2.polylines(img, poly, False, c, tk, cv2.LINE_AA)


def draw_box(img, bbox, color, thick, pad):
    x0, y0, x1, y1 = bbox
    cv2.rectangle(img, (int(x0 - pad), int(y0 - pad)), (int(x1 + pad), int(y1 + pad)),
                  color, thick, cv2.LINE_AA)


def draw_label(img, cx, y, text, color, S):
    font = cv2.FONT_HERSHEY_DUPLEX
    fs = S / 1300.0
    th = max(1, int(S / 900))
    (tw, thh), _ = cv2.getTextSize(text, font, fs, th)
    x = int(np.clip(cx - tw / 2, 8, S - tw - 8))
    y = int(np.clip(y, thh + 8, S - 8))
    cv2.putText(img, text, (x, y), font, fs, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, fs, color, th, cv2.LINE_AA)


def smoothstep(u):
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def draw_robot(img, center, s, accent):
    """A small mascot robot drawn with primitives (head, eyes, antenna, body, arms).

    `accent` (BGR) tints the eyes + antenna bulb so the robot can glow the verdict color.
    """
    x, y = int(center[0]), int(center[1])
    body = (225, 225, 235)
    edge = (35, 35, 45)
    er = max(2, int(s / 13))

    # antenna
    cv2.line(img, (x, y - int(0.40 * s)), (x, y - int(0.30 * s)), edge, max(1, int(s / 22)), cv2.LINE_AA)
    cv2.circle(img, (x, y - int(0.44 * s)), max(2, int(s / 11)), accent, -1, cv2.LINE_AA)

    # head
    hx, hy0, hy1 = int(0.30 * s), y - int(0.30 * s), y - int(0.02 * s)
    cv2.rectangle(img, (x - hx, hy0), (x + hx, hy1), body, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x - hx, hy0), (x + hx, hy1), edge, max(1, int(s / 26)), cv2.LINE_AA)
    ey = (hy0 + hy1) // 2
    cv2.circle(img, (x - int(0.14 * s), ey), er, accent, -1, cv2.LINE_AA)
    cv2.circle(img, (x + int(0.14 * s), ey), er, accent, -1, cv2.LINE_AA)

    # body
    bx, by0, by1 = int(0.26 * s), y + int(0.02 * s), y + int(0.34 * s)
    cv2.rectangle(img, (x - bx, by0), (x + bx, by1), body, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x - bx, by0), (x + bx, by1), edge, max(1, int(s / 26)), cv2.LINE_AA)

    # arms
    aw, ay = int(0.16 * s), by0 + int(0.10 * s)
    cv2.line(img, (x - bx, ay), (x - bx - aw, ay + int(0.08 * s)), body, max(2, int(s / 16)), cv2.LINE_AA)
    cv2.line(img, (x + bx, ay), (x + bx + aw, ay + int(0.08 * s)), body, max(2, int(s / 16)), cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output", nargs="?")
    ap.add_argument("--size", type=int, default=1440)
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--examples", type=int, default=8, help="how many curves get classified")
    ap.add_argument("--planet-prob", type=float, default=0.4, help="fraction judged 'planet'")
    ap.add_argument("--sec-per-example", type=float, default=1.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--intro", type=float, default=1.0, help="seconds of field before classifying")
    ap.add_argument("--outro", type=float, default=1.5, help="seconds holding the final field")
    ap.add_argument("--ang-size", type=float, default=8.0, help="mean curve width in degrees on the dome")
    ap.add_argument("--max-tilt", type=float, default=22.0)
    ap.add_argument("--background", help="azimuthal-equidistant (dome/fisheye) sky image; curves "
                                         "are placed only where it has coverage")
    args = ap.parse_args()

    S = args.size - (args.size % 2)            # even dims for the encoder
    output_file = output_path(args.output or f"lightcurve_classify_{S}.mp4", "videos")

    bg_fit = None
    if args.background:
        bg_file = existing_path(args.background)
        bg_img = cv2.imread(str(bg_file), cv2.IMREAD_COLOR)
        if bg_img is None:
            raise SystemExit(f"Could not read background image: {display_path(bg_file)}")
        bg_fit, _ = disc_fit(bg_img, S)

    rng = np.random.default_rng(args.seed)
    R = S / 2.0
    pad_px = int(np.ceil((args.ang_size / 90.0) * R * 0.7))   # curve half-extent, in px
    cov = coverage_mask(S, bg_fit, pad_px)
    field, curves = build_field(S, args.count, rng, args, bg_fit, cov)

    ex_idx = pick_examples(curves, args.examples, rng)
    n_planet = int(round(len(ex_idx) * args.planet_prob))   # guarantee a mix, not coin flips
    verdicts = [True] * n_planet + [False] * (len(ex_idx) - n_planet)
    rng.shuffle(verdicts)

    cx = cy = (S - 1) / 2.0
    j, i = np.meshgrid(np.arange(S), np.arange(S), indexing="xy")
    disc_inv = np.hypot(j - cx, i - cy) > R

    fpe = max(1, round(args.sec_per_example * args.fps))
    box_frames = max(1, round(fpe * 0.4))      # examining beat, then verdict
    box_pad = max(6, int(S / 180))
    box_th = max(2, int(S / 360))
    sym_th = max(3, int(S / 240))
    small_sz = 0.045 * S
    small_th = max(2, int(S / 600))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_file), fourcc, args.fps, (S, S))

    def write(img):
        f = img.copy()
        f[disc_inv] = 0
        writer.write(f)

    robot_s = 0.06 * S
    home = np.array([cx, cy])
    prev_perch = home.copy()

    # intro: field with the robot idling at the zenith
    for _ in range(round(args.intro * args.fps)):
        img = field.copy()
        draw_robot(img, home, robot_s, YELLOW)
        write(img)

    decided = []          # (center, verdict) marks that persist
    for ci, v in zip(ex_idx, verdicts):
        cobj = curves[ci]
        bbox = cobj["bbox"]
        vcol = GREEN if v else RED
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        sym = float(np.clip(max(bw, bh) * 1.2, 0.08 * S, 0.16 * S))
        label = "PLANET" if v else "NOT A PLANET"

        # robot perch: just outside the box, on the side toward the dome centre
        bc = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
        d = home - bc
        nrm = np.linalg.norm(d)
        d = d / nrm if nrm > 1e-6 else np.array([0.0, -1.0])
        perch = bc + d * (np.hypot(bw, bh) / 2.0 + box_pad + 0.7 * robot_s)
        off = perch - home                                # keep the robot inside the disc
        if np.linalg.norm(off) > 0.9 * R:
            perch = home + off / np.linalg.norm(off) * 0.9 * R

        for f in range(fpe):
            img = field.copy()
            for cen, dv in decided:                       # accumulated verdicts
                (draw_tick if dv else draw_cross)(img, cen, small_sz,
                                                  GREEN if dv else RED, small_th)
            cv2.polylines(img, [cobj["pts"]], False, (255, 255, 255), box_th + 1, cv2.LINE_AA)
            if f < box_frames:                            # examining: box yellow, robot glides in
                rp = prev_perch + (perch - prev_perch) * smoothstep((f + 1) / box_frames)
                draw_box(img, bbox, YELLOW, box_th, box_pad)
                draw_robot(img, rp, robot_s, YELLOW)
            else:                                         # verdict: box + mark + label, robot glows
                draw_box(img, bbox, vcol, box_th, box_pad)
                (draw_tick if v else draw_cross)(img, cobj["center"], sym, vcol, sym_th)
                draw_label(img, cobj["center"][0], bbox[3] + box_pad + 0.06 * S, label, vcol, S)
                draw_robot(img, perch, robot_s, vcol)
            write(img)
        decided.append((cobj["center"], v))
        prev_perch = perch

    # outro: final field with every verdict mark, robot parked at its last spot
    final = field.copy()
    for cen, dv in decided:
        (draw_tick if dv else draw_cross)(final, cen, small_sz, GREEN if dv else RED, small_th)
    for _ in range(round(args.outro * args.fps)):
        img = final.copy()
        draw_robot(img, prev_perch, robot_s, YELLOW)
        write(img)

    writer.release()
    n_planet = sum(verdicts)
    total = round(args.intro * args.fps) + len(ex_idx) * fpe + round(args.outro * args.fps)
    print(f"Wrote {display_path(output_file)}  ({S}x{S}, {total} frames @ {args.fps:g}fps, "
          f"{len(ex_idx)} classified: {n_planet} planet / {len(ex_idx) - n_planet} not)")


if __name__ == "__main__":
    main()
