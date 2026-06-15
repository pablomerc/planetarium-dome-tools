#!/usr/bin/env python3
"""
flat_to_dome.py — Convert a regular (rectilinear) photo into a planetarium
"fulldome master": a square image holding a 180-degree circular fisheye
(azimuthal-equidistant) projection.

  - Disc CENTER  = zenith (straight up on the dome)
  - Disc EDGE    = horizon
  - Top of image = "front" of the dome (where the audience faces)

Two placement modes:
  billboard  (default) — the flat photo is treated as a real perspective image
                         and mapped onto the dome like a flat panel/screen,
                         pointing at a chosen elevation + azimuth with a chosen
                         field of view. This keeps the picture looking natural.
  fill                 — the whole rectangle is squeezed to fill the fisheye
                         disc (heavy distortion; quick stylistic option).

Usage:
  python flat_to_dome.py INPUT OUTPUT [--size 2048] [--mode billboard]
         [--hfov 90] [--elevation 55] [--azimuth 0] [--bg 0,0,0]
"""
import argparse
import numpy as np
import cv2


def build_dome_directions(size):
    """Return (dirs, mask) for an SxS azimuthal-equidistant 180-deg fisheye.

    dirs: (S, S, 3) unit vectors in world space  (X=right, Y=front, Z=up/zenith)
    mask: (S, S) bool, True inside the 180-deg disc.
    """
    cx = cy = (size - 1) / 2.0
    R = size / 2.0
    j, i = np.meshgrid(np.arange(size), np.arange(size), indexing="xy")
    xn = (j - cx) / R            # +right
    yn = (cy - i) / R            # +up (image y grows downward, so flip)
    r = np.hypot(xn, yn)
    mask = r <= 1.0

    phi = r * (np.pi / 2.0)      # zenith angle: 0 at center -> 90deg at edge
    alpha = np.arctan2(xn, yn)   # screen angle: 0 at top, +toward right

    sphi = np.sin(phi)
    dirs = np.stack([sphi * np.sin(alpha),   # X (right)
                     sphi * np.cos(alpha),   # Y (front)
                     np.cos(phi)], axis=-1)  # Z (up)
    return dirs, mask


def billboard_maps(size, img_w, img_h, hfov_deg, elevation_deg, azimuth_deg):
    """Compute cv2.remap maps placing the flat image as a panel on the dome."""
    dirs, disc_mask = build_dome_directions(size)

    # Camera basis: forward at given elevation above horizon and azimuth from front.
    el = np.radians(elevation_deg)
    az = np.radians(azimuth_deg)
    f = np.array([np.cos(el) * np.sin(az),
                  np.cos(el) * np.cos(az),
                  np.sin(el)])
    zup = np.array([0.0, 0.0, 1.0])
    right = np.cross(f, zup); right /= np.linalg.norm(right)
    up = np.cross(right, f)

    with np.errstate(all="ignore"):               # silence spurious numpy-2 SIMD matmul warnings
        fc = dirs @ f                              # depth along view axis
        xc = dirs @ right
        yc = dirs @ up

    hfov = np.radians(hfov_deg)
    th = np.tan(hfov / 2.0)
    tv = th * (img_h / img_w)                      # vfov from aspect ratio

    with np.errstate(divide="ignore", invalid="ignore"):
        px = (xc / fc) / th                        # [-1, 1] across image width
        py = (yc / fc) / tv

    in_front = fc > 1e-6
    in_frame = (np.abs(px) <= 1) & (np.abs(py) <= 1)
    valid = disc_mask & in_front & in_frame

    map_x = ((px + 1) * 0.5 * (img_w - 1)).astype(np.float32)
    map_y = ((1 - (py + 1) * 0.5) * (img_h - 1)).astype(np.float32)
    map_x[~valid] = -1
    map_y[~valid] = -1
    return map_x, map_y, valid


def disc_fit(img, size):
    """Fit an already-circular all-sky / fisheye image into the SxS dome disc.

    Scales so the input's inscribed circle (diameter = min(w,h)) fills the disc,
    centers it, and masks outside the disc. No reprojection — use this when the
    source is already a circular azimuthal/fisheye image (e.g. an all-sky map).
    """
    h, w = img.shape[:2]
    scale = size / min(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    out = np.zeros((size, size, 3), dtype=img.dtype)
    x0 = (new_w - size) // 2
    y0 = (new_h - size) // 2
    out[:] = resized[y0:y0 + size, x0:x0 + size]

    j, i = np.meshgrid(np.arange(size), np.arange(size), indexing="xy")
    cx = cy = (size - 1) / 2.0
    valid = np.hypot(j - cx, i - cy) <= size / 2.0
    return out, valid


def fill_maps(size, img_w, img_h):
    """Squeeze the whole rectangle into the fisheye disc."""
    cx = cy = (size - 1) / 2.0
    R = size / 2.0
    j, i = np.meshgrid(np.arange(size), np.arange(size), indexing="xy")
    xn = (j - cx) / R
    yn = (cy - i) / R
    mask = np.hypot(xn, yn) <= 1.0
    map_x = ((xn + 1) * 0.5 * (img_w - 1)).astype(np.float32)
    map_y = ((1 - (yn + 1) * 0.5) * (img_h - 1)).astype(np.float32)
    map_x[~mask] = -1
    map_y[~mask] = -1
    return map_x, map_y, mask


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--size", type=int, default=2048, help="output square size (px)")
    ap.add_argument("--mode", choices=["billboard", "fill", "disc"], default="billboard")
    ap.add_argument("--hfov", type=float, default=90.0, help="billboard horizontal FOV (deg)")
    ap.add_argument("--elevation", type=float, default=55.0, help="billboard center elevation above horizon (deg)")
    ap.add_argument("--azimuth", type=float, default=0.0, help="billboard azimuth from front (deg)")
    ap.add_argument("--bg", default="0,0,0", help="background RGB outside content, e.g. 0,0,0")
    ap.add_argument("--mirror", action="store_true",
                    help="horizontally flip the master (East-West swap) so text/graphics "
                         "read correctly for an audience looking up; verify against your projector")
    args = ap.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not read image: {args.input}")
    h, w = img.shape[:2]

    if args.mode == "disc":
        out, valid = disc_fit(img, args.size)
    else:
        if args.mode == "billboard":
            map_x, map_y, valid = billboard_maps(args.size, w, h,
                                                 args.hfov, args.elevation, args.azimuth)
        else:
            map_x, map_y, valid = fill_maps(args.size, w, h)
        out = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LANCZOS4,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    bg = tuple(int(c) for c in args.bg.split(","))[::-1]  # RGB -> BGR
    out[~valid] = bg

    if args.mirror:
        out = out[:, ::-1]

    cv2.imwrite(args.output, out)
    print(f"Wrote {args.output}  ({args.size}x{args.size}, mode={args.mode})")


if __name__ == "__main__":
    main()
