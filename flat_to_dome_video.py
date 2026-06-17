#!/usr/bin/env python3
"""
flat_to_dome_video.py — Video version of flat_to_dome.py. Converts every frame of a
video into a fulldome master, writing a square fisheye dome-master video.

Modes (same as flat_to_dome.py):
  billboard  flat/rectilinear footage placed as a panel on the dome (default)
  fill       whole frame squeezed into the disc
  disc       footage that is ALREADY a circular all-sky/fisheye (just fit to disc)

NOTE: audio is not carried through (OpenCV writes video only). If you need the audio,
mux it back with ffmpeg afterwards.

Usage:
  python flat_to_dome_video.py IN.mov [OUT.mp4] [--size 2160] [--mode billboard]
         [--hfov 95] [--elevation 50] [--azimuth 0] [--copies 1] [--mirror] [--fps 0]

  --copies (billboard mode) repeats the footage on N evenly-spaced sides of the
  dome: 2 = opposite sides, 4 = four quarters.

Bare input filenames are found in resources/ and outputs/. Bare output filenames
are written to outputs/masters/. If OUT is omitted, a name is generated there.
"""
import argparse
import numpy as np
import cv2
from flat_to_dome import billboard_copies_maps, fill_maps
from project_paths import display_path, existing_path, output_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output", nargs="?")
    ap.add_argument("--size", type=int, default=2160)
    ap.add_argument("--mode", choices=["billboard", "fill", "disc"], default="billboard")
    ap.add_argument("--hfov", type=float, default=95.0)
    ap.add_argument("--elevation", type=float, default=50.0)
    ap.add_argument("--azimuth", type=float, default=0.0)
    ap.add_argument("--copies", type=int, default=1,
                    help="billboard mode: evenly-spaced copies around the dome "
                         "(1 = single panel, 2 = opposite sides, 4 = four quarters)")
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--fps", type=float, default=0.0, help="override output fps (0 = keep source)")
    ap.add_argument("--interp", choices=["linear", "lanczos"], default="linear",
                    help="remap interpolation; linear is ~10x faster and fine for motion video")
    args = ap.parse_args()
    interp = cv2.INTER_LANCZOS4 if args.interp == "lanczos" else cv2.INTER_LINEAR

    input_file = existing_path(args.input)
    default_output = f"{input_file.stem}_master_{args.size}.mp4"
    output_file = output_path(args.output or default_output, "masters")

    cap = cv2.VideoCapture(str(input_file))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {display_path(input_file)}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = args.fps if args.fps > 0 else (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    S = args.size

    # Precompute everything that's frame-independent.
    if args.mode == "billboard":
        map_x, map_y, valid = billboard_copies_maps(S, w, h, args.hfov, args.elevation,
                                                    args.azimuth, args.copies)
    elif args.mode == "fill":
        map_x, map_y, valid = fill_maps(S, w, h)
    else:  # disc: resize + center-crop + circular mask, all constant per frame
        scale = S / min(w, h)
        new_w, new_h = round(w * scale), round(h * scale)
        x0, y0 = (new_w - S) // 2, (new_h - S) // 2
        j, i = np.meshgrid(np.arange(S), np.arange(S), indexing="xy")
        c = (S - 1) / 2.0
        valid = np.hypot(j - c, i - c) <= S / 2.0

    inv = ~valid
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_file), fourcc, fps, (S, S))

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.mode == "disc":
            r = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            out = r[y0:y0 + S, x0:x0 + S].copy()
        else:
            out = cv2.remap(frame, map_x, map_y, interpolation=interp,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        out[inv] = 0
        if args.mirror:
            out = out[:, ::-1]
        writer.write(out)
        idx += 1
        if idx % 100 == 0:
            print(f"  {idx}/{n} frames")

    cap.release()
    writer.release()
    print(f"Wrote {display_path(output_file)}  ({S}x{S}, {idx} frames, {fps:.2f}fps, mode={args.mode})")


if __name__ == "__main__":
    main()
