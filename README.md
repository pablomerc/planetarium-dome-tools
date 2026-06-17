# planetarium-dome-tools

Small Python tools for preparing **fulldome planetarium masters** — square images and
videos holding a 180° circular fisheye (azimuthal-equidistant) projection, ready to play
on a dome.

In a dome master:

- **disc center** = zenith (straight up)
- **disc edge** = horizon
- **top of the image** = the "front" of the dome (where the audience faces)

## Setup

```bash
pip install -r requirements.txt          # numpy, opencv-python, Pillow
```

Inputs are read from `resources/` and outputs are written under `outputs/` (created
automatically). Both folders are git-ignored — supply your own `resources/`. Most scripts
accept bare filenames (resolved against `resources/` and `outputs/`) or full paths.

## Tools

| Script | What it does |
|--------|--------------|
| `flat_to_dome.py` | Turn a regular (rectilinear) **photo** into a fulldome master. Modes: `billboard` (place the photo as a flat panel at a chosen elevation/azimuth/FOV — looks natural), `fill` (squeeze the whole frame into the disc), `disc` (fit an already-circular all-sky image). `--copies N` repeats it on N evenly-spaced sides of the dome (2 = opposite sides, 4 = quarters). |
| `flat_to_dome_video.py` | Same projection, frame-by-frame, for a **video** (writes a square fisheye dome-master MP4). Shares the projection code with `flat_to_dome.py`. Audio is not carried through — re-mux with ffmpeg if needed. |
| `make_lightcurve_dome.py` | Generate a dome master filled with many synthetic **transit light curves** scattered across the dome (pre-distorted so they read as clean little plots when projected). `--background IMG` composites them over an azimuthal-equidistant sky image. |
| `make_lightcurve_classify_video.py` | Animated companion to the above: a "classification" sequence where each highlighted light curve is marked **PLANET** (green ✓) or **NOT A PLANET** (red ✗), with a mascot robot that visits each one. Curves are placed only within the background's sky coverage. `--poster` also writes a matching still (no robot) for a slide before pressing play. |
| `project_paths.py` | Shared helper for resolving input/output paths against `resources/` and `outputs/`. |
| `_make_reveal.py` | One-off helper that stitches an animation GIF and a final still into a single "reveal" GIF. |

## Examples

```bash
# A photo as a panel high on the dome, mirrored for an audience looking up
python flat_to_dome.py myphoto.jpg --mode billboard --elevation 55 --hfov 90 --mirror

# Same photo repeated on two opposite sides of the dome
python flat_to_dome.py myphoto.jpg --copies 2

# A field of light curves over a sky background
python make_lightcurve_dome.py lc.png --count 200 --colorful \
  --background resources/tess/TESS_north_hires_azeq_no_labels_4K.jpg

# 15s classification video + a matching poster still
python make_lightcurve_classify_video.py classify.mp4 --poster \
  --sec-per-example 1.5 --intro 1.2 --outro 1.8 \
  --background resources/tess/TESS_north_hires_azeq_no_labels_4K.jpg
```

Run any script with `-h` for the full list of options.

See `script.md` for show notes.
