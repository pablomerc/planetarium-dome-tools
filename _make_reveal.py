from PIL import Image
import os

SIZE = 1080
HOLD_SECONDS = 7.0
HOLD_FRAME_MS = 1000          # split hold into 1s frames (robust across players)
BEAT_MS = 700                 # brief pause on completed mosaic before candidates appear

g = Image.open('resources/tess/TESS_north_gif.gif')
frames, durs = [], []
for i in range(g.n_frames):
    g.seek(i)
    frames.append(g.convert('RGB'))
    durs.append(g.info.get('duration', 80))

# shorten the animation's built-in 3s end-hold so it doesn't double up with the reveal
durs[-1] = BEAT_MS

# candidates still, matched to the GIF canvas
cand = Image.open('resources/tess/TESS_north_candidates.jpg').convert('RGB').resize((SIZE, SIZE), Image.LANCZOS)
n_hold = max(1, round(HOLD_SECONDS * 1000 / HOLD_FRAME_MS))
frames += [cand] * n_hold
durs += [HOLD_FRAME_MS] * n_hold

out = 'outputs/TESS_north_reveal.gif'
frames[0].save(out, save_all=True, append_images=frames[1:],
               duration=durs, loop=0, optimize=True, disposal=2)

print(f"Wrote {out}")
print(f"frames: {len(frames)}  total: {sum(durs)/1000:.1f}s  "
      f"(anim ~{sum(durs[:-n_hold])/1000:.1f}s + reveal {n_hold*HOLD_FRAME_MS/1000:.0f}s)")
print(f"size: {os.path.getsize(out)/1e6:.1f} MB")
