import os
from PIL import Image

#─── USER CONFIG ───────────────────────────────────────────────────────────────
input_root  = '/home/tedlasai/genCamera/GOPRO_7'       # root of your GOPRO_7 folder
output_root = '/home/tedlasai/genCamera/Motion-ETR-main/Gopro_align_data'  # where to write train/ & test/
splits      = ['train', 'test']
#───────────────────────────────────────────────────────────────────────────────

for split in splits:
    blur_root   = os.path.join(input_root, split, 'blur')
    sharp_root  = os.path.join(input_root, split, 'sharp')
    out_split   = os.path.join(output_root, split)
    os.makedirs(out_split, exist_ok=True)

    count = 0
    # each sub-folder is like GOPR0384_11_00
    for seq in sorted(os.listdir(blur_root)):
        blur_seq  = os.path.join(blur_root, seq)
        sharp_seq = os.path.join(sharp_root, seq)
        if not os.path.isdir(blur_seq) or not os.path.isdir(sharp_seq):
            continue

        for fname in sorted(os.listdir(blur_seq)):
            bpath = os.path.join(blur_seq, fname)
            spath = os.path.join(sharp_seq, fname)
            if not os.path.exists(spath):
                print(f"⚠️  Missing sharp for {bpath}")
                continue

            # load and concat
            blur_img  = Image.open(bpath).convert('RGB')
            sharp_img = Image.open(spath).convert('RGB')

            w, h = blur_img.width + sharp_img.width, max(blur_img.height, sharp_img.height)
            canvas = Image.new('RGB', (w, h))
            canvas.paste(blur_img, (0, 0))
            canvas.paste(sharp_img, (blur_img.width, 0))

            # name: GOPR0384_11_00_000004.png
            out_name = f"{seq}_{fname}"
            canvas.save(os.path.join(out_split, out_name))

            count += 1

    print(f"[{split}] saved {count} merged pairs → {out_split}")

print("✅  Done!")
