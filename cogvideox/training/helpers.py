import torch 
import math
import random
import numpy as np
from PIL import Image

def random_insert_latent_frame(
    image_latent: torch.Tensor,
    noisy_model_input: torch.Tensor,
    target_latents: torch.Tensor,
    input_intervals: torch.Tensor,
    output_intervals: torch.Tensor,
    special_info
):
    """
    Inserts latent frames into noisy input, pads targets, and builds flattened intervals with flags.

    Args:
        image_latent:     [B, latent_count, C, H, W]
        noisy_model_input:[B, F, C, H, W]
        target_latents:   [B, F, C, H, W]
        input_intervals:  [B, N, frames_per_latent, L]
        output_intervals: [B, M, frames_per_latent, L]

    For each sample randomly choose:
    Mode A (50%):
        - Insert two image_latent frames at start of noisy input and targets.
        - Pad target_latents by prepending two zero-frames.
        - Pad input_intervals by repeating its last group once.
    Mode B (50%):
        - Insert one image_latent frame at start and repeat last noisy frame at end.
        - Pad target_latents by prepending one one-frame and appending last target frame.
        - Pad output_intervals by repeating its last group once.

    After padding intervals, flatten each group from [frames_per_latent, L] to [frames_per_latent * L],
    then append a 4-element flag (1 for input groups, 0 for output groups).

    Returns:
        outputs:     Tensor [B, F+2, C, H, W]
        new_targets: Tensor [B, F+2, C, H, W]
        masks:       Tensor [B, F+2] bool mask of latent inserts
        intervals:   Tensor [B, N+M+1, fpl * L + 4]
    """
    B, F, C, H, W = noisy_model_input.shape
    _, N, fpl, L = input_intervals.shape
    _, M, _, _ = output_intervals.shape
    device = noisy_model_input.device

    new_F = F + 1 if special_info == "just_one" else F + 2
    outputs = torch.empty((B, new_F, C, H, W), device=device)
    masks = torch.zeros((B, new_F), dtype=torch.bool, device=device)
    combined_groups = N + M #+ 1
    feature_len = fpl * L
    # intervals = torch.empty((B, combined_groups, feature_len + 4), device=device,
    #                         dtype=input_intervals.dtype)
    intervals = torch.empty((B, combined_groups, feature_len), device=device,
                            dtype=input_intervals.dtype)
    new_targets = torch.empty((B, new_F, C, H, W), device=device,
                            dtype=target_latents.dtype)

    for b in range(B):
        latent = image_latent[b, 0]
        frames = noisy_model_input[b]
        tgt = target_latents[b]

        limit = 10 if special_info == "use_a" else 0.5
        if special_info == "just_one": #ALWAYS_MODE_A
            # Mode A: two latent inserts, zero-prefixed targets
            outputs[b, 0] = latent
            masks[b, :1] = True
            outputs[b, 1:] = frames

            # pad targets: two large-numbers - these should be ignored
            large_number = torch.ones_like(tgt[0])*10000
            new_targets[b, 0] = large_number
            new_targets[b, 1:] = tgt

            # pad intervals: input + replicated last input group
            #pad_group = input_intervals[b, -1:].clone()
            in_groups = input_intervals[b] #torch.cat([input_intervals[b], pad_group], dim=0)
            out_groups = output_intervals[b]
        elif random.random() < limit: #ALWAYS_MODE_A
            # Mode A: two latent inserts, zero-prefixed targets
            outputs[b, 0] = latent
            outputs[b, 1] = latent
            masks[b, :2] = True
            outputs[b, 2:] = frames

            # pad targets: two large-numbers - these should be ignored
            large_number = torch.ones_like(tgt[0])*10000
            new_targets[b, 0] = large_number
            new_targets[b, 1] = large_number
            new_targets[b, 2:] = tgt

            # pad intervals: input + replicated last input group
            pad_group = input_intervals[b, -1:].clone()
            in_groups = torch.cat([input_intervals[b], pad_group], dim=0)
            out_groups = output_intervals[b]
        else:
            # Mode B: one latent insert & last-frame repeat, one-prefixed/appended targets
            outputs[b, 0] = latent
            masks[b, 0] = True
            outputs[b, 1:new_F-1] = frames
            outputs[b, new_F-1] = frames[-1]

            # pad targets: one one-frame then original then last frame
            zero = torch.zeros_like(tgt[0])
            new_targets[b, 0] = zero
            new_targets[b, 1:new_F-1] = tgt
            new_targets[b, new_F-1] = tgt[-1]

            # pad intervals: output + replicated last output group
            in_groups = input_intervals[b]
            pad_group = output_intervals[b, -1:].clone()
            out_groups = torch.cat([output_intervals[b], pad_group], dim=0)

        # flatten & flag groups
        flat_in = in_groups.reshape(-1, feature_len)
        proc_in = torch.cat([flat_in], dim=1)

        flat_out = out_groups.reshape(-1, feature_len)
        proc_out = torch.cat([flat_out], dim=1)

        intervals[b] = torch.cat([proc_in, proc_out], dim=0)

    return outputs, new_targets, masks, intervals




def transform_intervals(
    intervals: torch.Tensor,
    frames_per_latent: int = 4,
    repeat_first: bool = True
) -> torch.Tensor:
    """
    Pad and reshape intervals into [B, num_latent_frames, frames_per_latent, L].

    Args:
        intervals: Tensor of shape [B, N, L]
        frames_per_latent: number of frames per latent group (e.g., 4)
        repeat_first: if True, pad at the beginning by repeating the first row; otherwise pad at the end by repeating the last row.

    Returns:
        Tensor of shape [B, num_latent_frames, frames_per_latent, L]
    """
    B, N, L = intervals.shape
    num_latent = math.ceil(N / frames_per_latent)
    target_N = num_latent * frames_per_latent
    pad_count = target_N - N

    if pad_count > 0:
        # choose row to repeat
        pad_row = intervals[:, :1, :] if repeat_first else intervals[:, -1:, :]
        # replicate pad_row pad_count times
        pad = pad_row.repeat(1, pad_count, 1)
        # pad at beginning or end
        if repeat_first:
            expanded = torch.cat([pad, intervals], dim=1)
        else:
            expanded = torch.cat([intervals, pad], dim=1)
    else:
        expanded = intervals[:, :target_N, :]

    # reshape into latent-frame groups
    return expanded.view(B, num_latent, frames_per_latent, L)

import random
import numpy as np
import torch
from PIL import Image


import random
import numpy as np
import torch
from PIL import Image


def build_blur(frame_paths, gamma=2.2):
    """
    Simulate motion blur using inverse-gamma (linear-light) summation:
    - Load each image, convert to float32 sRGB [0,255]
    - Linearize via inverse gamma: linear = (img/255)^gamma
    - Sum linear values, average, then re-encode via gamma: (linear_avg)^(1/gamma)*255
    Returns a uint8 numpy array.
    """
    acc_lin = None
    for p in frame_paths:
        img = np.array(Image.open(p).convert('RGB'), dtype=np.float32)
        # normalize to [0,1] then linearize
        lin = np.power(img / 255.0, gamma)
        acc_lin = lin if acc_lin is None else acc_lin + lin
    # average in linear domain
    avg_lin = acc_lin / len(frame_paths)
    # gamma-encode back to sRGB domain
    srgb = np.power(avg_lin, 1.0 / gamma) * 255.0
    return np.clip(srgb, 0, 255).astype(np.uint8)



def generate_1x_sequence(frame_paths, window_max =16, output_len=17, base_rate=1, start = None):
    """
    1× mode at arbitrary base_rate (units of 1/240s):
      - Treat each output step as the sum of `base_rate` consecutive raw frames.
      - Pick window size W ∈ [1, output_len]
      - Randomly choose start index so W*base_rate frames fit
      - Group raw frames into W groups of length base_rate
      - Build blur image over all W*base_rate frames for input
      - For each group, build a blurred output frame by summing its base_rate frames
      - Pad sequence of W blurred frames to output_len by repeating last blurred frame
      - Input interval always [-0.5, 0.5]
      - Output intervals reflect each group’s coverage within [-0.5,0.5]
    """
    N = len(frame_paths)
    max_w = min(output_len, N // base_rate)
    max_w = min(max_w, window_max)
    W = random.randint(1, max_w)
    if start is not None:
        # choose start so that W*base_rate frames fit
        assert N >= W * base_rate, f"Not enough frames for base_rate={base_rate}, need {W * base_rate}, got {N}"
    else:
        start = random.randint(0, N - W * base_rate)
        

    # group start indices
    group_starts = [start + i * base_rate for i in range(W)]
    # flatten raw frame paths for blur input
    blur_paths = []
    for gs in group_starts:
        blur_paths.extend(frame_paths[gs:gs + base_rate])
    blur_img = build_blur(blur_paths)

    # build blurred output frames per group
    seq = []
    for gs in group_starts:
        group = frame_paths[gs:gs + base_rate]
        seq.append(build_blur(group))
    # pad with last blurred frame
    seq += [seq[-1]] * (output_len - len(seq))

    input_interval = torch.tensor([[-0.5, 0.5]], dtype=torch.float)
    # each group covers interval of length 1/W
    step = 1.0 / W
    intervals = [[-0.5 + i * step, -0.5 + (i + 1) * step] for i in range(W)]
    intervals += [intervals[-1]] * (output_len - W)
    output_intervals = torch.tensor(intervals, dtype=torch.float)

    return blur_img, seq, input_interval, output_intervals

#for each video do a 1x,2x, and large blur in the test list - choose at max two spots per video
# so you need to update the dataloader and these helpers to handle this setting
# and write a script that sets all of these settings and writes it to a file
# finally, I need to come up with a scheme to write all of this out blurry/Adobe240/lower_fps_frames/GOPRO9656/frame_1325_b_1320_1335_r_1320_1345_1x.png,
#This should take 2 hrs above

#then write a dataloader that handles_outside_files and just uses the naming convention to figure out how to deblur image_start_end_w4_240_1x.png, image_b_w7_240_1x.png, image_b_w8_240_1x.png, image_b_w12_240_1x.png, image_b_w16_240_1x.png, image_b_w8_240_2x.png, image_b_w8_120_2x.png, and image_b_w32_240_lb.png, image_start_end_w48_240_lb.png
#find points in each video with at least 24 frames on each side - thats the large blur (48 frame case)

def generate_2x_sequence(frame_paths, window_max =16, output_len=17, base_rate=1):
    """
    2× mode:
      - Logical window of W output-steps so that 2*W ≤ output_len
      - Raw window spans W*base_rate frames
      - Build blur only over that raw window (flattened) for input
      - before_count = W//2, after_count = W - before_count
      - Define groups for before, during, and after each of length base_rate
      - Build blurred frames for each group
      - Pad sequence of 2*W blurred frames to output_len by repeating last
      - Input interval always [-0.5,0.5]
      - Output intervals relative to window: each group’s center
    """
    N = len(frame_paths)
    max_w = min(output_len // 2, N // base_rate)
    max_w = min(max_w, window_max)
    W = random.randint(1, max_w)
    before_count = W // 2
    after_count = W - before_count
    # choose start so that before and after stay within bounds
    min_start = before_count * base_rate
    max_start = N - (W + after_count) * base_rate
    # ensure we can pick a valid start, else fail
    assert max_start >= min_start, f"Cannot satisfy before/after window for W={W}, base_rate={base_rate}, N={N}"
    start = random.randint(min_start, max_start)


    # window group starts
    window_starts = [start + i * base_rate for i in range(W)]
    # flatten for blur input
    blur_paths = []
    for gs in window_starts:
        print(f"gs: {gs}, base_rate: {base_rate}")
        blur_paths.extend(frame_paths[gs:gs + base_rate])


    blur_img = build_blur(blur_paths)

    # define before/after group starts
    before_count = W // 2
    after_count = W - before_count
    before_starts = [max(0, start - (i + 1) * base_rate) for i in range(before_count)][::-1]
    after_starts  = [min(N - base_rate, start + W * base_rate + i * base_rate) for i in range(after_count)]

    # all group starts in sequence
    group_starts = before_starts + window_starts + after_starts
    # build blurred frames per group
    print(f"Group starts: {group_starts}")
    seq = []
    for gs in group_starts:
        group = frame_paths[gs:gs + base_rate]
        seq.append(build_blur(group))
    # pad blurred frames to output_len
    seq += [seq[-1]] * (output_len - len(seq))

    input_interval = torch.tensor([[-0.5, 0.5]], dtype=torch.float)
    # each group covers 1/(2W) around its center within [-0.5,0.5]
    half = 0.5 / W
    centers = [((gs - start) / (W * base_rate)) - 0.5 + half
               for gs in group_starts]
    intervals = [[c - half, c + half] for c in centers]
    intervals += [intervals[-1]] * (output_len - len(intervals))
    output_intervals = torch.tensor(intervals, dtype=torch.float)

    return blur_img, seq, input_interval, output_intervals


def generate_large_blur_sequence(frame_paths, window_max=16, output_len=17, base_rate=1):
    """
    Large blur mode (fixed output_len=25) with instantaneous outputs:
      - Raw window spans 25 * base_rate consecutive frames
      - Build blur over that full raw window for input
      - For output sequence:
          • Pick 1 raw frame every `base_rate` (group_starts)
          • Each output frame is the instantaneous frame at that raw index
      - Input interval always [-0.5, 0.5]
      - Output intervals reflect each 1-frame slice’s coverage within the blur window,
        leaving gaps between.
    """
    N = len(frame_paths)
    total_raw = window_max * base_rate
    assert N >= total_raw, f"Not enough frames for base_rate={base_rate}, need {total_raw}, got {N}"
    start = random.randint(0, N - total_raw)

    # build blur input over the full raw block
    raw_block = frame_paths[start:start + total_raw]
    blur_img = build_blur(raw_block)

    # output sequence: instantaneous frames at each group_start
    seq = []
    group_starts = [start + i * base_rate for i in range(window_max)]
    for gs in group_starts:
        img = np.array(Image.open(frame_paths[gs]).convert('RGB'), dtype=np.uint8)
        seq.append(img)
     # pad blurred frames to output_len
    seq += [seq[-1]] * (output_len - len(seq))

    # compute intervals for each instantaneous frame:
    # each covers [gs, gs+1) over total_raw, normalized to [-0.5, 0.5]
    intervals = []
    for gs in group_starts:
        t0 = (gs - start) / total_raw - 0.5
        t1 = (gs + 1 - start) / total_raw - 0.5
        intervals.append([t0, t1])
    intervals += [intervals[-1]] * (output_len - len(intervals))
    output_intervals = torch.tensor(intervals, dtype=torch.float)

    # input interval
    input_interval = torch.tensor([[-0.5, 0.5]], dtype=torch.float)
    return blur_img, seq, input_interval, output_intervals

