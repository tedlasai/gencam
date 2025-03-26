import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
import sys
sys.path.append('.')
from training.utils import instantiate_from_config



class PositionalEncoding(nn.Module):
    def __init__(self, channels_per_dim, n_dims=3):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super().__init__()

        assert channels_per_dim % 2 == 0
        n_ch = channels_per_dim // 2
        freqs = 2. ** torch.linspace(0., n_ch - 1, steps=n_ch)

        self.register_buffer("freqs", freqs)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (b, h, w, [x, y, z]), x, y, z should be [0, 1]
        :return: Positional Encoding Matrix of size (b, h, w, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        pos_xyz = tensor[..., None] * self.freqs[None, None, None, None]

        pos_emb = torch.cat([torch.sin(pos_xyz), torch.cos(pos_xyz)], dim=-1)
        pos_emb = einops.rearrange(pos_emb, "b h w c f -> b h w (c f)")

        return pos_emb
    

def get_motion(verts: torch.Tensor, temporal_downscale: int):
    if verts.shape[1] % temporal_downscale == 1:
        return torch.cat([torch.zeros_like(verts[:, :1, :, :2]), get_motion(verts[:, 1:], temporal_downscale)], dim=1)

    # verts (b t n v)
    verts = einops.rearrange(verts[..., :2], 'b (t s) n v -> b t s n v', s=temporal_downscale, )
    motion = torch.zeros_like(verts)
    for i in range(temporal_downscale - 1):
        motion[:, :, i] = verts[:, :, temporal_downscale-1] - verts[:, :, i]

    motion = einops.rearrange(motion, 'b t s n v -> b (t s) n v')
    
    return motion


def temporal_subsample(map: torch.Tensor, temporal_downscale: int, mode="select"):
    # map (b t h w c)
    # mode ["select", "flatten"]
    if map.shape[1] % temporal_downscale == 1:
        if mode == "select":
            sub_first = map[:, :1]
            return torch.cat([sub_first, temporal_subsample(map[:, 1:], temporal_downscale)], dim=1)
        elif mode == "flatten":
            sub_first = map[:, :1].repeat(1, 1, 1, 1, temporal_downscale)
            sub_next = temporal_subsample(map[:, 1:], temporal_downscale, mode=mode)
            return torch.cat([sub_first, sub_next], dim=1)
        else:
            print("No such mode:", mode)
            assert False

    submap = einops.rearrange(map, 'b (t s) h w c -> b t s h w c', s=temporal_downscale)
    if mode == "select":
        return submap[:, :, -1]
    elif mode == "flatten":
        return einops.rearrange(submap, 'b t s h w c -> b t h w (s c)')
    else:
        print("No such mode:", mode)
        assert False


class CAPConditioning(nn.Module):
    def __init__(
        self,
        image_size=64,
        temporal_downscale=4,
        positional_channels=48,
        positional_multiplier=1.,
        super_resolution=2,
        use_ray_directions=False,
        use_displacements=True,
        use_border_mask=True,
        std_displacement=0.0104,
        std_motion=0.1,
        audio_model_config=None,
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.temporal_downscale=temporal_downscale
        assert super_resolution >=1 and super_resolution % 1 == 0
        self.super_resolution = super_resolution
        self.positional_channels = positional_channels
        self.positional_multiplier = positional_multiplier
        self.use_ray_directions = use_ray_directions
        self.use_displacements = use_displacements
        self.std_displacement = std_displacement
        self.use_border_mask = use_border_mask
        self.std_motion = std_motion

        assert positional_channels % 3 == 0
        self.pos_encoding = PositionalEncoding(positional_channels // 3)
        self.renderer = None

        if audio_model_config is not None:
            self.audio_model = instantiate_from_config(audio_model_config)
        else:
            self.audio_model = None

    # TODO: No grad!
    def forward(self, control, conditioned=True):
        verts = control["verts"]
        offsets = control["offsets"]
        is_ref = control["is_reference"]  # b t
        b_, t_ = verts.shape[:2]

        gt_z = None
        if "z" in control:
            gt_z = control["z"]

        img_size = self.image_size

        if not conditioned:
            total_channels = self.positional_channels + 1 # positional and ref mask
            if self.use_border_mask:
                total_channels += 1
            if self.use_ray_directions:
                total_channels += 3
            if self.use_displacements: 
                total_channels += 3
            if self.audio_model is not None:
                total_channels += self.audio_model.audio_features + 1
            pos_enc = torch.zeros((b_, t_, img_size, img_size, total_channels), device=verts.device)
            if gt_z is not None:
                gt_z = gt_z * 0.
        else:
            with torch.no_grad():
                motion = get_motion(verts, self.temporal_downscale)
                motion = einops.rearrange(motion, 'b t n v -> (b t) n v') / self.std_motion
                verts = einops.rearrange(verts, 'b t n v -> (b t) n v')
                offsets = einops.rearrange(offsets, 'b t n v -> (b t) n v') / self.std_displacement
                offsets = torch.cat([motion, offsets], dim=-1)

                uv_img, mask = self.renderer.render(
                    verts, 
                    (img_size * self.super_resolution, img_size * self.super_resolution),
                    prop=offsets if self.use_displacements else None,
                )

                if self.use_displacements:
                    # extract last three channels which are offsets
                    uv_img, motion, offsets = uv_img.split([3, 2, 3], dim=-1)

                pos_enc = self.pos_encoding(uv_img * self.positional_multiplier)
                pos_enc = pos_enc * mask
                pos_enc = einops.rearrange(pos_enc, '(b t) h w c -> b t h w c', b=b_)
                motion = motion * mask
                motion = einops.rearrange(motion, '(b t) h w c -> b t h w c', b=b_)
                offsets = offsets * mask
                offsets = einops.rearrange(offsets, '(b t) h w c -> b t h w c', b=b_)

                pos_enc = temporal_subsample(pos_enc, self.temporal_downscale)

                condition = pos_enc

                if self.use_displacements:
                    offsets = temporal_subsample(offsets, self.temporal_downscale)
                    motion = temporal_subsample(motion, self.temporal_downscale, "flatten")
                    condition = torch.cat([condition, motion, offsets], dim=-1)

                # after supersample downsample again
                condition = einops.rearrange(condition, 'b t h w c -> (b t) c h w')
                condition = F.interpolate(condition, (img_size, img_size), mode="area")
                condition = einops.rearrange(condition, '(b t) c h w -> b t h w c', b=b_)

                if self.use_ray_directions:
                    assert False
                    rays = control["rays"]
                    rays = einops.rearrange(rays, 'b t c h w -> b t h w c')
                    condition = torch.cat([condition, rays], dim=-1)

                # concat ref mask
                ref_mask = is_ref[..., None, None, None]
                ref_mask = ref_mask.repeat(1, 1, self.image_size, self.image_size, 1)
                ref_mask = temporal_subsample(ref_mask, self.temporal_downscale)
                condition = torch.cat([condition, ref_mask], dim=-1)

                if self.use_border_mask:
                    loss_mask = einops.rearrange(control["mask"], 'b t c h w -> (b t) c h w')
                    loss_mask = F.interpolate(loss_mask, (self.image_size, self.image_size), mode="area")
                    loss_mask = einops.rearrange(loss_mask, '(b t) c h w -> b t h w c', b=b_)
                    loss_mask = temporal_subsample(loss_mask, self.temporal_downscale)
                    condition = torch.cat([condition, loss_mask], dim=-1)

            # concat audio if available: caution this has to have grad!
            if self.audio_model is not None:
                audio_chunks = control["audio_chunks"]
                audio_chunks = einops.rearrange(audio_chunks, 'b t a -> (b t) a')
                audio_cond = self.audio_model(audio_chunks)
                audio_cond = einops.rearrange(audio_cond, '(b t) c h w -> b t h w c', b=b_)
                audio_mask = control["audio_mask"]
                audio_cond = einops.einsum(audio_cond, audio_mask, 'b t ..., b t -> b t ...')
                audio_mask = audio_mask[..., None, None, None].repeat(1, 1, self.image_size, self.image_size, 1)
                condition = torch.cat([condition, audio_cond, audio_mask], dim=-1)

        return {
            "condition": condition,
            "gt_z": gt_z,
            "is_ref": is_ref,
        }

    def get_vis(self, enc):
        visualizations = {}

        n_pos = self.positional_channels // 3

        counter = 0

        pos_enc = enc[..., 0:self.positional_channels]

        # for i in [n_pos-1]:
        for i in range(n_pos-2, n_pos):
            visualizations[f"pos_{i}"] = pos_enc[..., [i, i + n_pos, i + n_pos * 2]]

        counter = self.positional_channels

        if self.use_displacements:
            motion = enc[..., counter:counter+2*self.temporal_downscale]
            for i in range(self.temporal_downscale):
                motion = enc[..., counter:counter+2]
                motion = torch.cat([motion, torch.zeros_like(motion[..., :1])], dim=-1)
                visualizations[f"motion_{i}"] = motion
                counter += 2
            visualizations["disp"] = enc[..., counter:counter+3]
            counter += 3

        if self.use_ray_directions:
            visualizations["ray"] = enc[..., counter:counter+3]
            counter += 3

        visualizations["ref_mask"] = enc[..., [counter] * 3]
        counter += 1

        if self.use_border_mask:
            visualizations["loss_mask"] = enc[..., [counter] * 3]
            counter += 1

        if self.audio_model is not None:
            visualizations["audio_cond"] = enc[..., counter:counter+3]
            counter += self.audio_model.audio_features
            visualizations["audio_mask"] = enc[..., [counter] * 3]
            counter += 1

        return visualizations
