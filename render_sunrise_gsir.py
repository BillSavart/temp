import os
from argparse import ArgumentParser
from os import makedirs

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchvision

from scene import Scene
from gaussian_renderer import GaussianModel, render
from pbr import CubemapLight, get_brdf_lut, pbr_shading
from utils.general_utils import safe_state
from arguments import GroupParams, ModelParams, PipelineParams, get_combined_args

import nvdiffrast.torch as dr


def read_hdr(path: str) -> np.ndarray:
    """Reads an HDR map from disk."""
    with open(path, "rb") as h:
        buffer_ = np.frombuffer(h.read(), np.uint8)
    bgr = cv2.imdecode(buffer_, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def rotate_latlong(latlong: torch.Tensor, shift: float) -> torch.Tensor:
    """
    Rotate a latlong HDR horizontally.
    shift in [0,1], meaning 0~360 degrees.
    """
    W = latlong.shape[1]
    shift_px = int(W * shift)
    return torch.roll(latlong, shifts=shift_px, dims=1)


def latlong_to_cubemap(latlong_map: torch.Tensor, res: int = 256) -> torch.Tensor:
    cubemap = torch.zeros(6, res, res, 3, dtype=torch.float32, device="cuda")
    for s in range(6):
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
            torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
            indexing="ij",
        )
        if s == 0:
            rx, ry, rz = torch.ones_like(x), -y, -x
        elif s == 1:
            rx, ry, rz = -torch.ones_like(x), -y, x
        elif s == 2:
            rx, ry, rz = x, torch.ones_like(x), y
        elif s == 3:
            rx, ry, rz = x, -y, -y
        elif s == 4:
            rx, ry, rz = x, -y, torch.ones_like(x)
        elif s == 5:
            rx, ry, rz = -x, -y, -torch.ones_like(x)

        v = F.normalize(torch.stack((rx, ry, rz), dim=-1), p=2, dim=-1)

        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)

        cubemap[s] = dr.texture(latlong_map[None, ...], texcoord[None, ...], filter_mode="linear")[0]
    return cubemap


@torch.no_grad()
def launch(model_path, checkpoint, hdri_path, dataset, pipeline, fps=30, seconds=5):

    # Load scene + gaussians
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)

    # Load HDR
    print(f"Loading HDR: {hdri_path}")
    hdri_np = read_hdr(hdri_path)
    hdri = torch.from_numpy(hdri_np).cuda().float()

    hdri = hdri / hdri.max()  # normalize

    # Load checkpoint
    ckpt = torch.load(checkpoint)
    if isinstance(ckpt, dict):
        gaussians.restore(ckpt["gaussians"])
    else:
        gaussians.restore(ckpt)

    # Create output folder
    out_dir = os.path.join(model_path, "sunrise_anim")
    makedirs(out_dir, exist_ok=True)

    # Animation parameters
    total_frames = fps * seconds

    print("Generating sunrise frames...")
    for i in tqdm(range(total_frames)):
        t = i / (total_frames - 1)   # 0 → 1
        shift = t * 0.25             # rotate 0 → 90 degrees

        rotated = rotate_latlong(hdri, shift)
        cubemap = CubemapLight(base_res=256).cuda()
        cubemap.base.data = latlong_to_cubemap(rotated, 256)
        cubemap.eval()

        # Rendering first test camera only
        view = scene.getTestCameras()[0]

        rendering_result = render(
            viewpoint_camera=view,
            pc=scene.gaussians,
            pipe=pipeline,
            bg_color=torch.tensor([0, 0, 0], device="cuda", dtype=torch.float32),
            inference=True,
            pad_normal=True,
            derive_normal=True,
        )

        # PBR shading
        brdf_lut = get_brdf_lut().cuda()

        albedo = rendering_result["albedo_map"].permute(1, 2, 0)
        normal = rendering_result["normal_map"].permute(1, 2, 0)
        roughness = rendering_result["roughness_map"].permute(1, 2, 0)
        metallic = rendering_result["metallic_map"].permute(1, 2, 0)

        H, W = view.image_height, view.image_width
        c2w = torch.inverse(view.world_view_transform.T)
        canonical_rays = scene.get_canonical_rays()
        view_dirs = -(
            (F.normalize(canonical_rays[:, None, :], p=2, dim=-1) * c2w[None, :3, :3])
            .sum(dim=-1)
            .reshape(H, W, 3)
        )

        pbr_result = pbr_shading(
            light=cubemap,
            normals=normal,
            view_dirs=view_dirs,
            mask=rendering_result["normal_mask"].permute(1, 2, 0),
            albedo=albedo,
            roughness=roughness,
            metallic=metallic,
            brdf_lut=brdf_lut,
        )

        img = pbr_result["render_rgb"].clamp(0, 1).permute(2, 0, 1)

        torchvision.utils.save_image(img, os.path.join(out_dir, f"{i:04d}.png"))

    print("Done! Frames saved to:", out_dir)
    print("Use this to convert to video:")
    print(f"ffmpeg -framerate {fps} -i {out_dir}/%04d.png -c:v libx264 sunrise.mp4")
