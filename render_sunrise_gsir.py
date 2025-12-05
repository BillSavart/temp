import os
from argparse import ArgumentParser
from os import makedirs

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

import nvdiffrast.torch as dr

from gaussian_renderer import GaussianModel, render
from pbr import CubemapLight, get_brdf_lut, pbr_shading
from utils.general_utils import safe_state


# ---------------------------
# HDR 讀取
# ---------------------------
def read_hdr(path: str) -> np.ndarray:
    with open(path, "rb") as h:
        buffer_ = np.frombuffer(h.read(), np.uint8)
    bgr = cv2.imdecode(buffer_, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


# ---------------------------
# HDR lat-long 旋轉（水平旋轉）
# ---------------------------
def rotate_latlong(latlong: torch.Tensor, shift: float) -> torch.Tensor:
    W = latlong.shape[1]
    shift_px = int(W * shift)
    return torch.roll(latlong, shifts=shift_px, dims=1)


# ---------------------------
# Lat-long → Cubemap
# ---------------------------
def latlong_to_cubemap(latlong_map: torch.Tensor, res: int = 256) -> torch.Tensor:
    cubemap = torch.zeros(6, res, res, 3, dtype=torch.float32, device="cuda")

    def cube_to_dir_face(s, x, y):
        if s == 0:   rx, ry, rz = torch.ones_like(x), -y, -x
        elif s == 1: rx, ry, rz = -torch.ones_like(x), -y, x
        elif s == 2: rx, ry, rz = x, torch.ones_like(x), y
        elif s == 3: rx, ry, rz = x, -torch.ones_like(x), -y
        elif s == 4: rx, ry, rz = x, -y, torch.ones_like(x)
        else:        rx, ry, rz = -x, -y, -torch.ones_like(x)
        return torch.stack((rx, ry, rz), dim=-1)

    for s in range(6):
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0 + 1.0/res, 1.0 - 1.0/res, res, device="cuda"),
            torch.linspace(-1.0 + 1.0/res, 1.0 - 1.0/res, res, device="cuda"),
            indexing="ij"
        )
        v = F.normalize(cube_to_dir_face(s, gx, gy), p=2, dim=-1)

        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], -1, 1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)

        cubemap[s] = dr.texture(
            latlong_map[None], texcoord[None], filter_mode="linear"
        )[0]

    return cubemap


# ---------------------------
# 主程式：產生日出動畫（無 Scene 版本）
# ---------------------------
@torch.no_grad()
def launch(checkpoint_path, hdri_path, output_dir):

    print(">>> [START] Sunrise minimal renderer")
    checkpoint = torch.load(checkpoint_path)

    # --- 支援 GS-IR 兩種不同的 checkpoint 格式 ---
    if isinstance(checkpoint, dict):
    # 格式1：{"gaussians": {...}, "cubemap": {...}, ...}
        ckpt_gauss = checkpoint["gaussians"]
    elif isinstance(checkpoint, tuple):
    # 格式2：(gaussians, cubemap, irradiance_volumes)
        ckpt_gauss = checkpoint[0]
    else:
        raise TypeError("Unknown checkpoint format:", type(checkpoint))

    # 建立 GaussianModel
    gaussians = GaussianModel(sh_degree=ckpt_gauss["sh_degree"])
    gaussians.restore(ckpt_gauss)

    print(">>> Gaussians loaded. (checkpoint format OK)")

    # HDR
    hdri_np = read_hdr(hdri_path)
    hdri = torch.from_numpy(hdri_np).cuda().float()
    hdri = hdri / hdri.max()
    print(">>> HDR loaded:", hdri.shape)

    makedirs(output_dir, exist_ok=True)

    # 動畫配置
    fps = 30
    seconds = 5
    frames = fps * seconds

    # 建立一個固定 camera（不依賴 dataset）
    H, W = 1080, 1920
    fx = fy = 1200
    cx, cy = W / 2, H / 2

    class FakeCamera:
        pass

    cam = FakeCamera()
    cam.image_height = H
    cam.image_width = W
    cam.world_view_transform = torch.eye(4, device="cuda")
    cam.projection_matrix = torch.tensor([
        [fx, 0, cx, 0],
        [0, fy, cy, 0],
        [0, 0, 1, 0]
    ], device="cuda")

    brdf_lut = get_brdf_lut().cuda()

    canonical_rays = gaussians.get_xyz.detach().unsqueeze(0)  # fake rays placeholder

    # ---------------------------
    # 開始逐幀
    # ---------------------------
    for i in tqdm(range(frames), desc="Sunrise"):

        t = i / (frames - 1)
        shift = t * 0.25

        rotated = rotate_latlong(hdri, shift)
        cubemap = CubemapLight(base_res=256).cuda()
        cubemap.base.data = latlong_to_cubemap(rotated)

        rendering = render(
            viewpoint_camera=cam,
            pc=gaussians,
            pipe=None,
            bg_color=torch.tensor([0,0,0], device="cuda"),
            inference=True,
            pad_normal=True,
            derive_normal=True,
        )

        normal = rendering["normal_map"].permute(1,2,0)
        albedo = rendering["albedo_map"].permute(1,2,0)
        roughness = rendering["roughness_map"].permute(1,2,0)
        metallic = rendering["metallic_map"].permute(1,2,0)

        view_dirs = torch.zeros_like(normal)
        mask = rendering["normal_mask"].permute(1,2,0)

        shaded = pbr_shading(
            light=cubemap,
            normals=normal,
            view_dirs=view_dirs,
            mask=mask,
            albedo=albedo,
            roughness=roughness,
            metallic=metallic,
            brdf_lut=brdf_lut,
        )["render_rgb"]

        img = shaded.clamp(0,1).permute(2,0,1)
        torchvision.utils.save_image(img, os.path.join(output_dir, f"{i:04d}.png"))

    print(">>> Done. Saved to:", output_dir)


# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hdri", required=True)
    parser.add_argument("--out", default="sunrise_anim")
    args = parser.parse_args()

    safe_state(False)
    launch(args.checkpoint, args.hdri, args.out)
