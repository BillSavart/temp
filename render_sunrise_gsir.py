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
from scene import Scene
from pbr import CubemapLight, get_brdf_lut, pbr_shading
from utils.general_utils import safe_state
from arguments import GroupParams, ModelParams, PipelineParams, get_combined_args


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

    # 來源自 relight.py 的 cube_to_dir 對應面
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
        cubemap[s] = dr.texture(latlong_map[None], texcoord[None], filter_mode="linear")[0]

    return cubemap


# ---------------------------
# 主程式：產生日出動畫
# ---------------------------
@torch.no_grad()
def launch(model_path, checkpoint_path, hdri_path, dataset, pipeline):

    print(">>> [DEBUG] Sunrise renderer launching…")
    print("model_path =", model_path)
    print("checkpoint =", checkpoint_path)
    print("hdri =", hdri_path)

    # 讀取 GS-IR 模型
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)

    ckpt = torch.load(checkpoint_path)
    gaussians.restore(ckpt["gaussians"])
    print(">>> [DEBUG] Checkpoint loaded.")

    # 讀 HDR
    hdri_np = read_hdr(hdri_path)
    hdri = torch.from_numpy(hdri_np).cuda().float()
    hdri = hdri / hdri.max()
    print(">>> [DEBUG] HDR loaded:", hdri.shape)

    # 輸出目錄
    out_dir = os.path.join(model_path, "sunrise_anim")
    makedirs(out_dir, exist_ok=True)
    print(">>> [DEBUG] Saving frames to:", out_dir)

    # 動畫參數
    fps = 30
    seconds = 5
    total_frames = fps * seconds

    test_views = scene.getTestCameras()
    if len(test_views) == 0:
        print(">>> [ERROR] No test cameras found! Cannot render.")
        return

    view = test_views[0]

    # PBR LUT
    brdf_lut = get_brdf_lut().cuda()

    canonical_rays = scene.get_canonical_rays()

    # ---------------------------
    # 逐幀渲染
    # ---------------------------
    for i in tqdm(range(total_frames), desc="Rendering sunrise"):

        t = i / (total_frames - 1)
        shift = t * 0.25   # 太陽從左到右升起（0→90º）

        rotated = rotate_latlong(hdri, shift)

        cubemap = CubemapLight(base_res=256).cuda()
        cubemap.base.data = latlong_to_cubemap(rotated, 256)
        cubemap.eval()

        rendering = render(
            viewpoint_camera=view,
            pc=scene.gaussians,
            pipe=pipeline,
            bg_color=torch.tensor([0, 0, 0], device="cuda", dtype=torch.float32),
            inference=True,
            pad_normal=True,
            derive_normal=True,
        )

        H, W = view.image_height, view.image_width
        c2w = torch.inverse(view.world_view_transform.T)

        view_dirs = -(
            (F.normalize(canonical_rays[:, None], p=2, dim=-1) * c2w[None, :3, :3])
            .sum(dim=-1)
            .reshape(H, W, 3)
        )

        albedo = rendering["albedo_map"].permute(1, 2, 0)
        normal = rendering["normal_map"].permute(1, 2, 0)
        roughness = rendering["roughness_map"].permute(1, 2, 0)
        metallic = rendering["metallic_map"].permute(1, 2, 0)

        mask = rendering["normal_mask"].permute(1, 2, 0)

        pbr_img = pbr_shading(
            light=cubemap,
            normals=normal,
            view_dirs=view_dirs,
            mask=mask,
            albedo=albedo,
            roughness=roughness,
            metallic=metallic,
            brdf_lut=brdf_lut,
        )["render_rgb"]

        img = pbr_img.clamp(0, 1).permute(2, 0, 1)
        torchvision.utils.save_image(img, os.path.join(out_dir, f"{i:04d}.png"))

    print(">>> Done! Sunrise frames saved:", out_dir)
    print(f"ffmpeg -framerate 30 -i {out_dir}/%04d.png -c:v libx264 sunrise.mp4")


# ---------------------------
# 程式入口
# ---------------------------
if __name__ == "__main__":
    parser = ArgumentParser(description="Sunrise animation")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--hdri", type=str, required=True)

    args = get_combined_args(parser)

    model_path = os.path.dirname(args.checkpoint)
    safe_state(args.quiet)

    launch(
        model_path=model_path,
        checkpoint_path=args.checkpoint,
        hdri_path=args.hdri,
        dataset=model.extract(args),
        pipeline=pipeline.extract(args),
    )
