"""
PyTorch 可微 PSF Splatting 前向模型

物理基础：García et al. JBO 2026, Eq.(14)
将 3D Gaussian 荧光源投影到多视角 2D 图像
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

logger = logging.getLogger(__name__)


@dataclass
class TissueParams:
    """组织光学参数"""

    mu_a: float
    mu_sp: float
    n: float = 1.37

    @property
    def D(self) -> float:
        return 1.0 / (3.0 * (self.mu_a + self.mu_sp))

    @property
    def mu_eff(self) -> float:
        return np.sqrt(self.mu_a / self.D)

    @property
    def z_b(self) -> float:
        R_eff = 0.493
        A = (1 + R_eff) / (1 - R_eff)
        return 2 * A * self.D


class GaussianSource(nn.Module):
    """
    单个 3D Gaussian 荧光源，参数可优化

    可优化参数：
        center: (3,) 源中心 [x, y, z] (mm)
        log_sigma: (1,) log(σ) 保证正值 (mm)
        log_alpha: (1,) log(α) 保证正值（荧光强度）
    """

    def __init__(
        self, center_init: np.ndarray, sigma_init: float = 1.0, alpha_init: float = 1.0
    ):
        super().__init__()
        self.center = nn.Parameter(torch.tensor(center_init, dtype=torch.float32))
        self.log_sigma = nn.Parameter(
            torch.tensor(np.log(sigma_init), dtype=torch.float32)
        )
        self.log_alpha = nn.Parameter(
            torch.tensor(np.log(alpha_init), dtype=torch.float32)
        )

    @property
    def sigma(self) -> torch.Tensor:
        return torch.exp(self.log_sigma)

    @property
    def alpha(self) -> torch.Tensor:
        return torch.exp(self.log_alpha)

    def get_params(self) -> dict:
        return {
            "center": self.center.detach().cpu().numpy(),
            "sigma": self.sigma.item(),
            "alpha": self.alpha.item(),
        }


class PSFSplattingRenderer(nn.Module):
    """
    可微 PSF Splatting 渲染器

    对每个视角：
    1. 计算 Gaussian 源中心到组织表面的深度 d
    2. 计算源中心在该视角图像上的 2D 投影位置
    3. 用 depth-dependent PSF 参数（σ_PSF, T）渲染 2D 图像
    4. PSF 展宽 = Gaussian 自身 σ² + 散射 PSF σ_PSF²（卷积）
    """

    def __init__(
        self,
        tissue: TissueParams,
        psf_calibration: dict,
        image_size: int = 256,
        pixel_size_mm: float = 0.15,
        vol_size_mm: Tuple[float, float, float] = (30.0, 30.0, 20.0),
    ):
        super().__init__()
        self.mu_eff = tissue.mu_eff
        self.D = tissue.D
        self.z_b = tissue.z_b
        self.image_size = image_size
        self.pixel_size_mm = pixel_size_mm
        self.vol_size_mm = vol_size_mm

        self.psf_k = psf_calibration["k"]
        self.psf_p = psf_calibration["p"]

        coords = (
            torch.arange(image_size, dtype=torch.float32) - image_size / 2
        ) * pixel_size_mm
        grid_x = coords.unsqueeze(0).expand(image_size, -1)
        grid_y = coords.unsqueeze(1).expand(-1, image_size)
        self.register_buffer("grid_x", grid_x)
        self.register_buffer("grid_y", grid_y)

    def compute_psf_params(self, d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算 depth-dependent PSF 参数

        Args:
            d: 源深度 (mm), scalar tensor
        Returns:
            (sigma_psf, T_peak)
        """
        sigma_psf = self.psf_k * torch.pow(d, self.psf_p)

        r1 = d
        r2 = d + 2 * self.z_b
        T_peak = (
            torch.exp(-self.mu_eff * r1) / r1 - torch.exp(-self.mu_eff * r2) / r2
        ) / (4 * np.pi * self.D)
        T_peak = torch.clamp(T_peak, min=1e-20)

        return sigma_psf, T_peak

    def render_view(
        self,
        source: GaussianSource,
        view_matrix: torch.Tensor,
        surface_normal: torch.Tensor,
    ) -> torch.Tensor:
        """
        渲染单个视角的 2D 荧光图像

        Args:
            source: GaussianSource 对象
            view_matrix: (3,3) 旋转矩阵（世界坐标 → 相机坐标）
            surface_normal: (3,) 该视角下的表面法向量（指向相机）

        Returns:
            image: (H, W) 2D 图像
        """
        center_cam = view_matrix @ source.center

        d = torch.dot(source.center, surface_normal)

        if d < 0.05:
            return torch.zeros(
                self.image_size, self.image_size, device=center_cam.device
            )

        d = torch.clamp(d, min=0.1, max=15.0)

        sigma_psf, T_peak = self.compute_psf_params(d)

        sigma_total_sq = source.sigma**2 + sigma_psf**2

        proj_x = center_cam[0]
        proj_y = center_cam[1]

        dx = self.grid_x - proj_x
        dy = self.grid_y - proj_y
        dist_sq = dx**2 + dy**2

        image = source.alpha * T_peak * torch.exp(-dist_sq / (2 * sigma_total_sq))

        return image

    def render_all_views(
        self,
        source: GaussianSource,
        view_matrices: List[torch.Tensor],
        surface_normals: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        渲染所有视角

        Returns:
            images: (N_views, H, W)
        """
        images = []
        for vm, sn in zip(view_matrices, surface_normals):
            img = self.render_view(source, vm, sn)
            images.append(img)
        return torch.stack(images)


def build_turntable_views(
    n_views: int = 7,
    angles_deg: List[float] = None,
    device: torch.device = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    构建 turntable 多视角的旋转矩阵和表面法向量

    假设：
    - 鼠体沿 y 轴放置
    - 转台绕 y 轴旋转
    - 相机在 +z 方向观察（俯视）
    - 旋转角度 θ：绕 y 轴旋转鼠体

    Returns:
        (view_matrices, surface_normals)
    """
    if angles_deg is None:
        angles_deg = [-90, -60, -30, 0, 30, 60, 90]

    if device is None:
        device = torch.device("cpu")

    view_matrices = []
    surface_normals = []

    for angle in angles_deg:
        theta = np.radians(angle)
        R = torch.tensor(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ],
            dtype=torch.float32,
            device=device,
        )

        view_matrices.append(R)
        normal = R @ torch.tensor([0.0, 0.0, 1.0], device=device)
        surface_normals.append(normal)

    return view_matrices, surface_normals


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_renderer_from_config(
    config: dict, tissue_type: str, device: torch.device = None
) -> PSFSplattingRenderer:
    """从配置创建渲染器"""
    if device is None:
        device = torch.device("cpu")

    tissue_params = config["tissue_params"][tissue_type]
    psf_calib = config["psf_calibration"][tissue_type]

    tissue = TissueParams(
        mu_a=tissue_params["mu_a"],
        mu_sp=tissue_params["mu_sp"],
        n=tissue_params["n"],
    )

    renderer = PSFSplattingRenderer(
        tissue=tissue,
        psf_calibration=psf_calib,
        image_size=config["camera"]["image_size"],
        pixel_size_mm=config["camera"]["pixel_size_mm"],
        vol_size_mm=tuple(config["camera"]["vol_size_mm"]),
    )

    if device.type == "cuda":
        renderer = renderer.to(device)

    return renderer
