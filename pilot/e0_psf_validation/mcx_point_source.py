"""用 MCX 模拟荧光点源在组织中的 emission 传播

复用 FMT-SimGen 的 MCX 模块，但构建简单几何体而非 Digimouse atlas。
"""

import argparse
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def create_homogeneous_volume(
    tissue_mu_a: float,
    tissue_mu_sp: float,
    tissue_g: float,
    tissue_n: float,
    vol_size_mm: Tuple[float, float, float] = (30.0, 30.0, 20.0),
    voxel_size_mm: float = 0.1,
) -> Dict:
    """创建匀质半无限介质的 MCX 体积

    z=0 为组织表面，z>0 向组织内部
    表面以上（z<0）= 空气

    Args:
        tissue_mu_a: 吸收系数 (mm^-1)
        tissue_mu_sp: 约化散射系数 (mm^-1)
        tissue_g: 各向异性因子
        tissue_n: 折射率
        vol_size_mm: 体积尺寸 (x, y, z) mm
        voxel_size_mm: 体素大小 mm

    Returns:
        dict with 'volume' (3D uint8), 'props' (材料列表),
        'voxel_size', 'vol_shape'
    """
    nx = int(round(vol_size_mm[0] / voxel_size_mm))
    ny = int(round(vol_size_mm[1] / voxel_size_mm))
    nz = int(round(vol_size_mm[2] / voxel_size_mm))

    # 体积: 0=空气, 1=组织
    # z=0 为表面，所以 z>0 部分全是组织
    volume = np.zeros((nz, ny, nx), dtype=np.uint8)
    volume[:, :, :] = 1  # 全部为组织

    # 材料列表
    # mus = mu_sp / (1 - g)
    tissue_mus = tissue_mu_sp / (1 - tissue_g) if tissue_g < 1.0 else 0.0
    props = [
        {"mua": 0.0, "mus": 0.0, "g": 1.0, "n": 1.0},  # 0: 空气
        {
            "mua": tissue_mu_a,
            "mus": tissue_mus,
            "g": tissue_g,
            "n": tissue_n,
        },  # 1: 组织
    ]

    return {
        "volume": volume,
        "props": props,
        "voxel_size": voxel_size_mm,
        "vol_shape": (nz, ny, nx),
    }


def create_bilayer_volume(
    layer1_thickness_mm: float,
    layer1_props: Dict,
    layer2_props: Dict,
    vol_size_mm: Tuple[float, float, float] = (30.0, 30.0, 20.0),
    voxel_size_mm: float = 0.1,
) -> Dict:
    """创建双层介质的 MCX 体积

    Args:
        layer1_thickness_mm: 表层厚度 (mm)
        layer1_props: 表层参数 {mu_a, mu_sp, g, n}
        layer2_props: 底层参数 {mu_a, mu_sp, g, n}
        vol_size_mm: 体积尺寸 (x, y, z) mm
        voxel_size_mm: 体素大小 mm

    Returns:
        dict with volume info
    """
    nx = int(round(vol_size_mm[0] / voxel_size_mm))
    ny = int(round(vol_size_mm[1] / voxel_size_mm))
    nz = int(round(vol_size_mm[2] / voxel_size_mm))
    layer1_nz = int(round(layer1_thickness_mm / voxel_size_mm))

    # 体积: 0=空气, 1=表层, 2=底层
    volume = np.zeros((nz, ny, nx), dtype=np.uint8)
    volume[:layer1_nz, :, :] = 1  # 表层
    volume[layer1_nz:, :, :] = 2  # 底层

    # 材料列表
    layer1_mus = (
        layer1_props["mu_sp"] / (1 - layer1_props["g"])
        if layer1_props["g"] < 1.0
        else 0.0
    )
    layer2_mus = (
        layer2_props["mu_sp"] / (1 - layer2_props["g"])
        if layer2_props["g"] < 1.0
        else 0.0
    )

    props = [
        {"mua": 0.0, "mus": 0.0, "g": 1.0, "n": 1.0},  # 0: 空气
        {
            "mua": layer1_props["mu_a"],
            "mus": layer1_mus,
            "g": layer1_props["g"],
            "n": layer1_props["n"],
        },  # 1: 表层
        {
            "mua": layer2_props["mu_a"],
            "mus": layer2_mus,
            "g": layer2_props["g"],
            "n": layer2_props["n"],
        },  # 2: 底层
    ]

    return {
        "volume": volume,
        "props": props,
        "voxel_size": voxel_size_mm,
        "vol_shape": (nz, ny, nx),
    }


def generate_mcx_config_json(
    volume_dict: Dict,
    source_depth_mm: float,
    source_pos_xy: Tuple[float, float] = None,
    n_photons: int = 100_000_000,
    session_id: str = "point_source",
    output_dir: Path = None,
) -> Tuple[Path, Path]:
    """生成 MCX JSON 配置文件

    Args:
        volume_dict: 体积信息字典
        source_depth_mm: 点源深度 (mm)
        source_pos_xy: 点源 xy 坐标 (mm)，默认为体积中心
        n_photons: 光子数
        session_id: 会话 ID
        output_dir: 输出目录

    Returns:
        (json_path, volume_bin_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nz, ny, nx = volume_dict["vol_shape"]
    voxel_size = volume_dict["voxel_size"]

    # 点源位置（体素坐标）
    if source_pos_xy is None:
        source_x = nx / 2.0
        source_y = ny / 2.0
    else:
        source_x = source_pos_xy[0] / voxel_size
        source_y = source_pos_xy[1] / voxel_size

    # 深度转体素坐标（z=0 为表面）
    source_z = source_depth_mm / voxel_size

    # 保存体积二进制
    volume_bin_path = output_dir / "volume.bin"
    volume_dict["volume"].tofile(volume_bin_path)

    # MCX JSON 配置
    config = {
        "Domain": {
            "VolumeFile": "volume.bin",
            "Dim": [nz, ny, nx],
            "OriginType": 1,  # 原点在角落
            "LengthUnit": voxel_size,
            "Media": volume_dict["props"],
        },
        "Session": {
            "Photons": n_photons,
            "RNGSeed": 12345,
            "ID": session_id,
        },
        "Forward": {
            "T0": 0.0,
            "T1": 5.0e-9,
            "DT": 5.0e-9,
        },
        "Optode": {
            "Source": {
                "Pos": [float(source_x), float(source_y), float(source_z)],
                "Type": "isotropic",  # 各向同性点源，模拟荧光发射
            }
        },
    }

    json_path = output_dir / f"{session_id}.json"
    with open(json_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Generated MCX config: {json_path}")
    logger.info(
        f"  Volume: {nx}x{ny}x{nz} voxels = "
        f"{nx * voxel_size:.1f}x{ny * voxel_size:.1f}x{nz * voxel_size:.1f} mm"
    )
    logger.info(f"  Source depth: {source_depth_mm} mm (voxel z={source_z:.1f})")
    logger.info(f"  Photons: {n_photons:,}")

    return json_path, volume_bin_path


def run_mcx_simulation(
    json_path: Path,
    mcx_exec: str = "mcx",
    timeout: int = 600,
) -> Path:
    """运行 MCX 仿真

    Args:
        json_path: JSON 配置文件路径
        mcx_exec: MCX 可执行文件名
        timeout: 超时时间（秒）

    Returns:
        输出 .jnii 文件路径
    """
    work_dir = json_path.parent
    session_id = json_path.stem
    output_jnii = work_dir / f"{session_id}.jnii"

    if output_jnii.exists():
        logger.info(f"Skipping: {output_jnii} already exists")
        return output_jnii

    logger.info(f"Running MCX: {mcx_exec} -f {json_path.name}")
    try:
        result = subprocess.run(
            [mcx_exec, "-f", json_path.name],
            cwd=work_dir,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.stdout:
            logger.debug(f"MCX stdout: {result.stdout.strip()}")
        if result.stderr:
            logger.debug(f"MCX stderr: {result.stderr.strip()}")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"MCX timed out after {timeout}s")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MCX failed: {e.stderr or e.stdout}")

    if not output_jnii.exists():
        raise RuntimeError(f"MCX output not found: {output_jnii}")

    logger.info(f"MCX completed: {output_jnii}")
    return output_jnii


def load_mcx_fluence(jnii_path: Path) -> np.ndarray:
    """加载 MCX fluence 输出

    Args:
        jnii_path: .jnii 文件路径

    Returns:
        fluence 3D 数组 (nz, ny, nx)
    """
    import jdata as jd

    data = jd.loadjd(str(jnii_path))
    if isinstance(data, dict):
        fluence = data.get("NIFTIData", data.get("data", None))
        if fluence is None:
            raise KeyError(f"Could not find fluence data in {jnii_path}")
    else:
        fluence = data

    return np.asarray(fluence, dtype=np.float64)


def extract_surface_fluence(fluence: np.ndarray, surface_z: int = 0) -> np.ndarray:
    """提取表面 fluence

    Args:
        fluence: 3D fluence 数组 (nz, ny, nx)
        surface_z: 表面 z 索引（默认 0）

    Returns:
        2D surface image (ny, nx)
    """
    return fluence[surface_z, :, :]


def extract_radial_profile(
    surface_image: np.ndarray,
    center_xy: Tuple[int, int],
    voxel_size_mm: float = 0.1,
    rho_max_mm: float = 15.0,
    n_bins: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """从 2D surface image 提取径向平均分布

    对以 center_xy 为圆心的同心环做 azimuthal average

    Args:
        surface_image: 2D 表面 fluence (ny, nx)
        center_xy: 点源正上方的像素坐标 (x, y)
        voxel_size_mm: 体素大小 mm
        rho_max_mm: 最大径向距离 mm
        n_bins: 采样点数

    Returns:
        (rho_mm, intensity): 径向距离和平均强度
    """
    ny, nx = surface_image.shape
    cx, cy = center_xy

    # 创建距离网格
    y_indices, x_indices = np.ogrid[:ny, :nx]
    dist = np.sqrt((x_indices - cx) ** 2 + (y_indices - cy) ** 2)
    dist_mm = dist * voxel_size_mm

    # 径向平均
    rho_mm = np.linspace(0, rho_max_mm, n_bins)
    intensity = np.zeros(n_bins)

    for i, r in enumerate(rho_mm):
        if i == 0:
            # 第一个 bin: 包含 ρ=0 附近的体素
            mask = dist_mm <= rho_mm[1] / 2
        else:
            # 中间 bin: 半径 r ± delta/2
            delta = rho_mm[1] - rho_mm[0]
            mask = (dist_mm >= r - delta / 2) & (dist_mm < r + delta / 2)

        if mask.sum() > 0:
            intensity[i] = surface_image[mask].mean()
        else:
            # 如果没有体素，使用最近的非零值
            intensity[i] = intensity[i - 1] if i > 0 else 0

    return rho_mm, intensity


def run_point_source_mcx(
    config_id: str,
    tissue_type: str,
    depth_mm: float,
    tissue_params: Dict,
    output_dir: Path,
    n_photons: int = 100_000_000,
    voxel_size_mm: float = 0.1,
    vol_size_mm: Tuple[float, float, float] = (30.0, 30.0, 20.0),
    mcx_exec: str = "mcx",
) -> Dict:
    """运行单个点源 MCX 仿真

    Args:
        config_id: 配置 ID (e.g., "C01")
        tissue_type: 组织类型 ("muscle", "liver", "bilayer")
        depth_mm: 点源深度 mm
        tissue_params: 组织参数配置 (from config.yaml)
        output_dir: 输出目录
        n_photons: 光子数
        voxel_size_mm: 体素大小 mm
        vol_size_mm: 体积尺寸 mm
        mcx_exec: MCX 可执行文件

    Returns:
        结果字典
    """
    output_dir = Path(output_dir)
    sample_dir = output_dir / config_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running MCX for {config_id}: {tissue_type} @ {depth_mm}mm")

    # 创建体积
    if tissue_type in ["muscle", "soft_tissue"]:
        volume_dict = create_homogeneous_volume(
            tissue_mu_a=tissue_params["muscle"]["mu_a"],
            tissue_mu_sp=tissue_params["muscle"]["mu_sp"],
            tissue_g=tissue_params["muscle"]["g"],
            tissue_n=tissue_params["muscle"]["n"],
            vol_size_mm=vol_size_mm,
            voxel_size_mm=voxel_size_mm,
        )
    elif tissue_type == "liver":
        volume_dict = create_homogeneous_volume(
            tissue_mu_a=tissue_params["liver"]["mu_a"],
            tissue_mu_sp=tissue_params["liver"]["mu_sp"],
            tissue_g=tissue_params["liver"]["g"],
            tissue_n=tissue_params["liver"]["n"],
            vol_size_mm=vol_size_mm,
            voxel_size_mm=voxel_size_mm,
        )
    elif tissue_type == "bilayer":
        volume_dict = create_bilayer_volume(
            layer1_thickness_mm=1.0,  # 1mm skin
            layer1_props=tissue_params["muscle"],  # skin = soft tissue
            layer2_props=tissue_params["liver"],  # liver
            vol_size_mm=vol_size_mm,
            voxel_size_mm=voxel_size_mm,
        )
    else:
        raise ValueError(f"Unknown tissue type: {tissue_type}")

    # 生成 MCX 配置
    json_path, _ = generate_mcx_config_json(
        volume_dict=volume_dict,
        source_depth_mm=depth_mm,
        n_photons=n_photons,
        session_id=config_id,
        output_dir=sample_dir,
    )

    # 运行 MCX
    jnii_path = run_mcx_simulation(json_path, mcx_exec=mcx_exec)

    # 加载 fluence
    fluence = load_mcx_fluence(jnii_path)
    logger.info(f"  Fluence shape: {fluence.shape}, sum={fluence.sum():.2e}")

    # 提取表面 fluence
    surface = extract_surface_fluence(fluence, surface_z=0)

    # 提取径向分布
    nz, ny, nx = volume_dict["vol_shape"]
    center_xy = (nx // 2, ny // 2)
    rho_mm, intensity = extract_radial_profile(
        surface_image=surface,
        center_xy=center_xy,
        voxel_size_mm=voxel_size_mm,
        rho_max_mm=15.0,
        n_bins=500,
    )

    # 保存结果
    npz_path = output_dir / f"{config_id}_mcx.npz"
    np.savez(
        npz_path,
        rho=rho_mm,
        intensity=intensity,
        surface_image=surface,
        fluence_shape=fluence.shape,
        config_id=config_id,
        depth_mm=depth_mm,
        tissue_type=tissue_type,
    )
    logger.info(f"  Saved: {npz_path}")

    return {
        "config_id": config_id,
        "depth_mm": depth_mm,
        "tissue_type": tissue_type,
        "rho_mm": rho_mm,
        "intensity": intensity,
        "surface_image": surface,
    }


def run_all_mcx(config_path: str, output_dir: str, mcx_exec: str = "mcx") -> Dict:
    """运行所有配置的 MCX 仿真

    Args:
        config_path: 实验配置文件路径
        output_dir: 输出目录
        mcx_exec: MCX 可执行文件

    Returns:
        所有配置的结果字典
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    sim_params = config.get("simulation_params", {})
    n_photons = sim_params.get("n_photons", 100_000_000)
    voxel_size = sim_params.get("voxel_size_mm", 0.1)

    for cfg in config["configs"]:
        config_id = cfg["id"]
        depth_mm = cfg["depth_mm"]
        tissue_type = cfg["tissue_type"]

        try:
            result = run_point_source_mcx(
                config_id=config_id,
                tissue_type=tissue_type,
                depth_mm=depth_mm,
                tissue_params=config["tissue_params"],
                output_dir=output_dir,
                n_photons=n_photons,
                voxel_size_mm=voxel_size,
                mcx_exec=mcx_exec,
            )
            results[config_id] = result
        except Exception as e:
            logger.error(f"Failed {config_id}: {e}")
            results[config_id] = {"error": str(e)}

    # 保存汇总
    summary_path = output_dir / "mcx_summary.json"
    summary = {
        k: {
            "config_id": v.get("config_id"),
            "depth_mm": v.get("depth_mm"),
            "tissue_type": v.get("tissue_type"),
            "error": v.get("error"),
        }
        for k, v in results.items()
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run MCX point source simulations")
    parser.add_argument(
        "--config",
        default="pilot/e0_psf_validation/config.yaml",
        help="Experiment matrix config file",
    )
    parser.add_argument(
        "--output",
        default="pilot/e0_psf_validation/results/profiles/",
        help="Output directory",
    )
    parser.add_argument(
        "--mcx",
        default="mcx",
        help="MCX executable (mcx or mcxcl)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    run_all_mcx(args.config, args.output, mcx_exec=args.mcx)
    logger.info("MCX simulations complete!")


if __name__ == "__main__":
    main()
