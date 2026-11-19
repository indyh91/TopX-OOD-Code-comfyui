"""ComfyUI adapter for the unmodified TopX inference CLI.

The worker is a separate Python process. This avoids collisions with other
custom nodes that use a top-level package named src, and isolates TopX's RNG,
torch configuration, and optional LLM2Vec model from ComfyUI.
"""
from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import uuid
from pathlib import Path

import folder_paths

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "scripts" / "generate.py"
DEMO_ROOT = ROOT / "demo"
AXES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")


def demo_catalog() -> dict[str, dict]:
    """Read the bundled sample metadata, never synthesize captions or embeddings."""
    result = {}
    for path in sorted(DEMO_ROOT.glob("*/prompts.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for index, sample in enumerate(data.get("prompts", []), start=1):
            label = f"{path.parent.name} / {index}"
            result[label] = {
                "skeleton": path.parent / "skeleton.npz",
                "text_embedding": path.parent / sample["text_emb"],
                "joint_embedding": path.parent / "joint_sem.npy",
                "caption": sample["text"],
                "frames": int(sample["frames"]),
                "seed": int(sample["seed"]),
            }
    return result


def checkpoint_path(raw: str) -> Path:
    """Use ComfyUI model storage first, then the upstream weights directory."""
    if raw.strip():
        path = Path(raw.strip()).expanduser()
    else:
        preferred = Path(folder_paths.models_dir) / "topx" / "topx.pt"
        legacy = ROOT / "weights" / "topx.pt"
        path = preferred if preferred.is_file() or not legacy.is_file() else legacy
    if not path.is_file():
        raise FileNotFoundError(
            f"TopX checkpoint not found: {path}. Download Tevior/TopX-OOD "
            "topx.pt into ComfyUI/models/topx/topx.pt (or set checkpoint_path)."
        )
    return path.resolve()


def validate_inputs(
    skeleton_path: str,
    text_source: str,
    prompt: str,
    text_embedding_path: str,
    joint_embedding_path: str,
) -> tuple[Path, Path | None, Path | None]:
    sk = Path(skeleton_path.strip()).expanduser()
    if not sk.is_file() or sk.suffix.lower() not in (".bvh", ".npz"):
        raise ValueError("TopX requires an existing .bvh or TopX skeleton .npz file; GLB is not supported.")
    if text_source not in ("precomputed", "llm2vec"):
        raise ValueError(f"Unknown text source: {text_source!r}")
    text_emb = Path(text_embedding_path.strip()).expanduser() if text_embedding_path.strip() else None
    joint_emb = Path(joint_embedding_path.strip()).expanduser() if joint_embedding_path.strip() else None
    for label, path in (("text_embedding_path", text_emb), ("joint_embedding_path", joint_emb)):
        if path is not None and (not path.is_file() or path.suffix.lower() != ".npy"):
            raise ValueError(f"{label} must point to an existing .npy file: {path}")
    if text_source == "precomputed":
        if prompt.strip():
            raise ValueError(
                "A prompt would be ignored when using precomputed embeddings. "
                "Clear prompt or select llm2vec; TopX cannot use a ComfyUI CLIP embedding."
            )
        if text_emb is None or joint_emb is None:
            raise ValueError(
                "Precomputed mode requires both text_embedding_path [4096] and "
                "joint_embedding_path [joints,4096]. Use TopX Demo Inputs to test without LLM2Vec."
            )
    else:
        if not prompt.strip():
            raise ValueError("llm2vec mode requires a nonempty prompt.")
        if text_emb is not None:
            raise ValueError("llm2vec mode cannot also take text_embedding_path.")
    return sk.resolve(), text_emb.resolve() if text_emb else None, joint_emb.resolve() if joint_emb else None


def safe_stem(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_-")[:72]
    return stem or "topx"


def build_command(
    checkpoint: Path,
    skeleton: Path,
    output_dir: Path,
    name: str,
    text_source: str,
    prompt: str,
    text_emb: Path | None,
    joint_emb: Path | None,
    frames: int,
    seed: int,
    steps: int,
    cfg_text: float,
    up: str,
    forward: str,
    rest_frame: int,
    keep_rest_rotations: bool,
    allow_axis_mismatch: bool,
    tf32: bool,
    device: str,
) -> list[str]:
    if up not in AXES or forward not in AXES or up[-1] == forward[-1]:
        raise ValueError("Up and forward must be different orthogonal axes.")
    if not (1 <= int(frames) <= 240):
        raise ValueError("frames must be between 1 and 240.")
    if not (1 <= int(steps) <= 250):
        raise ValueError("steps must be between 1 and 250.")
    if not math.isfinite(float(cfg_text)) or not 0 <= cfg_text <= 20:
        raise ValueError("cfg_text must be finite and between 0 and 20.")
    if skeleton.suffix.lower() == ".npz" and not keep_rest_rotations:
        # This is valid for new NPZ rigs; leave the choice explicit.
        pass
    if device not in ("cuda", "cpu"):
        raise ValueError("device must be cuda or cpu.")
    cmd = [
        sys.executable, "-u", str(SCRIPT), "--ckpt", str(checkpoint),
        "--skeleton", str(skeleton), "--up", up, f"--forward={forward}",
        "--rest_frame", str(rest_frame), "--frames", str(frames),
        "--seed", str(seed), "--steps", str(steps),
        "--cfg_text", str(cfg_text), "--tf32", str(int(tf32)),
        "--device", device, "--out", str(output_dir), "--name", name,
    ]
    if text_source == "precomputed":
        cmd.extend(("--text_emb", str(text_emb)))
    else:
        cmd.extend(("--text", prompt.strip()))
    if joint_emb is not None:
        cmd.extend(("--joint_sem", str(joint_emb)))
    if keep_rest_rotations:
        cmd.append("--keep_rest_rotations")
    if allow_axis_mismatch:
        cmd.append("--allow_axis_mismatch")
    return cmd


def _stop_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _execute(cmd: list[str], log_file: Path, timeout_seconds: int) -> None:
    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    with log_file.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            cmd, cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, **kw,
        )
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            _stop_process_tree(process)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            raise RuntimeError(
                f"TopX exceeded {timeout_seconds}s; process stopped. Log: {log_file}"
            ) from exc
    if return_code:
        lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        raise RuntimeError(
            f"TopX exited with code {return_code}; log: {log_file}\n" +
            "\n".join(lines[-45:])
        )


def _results(output_dir: Path, name: str, caption: str):
    import numpy as np
    import torch
    from PIL import Image

    bvh = output_dir / f"{name}.bvh"
    npz = output_dir / f"{name}.npz"
    gif = output_dir / f"{name}.gif"
    for p in (bvh, npz, gif):
        if not p.is_file() or p.stat().st_size == 0:
            raise RuntimeError(f"TopX reported success but output is missing or empty: {p}")
    with np.load(npz, allow_pickle=False) as z:
        direct = z["positions_direct"]
        fk = z["positions_fk"]
        rest = z["P_rest"]
        if not (np.isfinite(direct).all() and np.isfinite(fk).all() and np.isfinite(rest).all()):
            raise RuntimeError("TopX output contains non-finite joint positions.")
        rig_scale = float(np.linalg.norm(np.ptp(rest, axis=0)))
        if not rig_scale > 0:
            raise RuntimeError("TopX output has zero-size rest pose.")
        gap = float(np.max(np.abs(direct - fk)) / rig_scale)
    with Image.open(gif) as picture:
        picture.seek(max(0, getattr(picture, "n_frames", 1) // 2))
        frame = np.asarray(picture.convert("RGB"), dtype=np.float32) / 255.0
    image = torch.from_numpy(frame.copy()).unsqueeze(0)
    warning = " WARNING: direct/FK mismatch exceeds upstream 0.3 threshold." if gap > 0.3 else ""
    diagnostic = f"Max direct/FK position gap = {gap:.4f} x rig size.{warning} This is not a calibrated quality score."
    log_file = output_dir / "topx.log"
    return {
        "ui": {"text": [f"TopX: {caption}", diagnostic, f"BVH: {bvh}", f"GIF: {gif}", f"Log: {log_file}"]},
        "result": (image, str(bvh), str(npz), str(gif), diagnostic),
    }


def run_topx(
    *, skeleton_path: str, text_source: str, prompt: str = "",
    text_embedding_path: str = "", joint_embedding_path: str = "",
    frames: int = 120, seed: int = 7, steps: int = 50, cfg_text: float = 2.0,
    checkpoint: str = "", output_name: str = "topx", up: str = "+Y", forward: str = "+Z",
    rest_frame: int = 0, keep_rest_rotations: bool = False,
    allow_axis_mismatch: bool = False, tf32: bool = True, device: str = "cuda",
    release_comfy_vram: bool = True, timeout_seconds: int = 1800,
    caption_label: str = "",
):
    sk, text_emb, joint_emb = validate_inputs(
        skeleton_path, text_source, prompt, text_embedding_path, joint_embedding_path
    )
    weights = checkpoint_path(checkpoint)
    name = f"{safe_stem(output_name)}_s{seed}_{uuid.uuid4().hex[:10]}"
    output_dir = Path(folder_paths.get_output_directory()) / "topx" / name
    cmd = build_command(
        weights, sk, output_dir, name, text_source, prompt, text_emb, joint_emb,
        frames, seed, steps, cfg_text, up, forward, rest_frame,
        keep_rest_rotations, allow_axis_mismatch, tf32, device,
    )
    if release_comfy_vram and device == "cuda":
        import comfy.model_management as management
        management.unload_all_models()
        management.soft_empty_cache()
    output_dir.mkdir(parents=True, exist_ok=False)
    log_file = output_dir / "topx.log"
    _execute(cmd, log_file, int(timeout_seconds))
    return _results(output_dir, name, caption_label or prompt or "precomputed embedding")


class TopXDemoInputs:
    """Use upstream bundled embeddings without downloading the gated text encoder."""

    CATEGORY = "TopX/motion"
    FUNCTION = "select"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = (
        "skeleton_path", "text_embedding_path", "joint_embedding_path",
        "reference_caption", "reference_frames", "reference_seed", "keep_rest_rotations",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"example": (list(demo_catalog()) or ["missing demos"],)}}

    def select(self, example):
        data = demo_catalog()[example]
        return (
            str(data["skeleton"]), str(data["text_embedding"]), str(data["joint_embedding"]),
            data["caption"], data["frames"], data["seed"], True,
        )


class TopXGenerateMotion:
    """Run original TopX sampler and export BVH motion into ComfyUI/output/topx."""

    CATEGORY = "TopX/motion"
    FUNCTION = "generate_motion"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("preview_image", "bvh_path", "npz_path", "gif_path", "diagnostic")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "skeleton_path": ("STRING", {"default": ""}),
                "text_source": (["precomputed", "llm2vec"],),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "text_embedding_path": ("STRING", {"default": ""}),
                "joint_embedding_path": ("STRING", {"default": ""}),
                "frames": ("INT", {"default": 120, "min": 1, "max": 240}),
                "seed": ("INT", {"default": 7, "min": 0, "max": 0x7FFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 250}),
                "cfg_text": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "up": (list(AXES), {"default": "+Y"}),
                "forward": (list(AXES), {"default": "+Z"}),
                "rest_frame": ("INT", {"default": 0, "min": -1, "max": 100000}),
                "keep_rest_rotations": ("BOOLEAN", {"default": False}),
                "allow_axis_mismatch": ("BOOLEAN", {"default": False}),
                "release_comfy_vram": ("BOOLEAN", {"default": True}),
                "device": (["cuda", "cpu"],),
                "tf32": ("BOOLEAN", {"default": True}),
                "checkpoint_path": ("STRING", {"default": ""}),
                "output_name": ("STRING", {"default": "motion"}),
                "timeout_seconds": ("INT", {"default": 1800, "min": 60, "max": 7200}),
            }
        }

    def generate_motion(
        self, skeleton_path, text_source, prompt, text_embedding_path,
        joint_embedding_path, frames, seed, steps, cfg_text, up, forward,
        rest_frame, keep_rest_rotations, allow_axis_mismatch, release_comfy_vram,
        device, tf32, checkpoint_path, output_name, timeout_seconds,
    ):
        return run_topx(
            skeleton_path=skeleton_path, text_source=text_source, prompt=prompt,
            text_embedding_path=text_embedding_path,
            joint_embedding_path=joint_embedding_path,
            frames=frames, seed=seed, steps=steps, cfg_text=cfg_text,
            up=up, forward=forward, rest_frame=rest_frame,
            keep_rest_rotations=keep_rest_rotations,
            allow_axis_mismatch=allow_axis_mismatch,
            release_comfy_vram=release_comfy_vram, device=device, tf32=tf32,
            checkpoint=checkpoint_path, output_name=output_name,
            timeout_seconds=timeout_seconds,
        )


class TopXGenerateDemo:
    """Single-node demo with the exact published prompt and paired embeddings."""

    CATEGORY = "TopX/motion"
    FUNCTION = "generate_demo"
    OUTPUT_NODE = True
    RETURN_TYPES = TopXGenerateMotion.RETURN_TYPES
    RETURN_NAMES = TopXGenerateMotion.RETURN_NAMES

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "example": (list(demo_catalog()) or ["missing demos"],),
                "seed_override": ("INT", {"default": -1, "min": -1, "max": 0x7FFFFFFFFFFFFFFF}),
                "frames_override": ("INT", {"default": 0, "min": 0, "max": 240}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 250}),
                "cfg_text": ("FLOAT", {"default": 2.0, "min": 0, "max": 20, "step": 0.1}),
                "release_comfy_vram": ("BOOLEAN", {"default": True}),
                "checkpoint_path": ("STRING", {"default": ""}),
                "timeout_seconds": ("INT", {"default": 1800, "min": 60, "max": 7200}),
            }
        }

    def generate_demo(
        self, example, seed_override, frames_override, steps,
        cfg_text, release_comfy_vram, checkpoint_path, timeout_seconds,
    ):
        sample = demo_catalog()[example]
        return run_topx(
            skeleton_path=str(sample["skeleton"]), text_source="precomputed",
            text_embedding_path=str(sample["text_embedding"]),
            joint_embedding_path=str(sample["joint_embedding"]),
            frames=frames_override or sample["frames"],
            seed=sample["seed"] if seed_override < 0 else seed_override,
            steps=steps, cfg_text=cfg_text, keep_rest_rotations=True,
            release_comfy_vram=release_comfy_vram, checkpoint=checkpoint_path,
            output_name=example.replace(" / ", "_"), caption_label=sample["caption"],
            timeout_seconds=timeout_seconds,
        )
