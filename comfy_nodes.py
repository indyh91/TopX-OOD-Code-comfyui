"""ComfyUI TopX nodes: cached LLM2Vec, model-resident multi-clip sampling, skinned GLB."""
from __future__ import annotations
import json
import re
import uuid
from pathlib import Path

import folder_paths
from . import glb_bridge
from . import topx_runtime as runtime

AXES = list(glb_bridge.AXES)


def demo_catalog():
    found = {}
    for path in sorted((Path(__file__).parent / "demo").glob("*/prompts.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for i, sample in enumerate(data.get("prompts", []), 1):
            found[f"{path.parent.name} / {i}"] = dict(
                skeleton=path.parent/"skeleton.npz",
                text_embedding=path.parent/sample["text_emb"],
                joint_embedding=path.parent/"joint_sem.npy",
                caption=sample["text"],
                frames=int(sample["frames"]),
                seed=int(sample["seed"]),
            )
    return found


def _cache(raw=""):
    return (Path(raw).expanduser() if raw.strip() else
            Path(folder_paths.get_output_directory())/"topx"/"_cache").resolve()


def checkpoint_path(raw):
    path = Path(raw.strip()).expanduser() if raw.strip() else (
        Path(folder_paths.models_dir)/"topx"/"topx.pt")
    legacy = Path(__file__).parent/"weights"/"topx.pt"
    if not raw.strip() and not path.is_file() and legacy.is_file():
        path = legacy
    if not path.is_file():
        raise FileNotFoundError(f"Missing official TopX checkpoint: {path}")
    return path.resolve()


def _unload(device, enabled):
    if enabled and device == "cuda":
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache()


def _preview(npz_path, gif_path):
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    if gif_path and Path(gif_path).is_file():
        with Image.open(gif_path) as gif:
            gif.seek(max(0, getattr(gif, "n_frames", 1)//2))
            return torch.from_numpy(np.asarray(gif.convert("RGB"), dtype=np.float32).copy()/255).unsqueeze(0)
    with np.load(npz_path, allow_pickle=False) as z:
        points = np.asarray(z["positions_fk"], dtype=np.float64)
        parents = list(map(int, z["parents"]))
    p = points[len(points)//2]
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite motion positions.")
    mid = (p[:, :2].max(0)+p[:, :2].min(0))/2
    scl = 425/max(float(np.ptp(p[:, 0])), float(np.ptp(p[:, 1])), 1e-6)
    coords = (p[:, :2]-mid)*scl+256
    coords[:, 1] = 512-coords[:, 1]
    img = Image.new("RGB", (512, 512), (247, 248, 249))
    draw = ImageDraw.Draw(img)
    for j, k in enumerate(parents):
        if k >= 0:
            draw.line((*coords[j], *coords[k]), fill=(32, 80, 140), width=3)
    return torch.from_numpy(np.asarray(img, dtype=np.float32).copy()/255).unsqueeze(0)


def _outputs(batch, description="TopX motion"):
    first = batch["results"][0]
    diagnostic = ("Direct/FK max difference: "
                  f"{first['fk_diag_score']:.4f} x skeleton size"
                  + (" (over 0.3 warning threshold)" if first["fk_diag_score"] > .3 else "")
                  + "; not a calibrated quality score")
    return {"ui": {"text": [description, diagnostic, first["bvh_path"]]},
            "result": (_preview(first["npz_path"], first["gif_path"]),
                       first["bvh_path"], first["npz_path"], first["gif_path"] or "",
                       diagnostic, json.dumps(first), json.dumps(batch))}


class TopXDemoInputs:
    CATEGORY = "TopX/motion"
    FUNCTION = "select"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = ("skeleton_path", "text_embedding_path", "joint_embedding_path",
                    "reference_caption", "reference_frames", "reference_seed",
                    "keep_rest_rotations")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"example": (list(demo_catalog()),)}}

    def select(self, example):
        d = demo_catalog()[example]
        return (str(d["skeleton"]), str(d["text_embedding"]), str(d["joint_embedding"]),
                d["caption"], d["frames"], d["seed"], True)


class TopXEncodeEmbeddings:
    CATEGORY = "TopX/motion"
    FUNCTION = "encode"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("text_embedding_path", "joint_embedding_path", "skeleton_path",
                    "text_embedding_paths_json", "metadata_json", "cache_hit")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "prompt": ("STRING", {"default": "An object walks.", "multiline": True}),
            "skeleton_path": ("STRING", {"default": ""}),
            "cache_dir": ("STRING", {"default": ""}),
            "up": (AXES, {"default": "+Y"}),
            "forward": (AXES, {"default": "+Z"}),
            "rest_frame": ("INT", {"default": 0, "min": -1, "max": 100000}),
            "keep_rest_rotations": ("BOOLEAN", {"default": False}),
            "auto_prefix_caption": ("BOOLEAN", {"default": True}),
            "device": (["cuda", "cpu"],),
            "release_comfy_vram": ("BOOLEAN", {"default": True}),
            "timeout_seconds": ("INT", {"default": 3600, "min": 60, "max": 14400}),
        }}

    def encode(self, prompt, skeleton_path, cache_dir, up, forward, rest_frame,
               keep_rest_rotations, auto_prefix_caption, device,
               release_comfy_vram, timeout_seconds):
        captions = [x.strip() for x in prompt.splitlines() if x.strip()]
        if not captions:
            raise ValueError("A prompt is required.")
        meta = runtime.encode_embeddings(
            captions, None, _cache(cache_dir),
            skeleton_path=Path(skeleton_path) if skeleton_path.strip() else None,
            up=up, forward=forward, rest_frame=rest_frame,
            keep_rest_rotations=keep_rest_rotations,
            auto_prefix_caption=auto_prefix_caption, timeout_seconds=timeout_seconds,
            device=device,
            before_worker=(lambda: _unload(device, release_comfy_vram)),
        )
        return (meta["text_embedding_paths"][0], meta["joint_embedding_path"],
                meta["skeleton_path"], json.dumps(meta["text_embedding_paths"]),
                json.dumps(meta), meta["cache_hit"])


class TopXGenerateMotion:
    CATEGORY = "TopX/motion"
    FUNCTION = "generate_motion"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("preview_image", "bvh_path", "npz_path", "gif_path",
                    "diagnostic", "result_json", "batch_result_json")

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
                "cfg_text": ("FLOAT", {"default": 2.0, "min": 0, "max": 20, "step": .1}),
                "up": (AXES, {"default": "+Y"}),
                "forward": (AXES, {"default": "+Z"}),
                "rest_frame": ("INT", {"default": 0, "min": -1, "max": 100000}),
                "keep_rest_rotations": ("BOOLEAN", {"default": False}),
                "allow_axis_mismatch": ("BOOLEAN", {"default": False}),
                "release_comfy_vram": ("BOOLEAN", {"default": True}),
                "device": (["cuda", "cpu"],),
                "tf32": ("BOOLEAN", {"default": True}),
                "checkpoint_path": ("STRING", {"default": ""}),
                "output_name": ("STRING", {"default": "motion"}),
                "timeout_seconds": ("INT", {"default": 3600, "min": 60, "max": 14400}),
                "render_gif": ("BOOLEAN", {"default": True}),
                "auto_prefix_caption": ("BOOLEAN", {"default": True}),
                "cache_dir": ("STRING", {"default": ""}),
            },
            "optional": {"specs_json": ("STRING", {"default": "", "multiline": True})},
        }

    def generate_motion(self, skeleton_path, text_source, prompt, text_embedding_path,
                        joint_embedding_path, frames, seed, steps, cfg_text, up, forward,
                        rest_frame, keep_rest_rotations, allow_axis_mismatch,
                        release_comfy_vram, device, tf32, checkpoint_path, output_name,
                        timeout_seconds, render_gif=True, auto_prefix_caption=True,
                        cache_dir="", specs_json=""):
        cache = _cache(cache_dir)
        if not skeleton_path.strip():
            raise ValueError("Provide .bvh, TopX skeleton .npz, or rigged .glb.")
        sk = runtime.ensure_bvh_skeleton(Path(skeleton_path), cache, up, forward)
        joint = Path(joint_embedding_path) if joint_embedding_path.strip() else None
        hit = True
        if specs_json.strip():
            if text_source != "precomputed" or prompt.strip():
                raise ValueError("specs_json requires precomputed mode, with empty prompt.")
            specs = json.loads(specs_json)
            if not isinstance(specs, list):
                raise ValueError("specs_json must be a JSON array of candidate objects.")
        elif text_source == "llm2vec":
            if text_embedding_path.strip():
                raise ValueError("Leave text_embedding_path empty in llm2vec mode.")
            captions = [x.strip() for x in prompt.splitlines() if x.strip()]
            if not captions:
                raise ValueError("Enter a prompt for llm2vec mode.")
            meta = runtime.encode_embeddings(
                captions, None, cache, skeleton_path=sk, up=up, forward=forward,
                rest_frame=rest_frame, keep_rest_rotations=keep_rest_rotations,
                auto_prefix_caption=auto_prefix_caption, device=device,
                timeout_seconds=timeout_seconds,
                before_worker=(lambda: _unload(device, release_comfy_vram)),
            )
            hit = meta["cache_hit"]
            if joint is None:
                joint = Path(meta["joint_embedding_path"])
            specs = [dict(text_embedding_path=emb, seed=seed+i, num_frames=frames,
                          steps=steps, cfg_text=cfg_text, prompt=meta["effective_prompts"][i],
                          raw_prompt=captions[i], cache_hit=hit)
                     for i, emb in enumerate(meta["text_embedding_paths"])]
        elif text_source == "precomputed":
            if prompt.strip():
                raise ValueError("Precomputed embeddings cannot be combined with a new prompt.")
            if not text_embedding_path.strip():
                raise ValueError("Provide text_embedding_path or specs_json.")
            specs = [dict(text_embedding_path=text_embedding_path, seed=seed, num_frames=frames,
                          steps=steps, cfg_text=cfg_text, cache_hit=True)]
        else:
            raise ValueError(f"Unknown text_source: {text_source}")
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", output_name)[:48].strip("_") or "motion"
        destination = Path(folder_paths.get_output_directory())/"topx"/(
            safe+"_"+uuid.uuid4().hex[:12])
        _unload(device, release_comfy_vram)
        result = runtime.sample_motion_batch(
            sk, specs, checkpoint_path=globals()["checkpoint_path"](checkpoint_path),
            output_dir=destination, joint_embedding_path=joint, up=up, forward=forward,
            rest_frame=rest_frame, keep_rest_rotations=keep_rest_rotations,
            allow_axis_mismatch=allow_axis_mismatch, render_gif=render_gif,
            tf32=tf32, device=device, timeout_seconds=timeout_seconds, cache_hit=hit,
        )
        return _outputs(result)


class TopXGenerateDemo:
    CATEGORY = "TopX/motion"
    FUNCTION = "generate_demo"
    OUTPUT_NODE = True
    RETURN_TYPES = TopXGenerateMotion.RETURN_TYPES
    RETURN_NAMES = TopXGenerateMotion.RETURN_NAMES

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "example": (list(demo_catalog()),),
            "seed_override": ("INT", {"default": -1, "min": -1, "max": 0x7FFFFFFFFFFFFFFF}),
            "frames_override": ("INT", {"default": 0, "min": 0, "max": 240}),
            "steps": ("INT", {"default": 50, "min": 1, "max": 250}),
            "cfg_text": ("FLOAT", {"default": 2., "min": 0., "max": 20., "step": .1}),
            "release_comfy_vram": ("BOOLEAN", {"default": True}),
            "checkpoint_path": ("STRING", {"default": ""}),
            "timeout_seconds": ("INT", {"default": 3600, "min": 60, "max": 14400}),
            "render_gif": ("BOOLEAN", {"default": True}),
        }}

    def generate_demo(self, example, seed_override, frames_override, steps,
                      cfg_text, release_comfy_vram, checkpoint_path, timeout_seconds,
                      render_gif=True):
        d = demo_catalog()[example]
        seed = d["seed"] if seed_override < 0 else seed_override
        specs = [dict(text_embedding_path=str(d["text_embedding"]), seed=seed,
                      num_frames=frames_override or d["frames"], steps=steps,
                      cfg_text=cfg_text, prompt=d["caption"], cache_hit=True)]
        folder = Path(folder_paths.get_output_directory())/"topx"/(
            example.replace(" / ", "_")+"_"+uuid.uuid4().hex[:12])
        _unload("cuda", release_comfy_vram)
        result = runtime.sample_motion_batch(
            d["skeleton"], specs, checkpoint_path=globals()["checkpoint_path"](checkpoint_path),
            output_dir=folder, joint_embedding_path=d["joint_embedding"],
            keep_rest_rotations=True, render_gif=render_gif,
            timeout_seconds=timeout_seconds,
        )
        return _outputs(result, d["caption"])


class TopXExtractSkeletonGLB:
    CATEGORY = "TopX/GLB"
    FUNCTION = "extract"
    RETURN_TYPES = ("STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("bvh_path", "joint_names_json", "joint_count", "rest_frame")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "glb_path": ("STRING", {"default": ""}),
            "up_axis": (AXES, {"default": "+Y"}),
            "forward_axis": (AXES, {"default": "+Z"}),
        }}

    def extract(self, glb_path, up_axis, forward_axis):
        path = Path(glb_path).expanduser().resolve()
        key = runtime._sha1({"sha256": runtime._sha256_file(path),
                             "up": up_axis, "forward": forward_axis})
        target = _cache()/"skeletons"/(key+".bvh")
        info = glb_bridge.extract_skeleton(path, target, up_axis=up_axis,
                                           forward_axis=forward_axis)
        return (info["bvh_path"], json.dumps(info["joint_names"]),
                info["joint_count"], info["rest_frame"])


class TopXBakeMotionGLB:
    CATEGORY = "TopX/GLB"
    FUNCTION = "bake"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("animated_glb_path", "result_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "source_glb_path": ("STRING", {"default": ""}),
            "motion_path": ("STRING", {"default": ""}),
            "clip_name": ("STRING", {"default": "TopX"}),
            "output_glb_path": ("STRING", {"default": ""}),
            "up_axis": (AXES, {"default": "+Y"}),
            "forward_axis": (AXES, {"default": "+Z"}),
        }}

    def bake(self, source_glb_path, motion_path, clip_name,
             output_glb_path, up_axis, forward_axis):
        result = glb_bridge.bake_motion(
            source_glb_path, motion_path, clip_name,
            output_glb_path or None, up_axis=up_axis, forward_axis=forward_axis)
        return result["animated_glb_path"], json.dumps(result)
