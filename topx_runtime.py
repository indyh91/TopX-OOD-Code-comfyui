"""Isolated TopX encoding and multi-clip sampling runtime.

The imported API is cheap and never imports torch/LLM2Vec. Each worker
runs in a bounded child Python process; the encoder exits before the
motion-sampling worker starts. No stdout/stderr pipes are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ENCODER_ID = (
    "meta-llama/Meta-Llama-3-8B-Instruct|"
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp|"
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised|"
    "TopX-caption-span-mean-v1|TopX-joint-pool-v1"
)


def _sha1(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                  separators=(",", ":")).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _valid_npy(path: Path, shape: tuple[int | None, ...]) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False, mmap_mode="r") as a:
            pass
    except TypeError:
        # mmap returns an ndarray, not a context manager.
        try:
            a = np.load(path, allow_pickle=False, mmap_mode="r")
            return (a.dtype == np.float32 and a.ndim == len(shape)
                    and all(b is None or a.shape[i] == b for i, b in enumerate(shape))
                    and np.isfinite(a).all())
        except (OSError, ValueError):
            return False
    except (OSError, ValueError):
        return False
    return False


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with tmp.open("wb") as f:
            np.save(f, np.asarray(array, dtype=np.float32), allow_pickle=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _stop_process_tree(process) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL, timeout=15, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _run_worker(mode: str, request: dict, work_dir: Path, timeout_seconds: int) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    request_path = work_dir / "request.json"
    manifest_path = work_dir / "result.json"
    log_path = work_dir / "topx.log"
    _atomic_json(request_path, request)
    kwargs = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
              if os.name == "nt" else {"start_new_session": True})
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        p = subprocess.Popen(
            [sys.executable, "-u", str(ROOT / "topx_runtime.py"),
             "--worker", mode, "--request", str(request_path), "--result", str(manifest_path)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            **kwargs,
        )
        try:
            ret = p.wait(timeout=int(timeout_seconds))
        except subprocess.TimeoutExpired as exc:
            _stop_process_tree(p)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
            raise RuntimeError(f"TopX {mode} timed out after {timeout_seconds}s; log: {log_path}") from exc
    if ret != 0 or not manifest_path.is_file():
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-35:]
        raise RuntimeError(f"TopX {mode} exited {ret}, log: {log_path}\n" + "\n".join(tail))
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _preflight_prompt(prompt: str, auto_prefix_caption: bool) -> str:
    text = " ".join(prompt.split())
    if not text:
        raise ValueError("Empty caption.")
    if auto_prefix_caption and not text.lower().startswith("an object"):
        text = "An object " + text[0].lower() + text[1:]
    return text


def ensure_bvh_skeleton(skeleton_path: Path, cache_dir: Path,
                        up: str = "+Y", forward: str = "+Z") -> Path:
    from .glb_bridge import canonical_basis, extract_skeleton
    canonical_basis(up, forward)
    path = Path(skeleton_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in (".bvh", ".npz"):
        return path
    if path.suffix.lower() != ".glb":
        raise ValueError("Skeleton must be .bvh, .npz, or skinned .glb.")
    key = _sha1({"glb_sha256": _sha256_file(path), "bridge": "rigid-bind-v1",
                 "up": up, "forward": forward})
    target = cache_dir / "skeletons" / (key + ".bvh")
    if not target.is_file():
        extract_skeleton(path, target, up_axis=up, forward_axis=forward)
    return target.resolve()


def encode_embeddings(
    prompts: list[str],
    joint_names: list[str] | None,
    cache_dir: Path,
    *,
    skeleton_path: Path | None = None,
    up: str = "+Y",
    forward: str = "+Z",
    rest_frame: int = 0,
    keep_rest_rotations: bool = False,
    body_plan: str | None = None,
    auto_prefix_caption: bool = True,
    timeout_seconds: int = 3600,
    device: str = "cuda",
) -> dict:
    """Return content-addressed cached embeddings; never launch worker on a full hit.

    Joint descriptions depend on geometry and lexicon as well as names. The
    joint key therefore includes the entire canonical skeleton input identity
    and descriptive settings, rather than dangerously using names alone.
    """
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    skeleton = (ensure_bvh_skeleton(Path(skeleton_path), cache_dir, up, forward)
                if skeleton_path else None)
    if joint_names is not None and skeleton is None:
        raise ValueError("Joint embeddings need a skeleton to derive geometric descriptions.")
    cleaned = [_preflight_prompt(p, auto_prefix_caption) for p in prompts]
    paths = [cache_dir / "text" / (_sha1({"prompt": p, "model": ENCODER_ID}) + ".npy")
             for p in cleaned]
    missing_text = [{"prompt": p, "path": str(path)} for p, path in zip(cleaned, paths)
                    if not _valid_npy(path, (4096,))]
    joints_path = None
    missing_joints = False
    if skeleton:
        key = _sha1({
            "skeleton_sha256": _sha256_file(skeleton),
            "joint_names": joint_names,
            "model": ENCODER_ID, "up": up, "forward": forward,
            "rest_frame": rest_frame, "keep_rest_rotations": keep_rest_rotations,
            "body_plan": body_plan,
            "lexicon_sha256": _sha256_file(ROOT / "configs/joint_name_lexicon.json"),
        })
        joints_path = cache_dir / "joints" / (key + ".npy")
        # The encoder validates joint count against the rig on a cache miss;
        # a cache hit is only possible for an identical skeleton hash.
        missing_joints = not _valid_npy(joints_path, (None, 4096))
    need = bool(missing_text or missing_joints)
    if need:
        req = {
            "text": missing_text, "joint_output": str(joints_path) if missing_joints else None,
            "skeleton_path": str(skeleton) if skeleton else None,
            "expected_joint_names": joint_names, "up": up, "forward": forward,
            "rest_frame": int(rest_frame), "keep_rest_rotations": keep_rest_rotations,
            "body_plan": body_plan, "device": device,
        }
        job_dir = cache_dir / "logs" / ("encode_" + uuid.uuid4().hex)
        _run_worker("encode", req, job_dir, timeout_seconds)
    for path in paths:
        if not _valid_npy(path, (4096,)):
            raise RuntimeError(f"Missing or malformed cached text embedding: {path}")
    if joints_path and not _valid_npy(joints_path, (None, 4096)):
        raise RuntimeError(f"Missing or malformed cached joint embedding: {joints_path}")
    return {
        "text_embedding_paths": [str(x) for x in paths],
        "joint_embedding_path": str(joints_path) if joints_path else "",
        "effective_prompts": cleaned, "raw_prompts": prompts,
        "skeleton_path": str(skeleton) if skeleton else "",
        "cache_hit": not need, "model_id": ENCODER_ID,
    }


def sample_motion_batch(
    skeleton_path: Path,
    specs: list[dict],
    checkpoint_path: Path,
    *,
    output_dir: Path,
    joint_embedding_path: Path | None = None,
    up: str = "+Y",
    forward: str = "+Z",
    rest_frame: int = 0,
    keep_rest_rotations: bool = False,
    allow_axis_mismatch: bool = False,
    render_gif: bool = False,
    tf32: bool = True,
    device: str = "cuda",
    timeout_seconds: int = 3600,
    cache_hit: bool = False,
) -> dict:
    """Sample multiple candidates with one checkpoint load in one child worker.

    Each seed is still sampled sequentially, preserving upstream torch RNG
    and per-clip shape semantics; this is model-resident batch processing,
    not a claimed vectorized speedup.
    """
    skeleton_path = Path(skeleton_path).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not skeleton_path.is_file() or skeleton_path.suffix.lower() not in (".bvh", ".npz"):
        raise ValueError("Sampling requires a BVH or TopX skeleton NPZ.")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not specs:
        raise ValueError("specs must contain at least one motion candidate.")
    if len(specs) > 128:
        raise ValueError("At most 128 candidates per worker run.")
    joint_file = Path(joint_embedding_path).expanduser().resolve() if joint_embedding_path else None
    if joint_file is None or not _valid_npy(joint_file, (None, 4096)):
        raise ValueError("A valid joint embedding file [joints,4096] is mandatory.")
    normalized = []
    for i, spec in enumerate(specs):
        if not isinstance(spec, dict):
            raise ValueError(f"Candidate {i} must be a mapping.")
        text_path = Path(spec.get("text_embedding_path", "")).expanduser().resolve()
        if not _valid_npy(text_path, (4096,)):
            raise ValueError(f"Candidate {i}: invalid text embedding {text_path}")
        seed = int(spec.get("seed", 7))
        frames = int(spec.get("num_frames", spec.get("frames", 120)))
        steps = int(spec.get("steps", 50))
        cfg = float(spec.get("cfg_text", 2.0))
        if not 0 <= seed < 2**63 or not 1 <= frames <= 240 or not 1 <= steps <= 250:
            raise ValueError(f"Candidate {i}: invalid seed, frames or steps.")
        if not math.isfinite(cfg) or not 0 <= cfg <= 20:
            raise ValueError(f"Candidate {i}: invalid cfg_text.")
        normalized.append({
            "text_embedding_path": str(text_path), "seed": seed, "num_frames": frames,
            "steps": steps, "cfg_text": cfg, "prompt": str(spec.get("prompt", "")),
            "raw_prompt": str(spec.get("raw_prompt", spec.get("prompt", ""))),
            "cache_hit": bool(spec.get("cache_hit", cache_hit)),
        })
    if device not in ("cuda", "cpu"):
        raise ValueError("device must be cuda or cpu.")
    # Avoid creating outputs over existing files; caller chooses a unique run directory.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    request = {
        "skeleton": str(skeleton_path), "checkpoint": str(checkpoint_path),
        "joint_embedding_path": str(joint_file), "specs": normalized,
        "output_dir": str(output_dir), "up": up, "forward": forward,
        "rest_frame": int(rest_frame),
        "keep_rest_rotations": bool(keep_rest_rotations),
        "allow_axis_mismatch": bool(allow_axis_mismatch),
        "render_gif": bool(render_gif), "tf32": bool(tf32), "device": device,
    }
    return _run_worker("sample", request, output_dir, timeout_seconds)


def _worker_encode(req: dict) -> dict:
    import torch
    from llm2vec.llm2vec import batch_to_device
    from scripts import generate as g
    from src.data.text_encoders import LLM2VecEncoder

    device = req["device"]
    needs_joint = bool(req["joint_output"])
    desc = None
    if needs_joint:
        sk = Path(req["skeleton_path"])
        if sk.suffix.lower() == ".bvh":
            rig = g.rig_from_bvh(sk, req["up"], req["forward"], req["rest_frame"])
        else:
            rig = g.rig_from_npz(sk, reconvention=not req["keep_rest_rotations"])
        expected = req.get("expected_joint_names")
        if expected is not None and list(expected) != rig["joint_names"]:
            raise ValueError("Joint names do not match skeleton file.")
        if rig.get("joint_descriptions") and req["keep_rest_rotations"]:
            desc = rig["joint_descriptions"]
        else:
            desc, _ = g.describe_joints(
                rig, g.load_lexicon(g.DEFAULT_LEXICON), body_plan=req["body_plan"],
            )
    enc = LLM2VecEncoder(device=device, max_length=g.DESC_MAX_LENGTH)
    dev = next(enc.l2v.model.parameters()).device
    if needs_joint:
        uniq = sorted(set(desc))
        table = np.zeros((len(uniq), enc.dim), dtype=np.float32)
        for i, description in enumerate(uniq):
            features = enc.l2v.tokenize(
                [enc.l2v.prepare_for_tokenization(enc.l2v._convert_to_str("", description))]
            )
            with torch.no_grad():
                table[i] = enc.l2v.forward(batch_to_device(features, dev)).float().cpu().numpy()[0]
        indexed = {d: i for i, d in enumerate(uniq)}
        joint_sem = np.stack([table[indexed[d]] for d in desc]).astype(np.float32)
        if not np.isfinite(joint_sem).all() or joint_sem.shape[1] != 4096:
            raise RuntimeError("Invalid joint-description embedding.")
        _atomic_save_npy(Path(req["joint_output"]), joint_sem)
    enc.l2v.max_length = g.CAPTION_MAX_LENGTH
    enc.max_length = g.CAPTION_MAX_LENGTH
    done = {}
    for item in req["text"]:
        caption = item["prompt"]
        path = Path(item["path"])
        if path.as_posix() in done or _valid_npy(path, (4096,)):
            continue
        length = int(enc.encoded_token_lengths([caption])[0])
        if length > g.CAPTION_STORE_MAX_TOKENS:
            raise ValueError(f"Caption exceeds {g.CAPTION_STORE_MAX_TOKENS} TopX token limit.")
        hidden, mask = enc.encode_tokens([caption], bs=1)
        tokens = hidden[0][mask[0]]
        if tokens.shape[0] == 0 or not np.isfinite(tokens).all():
            raise RuntimeError("No valid caption token states.")
        vector = tokens.mean(0).astype(np.float32)
        if vector.shape != (4096,):
            raise RuntimeError(f"Unexpected caption embedding shape: {vector.shape}")
        _atomic_save_npy(path, vector)
        done[path.as_posix()] = True
    return {"text_encoded": len(done), "joint_encoded": needs_joint}


def _worker_sample(req: dict) -> dict:
    import torch
    from scripts import generate as g

    if req["tf32"]:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    sk = Path(req["skeleton"])
    if sk.suffix.lower() == ".bvh":
        rig = g.rig_from_bvh(sk, req["up"], req["forward"], req["rest_frame"])
    else:
        rig = g.rig_from_npz(sk, reconvention=not req["keep_rest_rotations"])
    agree, sided, _ = g.side_check(rig)
    if (sided >= 4 and agree < 0.8*sided and sk.suffix.lower() == ".bvh"
            and not req["allow_axis_mismatch"]):
        raise ValueError("BVH axes contradict joint names; set correct up/forward axes.")
    joint_sem = np.load(req["joint_embedding_path"], allow_pickle=False).astype(np.float32)
    if joint_sem.shape != (len(rig["parents"]), 4096) or not np.isfinite(joint_sem).all():
        raise ValueError("Joint embedding shape/values do not match loaded skeleton.")
    # Load TopX exactly once for the whole seed/clip batch.
    model, ca, _ = g.load_model(req["checkpoint"], req["device"])
    checkpoint_hash = _sha256_file(Path(req["checkpoint"]))
    P = np.asarray(rig["P_rest_global"], dtype=np.float64)
    if rig.get("joint_descriptions") is not None and req["keep_rest_rotations"]:
        descriptions = list(rig["joint_descriptions"])
    else:
        descriptions, _ = g.describe_joints(rig, g.load_lexicon(g.DEFAULT_LEXICON))
    out = Path(req["output_dir"])
    results = []
    for index, spec in enumerate(req["specs"]):
        start = time.monotonic()
        name = f"clip_{index:03d}_s{spec['seed']}"
        emb = np.load(spec["text_embedding_path"], allow_pickle=False).astype(np.float32)
        if emb.shape != (4096,) or not np.isfinite(emb).all():
            raise ValueError(f"Invalid embedding for candidate {index}.")
        segment, decoded = g.generate(
            model, ca, rig, emb, joint_sem,
            spec["num_frames"], spec["seed"], spec["steps"],
            spec["cfg_text"], req["device"],
        )
        gap = float(np.max(np.abs(decoded["positions_direct"] - decoded["positions_fk"]))
                    / rig["s_rig"])
        if not math.isfinite(gap):
            raise ValueError("Non-finite direct/FK diagnostic.")
        npz_file = out / (name + ".npz")
        bvh_file = out / (name + ".bvh")
        gif_file = out / (name + ".gif") if req["render_gif"] else None
        np.savez(
            npz_file, positions_direct=decoded["positions_direct"],
            positions_fk=decoded["positions_fk"],
            rest_delta_rotations=decoded["rest_delta"], model_output=segment,
            joint_names=np.asarray(rig["joint_names"]), parents=np.asarray(rig["parents"], np.int64),
            P_rest=P, fps=np.array(g.FPS), text=np.array(spec["prompt"]),
            raw_prompt=np.array(spec["raw_prompt"]), seed=np.array(spec["seed"]),
            steps=np.array(spec["steps"]), cfg_text=np.array(spec["cfg_text"]),
            ckpt=np.array(req["checkpoint"]), ckpt_sha256=np.array(checkpoint_hash),
            skeleton=np.array(rig["source"]), joint_descriptions=np.array(descriptions),
            frame=np.array("canonical: +Y up, +Z forward, +X left; rest pose grounded"),
        )
        err = g.write_bvh(bvh_file, rig, decoded)
        if gif_file is not None:
            from src.render import render_gif
            render_gif(
                gif_file, [
                    ("demo", "REST", P[None]),
                    ("gen_ric", "GEN position channels", decoded["positions_direct"]),
                    ("gen_fk", "GEN rotation FK", decoded["positions_fk"]),
                ], list(map(int, rig["parents"])), spec["prompt"], name, fps=int(g.FPS),
            )
        elapsed = time.monotonic() - start
        results.append({
            "bvh_path": str(bvh_file), "npz_path": str(npz_file),
            "gif_path": str(gif_file) if gif_file else None,
            "fk_diag_score": gap, "joint_count": len(rig["parents"]),
            "frame_count": spec["num_frames"], "fps": float(g.FPS),
            "seed": spec["seed"], "elapsed_sec": elapsed,
            "cache_hit": spec["cache_hit"], "prompt": spec["prompt"],
            "raw_prompt": spec["raw_prompt"], "bvh_readback_error": err,
        })
        print(f"[topx] {name}: frame={spec['num_frames']}, "
              f"FK gap={gap:.5f}, read-back={err:.4e}, elapsed={elapsed:.2f}s", flush=True)
    return {"results": results, "checkpoint_loads": 1}


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("encode", "sample"), required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    result = _worker_encode(request) if args.worker == "encode" else _worker_sample(request)
    _atomic_json(args.result, result)


if __name__ == "__main__":
    _main()
