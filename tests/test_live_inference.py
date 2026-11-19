"""Opt-in live CUDA smoke test. NEVER runs in ordinary CI.

Run on the target ComfyUI Python environment with the original checkpoint:
  $env:TOPX_RUN_LIVE_TESTS = "1"
  python -m pytest tests/test_live_inference.py -v
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import topx_runtime as runtime  # noqa: E402


@pytest.mark.skipif(os.getenv("TOPX_RUN_LIVE_TESTS") != "1",
                    reason="Opt-in only: TOPX_RUN_LIVE_TESTS=1")
def test_spider_cuda_inference_without_gif():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable.")
    ckpt = Path(os.getenv("TOPX_CHECKPOINT", str(ROOT/"models"/"topx"/"topx.pt")))
    if not ckpt.is_file():
        pytest.skip(f"Original TopX checkpoint missing: {ckpt}")
    spider = ROOT/"demo"/"spider"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)/"run"
        manifest = runtime.sample_motion_batch(
            spider/"skeleton.npz",
            [{"text_embedding_path": str(spider/"text_emb_1.npy"),
              "seed": 17, "num_frames": 31, "steps": 50,
              "cfg_text": 2., "cache_hit": True}],
            ckpt,
            output_dir=out, joint_embedding_path=spider/"joint_sem.npy",
            keep_rest_rotations=True, render_gif=False, tf32=True,
            timeout_seconds=3600,
        )
        assert manifest["checkpoint_loads"] == 1
        assert len(manifest["results"]) == 1
        result = manifest["results"][0]
        assert result["fk_diag_score"] < 0.3, result
        assert result["elapsed_sec"] > 0
        assert result["frame_count"] == 31
        assert result["gif_path"] is None
        assert Path(result["bvh_path"]).stat().st_size > 0
        assert Path(result["npz_path"]).stat().st_size > 0
        assert not any(out.glob("*.gif"))
        with np.load(result["npz_path"], allow_pickle=False) as motion:
            assert np.isfinite(motion["rest_delta_rotations"]).all()
            assert np.isfinite(motion["positions_fk"]).all()
            assert motion["rest_delta_rotations"].shape == (31, 24, 3, 3)
        print("TOPX_LIVE_RESULT", result)
