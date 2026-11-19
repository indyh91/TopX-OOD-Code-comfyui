"""Dependency-free ComfyUI packaging and input-contract tests.

These tests cannot establish GPU inference correctness or generated-motion quality.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

fake_paths = types.ModuleType("folder_paths")
fake_paths.models_dir = str(ROOT / "nonexistent_models_for_tests")
fake_paths.get_output_directory = lambda: str(ROOT / "out")
sys.modules.setdefault("folder_paths", fake_paths)

spec = importlib.util.spec_from_file_location(
    "topx_comfy_test_package", ROOT / "__init__.py",
    submodule_search_locations=[str(ROOT)],
)
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from topx_comfy_test_package import comfy_nodes as node  # noqa: E402


class RegistrationTests(unittest.TestCase):
    def test_registration(self):
        self.assertEqual(
            set(package.NODE_CLASS_MAPPINGS),
            {"TopXDemoInputs", "TopXGenerateMotion", "TopXGenerateDemo"},
        )
        for cls in package.NODE_CLASS_MAPPINGS.values():
            self.assertTrue(cls.INPUT_TYPES()["required"])
            self.assertEqual(len(cls.RETURN_TYPES), len(cls.RETURN_NAMES))

    def test_demo_metadata_is_complete(self):
        samples = node.demo_catalog()
        self.assertEqual(len(samples), 12)
        for key, sample in samples.items():
            self.assertTrue(sample["skeleton"].is_file(), key)
            self.assertTrue(sample["text_embedding"].is_file(), key)
            self.assertTrue(sample["joint_embedding"].is_file(), key)
            self.assertTrue(sample["caption"].startswith("An object "), key)
            self.assertGreater(sample["frames"], 0)
            self.assertLessEqual(sample["frames"], 240)

    def test_demo_outputs(self):
        values = node.TopXDemoInputs().select("spider / 1")
        self.assertEqual(len(values), 7)
        self.assertTrue(values[0].endswith("skeleton.npz"))
        self.assertEqual(values[4], 31)
        self.assertEqual(values[5], 17)
        self.assertTrue(values[6])

    def test_preview_is_an_image(self):
        self.assertEqual(node.TopXGenerateMotion.RETURN_TYPES[0], "IMAGE")
        self.assertEqual(node.TopXGenerateMotion.RETURN_TYPES[1], "STRING")

    def test_requirements_do_not_replace_comfy_torch(self):
        base = (ROOT / "requirements.txt").read_text().lower()
        self.assertNotIn("torch>=", base)
        self.assertNotIn("llm2vec", [s.strip() for s in base.splitlines() if not s.startswith("#")])
        self.assertIn("llm2vec", (ROOT / "requirements-text.txt").read_text())


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.sk = root / "rig.bvh"
        self.te = root / "prompt.npy"
        self.je = root / "joints.npy"
        for path in (self.sk, self.te, self.je):
            path.write_bytes(b"test")

    def tearDown(self):
        self.tmp.cleanup()

    def test_precomputed_paths(self):
        sk, te, je = node.validate_inputs(
            str(self.sk), "precomputed", "", str(self.te), str(self.je)
        )
        self.assertEqual((sk, te, je), (self.sk.resolve(), self.te.resolve(), self.je.resolve()))

    def test_refuse_mismatched_new_prompt(self):
        with self.assertRaisesRegex(ValueError, "ignored"):
            node.validate_inputs(
                str(self.sk), "precomputed", "An object walks.", str(self.te), str(self.je)
            )

    def test_refuse_missing_precomputed_embeddings(self):
        with self.assertRaisesRegex(ValueError, "requires both"):
            node.validate_inputs(str(self.sk), "precomputed", "", "", "")

    def test_new_prompt_mode(self):
        sk, te, je = node.validate_inputs(
            str(self.sk), "llm2vec", "An object jumps.", "", ""
        )
        self.assertEqual(sk, self.sk.resolve())
        self.assertIsNone(te)
        self.assertIsNone(je)

    def test_reject_glb(self):
        glb = self.sk.with_suffix(".glb")
        glb.write_bytes(b"test")
        with self.assertRaisesRegex(ValueError, "GLB"):
            node.validate_inputs(str(glb), "llm2vec", "jump", "", "")

    def test_safe_stem_never_escapes_output(self):
        self.assertEqual(node.safe_stem("../_evil\\stuff"), "evil_stuff")
        self.assertEqual(node.safe_stem(""), "topx")

    def test_windows_command_uses_argv_not_shell(self):
        args = node.build_command(
            checkpoint=Path("C:/models/topx/topx.pt"),
            skeleton=Path("C:/My Rig/rig.bvh"),
            output_dir=Path("C:/My Output/topx"),
            name="motion_1",
            text_source="precomputed",
            prompt="",
            text_emb=Path("C:/My Rig/text.npy"),
            joint_emb=Path("C:/My Rig/joint.npy"),
            frames=120,
            seed=17,
            steps=50,
            cfg_text=2.0,
            up="+Y",
            forward="-Z",
            rest_frame=-1,
            keep_rest_rotations=False,
            allow_axis_mismatch=False,
            tf32=True,
            device="cuda",
        )
        self.assertEqual(args[0], sys.executable)
        self.assertIn("--forward=-Z", args)
        self.assertIn("C:/My Rig/rig.bvh", args)
        self.assertNotIn("--text", args)
        self.assertIn("--text_emb", args)

    def test_axes_are_perpendicular(self):
        with self.assertRaisesRegex(ValueError, "orthogonal"):
            node.build_command(
                Path("x.pt"), self.sk, Path("out"), "name", "precomputed",
                "", self.te, self.je, 60, 7, 50, 2.0,
                "+Z", "-Z", 0, False, False, True, "cuda"
            )

    def test_subprocess_log_has_no_pipes(self):
        log = Path(self.tmp.name) / "subprocess.log"
        node._execute([sys.executable, "-c", "print('TopX-test-process')"], log, 10)
        self.assertIn("TopX-test-process", log.read_text())


if __name__ == "__main__":
    unittest.main()
