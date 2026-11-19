"""CPU-only packaging, cache, GLB skinning and API-contract regressions."""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
fake_paths = types.ModuleType("folder_paths")
fake_paths.models_dir = str(ROOT / "nonexistent_models_for_tests")
fake_paths.get_output_directory = lambda: str(ROOT / "out")
sys.modules.setdefault("folder_paths", fake_paths)

import importlib.util

spec = importlib.util.spec_from_file_location(
    "topx_comfy_test_package", ROOT / "__init__.py",
    submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from topx_comfy_test_package import comfy_nodes, topx_runtime, glb_bridge  # noqa: E402


def create_skinned_glb(path: Path):
    """Synthetic GLB includes skin hierarchy, nonidentity rest rotation,
    normals, tangents, skin indices/weights and a normal-map material.
    """
    from pygltflib import (
        GLTF2, Asset, Accessor, Attributes, Buffer, BufferView, Image,
        Material, Mesh, Node, NormalMaterialTexture, Primitive, Scene,
        Skin, Texture,
    )
    gltf = GLTF2()
    gltf.asset = Asset(version="2.0")
    gltf.scene = 0
    gltf.scenes = [Scene(nodes=[0, 2])]
    # Root's 90-degree Z rotation verifies world-space rest bone offset and
    # nonidentity bind-pose rotations when baking motion to joint-local tracks.
    rz90 = [0.0, 0.0, 0.7071067811865475, 0.7071067811865476]
    gltf.nodes = [
        Node(name="Hips", children=[1], rotation=rz90, translation=[0, 0, 0]),
        Node(name="Neck", translation=[0, 1, 0], rotation=[0, 0, 0, 1]),
        Node(name="BodyMesh", mesh=0, skin=0),
    ]
    gltf.skins = [Skin(joints=[0, 1], skeleton=0)]
    payload = bytearray()
    gltf.bufferViews = []
    gltf.accessors = []

    def add(array, component, typ):
        data = np.ascontiguousarray(array)
        while len(payload) % 4:
            payload.append(0)
        start = len(payload)
        payload.extend(data.tobytes())
        view = len(gltf.bufferViews)
        gltf.bufferViews.append(BufferView(buffer=0, byteOffset=start, byteLength=data.nbytes))
        acc = len(gltf.accessors)
        gltf.accessors.append(Accessor(
            bufferView=view, byteOffset=0, componentType=component,
            count=len(data), type=typ,
        ))
        return acc

    pos = add(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32), 5126, "VEC3")
    normal = add(np.tile(np.array([0, 0, 1], np.float32), (3, 1)), 5126, "VEC3")
    tangent = add(np.tile(np.array([1, 0, 0, 1], np.float32), (3, 1)), 5126, "VEC4")
    joints = add(np.array([[0, 1, 0, 0]] * 3, np.uint16), 5123, "VEC4")
    weights = add(np.array([[.75, .25, 0, 0]] * 3, np.float32), 5126, "VEC4")
    ids = add(np.array([0, 1, 2], np.uint16), 5123, "SCALAR")
    gltf.meshes = [Mesh(primitives=[Primitive(
        attributes=Attributes(POSITION=pos, NORMAL=normal, TANGENT=tangent,
                              JOINTS_0=joints, WEIGHTS_0=weights),
        indices=ids, material=0,
    )])]
    gltf.materials = [Material(normalTexture=NormalMaterialTexture(index=0))]
    gltf.textures = [Texture(source=0)]
    gltf.images = [Image(uri="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwsB9WVa5i8AAAAASUVORK5CYII=")]
    gltf.buffers = [Buffer(byteLength=len(payload))]
    gltf.set_binary_blob(bytes(payload))
    gltf.save_binary(str(path))
    return path


def read_accessor(gltf, aid):
    accessor = gltf.accessors[aid]
    view = gltf.bufferViews[accessor.bufferView]
    num = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[accessor.type]
    offset = view.byteOffset + (accessor.byteOffset or 0)
    payload = gltf.binary_blob()
    length = accessor.count * num
    return np.frombuffer(payload, dtype="<f4", count=length, offset=offset).reshape(accessor.count, num)


class RegistrationTests(unittest.TestCase):
    def test_node_registration_and_types(self):
        self.assertEqual(set(package.NODE_CLASS_MAPPINGS), {
            "TopXDemoInputs", "TopXEncodeEmbeddings", "TopXGenerateMotion",
            "TopXGenerateDemo", "TopXExtractSkeletonGLB", "TopXBakeMotionGLB",
        })
        for cls in package.NODE_CLASS_MAPPINGS.values():
            self.assertTrue(cls.INPUT_TYPES()["required"])
            self.assertEqual(len(cls.RETURN_TYPES), len(cls.RETURN_NAMES))
        self.assertEqual(comfy_nodes.TopXGenerateMotion.RETURN_NAMES[5:], (
            "result_json", "batch_result_json",
        ))
        self.assertEqual(comfy_nodes.TopXGenerateMotion.RETURN_TYPES[0], "IMAGE")

    def test_twelve_demo_pairs(self):
        samples = comfy_nodes.demo_catalog()
        self.assertEqual(len(samples), 12)
        for name, d in samples.items():
            self.assertTrue(d["skeleton"].is_file(), name)
            self.assertTrue(d["text_embedding"].is_file(), name)
            self.assertTrue(d["joint_embedding"].is_file(), name)

    def test_no_host_torch_reinstall(self):
        base = (ROOT / "requirements.txt").read_text()
        optional = (ROOT / "requirements-text.txt").read_text()
        for forbidden in ("torch", "torchvision", "torchaudio", "peft", "transformers"):
            self.assertFalse(any(line.strip().startswith(forbidden)
                                 for line in base.splitlines()+optional.splitlines()
                                 if not line.strip().startswith("#")))
        self.assertIn("pygltflib", base)
        self.assertIn("llm2vec", optional)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _fake_encode(self, mode, req, workdir, timeout):
        self.assertEqual(mode, "encode")
        for text in req["text"]:
            topx_runtime._atomic_save_npy(Path(text["path"]), np.zeros(4096, np.float32))
        if req["joint_output"]:
            topx_runtime._atomic_save_npy(
                Path(req["joint_output"]), np.zeros((24, 4096), np.float32))
        return {"text_encoded": len(req["text"])}

    def test_text_and_joint_cache_hit_miss_and_encoder_unload(self):
        skeleton = ROOT / "demo" / "spider" / "skeleton.npz"
        hook = mock.Mock()
        with mock.patch.object(topx_runtime, "_run_worker", side_effect=self._fake_encode) as worker:
            first = topx_runtime.encode_embeddings(
                ["scuttles", "jumps"], None, self.directory, skeleton_path=skeleton,
                keep_rest_rotations=True, before_worker=hook)
            self.assertFalse(first["cache_hit"])
            self.assertEqual(len(first["text_embedding_paths"]), 2)
            self.assertEqual(worker.call_count, 1)
            self.assertEqual(hook.call_count, 1)
            second = topx_runtime.encode_embeddings(
                ["scuttles", "jumps"], None, self.directory, skeleton_path=skeleton,
                keep_rest_rotations=True, before_worker=hook)
            self.assertTrue(second["cache_hit"])
            self.assertEqual(worker.call_count, 1)
            self.assertEqual(hook.call_count, 1)
            self.assertEqual(first["text_embedding_paths"], second["text_embedding_paths"])
            third = topx_runtime.encode_embeddings(
                ["scuttles", "walks"], None, self.directory, skeleton_path=skeleton,
                keep_rest_rotations=True, before_worker=hook)
            self.assertFalse(third["cache_hit"])
            self.assertEqual(worker.call_count, 2)
            self.assertEqual(len(worker.call_args[0][1]["text"]), 1)
            self.assertFalse(bool(worker.call_args[0][1]["joint_output"]))

    def test_cache_key_changes_with_rig_geometry_and_axes(self):
        one = self.directory/"a.npz"
        two = self.directory/"b.npz"
        one.write_bytes(b"rig_1")
        two.write_bytes(b"rig_2")
        self.assertNotEqual(topx_runtime._sha256_file(one), topx_runtime._sha256_file(two))
        self.assertNotEqual(topx_runtime._sha1({"rig": 1, "up": "+Y"}),
                            topx_runtime._sha1({"rig": 1, "up": "+Z"}))

    def test_caption_prefix_preserves_raw(self):
        self.assertEqual(
            topx_runtime._preflight_prompt("Walks forward.", True),
            "An object walks forward.",
        )
        self.assertEqual(
            topx_runtime._preflight_prompt("Walks forward.", False),
            "Walks forward.",
        )

    def test_atomic_cache_rejects_corruption(self):
        target = self.directory/"vec.npy"
        target.write_bytes(b"invalid")
        self.assertFalse(topx_runtime._valid_npy(target, (4096,)))
        topx_runtime._atomic_save_npy(target, np.ones(4096, np.float32))
        self.assertTrue(topx_runtime._valid_npy(target, (4096,)))


class GLBTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.source = create_skinned_glb(self.dir/"character.glb")

    def tearDown(self):
        self.temp.cleanup()

    def test_extract_bvh_hierarchy_and_rest_pose(self):
        info = glb_bridge.extract_skeleton(self.source, self.dir/"rig.bvh")
        self.assertEqual(info["joint_count"], 2)
        self.assertEqual(info["joint_names"], ["Hips", "Neck"])
        self.assertEqual(info["rest_frame"], -1)
        parsed = glb_bridge._parse_bvh_motion(Path(info["bvh_path"]))
        self.assertEqual(parsed["parents"], [-1, 0])
        self.assertEqual(parsed["frames"], 2)
        # Neck is rotated to world -X by original glTF root's bind rotation.
        np.testing.assert_allclose(parsed["offsets"][1], [-1, 0, 0], atol=1e-6)

    def test_hemisphere_continuity(self):
        q = np.array([[0., 0., 0., 1.], [0., 0., 0., -1.],
                      [0., 0., .3, -.9539392]])
        result = glb_bridge.hemisphere_continuous(q)
        self.assertTrue((np.sum(result[1:]*result[:-1], axis=-1) >= 0).all())
        np.testing.assert_allclose(np.linalg.norm(result, axis=-1), 1, atol=1e-6)

    def test_roundtrip_preserves_geometry_normals_tangents_and_material(self):
        from pygltflib import GLTF2
        bvh = self.dir/"skeleton.bvh"
        glb_bridge.extract_skeleton(self.source, bvh)
        text = bvh.read_text()
        # Modify second rest frame: root Z 45 degrees, Neck X 20 degrees,
        # giving nontrivial local quaternion keyframes with a fixed bind pose.
        lines = text.splitlines()
        last = lines[-1].split()
        self.assertEqual(len(last), 9)
        last[5] = "45.0"
        last[6] = "20.0"
        lines[-1] = " ".join(last)
        bvh.write_text("\n".join(lines)+"\n")
        before = GLTF2().load_binary(str(self.source))
        original_bin = before.binary_blob()
        original_views = [v.to_dict() for v in before.bufferViews]
        original_materials = [m.to_dict() for m in before.materials]
        original_primitive = before.meshes[0].primitives[0].to_dict()
        animated = self.dir/"animated.glb"
        result = glb_bridge.bake_motion(self.source, bvh, "walk", animated)
        self.assertEqual(result["joint_count"], 2)
        self.assertEqual(result["frame_count"], 2)
        out = GLTF2().load_binary(str(animated))
        self.assertEqual(out.binary_blob()[:len(original_bin)], original_bin)
        self.assertEqual([v.to_dict() for v in out.bufferViews[:len(original_views)]],
                         original_views)
        self.assertEqual([m.to_dict() for m in out.materials], original_materials)
        self.assertEqual(out.meshes[0].primitives[0].to_dict(), original_primitive)
        for attr in ("NORMAL", "TANGENT", "JOINTS_0", "WEIGHTS_0"):
            self.assertEqual(getattr(out.meshes[0].primitives[0].attributes, attr),
                             getattr(before.meshes[0].primitives[0].attributes, attr))
        self.assertEqual(out.materials[0].normalTexture.index, 0)
        animation = out.animations[-1]
        self.assertEqual(animation.name, "walk")
        self.assertEqual(len(animation.channels), 3)  # 2 rotations and 1 root translation
        for ch in animation.channels:
            if ch.target.path == "rotation":
                track = read_accessor(out, animation.samplers[ch.sampler].output)
                np.testing.assert_allclose(np.linalg.norm(track, axis=-1), 1, atol=1e-5)
                self.assertTrue((np.sum(track[:-1]*track[1:], axis=-1) >= 0).all())

    def test_refuse_nonrigid_bind_transforms(self):
        from pygltflib import GLTF2
        gltf = GLTF2().load_binary(str(self.source))
        gltf.nodes[1].scale = [2., 1., 1.]
        path = self.dir/"scaled.glb"
        gltf.save_binary(str(path))
        with self.assertRaisesRegex(ValueError, "scale"):
            glb_bridge.extract_skeleton(path, self.dir/"bad.bvh")


class APITests(unittest.TestCase):
    def test_json_schema(self):
        fields = {
            "bvh_path": "a.bvh", "npz_path": "a.npz", "gif_path": None,
            "fk_diag_score": .12, "joint_count": 24, "frame_count": 31,
            "fps": 30., "seed": 17, "elapsed_sec": 1.5, "cache_hit": True,
        }
        with mock.patch.object(comfy_nodes, "_preview", return_value="preview"):
            out = comfy_nodes._outputs({"results": [fields], "checkpoint_loads": 1})
        result = json.loads(out["result"][5])
        self.assertTrue(set(fields).issubset(result))
        self.assertEqual(out["result"][3], "")
        self.assertEqual(json.loads(out["result"][6])["checkpoint_loads"], 1)

    def test_render_gif_false_passes_to_worker(self):
        sample = "spider / 1"
        info = comfy_nodes.demo_catalog()[sample]
        record = {"results": [dict(bvh_path="x.bvh", npz_path="x.npz",
                                   gif_path=None, fk_diag_score=.1)],
                  "checkpoint_loads": 1}
        with (mock.patch.object(comfy_nodes, "_unload"),
              mock.patch.object(comfy_nodes, "checkpoint_path", return_value=Path("x.pt")),
              mock.patch.object(topx_runtime, "sample_motion_batch", return_value=record) as worker,
              mock.patch.object(comfy_nodes, "_outputs", return_value="ok")):
            node = comfy_nodes.TopXGenerateDemo()
            got = node.generate_demo(sample, -1, 0, 50, 2., False, "", 1800,
                                     render_gif=False)
        self.assertEqual(got, "ok")
        self.assertFalse(worker.call_args.kwargs["render_gif"])


if __name__ == "__main__":
    unittest.main()
