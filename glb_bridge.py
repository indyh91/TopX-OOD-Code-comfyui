"""Strict, Blender-free glTF 2.0 skin/BVH bridge for TopX.

Only rigid TRS (unit scale) joint chains are accepted. This is intentional:
discarding bind transforms or interpolating a node's matrix would silently
damage skinning. Original mesh, material and buffer-view payloads are untouched.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pygltflib import (
    GLTF2, Accessor, Animation, AnimationChannel, AnimationChannelTarget,
    AnimationSampler, BufferView,
)

AXES = {
    "+X": np.array([1., 0., 0.]), "-X": np.array([-1., 0., 0.]),
    "+Y": np.array([0., 1., 0.]), "-Y": np.array([0., -1., 0.]),
    "+Z": np.array([0., 0., 1.]), "-Z": np.array([0., 0., -1.]),
}


def canonical_basis(up: str, forward: str) -> np.ndarray:
    if up not in AXES or forward not in AXES or up[-1] == forward[-1]:
        raise ValueError("Up and forward axes must be orthogonal.")
    u, f = AXES[up], AXES[forward]
    return np.stack((np.cross(u, f), u, f))


def _unit_quaternion(q):
    q = np.asarray(q, dtype=np.float64)
    mag = float(np.linalg.norm(q))
    if q.shape != (4,) or mag < 1e-12 or not np.isfinite(q).all():
        raise ValueError("Invalid joint quaternion.")
    return q / mag


def quat_to_mat(q):
    x, y, z, w = _unit_quaternion(q)
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)],
    ])


def mat_to_quat(r):
    r = np.asarray(r, dtype=np.float64)
    if r.shape != (3, 3):
        raise ValueError("Rotation matrix has wrong shape.")
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or np.linalg.det(r) < 0.999:
        raise ValueError("Non-rigid rotation matrix encountered.")
    # Stable matrix-to-xyzw conversion, including 180-degree rotations.
    tr = float(np.trace(r))
    if tr > 0:
        s = 2. * math.sqrt(tr + 1.)
        q = [(r[2, 1]-r[1, 2])/s, (r[0, 2]-r[2, 0])/s,
             (r[1, 0]-r[0, 1])/s, s/4.]
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2. * math.sqrt(max(0., 1. + r[i, i] - r[j, j] - r[k, k]))
        if s < 1e-9:
            raise ValueError("Degenerate rotation matrix.")
        xyz = [0., 0., 0.]
        xyz[i] = s / 4.
        xyz[j] = (r[j, i] + r[i, j]) / s
        xyz[k] = (r[k, i] + r[i, k]) / s
        q = xyz + [(r[k, j] - r[j, k]) / s]
    return _unit_quaternion(q)


def hemisphere_continuous(quats):
    result = np.asarray(quats, dtype=np.float64).copy()
    for frame in range(len(result)):
        result[frame] = _unit_quaternion(result[frame])
        if frame and np.dot(result[frame], result[frame - 1]) < 0:
            result[frame] *= -1.
    return result


def _trs_field(value, length, default, name, node_index):
    # pygltflib represents omitted glTF TRS properties as either None or [].
    if value is None or len(value) == 0:
        return np.asarray(default, dtype=np.float64)
    out = np.asarray(value, dtype=np.float64)
    if out.shape != (length,) or not np.isfinite(out).all():
        raise ValueError(f"Node {node_index}: invalid {name} transform.")
    return out


def _local_matrix(node, *, node_index: int):
    if node.matrix is not None and len(node.matrix):
        m = np.array(node.matrix, dtype=np.float64).reshape(4, 4).T
        # Matrix-form joints cannot receive glTF animation channels. Ancestor
        # nodes are allowed if they contain a rigid matrix.
    else:
        translation = _trs_field(node.translation, 3, [0., 0., 0.], "translation", node_index)
        scale = _trs_field(node.scale, 3, [1., 1., 1.], "scale", node_index)
        if not np.allclose(scale, 1., atol=1e-6):
            raise ValueError(f"Node {node_index} has non-unit scale; refuse skin distortion.")
        rotation = quat_to_mat(_trs_field(node.rotation, 4, [0, 0, 0, 1], "rotation", node_index))
        m = np.eye(4)
        m[:3, :3] = rotation
        m[:3, 3] = translation
    if not np.isfinite(m).all() or not np.allclose(m[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"Non-affine transform at node {node_index}.")
    if not np.allclose(m[:3, :3].T @ m[:3, :3], np.eye(3), atol=1e-5) or np.linalg.det(m[:3, :3]) < .999:
        raise ValueError(f"Non-rigid transform at node {node_index}; cannot safely bake.")
    return m


@dataclass
class SkinRig:
    gltf: GLTF2
    joints: list[int]                 # DFS order (not necessarily skin.joints order)
    names: list[str]
    parents: list[int]                # parent indices in DFS order, root -1
    node_parents: list[int]           # glTF parents (root may have non-skin parent)
    world: np.ndarray                # original 4x4 world rest matrices in joint order
    positions: np.ndarray
    rest_rotations: np.ndarray
    skin_index: int


def load_skin(path: str | Path, skin_index: int = 0) -> SkinRig:
    path = Path(path)
    if not path.is_file() or path.suffix.lower() != ".glb":
        raise ValueError("Requires an existing .glb file (not an unrigged mesh).")
    gltf = GLTF2().load_binary(str(path))
    if not gltf.skins or not (0 <= skin_index < len(gltf.skins)):
        raise ValueError("GLB contains no glTF skin.")
    skin = gltf.skins[skin_index]
    members = set(map(int, skin.joints))
    if not members:
        raise ValueError("Skin has no joints.")
    parent = [-1] * len(gltf.nodes)
    for p, node in enumerate(gltf.nodes):
        for c in node.children or []:
            if not 0 <= c < len(gltf.nodes) or parent[c] != -1:
                raise ValueError("Invalid or multiply parented glTF node hierarchy.")
            parent[c] = p
    roots = [i for i in members if parent[i] not in members]
    if len(roots) != 1:
        raise ValueError("Skin must have exactly one joint root.")
    root = roots[0]
    ordered = []
    def visit(k):
        ordered.append(k)
        for child in gltf.nodes[k].children or []:
            if child in members:
                visit(child)
            elif any(_descendant_in_skin(child, gltf.nodes, members)):
                raise ValueError("Non-joint nodes between skin joints are unsupported (unsafe bind transform).")
    visit(root)
    if set(ordered) != members:
        raise ValueError("Skin joints are disconnected or cannot be ordered as one tree.")
    if any(gltf.nodes[i].matrix is not None and len(gltf.nodes[i].matrix) for i in members):
        raise ValueError("Matrix-form glTF joints cannot be animated via TRS channels.")
    names = [(gltf.nodes[i].name or f"joint_{i}").strip() for i in ordered]
    if len(set(names)) != len(names) or any(not n or re.search(r"[\r\n{}]", n) for n in names):
        raise ValueError("Joint names must be unique and valid BVH identifiers.")
    world_all = {}
    active = set()
    def world(k):
        if k in world_all:
            return world_all[k]
        if k in active:
            raise ValueError("Cycle in glTF node hierarchy.")
        active.add(k)
        m = _local_matrix(gltf.nodes[k], node_index=k)
        if parent[k] >= 0:
            m = world(parent[k]) @ m
        world_all[k] = m
        active.remove(k)
        return m
    matrices = np.stack([world(i) for i in ordered])
    index = {v: i for i, v in enumerate(ordered)}
    parents = [index.get(parent[i], -1) for i in ordered]
    if parents[0] != -1 or any(p < 0 or p >= j for j, p in enumerate(parents) if j):
        raise ValueError("Non-topological skin joint order.")
    return SkinRig(
        gltf=gltf, joints=ordered, names=names, parents=parents,
        node_parents=parent, world=matrices,
        positions=matrices[:, :3, 3].copy(),
        rest_rotations=matrices[:, :3, :3].copy(),
        skin_index=skin_index,
    )


def _descendant_in_skin(node_id, nodes, members):
    if node_id in members:
        yield node_id
    for kid in nodes[node_id].children or []:
        yield from _descendant_in_skin(kid, nodes, members)


def _render_skeleton_bvh(rig: SkinRig) -> str:
    children = {j: [] for j in range(len(rig.joints))}
    for j in range(1, len(rig.joints)):
        children[rig.parents[j]].append(j)
    lines = ["HIERARCHY"]
    def render(j, depth):
        pref = "  " * depth
        off = rig.positions[j] if j == 0 else rig.positions[j] - rig.positions[rig.parents[j]]
        lines.extend([f"{pref}{'ROOT' if j == 0 else 'JOINT'} {rig.names[j]}",
                      f"{pref}{{",
                      f"{pref}  OFFSET {off[0]:.9f} {off[1]:.9f} {off[2]:.9f}",
                      f"{pref}  CHANNELS {6 if j == 0 else 3} " +
                      ("Xposition Yposition Zposition " if j == 0 else "") +
                      "Zrotation Xrotation Yrotation"])
        if children[j]:
            for child in children[j]:
                render(child, depth + 1)
        else:
            direction = off if j else np.array([0., .1, 0.])
            if np.linalg.norm(direction) < 1e-7:
                direction = np.array([0., .1, 0.])
            direction = .25 * direction
            lines.extend([f"{pref}  End Site", f"{pref}  {{",
                          f"{pref}    OFFSET {direction[0]:.9f} {direction[1]:.9f} {direction[2]:.9f}",
                          f"{pref}  }}"])
        lines.append(f"{pref}}}")
    render(0, 0)
    # Zero rotations, two identical frames; root has explicit coordinates.
    root = rig.positions[0]
    frame = np.zeros(3 + 3 * len(rig.joints), dtype=np.float64)
    frame[:3] = root
    line = " ".join(f"{v:.9f}" for v in frame)
    lines.extend(["MOTION", "Frames: 2", "Frame Time: 0.033333", line, line])
    return "\n".join(lines) + "\n"


def extract_skeleton(glb_path: str | Path, output_bvh: str | Path,
                     up_axis: str = "+Y", forward_axis: str = "+Z") -> dict:
    canonical_basis(up_axis, forward_axis)
    rig = load_skin(glb_path)
    output = Path(output_bvh)
    if output.resolve() == Path(glb_path).resolve():
        raise ValueError("Output path must differ from input.")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(_render_skeleton_bvh(rig), encoding="utf-8")
    tmp.replace(output)
    return {
        "bvh_path": str(output.resolve()),
        "joint_names": rig.names,
        "joint_count": len(rig.joints),
        "up_axis": up_axis, "forward_axis": forward_axis, "rest_frame": -1,
        "skin_index": rig.skin_index,
    }


def _axis_rotation(axis: str, degrees: np.ndarray):
    theta = np.deg2rad(degrees)
    c, s = np.cos(theta), np.sin(theta)
    a = axis.lower()
    result = np.zeros((len(theta), 3, 3), dtype=np.float64)
    if a == "x":
        result[:, 0, 0] = 1.; result[:, 1, 1] = c; result[:, 1, 2] = -s
        result[:, 2, 1] = s; result[:, 2, 2] = c
    elif a == "y":
        result[:, 1, 1] = 1.; result[:, 0, 0] = c; result[:, 0, 2] = s
        result[:, 2, 0] = -s; result[:, 2, 2] = c
    else:
        result[:, 2, 2] = 1.; result[:, 0, 0] = c; result[:, 0, 1] = -s
        result[:, 1, 0] = s; result[:, 1, 1] = c
    return result


def _parse_bvh_motion(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    names, parents, offsets, channels, stack = [], [], [], [], []
    idx = 0
    if not lines or lines[0].strip().upper() != "HIERARCHY":
        raise ValueError("Invalid BVH hierarchy.")
    idx = 1
    pending = None
    while idx < len(lines):
        line = lines[idx].strip()
        idx += 1
        if not line:
            continue
        parts = line.split()
        op = parts[0].upper()
        if op == "MOTION":
            break
        if op in ("ROOT", "JOINT", "END"):
            pending = (None if op == "END" else line.split(None, 1)[1].strip(), op == "END")
        elif line == "{":
            if pending is None:
                raise ValueError("Unexpected BVH brace.")
            name, end = pending
            names.append(name)
            parents.append(stack[-1] if stack else -1)
            channels.append([])
            offsets.append(np.zeros(3))
            stack.append(len(names)-1)
            pending = None
        elif line == "}":
            if not stack:
                raise ValueError("Unexpected BVH closing brace.")
            stack.pop()
        elif op == "OFFSET":
            offsets[stack[-1]] = np.array(list(map(float, parts[1:4])))
        elif op == "CHANNELS":
            n = int(parts[1])
            channels[stack[-1]] = parts[2:2+n]
            if len(channels[stack[-1]]) != n:
                raise ValueError("Malformed BVH channels.")
        else:
            raise ValueError(f"Unsupported BVH syntax: {line}")
    if stack or idx >= len(lines):
        raise ValueError("Incomplete BVH.")
    if not lines[idx].strip().lower().startswith("frames:"):
        raise ValueError("Missing BVH frame count.")
    nframes = int(lines[idx].split(":", 1)[1].strip()); idx += 1
    if not lines[idx].strip().lower().startswith("frame time:"):
        raise ValueError("Missing BVH frame time.")
    dt = float(lines[idx].split(":", 1)[1].strip()); idx += 1
    nch = sum(len(ch) for ch in channels)
    numbers = np.fromstring(" ".join(lines[idx:]), sep=" ")
    if nframes < 1 or dt <= 0 or numbers.size != nframes*nch or not np.isfinite(numbers).all():
        raise ValueError("Invalid BVH motion frames.")
    rows = numbers.reshape(nframes, nch)
    joint_ids = [i for i, n in enumerate(names) if n is not None]
    m = {index: pos for pos, index in enumerate(joint_ids)}
    joint_names = [names[i] for i in joint_ids]
    joint_parents = [m.get(parents[i], -1) for i in joint_ids]
    local_r = np.zeros((nframes, len(joint_ids), 3, 3), dtype=np.float64)
    local_r[:] = np.eye(3)
    root_pos = None
    cursor = 0
    for i, ch in enumerate(channels):
        values = rows[:, cursor:cursor+len(ch)]
        cursor += len(ch)
        if i not in m:
            continue
        j = m[i]
        rotation_channels = [c for c in ch if c.lower().endswith("rotation")]
        if len(rotation_channels) != 3 or sorted(c[0].lower() for c in rotation_channels) != ["x", "y", "z"]:
            raise ValueError("BVH joint rotation must have X/Y/Z channels.")
        for k, c in enumerate(ch):
            if c.lower().endswith("rotation"):
                local_r[:, j] = local_r[:, j] @ _axis_rotation(c[0], values[:, k])
        if j == 0:
            coordinate = {c[0].lower(): values[:, k] for k, c in enumerate(ch)
                          if c.lower().endswith("position")}
            if sorted(coordinate) != ["x", "y", "z"]:
                raise ValueError("BVH root must contain X/Y/Z position channels.")
            root_pos = np.stack([coordinate[a] for a in "xyz"], axis=1)
    world_r = local_r.copy()
    for j in range(1, len(joint_ids)):
        world_r[:, j] = world_r[:, joint_parents[j]] @ local_r[:, j]
    return {
        "names": joint_names, "parents": joint_parents,
        "offsets": np.stack([offsets[i] for i in joint_ids]),
        "rotation_world_delta": world_r, "root_world": root_pos,
        "fps": 1./dt, "frames": nframes,
    }


def _load_motion(path: Path, rig: SkinRig, *, up_axis: str, forward_axis: str):
    C = canonical_basis(up_axis, forward_axis)
    if path.suffix.lower() == ".bvh":
        data = _parse_bvh_motion(path)
        if data["names"] != rig.names or data["parents"] != rig.parents:
            raise ValueError("BVH joint names/order or hierarchy do not match source GLB skin.")
        desired = np.stack([rig.positions[j] if j == 0 else
                            rig.positions[j]-rig.positions[rig.parents[j]]
                            for j in range(len(rig.names))])
        if not np.allclose(data["offsets"], desired, atol=2e-4, rtol=1e-5):
            raise ValueError("BVH rest offsets do not match source GLB skin.")
        return data["rotation_world_delta"], data["root_world"], data["fps"]
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as z:
            names = [str(x) for x in z["joint_names"]]
            parents = list(map(int, z["parents"]))
            if names != rig.names or parents != rig.parents:
                raise ValueError("NPZ joint names/order or hierarchy do not match GLB skin.")
            P = np.asarray(z["P_rest"], dtype=np.float64)
            W = np.asarray(z["rest_delta_rotations"], dtype=np.float64)
            direct = np.asarray(z["positions_direct"], dtype=np.float64)
            fps = float(z["fps"])
        expected = (rig.positions @ C.T)
        expected -= np.array([expected[0, 0], expected[:, 1].min(), expected[0, 2]])
        if P.shape != expected.shape or not np.allclose(P, expected, atol=2e-4, rtol=1e-5):
            raise ValueError("NPZ rest pose does not match the source GLB and selected axes.")
        if W.ndim != 4 or W.shape[1:] != (len(rig.names), 3, 3) or direct.shape != (W.shape[0], len(rig.names), 3):
            raise ValueError("Malformed TopX NPZ motion data.")
        world_delta = np.einsum("ab,tjbc,cd->tjad", C.T, W, C)
        delta = (direct[:, 0] - P[0]) @ C
        root_world = rig.positions[0] + delta
        return world_delta, root_world, fps
    raise ValueError("Motion must be TopX-generated .bvh or .npz.")


def bake_motion(source_glb_path: str | Path, motion_path: str | Path,
                clip_name: str = "TopX", output_glb_path: str | Path | None = None,
                up_axis: str = "+Y", forward_axis: str = "+Z") -> dict:
    src = Path(source_glb_path)
    motion = Path(motion_path)
    rig = load_skin(src)
    if not motion.is_file():
        raise FileNotFoundError(motion)
    delta, root_world, fps = _load_motion(motion, rig, up_axis=up_axis, forward_axis=forward_axis)
    if not np.isfinite(delta).all() or not np.isfinite(root_world).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Motion contains non-finite or invalid values.")
    if delta.shape[0] < 1:
        raise ValueError("No animation frames.")
    gltf = rig.gltf
    if len(gltf.buffers) != 1 or gltf.buffers[0].uri:
        raise ValueError("Only GLB assets with a single embedded buffer are supported.")
    original = gltf.binary_blob() or b""
    if len(original) < int(gltf.buffers[0].byteLength):
        raise ValueError("Truncated GLB binary buffer.")
    # Original BIN prefix, buffer views, skin inverse bind matrices, all geometry
    # and materials remain untouched. New animation data only ever appends.
    original_views = [v.to_dict() for v in gltf.bufferViews]
    original_materials = [m.to_dict() for m in gltf.materials]
    original_primitives = [[p.to_dict() for p in mesh.primitives] for mesh in gltf.meshes]
    original_skin = [s.to_dict() for s in gltf.skins]
    original_accessors = [a.to_dict() for a in gltf.accessors]
    nframes = delta.shape[0]
    target_world = np.einsum("tjab,jbc->tjac", delta, rig.rest_rotations)
    rotations = np.empty((nframes, len(rig.joints), 4), dtype=np.float32)
    for j, nodeid in enumerate(rig.joints):
        parent_j = rig.parents[j]
        # Non-joint ancestry before the skin root stays at its rest transform.
        parent_world = (target_world[:, parent_j] if parent_j >= 0 else
                        np.broadcast_to(
                            np.linalg.inv(rig.world[0])[:3, :3] @ rig.rest_rotations[0],
                            (nframes, 3, 3),
                        ))
        if parent_j < 0:
            ancestor = rig.node_parents[nodeid]
            if ancestor < 0:
                parent_world = np.broadcast_to(np.eye(3), (nframes, 3, 3))
            else:
                # Reconstruct rigid ancestor world matrix by inverting root local.
                node_local = _local_matrix(gltf.nodes[nodeid], node_index=nodeid)
                parent_world = np.broadcast_to(
                    rig.rest_rotations[0] @ node_local[:3, :3].T, (nframes, 3, 3)
                )
        local_target = np.swapaxes(parent_world, -1, -2) @ target_world[:, j]
        rotations[:, j] = hemisphere_continuous([mat_to_quat(r) for r in local_target])
    rootid = rig.joints[0]
    root_parent = rig.node_parents[rootid]
    if root_parent < 0:
        parent_r = np.eye(3)
    else:
        local_root = _local_matrix(gltf.nodes[rootid], node_index=rootid)
        parent_r = rig.rest_rotations[0] @ local_root[:3, :3].T
    original_translation = _trs_field(gltf.nodes[rootid].translation, 3, [0., 0., 0.],
                                      "translation", rootid)
    root_local = original_translation + (root_world - rig.positions[0]) @ parent_r
    if not np.isfinite(rotations).all() or not np.isfinite(root_local).all():
        raise ValueError("Invalid resulting GLB animation tracks.")
    blob = bytearray(original)
    def append_accessor(array, kind):
        while len(blob) % 4:
            blob.append(0)
        start = len(blob)
        data = np.ascontiguousarray(array, dtype="<f4")
        blob.extend(data.tobytes())
        viewid = len(gltf.bufferViews)
        gltf.bufferViews.append(BufferView(buffer=0, byteOffset=start, byteLength=data.nbytes))
        aid = len(gltf.accessors)
        rows = data.reshape(data.shape[0], -1)
        kwargs = dict(bufferView=viewid, byteOffset=0, componentType=5126,
                      count=len(rows), type=kind)
        if kind == "SCALAR":
            kwargs["min"] = [float(rows.min())]
            kwargs["max"] = [float(rows.max())]
        gltf.accessors.append(Accessor(**kwargs))
        return aid
    times = np.arange(nframes, dtype=np.float32) / float(fps)
    timeid = append_accessor(times, "SCALAR")
    samplers, channels = [], []
    def channel(nodeid, path, values, kind):
        aid = append_accessor(values, kind)
        si = len(samplers)
        samplers.append(AnimationSampler(input=timeid, output=aid, interpolation="LINEAR"))
        channels.append(AnimationChannel(sampler=si, target=AnimationChannelTarget(node=nodeid, path=path)))
    for j, nodeid in enumerate(rig.joints):
        channel(nodeid, "rotation", rotations[:, j], "VEC4")
    channel(rootid, "translation", root_local.astype(np.float32), "VEC3")
    gltf.animations.append(Animation(name=clip_name or "TopX", samplers=samplers, channels=channels))
    gltf.buffers[0].byteLength = len(blob)
    gltf.set_binary_blob(bytes(blob))
    if [v.to_dict() for v in gltf.bufferViews[:len(original_views)]] != original_views:
        raise RuntimeError("Original buffer views changed unexpectedly.")
    changed = []
    if [m.to_dict() for m in gltf.materials] != original_materials:
        changed.append("materials")
    if [[p.to_dict() for p in mesh.primitives] for mesh in gltf.meshes] != original_primitives:
        changed.append("mesh primitives")
    if [skin.to_dict() for skin in gltf.skins] != original_skin:
        changed.append("skins")
    if [a.to_dict() for a in gltf.accessors[:len(original_accessors)]] != original_accessors:
        changed.append("accessors")
    if gltf.binary_blob()[:len(original)] != original:
        changed.append("original BIN payload")
    if changed:
        raise RuntimeError("Original-data preservation invariant failed: " + ", ".join(changed))
    dest = Path(output_glb_path) if output_glb_path else src.with_name(src.stem + "_topx.glb")
    if dest.resolve() == src.resolve():
        raise ValueError("Refusing to overwrite the original GLB.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.glb")
    gltf.save_binary(str(tmp))
    check = GLTF2().load_binary(str(tmp))
    if ((check.binary_blob() or b"")[:len(original)] != original or
        [m.to_dict() for m in check.materials] != original_materials or
        [[p.to_dict() for p in mesh.primitives] for mesh in check.meshes] != original_primitives or
        [s.to_dict() for s in check.skins] != original_skin):
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Serialized GLB failed original-data preservation check.")
    tmp.replace(dest)
    return {
        "animated_glb_path": str(dest.resolve()),
        "clip_name": clip_name or "TopX",
        "joint_count": len(rig.joints), "frame_count": nframes,
        "fps": float(fps), "preserved_binary_prefix_bytes": len(original),
        "quaternion_continuity": True,
        "skin_index": rig.skin_index,
    }
