# TopX for ComfyUI (Windows-native)

This fork is a ComfyUI custom-node package. It wraps the upstream TopX inference
script without changing model weights, motion representation, skeleton axis
conventions, or the BVH exporter.

## 1. Install the node package

Place this repository **directly** inside the custom_nodes directory used by
your active ComfyUI Desktop installation:

~~~text
<ComfyUI custom_nodes>/
  TopX-OOD-Code-comfyui/
    __init__.py
    comfy_nodes.py
    scripts/generate.py
    src/
    demo/
~~~

For a fresh installation, clone it into that directory:

~~~powershell
cd "<your ComfyUI custom_nodes directory>"
git clone https://github.com/indyh91/TopX-OOD-Code-comfyui.git
~~~

If using the integration pull-request branch before it is merged:

~~~powershell
git -C ".\TopX-OOD-Code-comfyui" switch comfyui-native-topx-nodes
~~~

Install only the small runtime dependencies **using the exact Python environment
that launches ComfyUI**, not system Python:

~~~powershell
& "<ComfyUI Python executable>" -m pip install -r "<ComfyUI custom_nodes>\TopX-OOD-Code-comfyui\requirements.txt"
~~~

ComfyUI provides its own CUDA-enabled PyTorch; this package does not
reinstall or upgrade it. Restart ComfyUI after installation.

No WSL, Linux VM, or separate inference server is required.

## 2. Download the TopX checkpoint

Get the original authors' topx.pt from:

https://huggingface.co/Tevior/TopX-OOD/blob/main/topx.pt

Put it at **folder_paths.models_dir/topx/topx.pt** in the active ComfyUI
installation. Commonly, this is the models/topx/topx.pt subdirectory under
ComfyUI's configured model directory. Create topx if needed. The nodes also
accept an explicit absolute checkpoint_path and can use this repository's
weights/topx.pt if present.

Only download weights from a source you trust: the upstream loader uses
PyTorch checkpoint deserialization.

## 3. First smoke test (no LLM2Vec required)

1. Add **TopX > motion > TopX Generate Demo Motion**.
2. Choose **spider / 1** (or another included example).
3. Leave the reference seed and frame-count overrides unchanged.
4. Queue the prompt.

This uses the exact upstream demonstration skeleton and matching
text_emb_1.npy and joint_sem.npy. It does not load Meta-Llama-3-8B, or
pretend that a CLIP/other embedding is equivalent.

Outputs appear under ComfyUI's output directory:

~~~text
output/topx/<run name>/
  <run name>.bvh          # animation for importing into Blender
  <run name>.npz          # positions, rotations, metadata
  <run name>.gif          # motion preview
  <run name>.descriptions.json
  topx.log                # complete worker logs
~~~

The node returns an IMAGE preview, BVH path, NPZ path, GIF path and a diagnostic
string. The IMAGE is a still from the GIF, **not** a playable video. The BVH is
the animated skeleton, **not** a skinned mesh. Each run writes to a new folder;
previous results are never silently overwritten.

For a more modular graph, connect the outputs of **TopX Demo Inputs** to
**TopX Generate Motion (BVH)**: skeleton_path, text_embedding_path,
joint_embedding_path, reference_frames, reference_seed, and
keep_rest_rotations. Set text_source to precomputed and leave prompt
**empty**. The reference_caption output is documentation only; connecting it
to prompt in precomputed mode is deliberately rejected.

## 4. A custom skeleton and prompt

**TopX Generate Motion (BVH)** accepts a BVH skeleton (.bvh) or a TopX
skeleton .npz in the upstream format. Set up and forward to the axes of
the source BVH, and set rest_frame to the frame that actually contains the
rest pose. Use -1 only if the BVH OFFSET hierarchy with zero rotations
represents the rest pose.

For **arbitrary new text** or missing joint-semantic embeddings, first
install the *optional* LLM2Vec dependencies in the same ComfyUI Python
environment:

~~~powershell
& "<ComfyUI Python executable>" -m pip install -r "<ComfyUI custom_nodes>\TopX-OOD-Code-comfyui\requirements-text.txt"
~~~

You must also have access to the gated Meta-Llama-3-8B base model and the
LLM2Vec MNTP + supervised adapters required by upstream TopX. This is a
large additional model (about 16 GB in bf16). Switch text_source to llm2vec,
enter a prompt (the training-style caption begins with "An object"), and
leave text_embedding_path blank. Either provide a correct
joint_embedding_path or leave it blank to encode joint descriptions. The
model may download encoder weights using the Hugging Face credentials of
the ComfyUI Python process. This mode can have substantial memory usage.

For **precomputed** mode both embeddings are mandatory, with exact shapes
[4096] and [joint count,4096] respectively. Embeddings from ComfyUI's
CLIP, Qwen, Wan or other text encoders are **not compatible**. Entering a
new prompt while providing a stored embedding would silently generate
the *stored* prompt's motion; this node therefore rejects that combination.

The TopX generator itself supports up to the checkpoint's target length
(typically 240 frames) and produces motion at 30 FPS.

## 5. VRAM, safety and diagnostics

The integration runs upstream scripts/generate.py as a bounded child process
with the **same Python executable as ComfyUI**. This avoids Python module
namespace collisions with other custom-node packages, and keeps Torch RNG
and optional LLM2Vec state isolated. Worker output goes to topx.log instead
of pipes, so inherited pipe handles cannot wedge ComfyUI on timeout.

release_comfy_vram=true calls ComfyUI's registered-model unload/cache-clear
functions before launching TopX. This can make other ComfyUI models reload
later. It does **not** guarantee memory held by non-ComfyUI Python objects
or other applications is released. The timeout terminates the child process
tree on Windows; an unsuccessful run keeps its log for diagnosis.

The diagnostic is the maximum absolute direct-vs-forward-kinematics position
difference divided by the rest rig's bounding-box diagonal. Upstream prints
a warning above 0.3. It is **not** a calibrated motion-quality score and
should not be used as the sole criterion for auto-selecting a generated clip.

TopX will not import or rig a GLB/FBX mesh for you. To animate a ComfyUI
generated GLB, first extract or build a valid **BVH skeleton** with correct
rest pose, forward axis and joint naming, then transfer the resulting motion
to the skinned character in Blender. Skeleton correspondence and skinning
are separate operations; directly applying BVH rotations to an unrelated
armature is not guaranteed to work.

## 6. Test and licensing status

Static/contract checks are in tests/test_comfy_nodes.py. These tests do not
validate visual quality or run the actual model. A complete inference test
requires the downloaded checkpoint and a working CUDA/PyTorch environment.

**Licensing:** source code is MIT. The TopX weights are CC BY-NC 4.0
(noncommercial), so integrating the model into a commercial game pipeline
does not confer commercial model-use rights. The demo skeletons retain
their upstream third-party licences.
