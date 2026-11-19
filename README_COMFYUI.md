# TopX ComfyUI — cached encoding, batch motion and rigged GLB

Windows-native custom nodes. Core TopX model weights, the KTJD-17 motion
representation, skeleton coordinate conventions, the original BVH exporter,
and the upstream scripts/generate.py remain unchanged.

## Installation

Clone or update this repository directly in the active ComfyUI custom_nodes
directory, using the existing ComfyUI Python environment.

~~~powershell
git switch main
git pull origin main
& "<ComfyUI Python.exe>" -m pip install -r ".\requirements.txt"
~~~

During development, switch to the production-cache-batch-glb-bridge branch,
which contains this pull request. Restart ComfyUI after updating.

Download the official checkpoint from
https://huggingface.co/Tevior/TopX-OOD/blob/main/topx.pt
into the configured ComfyUI models/topx/topx.pt directory. Alternatively
specify an absolute checkpoint_path on the node. Only use trusted PyTorch
checkpoints.

The core requirements include pygltflib, NumPy, SciPy and Pillow, but do
not install or replace ComfyUI's PyTorch CUDA stack.

## Six nodes

| Node | Function |
| --- | --- |
| TopX Generate Demo Motion | Generate with the included demonstration skeleton and embeddings |
| TopX Demo Inputs | Select stored demonstration inputs for a modular graph |
| TopX Encode Embeddings (LLM2Vec) | Encode one or many prompts and optional joint semantics with disk caching |
| TopX Generate Motion (BVH) | Generate one or many BVH clips and structured JSON responses |
| TopX Extract Skeleton from GLB | Extract rest pose from an already skinned GLB |
| TopX Bake Motion to GLB | Add glTF rotation/translation animation to the original skinned GLB |

### Demo smoke test

Load workflows/topx_demo_spider.json in ComfyUI. Choose spider / 1, queue,
and examine the generated BVH/NPZ/GIF under ComfyUI/output/topx/.

Demos use the published precomputed embeddings; no LLM2Vec is required.

### Arbitrary text with a custom skeleton

Text encoder installation is optional. To minimize dependency disruption,
install LLM2Vec without pip's dependency resolver:

~~~powershell
& "<ComfyUI Python.exe>" -m pip install --no-deps -r ".\requirements-text.txt"
~~~

This does not guarantee compatibility with the ComfyUI environment's
existing transformers, peft, tokenizers, or other dependencies. Check those
against the encoder's runtime requirements independently. The original
Meta-Llama-3-8B-Instruct model is gated and the published McGill-NLP MNTP
and supervised LLM2Vec adapters are also required.

The encoding node accepts a single prompt or one prompt per line.
The optional skeleton_path may point to a BVH, TopX skeleton NPZ or already
rigged GLB. It returns the first text embedding file, joint embedding file,
a resolved skeleton path, all text embedding paths as JSON, cache metadata
and a cache_hit boolean.

Default cache: ComfyUI/output/topx/_cache/; you can override cache_dir.
Text SHA-1 identifiers include the **effective** prompt and complete
encoder model ID. Joint SHA-1 identifiers also include the skeleton content
digest, axis settings, rest pose options, body plan and lexicon digest.
Joint names alone are NOT a valid joint-cache identity: different rigs
can share names but have different geometry-dependent descriptions.

Before spawning LLM2Vec, the integration checks stored arrays for float32
dtype, expected shape and finite values. A full hit returns immediately
without loading LLM2Vec or unnecessarily unloading ComfyUI GPU models.
Any missing captions and joint descriptions are encoded in one isolated
child process, saved with atomic replacement and released when the worker
exits. Encoding and sampling do not coexist in a single process.

The default auto_prefix_caption option converts "walks forward." to
"An object walks forward." before hashing and embedding; the original
prompt is also retained in metadata.

To use the encoding node with TopX Generate Motion (BVH), set the
generation node to precomputed mode, leave its prompt empty, and link
the text_embedding_path, joint_embedding_path and skeleton_path outputs.

### Batch generation and headless results

The generator also supports llm2vec directly, launching an isolated
encoding worker only on a cache miss. Multiple newline-separated captions
generate one candidate per caption with incrementing seeds.

For batch sweeps, select precomputed mode and supply specs_json as an
array of dictionaries:

~~~json
[
  {"text_embedding_path": "F:\\topx_cache\\text\\example.npy", "seed": 7, "num_frames": 120},
  {"text_embedding_path": "F:\\topx_cache\\text\\example.npy", "seed": 19, "num_frames": 90}
]
~~~

Set joint_embedding_path and skeleton_path once for the batch.
Per-clip overrides include steps, cfg_text, prompt, raw_prompt and cache_hit.

The checkpoint loads once in the sampling worker and is reused sequentially
for every candidate. This is checkpoint-resident batching, **not**
simultaneous/vectorized generation. It preserves the one-active-runtime
assumption.

The first result is returned as result_json; batch_result_json contains
all candidates. Each candidate object includes:

- bvh_path, npz_path, gif_path (null when skipped)
- fk_diag_score, joint_count, frame_count, fps
- seed, elapsed_sec, cache_hit
- prompt and raw_prompt metadata where available

render_gif defaults to True in the UI and False for the Python batch API.
With rendering disabled, only BVH/NPZ are written; the preview IMAGE is
rendered as a lightweight static skeleton preview in memory.

Worker stdout/stderr go to topx.log, never to capture pipes. Workers have
bounded timeouts with Windows process-tree termination. New runs use
unique output directories to prevent overwriting previous results.

**Do not treat fk_diag_score as a perceptual quality score.** It is the
maximum absolute direct-vs-FK joint-position difference normalized by
skeleton size. Upstream warns above 0.3. A clip with a low score can still
have bad foot contact, penetration, motion ambiguity, or animation quality.

## GLB end-to-end workflow

Load workflows/topx_glb_end_to_end.json and set the source GLB path in both
the Extract and Bake nodes.

1. Use a character GLB that ALREADY has a valid skin and weights.
2. Extract Skeleton reads skins[0] and the joint hierarchy, derives
   world-space rest offsets from actual rigid node transforms and writes
   a BVH HIERARCHY plus a two-frame zero-motion rest stub. Use rest_frame -1.
3. Encode the caption and joint semantics from the extracted BVH.
4. Generate BVH with TopX in precomputed mode.
5. Bake onto the original GLB, preserving the mesh, original binary
   payload, primitive attributes, skin mappings, inverse bind matrices,
   materials and normalTexture references.

The baker composes TopX's world-space rest-relative rotations with the
original glTF bind rotations, derives local joint rotations, normalizes
XYZW-order **[x,y,z,w]** quaternions and ensures adjacent hemisphere
continuity. Root displacement is converted to the source glTF parent frame.
The output is a NEW GLB; the source is never overwritten.

### Fail-closed GLB limitations

The source must have one connected joint root and rigid skin transforms
(unit scale). Matrix-form joints, non-joint nodes interleaved between
skin joints, disconnected skins, and unrigged meshes are rejected instead
of approximated. This package is not a rigging system. Source GLB and
TopX BVH/NPZ must have matching skeleton identities, topology and rest
offsets. The initial GLB implementation is for the first skin only.
Always inspect an animated result before production use.

## Workflow examples

- workflows/topx_demo_spider.json
- workflows/topx_custom_bvh_llm2vec.json
- workflows/topx_glb_end_to_end.json

The example paths beginning C:\path\to\... are placeholders. Change to
your actual file paths. The visual UI graphs need an on-machine ComfyUI
import/inference validation in addition to their JSON syntax checks.

## Tests

Run tests through the Python environment that starts ComfyUI:

~~~powershell
& "<ComfyUI Python.exe>" -m compileall -q __init__.py comfy_nodes.py topx_runtime.py glb_bridge.py scripts src tests
& "<ComfyUI Python.exe>" -m pytest tests/test_comfy_nodes.py -q
~~~

The CPU suite includes an artificially constructed skinned GLB and checks
joint topology, nonidentity rest rotations, quaternion continuity, NORMAL,
TANGENT, JOINTS_0, WEIGHTS_0, normalTexture, and unchanged original BIN bytes.

Optional **live CUDA** test, when the official checkpoint exists locally:

~~~powershell
$env:TOPX_RUN_LIVE_TESTS = "1"
$env:TOPX_CHECKPOINT = "<ComfyUI models>\topx\topx.pt"
& "<ComfyUI Python.exe>" -m pytest tests/test_live_inference.py -v -s
~~~

Runs the published spider / 1 input with render_gif disabled, asserts
finite rotations, nonempty output files and FK diagnostic less than 0.3,
and prints elapsed time. Without a CUDA environment and real checkpoint,
these quality/performance claims remain unverified.

## Licensing

Code: MIT. Original TopX model weights: CC BY-NC 4.0 (noncommercial).
Demo assets retain their upstream licences. The integration does not grant
commercial licensing for the published TopX weights.
