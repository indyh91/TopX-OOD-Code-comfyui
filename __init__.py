"""Native ComfyUI integration for TopX motion sampling and skinned glTF."""
from .comfy_nodes import (
    TopXDemoInputs, TopXEncodeEmbeddings, TopXGenerateMotion,
    TopXGenerateDemo, TopXExtractSkeletonGLB, TopXBakeMotionGLB,
)

NODE_CLASS_MAPPINGS = {
    "TopXDemoInputs": TopXDemoInputs,
    "TopXEncodeEmbeddings": TopXEncodeEmbeddings,
    "TopXGenerateMotion": TopXGenerateMotion,
    "TopXGenerateDemo": TopXGenerateDemo,
    "TopXExtractSkeletonGLB": TopXExtractSkeletonGLB,
    "TopXBakeMotionGLB": TopXBakeMotionGLB,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TopXDemoInputs": "TopX Demo Inputs",
    "TopXEncodeEmbeddings": "TopX Encode Embeddings (LLM2Vec)",
    "TopXGenerateMotion": "TopX Generate Motion (BVH)",
    "TopXGenerateDemo": "TopX Generate Demo Motion",
    "TopXExtractSkeletonGLB": "TopX Extract Skeleton from GLB",
    "TopXBakeMotionGLB": "TopX Bake Motion to GLB",
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
