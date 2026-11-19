"""TopX ComfyUI package.

ComfyUI loads this directory as a package. Pytest also imports the root
__init__.py as a top-level module; only register nodes in package context.
"""
if __package__:
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
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
