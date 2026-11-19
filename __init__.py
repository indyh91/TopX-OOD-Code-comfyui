"""TopX motion generation custom nodes for ComfyUI."""

from .comfy_nodes import TopXDemoInputs, TopXGenerateDemo, TopXGenerateMotion

NODE_CLASS_MAPPINGS = {
    "TopXDemoInputs": TopXDemoInputs,
    "TopXGenerateMotion": TopXGenerateMotion,
    "TopXGenerateDemo": TopXGenerateDemo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TopXDemoInputs": "TopX Demo Inputs",
    "TopXGenerateMotion": "TopX Generate Motion (BVH)",
    "TopXGenerateDemo": "TopX Generate Demo Motion",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
