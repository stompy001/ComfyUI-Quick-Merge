import sys
import subprocess
import importlib

def install_dependencies():
    required_packages = {
        "transformers": "transformers",
        "torchvision": "torchvision",
        "huggingface_hub": "huggingface-hub"
    }
    
    missing_packages = []
    for module_name, pip_name in required_packages.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing_packages.append(pip_name)
            
    if missing_packages:
        print(f"\n[🎨 Quick Merge] Missing dependencies detected: {', '.join(missing_packages)}")
        print(f"[🎨 Quick Merge] Auto-installing now. Please wait...\n")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
        except subprocess.CalledProcessError:
            print(f"\n[🎨 Quick Merge] ERROR: Failed auto-installation. Run: pip install {' '.join(missing_packages)}\n")

install_dependencies()

from .quick_merge_node import QuickMergeNode

NODE_CLASS_MAPPINGS = {
    "QuickMergeNode": QuickMergeNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QuickMergeNode": "🎨 Quick Merge"
}

WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]