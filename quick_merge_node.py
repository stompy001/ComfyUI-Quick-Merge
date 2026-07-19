import torch
import torch.nn.functional as F
import os
import math
import urllib.request
from PIL import Image, ImageOps
import numpy as np
import folder_paths
import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar
import spandrel

try:
    from transformers import AutoModelForImageSegmentation
    from torchvision import transforms
    import torchvision.transforms.functional as TF
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False

class QuickMergeNode:
    def __init__(self):
        self.cached_bg_model = None
        self.cached_bg_model_name = None
        self.cached_upscaler = None
        self.cached_upscaler_name = None
        self.onnx_session = None

    @classmethod
    def INPUT_TYPES(cls):
        upscale_models = folder_paths.get_filename_list("upscale_models")
        auto_dl_name = "RealESRGAN_x2plus.pth (Auto-Download)"
        if not upscale_models:
            upscale_models = [auto_dl_name]
        elif "RealESRGAN_x2plus.pth" not in upscale_models:
            upscale_models.insert(0, auto_dl_name)
            
        resize_modes = ["Crop to Fill (Preserve Aspect)", "Stretch to Fit", "Fit Inside (Letterbox)", "Fit Inside (Pad with Blur)"]
        align_modes = ["center", "top", "bottom", "left", "right"]

        return {
            "required": {},
            "optional": {
                "Background": ("IMAGE",),
                "Foreground (optional)": ("IMAGE",),
                "mask (optional)": ("MASK",),
                
                # --- 0. BATCH PROCESSING ---
                "📁 batch_processing": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "CRITICAL: Ensure all images processed in a folder batch share identical resolutions and aspect ratios to prevent platform layout shifts."
                }),
                "📁 foreground_batch": ("BOOLEAN", {"default": False}),
                "📁 fg_batch_folder": ("STRING", {"default": "📁 Type or paste folder path here..."}),
                "📁 background_batch": ("BOOLEAN", {"default": False}),
                "📁 bg_batch_folder": ("STRING", {"default": "📁 Type or paste folder path here..."}),
                "📁 batch_pairing_mode": (["1-to-1 Match (Sequential)", "Cross-Multiply (All Combinations)"],),
                "📁 batch_chunk_size": ("INT", {
                    "default": 4, "min": 1, "max": 128, "step": 1,
                    "tooltip": "Max images processed at once through the heavy pipeline. 0 = no limit."
                }),

                # --- 1. INITIAL RESIZE ---
                "📏 initial_resize": ("BOOLEAN", {"default": False}),
                "📏 fg_resize_mode": (resize_modes, {"default": "Crop to Fill (Preserve Aspect)"}),
                "📏 fg_crop_align": (align_modes, {"default": "center"}),
                "📏 fg_initial_width": ("INT", {"default": 720, "min": 8, "max": 16384, "step": 8}),
                "📏 fg_initial_height": ("INT", {"default": 1280, "min": 8, "max": 16384, "step": 8}),
                "📏 bg_resize_mode": (resize_modes, {"default": "Crop to Fill (Preserve Aspect)"}),
                "📏 bg_crop_align": (align_modes, {"default": "center"}),
                "📏 bg_initial_width": ("INT", {"default": 1920, "min": 8, "max": 16384, "step": 8}),
                "📏 bg_initial_height": ("INT", {"default": 1080, "min": 8, "max": 16384, "step": 8}),

                # --- 2. FOREGROUND MERGE ---
                "✂️ merge_image": ("BOOLEAN", {"default": False}),
                "✂️ cutout_model": (["BiRefNet_lite (Fast/Accurate)", "RMBG-2.0 (New/Accurate)", "RMBG-1.4 (Fast)", "BiRefNet (High Quality)", "BiRefNet_HR (4K/8K High Quality)", "None (Use Alpha)"],),
                "✂️ auto-position (WiP)": ("BOOLEAN", {"default": False}),
                "✂️ foreground_size": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}), 
                "✂️ foreground_left_right": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1}),
                "✂️ foreground_up_down": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1}),
                "✂️ flip_horizontal": ("BOOLEAN", {"default": False}),
                "✂️ flip_vertical": ("BOOLEAN", {"default": False}),
                "✂️ rotation_degrees": ("INT", {"default": 0, "min": -360, "max": 360, "step": 1}),
                "✂️ edge_shrink_grow": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "✂️ edge_softness": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "✂️ foreground_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "✂️ blend_mode": (["Normal", "Multiply", "Screen", "Overlay", "Soft Light"],),
                "✂️ blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),

                # --- 3. AI SUPERSAMPLING ---
                "⚡ ai_upscale": ("BOOLEAN", {"default": False}),
                "⚡ pipeline_order": (["upscale composite", "upscale foreground only", "upscale background only"],),
                "⚡ choose_upscale_model": (upscale_models,),
                "⚡ upscale_by": ("FLOAT", {
                    "default": 2.0, "min": 0.25, "max": 32.0, "step": 0.25,
                }),
                "⚡ tile_size (0 means off/native)": ("INT", {"default": 512, "min": 0, "max": 8192, "step": 32}),
                "⚡ post_upscale_edge_soften": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                
                # --- 4. COLORS & FILTERS ---
                "🎨 colors_and_filters": ("BOOLEAN", {"default": False}),
                "🎨 filter_target": (["Foreground", "Background", "Both"],),
                "🎨 filter_type": (["Gaussian Blur", "Auto-Color Match", "Laplacian Sharpen", "Edge Detect", "Sepia", "Invert", "None"],),
                "🎨 filter_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "🎨 brightness": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "🎨 contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                "🎨 saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                "🎨 filter_size": ("INT", {"default": 3, "min": 1, "max": 127, "step": 2}),
                "🎨 background_blur": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                
                # --- 5. SHADOWS ---
                "🌗 shadows": ("BOOLEAN", {"default": False}),
                "🌗 shadow_color_hex": ("STRING", {"default": "#000000"}),
                "🌗 shadow_darkness": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "🌗 shadow_softness": ("INT", {"default": 15, "min": 1, "max": 255, "step": 1}),
                "🌗 shadow_position_x": ("INT", {"default": 10, "min": -1000, "max": 1000, "step": 1}),
                "🌗 shadow_position_y": ("INT", {"default": 10, "min": -1000, "max": 1000, "step": 1}),
                
                # --- 6. TARGET RESOLUTION ---
                "🎯 target_resolution": ("BOOLEAN", {"default": False}),
                "🎯 target_resize_mode": (resize_modes,),
                "🎯 target_crop_align": (align_modes,),
                "🎯 target_width": ("INT", {"default": 1920, "min": 64, "max": 16384, "step": 8}),
                "🎯 target_height": ("INT", {"default": 1080, "min": 64, "max": 16384, "step": 8}),
                
                # --- GLOBAL ---
                "🖼️ pixel_smoothing": (["bicubic", "lanczos", "bilinear", "nearest"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("composited_image", "light_mask", "relight_fg_crop", "relight_bg_crop")
    FUNCTION = "process"
    CATEGORY = "image/compositing"

    def color_match(self, fg, bg, mask):
        mask_sum = mask.sum(dim=(2, 3), keepdim=True) + 1e-6
        mu_fg = (fg * mask).sum(dim=(2, 3), keepdim=True) / mask_sum
        std_fg = torch.sqrt(torch.sum((((fg - mu_fg) * mask) ** 2), dim=(2, 3), keepdim=True) / mask_sum) + 1e-6
        mu_bg = bg.mean(dim=(2, 3), keepdim=True)
        std_bg = bg.std(dim=(2, 3), keepdim=True) + 1e-6
        matched_fg = (fg - mu_fg) * (std_bg / std_fg) + mu_bg
        return torch.clamp(matched_fg, 0.0, 1.0)

    def load_bg_model(self, model_choice, device):
        if not TRANSFORMERS_AVAILABLE:
            raise Exception("Please install transformers via ComfyUI Manager")
        if self.cached_bg_model_name == model_choice:
            return self.cached_bg_model

        mm.soft_empty_cache()
        original_getattr = torch.nn.Module.__getattr__
        def safe_getattr(module_self, name):
            if name == "all_tied_weights_keys": return {}
            return original_getattr(module_self, name)
            
        try:
            torch.nn.Module.__getattr__ = safe_getattr
            if model_choice == "RMBG-1.4 (Fast)":
                model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True)
            elif model_choice == "RMBG-2.0 (New/Accurate)":
                model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-2.0", trust_remote_code=True)
            elif model_choice == "BiRefNet_lite (Fast/Accurate)":
                model = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet_lite", trust_remote_code=True)
            elif model_choice == "BiRefNet (High Quality)":
                model = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
            elif model_choice == "BiRefNet_HR (4K/8K High Quality)":
                model = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet_HR", trust_remote_code=True)
        finally:
            torch.nn.Module.__getattr__ = original_getattr
        
        model.to(device)
        model.eval()
        self.cached_bg_model = model
        self.cached_bg_model_name = model_choice
        return model

    def apply_bg_removal(self, image, model_choice, device):
        model = self.load_bg_model(model_choice, device)
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img_t = image.permute(0, 3, 1, 2)
        if img_t.shape[1] == 4:
            img_t = img_t[:, :3, :, :]
            
        model_dtype = next(model.parameters()).dtype
        b_size = img_t.shape[0]
        out_masks = []
        
        for i in range(b_size):
            chunk = img_t[i:i+1].to(device)
            input_tensor = transform(chunk).to(dtype=model_dtype)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                def get_all_tensors(obj):
                    if isinstance(obj, torch.Tensor): return [obj]
                    elif isinstance(obj, (list, tuple)):
                        res = []
                        for item in obj: res.extend(get_all_tensors(item))
                        return res
                    elif isinstance(obj, dict):
                        res = []
                        for val in obj.values(): res.extend(get_all_tensors(val))
                        return res
                    return []
                    
                all_tensors = get_all_tensors(outputs)
                preds_tensor = all_tensors[0] if "RMBG" in model_choice else all_tensors[-1]
                preds_tensor = preds_tensor.float().sigmoid()
                preds_tensor = (preds_tensor - preds_tensor.min()) / (preds_tensor.max() - preds_tensor.min() + 1e-8)
                preds_tensor = torch.where(preds_tensor < 0.05, torch.zeros_like(preds_tensor), preds_tensor)
                
            mask = F.interpolate(preds_tensor, size=(image.shape[1], image.shape[2]), mode='bilinear', align_corners=False)
            if mask.shape[1] > 1: mask = mask[:, 0:1, :, :]
            out_masks.append(mask.cpu())  
            mm.soft_empty_cache()
            
        return torch.cat(out_masks, dim=0)

    def tiled_upscale(self, model, tensor, tile_size=512, overlap=32, pbar=None, start_pct=0, end_pct=100):
        b, c, h, w = tensor.shape
        device = tensor.device
        is_mask = (c == 1)
        if is_mask:
            tensor = tensor.repeat(1, 3, 1, 1)

        with torch.no_grad():
            dummy = torch.zeros((1, 3, 16, 16), device=device)
            scale = model(dummy).shape[-1] // 16

        if tile_size == 0:
            with torch.no_grad():
                result = model(tensor)
            if is_mask:
                result = result.mean(dim=1, keepdim=True)
            if pbar:
                pbar.update_absolute(end_pct)
            return result

        out_h, out_w = h * scale, w * scale
        output  = torch.zeros((b, 3, out_h, out_w), device=device)
        weights = torch.zeros((b, 1, out_h, out_w), device=device)
        stride  = max(1, tile_size - overlap)

        y_range     = list(range(0, h, stride))
        x_range     = list(range(0, w, stride))
        total_tiles = max(1, len(y_range) * len(x_range))
        tiles_done  = 0

        for y in y_range:
            for x in x_range:
                y1 = max(0, min(y, h - tile_size)) if h > tile_size else 0
                x1 = max(0, min(x, w - tile_size)) if w > tile_size else 0
                y2 = min(y1 + tile_size, h)
                x2 = min(x1 + tile_size, w)

                tile = tensor[:, :, y1:y2, x1:x2]
                tile_h, tile_w = tile.shape[2], tile.shape[3]

                gy = torch.linspace(-1.0, 1.0, tile_h, device=device)
                gx = torch.linspace(-1.0, 1.0, tile_w, device=device)
                yy, xx = torch.meshgrid(gy, gx, indexing='ij')
                g_map = torch.exp(-(xx ** 2 + yy ** 2) / (2 * 0.5 ** 2))
                g_map = g_map.unsqueeze(0).unsqueeze(0)

                with torch.no_grad():
                    up_tile = model(tile)

                up_th, up_tw = up_tile.shape[2], up_tile.shape[3]
                g_map_up = F.interpolate(g_map, size=(up_th, up_tw), mode='bilinear', align_corners=False)

                out_y1, out_x1 = y1 * scale, x1 * scale
                out_y2, out_x2 = out_y1 + up_th, out_x1 + up_tw

                output[:, :, out_y1:out_y2, out_x1:out_x2] += up_tile * g_map_up
                weights[:, :, out_y1:out_y2, out_x1:out_x2] += g_map_up
                del tile, up_tile, g_map, g_map_up

                tiles_done += 1
                if pbar:
                    pbar.update_absolute(int(start_pct + (tiles_done / total_tiles) * (end_pct - start_pct)))

        final_out = output / weights.clamp(min=1e-6)
        if is_mask:
            final_out = final_out.mean(dim=1, keepdim=True)
        return final_out

    def process_alpha(self, mask, dilate_erode, feather):
        if dilate_erode > 0:
            mask = F.max_pool2d(mask, kernel_size=dilate_erode*2+1, stride=1, padding=dilate_erode)
        elif dilate_erode < 0:
            erode_val = abs(dilate_erode)
            mask = -F.max_pool2d(-mask, kernel_size=erode_val*2+1, stride=1, padding=erode_val)
        if feather > 0:
            kernel_size = feather * 2 + 1
            sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
            x = torch.arange(kernel_size, dtype=torch.float32, device=mask.device) - kernel_size // 2
            x = x.expand(kernel_size, kernel_size)
            y = x.t()
            kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
            kernel = kernel / kernel.sum()
            kernel = kernel.view(1, 1, kernel_size, kernel_size)
            mask = F.conv2d(mask, kernel, padding=feather)
        return torch.clamp(mask, 0.0, 1.0)

    def generate_gaussian_blur(self, t_img, blur_amount, device):
        k_size = blur_amount if blur_amount % 2 != 0 else blur_amount + 1
        sigma = max(0.3 * ((k_size - 1) * 0.5 - 1) + 0.8, 0.1)
        coords = torch.arange(k_size, dtype=torch.float32, device=device) - k_size // 2
        x_g = coords.expand(k_size, k_size)
        y_g = x_g.t()
        g_kernel = torch.exp(-(x_g**2 + y_g**2) / (2 * sigma**2))
        g_kernel = (g_kernel / g_kernel.sum()).view(1, 1, k_size, k_size).expand(t_img.shape[1], 1, k_size, k_size)
        return F.conv2d(t_img, g_kernel, padding=k_size//2, groups=t_img.shape[1])

    # --- RE-INSTATED METHODS: RESIZE AND GRADIENT ENGINE SLOTS ---
    def generate_smart_light_gradient(self, mask_tensor, target_h, target_w, device):
        if mask_tensor is None:
            return torch.zeros((1, target_h, target_w), dtype=torch.float32).cpu()
            
        b_size = mask_tensor.shape[0]
        tiny_size = 128
        out_masks = []
        for i in range(b_size):
            m_tiny = F.interpolate(mask_tensor[i:i+1].float().to(device), size=(tiny_size, tiny_size), mode='area')
            m_blurred = self.generate_gaussian_blur(m_tiny, 31, device)
            m_final = F.interpolate(m_blurred, size=(target_h, target_w), mode='bicubic', align_corners=False)
            out_masks.append(torch.clamp(m_final, 0.0, 1.0).squeeze(1).cpu())
        return torch.cat(out_masks, dim=0)

    def resize_tensor(self, tensor, target_w, target_h, mode, align, interpolation):
        b, c, h, w = tensor.shape
        target_ratio = target_w / target_h
        tensor_ratio = w / h

        if mode == "Crop to Fill (Preserve Aspect)":
            if abs(tensor_ratio - target_ratio) > 0.001:
                if tensor_ratio > target_ratio:
                    crop_w = int(h * target_ratio)
                    if align == "left": start_x = 0
                    elif align == "right": start_x = w - crop_w
                    else: start_x = (w - crop_w) // 2 
                    tensor = tensor[:, :, :, start_x:start_x + crop_w]
                else:
                    crop_h = int(w / target_ratio)
                    if align == "top": start_y = 0
                    elif align == "bottom": start_y = h - crop_h
                    else: start_y = (h - crop_h) // 2 
                    tensor = tensor[:, :, start_y:start_y + crop_h, :]
            return F.interpolate(tensor, size=(target_h, target_w), mode=interpolation, align_corners=False)

        elif mode == "Stretch to Fit":
            return F.interpolate(tensor, size=(target_h, target_w), mode=interpolation, align_corners=False)

        elif mode == "Fit Inside (Letterbox)":
            if tensor_ratio > target_ratio:
                new_w = target_w
                new_h = int(target_w / tensor_ratio)
            else:
                new_h = target_h
                new_w = int(target_h * tensor_ratio)
            tensor_scaled = F.interpolate(tensor, size=(max(1, new_h), max(1, new_w)), mode=interpolation, align_corners=False)
            padded = torch.zeros((b, c, target_h, target_w), device=tensor.device)
            start_y = (target_h - new_h) // 2
            start_x = (target_w - new_w) // 2
            padded[:, :, start_y:start_y+new_h, start_x:start_x+new_w] = tensor_scaled
            return padded
            
        elif mode == "Fit Inside (Pad with Blur)":
            bg_fill = F.interpolate(tensor, size=(target_h, target_w), mode=interpolation, align_corners=False)
            bg_blur = self.generate_gaussian_blur(bg_fill.to(mm.get_torch_device()), 51, mm.get_torch_device()).cpu() if tensor.device.type == 'cpu' else self.generate_gaussian_blur(bg_fill, 51, tensor.device)
            if tensor_ratio > target_ratio:
                new_w = target_w
                new_h = int(target_w / tensor_ratio)
            else:
                new_h = target_h
                new_w = int(target_h * tensor_ratio)
            tensor_scaled = F.interpolate(tensor, size=(max(1, new_h), max(1, new_w)), mode=interpolation, align_corners=False)
            start_y = (target_h - new_h) // 2
            start_x = (target_w - new_w) // 2
            bg_blur[:, :, start_y:start_y+new_h, start_x:start_x+new_w] = tensor_scaled
            return bg_blur

    def load_folder_images(self, folder_path, device, is_active):
        if not is_active or not os.path.isdir(folder_path):
            return None
            
        valid_exts = ['.jpg', '.jpeg', '.png', '.webp']
        processed_tensors = []
        first_w, first_h = None, None
        
        for file in sorted(os.listdir(folder_path)):
            if any(file.lower().endswith(ext) for ext in valid_exts):
                img_path = os.path.join(folder_path, file)
                try:
                    i = Image.open(img_path)
                    i = ImageOps.exif_transpose(i)
                    
                    i = i.convert("RGBA")
                    img_arr = np.array(i).astype(np.float32) / 255.0
                        
                    t = torch.from_numpy(img_arr).unsqueeze(0).cpu() 
                    t_perm = t.permute(0, 3, 1, 2) 
                    
                    if first_w is None:
                        first_w, first_h = t_perm.shape[3], t_perm.shape[2]
                    elif t_perm.shape[3] != first_w or t_perm.shape[2] != first_h:
                        t_perm = self.resize_tensor(t_perm, first_w, first_h, "Fit Inside (Letterbox)", "center", "bicubic")
                        
                    processed_tensors.append(t_perm.permute(0, 2, 3, 1))
                except Exception as e:
                    print(f"Quick Merge Image Pack Alignment Failed: {file} - {e}")
                    
        if not processed_tensors:
            return None
            
        return torch.cat(processed_tensors, dim=0)

    def process(self, **kwargs):
        device = mm.get_torch_device()
        
        alpha = None
        fg = None
        b_size = 1
        effective_chunk = 1
        num_chunks = 1
        multi_chunk = False
        pre_transform_bg_h = 1080
        pre_transform_bg_w = 1920
        pbar = ProgressBar(100)
        
        # --- MAP SETTINGS BLOCK CONTEXT ---
        batch_processing   = kwargs.get("📁 batch_processing", False)
        foreground_batch   = kwargs.get("📁 foreground_batch", False)
        background_batch   = kwargs.get("📁 background_batch", False)
        fg_batch_folder    = kwargs.get("📁 fg_batch_folder", "")
        bg_batch_folder    = kwargs.get("📁 bg_batch_folder", "")
        batch_pairing_mode = kwargs.get("📁 batch_pairing_mode", "1-to-1 Match (Sequential)")
        batch_chunk_size   = kwargs.get("📁 batch_chunk_size", 4)

        initial_resize    = kwargs.get("📏 initial_resize", False)
        fg_resize_mode    = kwargs.get("📏 fg_resize_mode", "Crop to Fill (Preserve Aspect)")
        fg_crop_align     = kwargs.get("📏 fg_crop_align", "center")
        fg_initial_width  = kwargs.get("📏 fg_initial_width", 720)
        fg_initial_height = kwargs.get("📏 fg_initial_height", 1280)
        bg_resize_mode    = kwargs.get("📏 bg_resize_mode", "Crop to Fill (Preserve Aspect)")
        bg_crop_align     = kwargs.get("📏 bg_crop_align", "center")
        bg_initial_width  = kwargs.get("📏 bg_initial_width", 1920)
        bg_initial_height = kwargs.get("📏 bg_initial_height", 1080)
            
        cutout_model          = kwargs.get("✂️ cutout_model", "BiRefNet_lite (Fast/Accurate)")
        foreground_size       = kwargs.get("✂️ foreground_size", 1.0)
        foreground_left_right = kwargs.get("✂️ foreground_left_right", 0)
        foreground_up_down    = kwargs.get("✂️ foreground_up_down", 0)
        edge_shrink_grow      = kwargs.get("✂️ edge_shrink_grow", 0)
        edge_softness         = kwargs.get("✂️ edge_softness", 0)
        flip_horizontal       = kwargs.get("✂️ flip_horizontal", False)
        flip_vertical         = kwargs.get("✂️ flip_vertical", False)
        rotation_degrees      = kwargs.get("✂️ rotation_degrees", 0)
        foreground_opacity    = kwargs.get("✂️ foreground_opacity", 1.0)
        blend_mode            = kwargs.get("✂️ blend_mode", "Normal")
        blend_strength        = kwargs.get("✂️ blend_strength", 1.0)

        auto_position = False

        ai_upscale               = kwargs.get("⚡ ai_upscale", False)
        pipeline_order           = kwargs.get("⚡ pipeline_order", "upscale composite")
        ai_upscale_model         = kwargs.get("⚡ choose_upscale_model", "None Found")
        upscale_by               = kwargs.get("⚡ upscale_by", 2.0)
        ai_upscale_tile_size     = kwargs.get("⚡ tile_size (0 means off/native)", 512)
        post_upscale_edge_soften = kwargs.get("⚡ post_upscale_edge_soften", 0)

        colors_and_filters = kwargs.get("🎨 colors_and_filters", False)
        filter_target      = kwargs.get("🎨 filter_target", "Foreground")
        color_and_filters  = kwargs.get("🎨 filter_type", "Auto-Color Match")
        filter_strength    = kwargs.get("🎨 filter_strength", 1.0)
        filter_size        = kwargs.get("🎨 filter_size", 3)
        brightness         = kwargs.get("🎨 brightness", 0.0)
        contrast           = kwargs.get("🎨 contrast", 1.0)
        saturation         = kwargs.get("🎨 saturation", 1.0)
        background_blur    = kwargs.get("🎨 background_blur", 0)

        shadows             = kwargs.get("🌗 shadows", False)
        shadow_color_hex    = kwargs.get("🌗 shadow_color_hex", "#000000").lstrip('#')
        if len(shadow_color_hex) != 6: shadow_color_hex = "000000"
        shadow_R = int(shadow_color_hex[0:2], 16) / 255.0
        shadow_G = int(shadow_color_hex[2:4], 16) / 255.0
        shadow_B = int(shadow_color_hex[4:6], 16) / 255.0
        shadow_darkness     = kwargs.get("🌗 shadow_darkness", 0.5)
        shadow_softness     = kwargs.get("🌗 shadow_softness", 15)
        shadow_position_x   = kwargs.get("🌗 shadow_position_x", 10)
        shadow_position_y   = kwargs.get("🌗 shadow_position_y", 10)

        target_resolution   = kwargs.get("🎯 target_resolution", False)
        target_resize_mode  = kwargs.get("🎯 target_resize_mode", "Crop to Fill (Preserve Aspect)")
        target_crop_align   = kwargs.get("🎯 target_crop_align", "center")
        target_width        = kwargs.get("🎯 target_width", 1920)
        target_height       = kwargs.get("🎯 target_height", 1080)
        
        pixel_smoothing  = kwargs.get("🖼️ pixel_smoothing", "bicubic")
        interpolation    = pixel_smoothing if pixel_smoothing != "lanczos" else "bicubic"

        # --- 1. DIRECTORY CONDITIONAL PROCESSING OVERRIDES ---
        bg_input   = kwargs.get("Background", None)
        fg_input   = kwargs.get("Foreground (optional)", None)
        mask_input = kwargs.get("mask (optional)", None)
        
        default_folder_text = "📁 Type or paste folder path here..."
        if fg_batch_folder == default_folder_text: fg_batch_folder = ""
        if bg_batch_folder == default_folder_text: bg_batch_folder = ""
        
        if batch_processing:
            if background_batch and bg_batch_folder and os.path.isdir(bg_batch_folder):
                bg_input = self.load_folder_images(bg_batch_folder, device, True)
            if foreground_batch and fg_batch_folder and os.path.isdir(fg_batch_folder):
                fg_input = self.load_folder_images(fg_batch_folder, device, True)
                
        has_bg   = bg_input is not None
        has_fg   = fg_input is not None
        has_mask = mask_input is not None
        
        if not has_bg and not has_fg and not has_mask:
            empty_img = torch.zeros((1, 64, 64, 3), dtype=torch.float32).cpu()
            empty_msk = torch.zeros((1, 64, 64),    dtype=torch.float32).cpu()
            return (empty_img, empty_msk, empty_img, empty_img)

        if has_bg:
            bg = bg_input.clone().detach().cpu().permute(0, 3, 1, 2) if bg_input.shape[3] in [3, 4] else bg_input.clone().detach().cpu()
            if bg.shape[1] == 4: bg = bg[:, :3, :, :]
        if has_fg:
            fg = fg_input.clone().detach().cpu().permute(0, 3, 1, 2) if fg_input.shape[3] in [3, 4] else fg_input.clone().detach().cpu()
        if has_mask:
            m_tensor = mask_input.clone().detach().cpu()
            if len(m_tensor.shape) == 3: m_tensor = m_tensor.unsqueeze(1)

        merge_image = kwargs.get("✂️ merge_image", True)
        if not has_fg:
            merge_image = False
            
        is_real_bg = has_bg
        if has_bg:
            b_size = bg.shape[0]
        elif has_fg:
            if merge_image:
                b_size = fg.shape[0]
                bg = torch.zeros((b_size, 3, fg.shape[2], fg.shape[3]), device='cpu')
                has_bg = True
            else:
                bg = fg.clone()
                has_bg = True
                is_real_bg = True
                b_size = bg.shape[0]
        elif has_mask:
            b_size = m_tensor.shape[0]
            bg = torch.zeros((b_size, 3, m_tensor.shape[2], m_tensor.shape[3]), device='cpu')
            has_bg = True

        if not merge_image:
            has_fg = False

        # --- PRE-PROCESS FOREGROUND CUTOUT ENGINE ---
        if has_fg and merge_image:
            if cutout_model != "None (Use Alpha)":
                pass_fg = fg.permute(0, 2, 3, 1)
                alpha = self.apply_bg_removal(pass_fg, cutout_model, device)
                del pass_fg
                if fg.shape[1] == 4: fg = fg[:, :3, :, :]
            else:
                if fg.shape[1] == 4:
                    alpha = fg[:, 3:4, :, :]
                    fg    = fg[:, :3, :, :]
                else:
                    alpha = torch.ones((fg.shape[0], 1, fg.shape[2], fg.shape[3]), dtype=torch.float32, device='cpu')

            alpha = self.process_alpha(alpha, edge_shrink_grow, edge_softness)
            if flip_horizontal:
                fg, alpha = torch.flip(fg, [3]), torch.flip(alpha, [3])
            if flip_vertical:
                fg, alpha = torch.flip(fg, [2]), torch.flip(alpha, [2])
            if rotation_degrees != 0:
                fg    = TF.rotate(fg,    rotation_degrees, interpolation=TF.InterpolationMode.BILINEAR, expand=True)
                alpha = TF.rotate(alpha, rotation_degrees, interpolation=TF.InterpolationMode.BILINEAR, expand=True)

        # --- COMBINATORIAL PAIRING SYNCS ---
        if is_real_bg and merge_image and has_fg:
            b_bg = bg.shape[0]
            b_fg = fg.shape[0]
            
            if batch_pairing_mode == "1-to-1 Match (Sequential)":
                b_size = max(b_bg, b_fg)
                if b_bg < b_size:
                    repeats_bg = math.ceil(b_size / b_bg)
                    bg = bg.repeat(repeats_bg, 1, 1, 1)[:b_size]
                if b_fg < b_size:
                    repeats_fg = math.ceil(b_size / b_fg)
                    fg = fg.repeat(repeats_fg, 1, 1, 1)[:b_size]
                    if alpha is not None:
                        alpha = alpha.repeat(repeats_fg, 1, 1, 1)[:b_size]
            else:
                b_size = b_bg * b_fg
                bg = bg.repeat_interleave(b_fg, dim=0)
                fg = fg.repeat(b_bg, 1, 1, 1)
                if alpha is not None:
                    alpha = alpha.repeat(b_bg, 1, 1, 1)

        # --- UNCONDITIONAL CHUNK EVALUATION METRICS ---
        pre_transform_bg_h, pre_transform_bg_w = bg.shape[2], bg.shape[3]
        effective_chunk = batch_chunk_size if (batch_chunk_size > 0 and b_size > batch_chunk_size) else b_size
        num_chunks  = math.ceil(b_size / effective_chunk)
        multi_chunk = num_chunks > 1

        if has_mask:
            if m_tensor.shape[0] != b_size:
                m_tensor = m_tensor.repeat(math.ceil(b_size / m_tensor.shape[0]), 1, 1, 1)[:b_size]

        if initial_resize:
            bg = self.resize_tensor(bg, bg_initial_width, bg_initial_height, bg_resize_mode, bg_crop_align, interpolation)
            if merge_image and has_fg:
                fg = self.resize_tensor(fg, fg_initial_width, fg_initial_height, fg_resize_mode, fg_crop_align, interpolation)
                alpha = self.resize_tensor(alpha, fg_initial_width, fg_initial_height, fg_resize_mode, fg_crop_align, interpolation)
        pbar.update_absolute(5)

        # ---- CLOSURES ----
        def execute_filters(t_fg, t_alpha, t_bg):
            if not colors_and_filters: return t_fg
            if brightness != 0.0 or contrast != 1.0:
                t_fg = t_fg + brightness
                t_fg = torch.clamp((t_fg - 0.5) * contrast + 0.5, 0.0, 1.0)
            if saturation != 1.0:
                lum = 0.299 * t_fg[:, 0:1] + 0.587 * t_fg[:, 1:2] + 0.114 * t_fg[:, 2:3]
                t_fg = torch.clamp(torch.lerp(lum.repeat(1, 3, 1, 1), t_fg, saturation), 0.0, 1.0)
            if color_and_filters == "Laplacian Sharpen":
                lap_kernel = torch.tensor([[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]], device=device).expand(t_fg.shape[1], 1, 3, 3)
                edges = F.conv2d(t_fg, lap_kernel, padding=1, groups=t_fg.shape[1])
                return torch.clamp(torch.lerp(t_fg, t_fg - (edges * filter_strength), min(filter_strength, 1.0)), 0.0, 1.0)
            elif color_and_filters == "Gaussian Blur":
                blur_fg = self.generate_gaussian_blur(t_fg, filter_size, device)
                return torch.lerp(t_fg, blur_fg, min(filter_strength, 1.0))
            elif color_and_filters == "Edge Detect":
                sobel_x = torch.tensor([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]], device=device).expand(t_fg.shape[1], 1, 3, 3)
                sobel_y = torch.tensor([[[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]], device=device).expand(t_fg.shape[1], 1, 3, 3)
                edge_x  = F.conv2d(t_fg, sobel_x, padding=1, groups=t_fg.shape[1])
                edge_y  = F.conv2d(t_fg, sobel_y, padding=1, groups=t_fg.shape[1])
                edges   = torch.clamp(torch.sqrt(edge_x**2 + edge_y**2), 0.0, 1.0)
                return torch.lerp(t_fg, edges, min(filter_strength, 1.0))
            elif color_and_filters == "Sepia":
                sepia_w = torch.tensor([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]], device=device)
                sepia_fg = torch.einsum('b c h w, d c -> b d h w', t_fg, sepia_w)
                return torch.lerp(t_fg, torch.clamp(sepia_fg, 0.0, 1.0), min(filter_strength, 1.0))
            elif color_and_filters == "Invert":
                return torch.lerp(t_fg, 1.0 - t_fg, min(filter_strength, 1.0))
            elif color_and_filters == "Auto-Color Match":
                if t_alpha is not None and is_real_bg:
                    matched_fg = self.color_match(t_fg, t_bg, t_alpha)
                    return torch.lerp(t_fg, matched_fg, min(filter_strength, 1.0))
                return t_fg
            return t_fg

        def execute_upscale(t_img, t_is_mask=False, start_pct=0, end_pct=100):
            if not ai_upscale or ai_upscale_model == "None Found":
                pbar.update_absolute(end_pct)
                return t_img

            chunk_b, orig_c, orig_h, orig_w = t_img.shape
            target_out_h   = max(8, round(orig_h * upscale_by))
            target_out_w   = max(8, round(orig_w * upscale_by))

            mm.soft_empty_cache()
            model_name_clean = ai_upscale_model.replace(" (Auto-Download)", "")
            
            if self.cached_upscaler_name != model_name_clean:
                model_path = folder_paths.get_full_path("upscale_models", model_name_clean)
                if not model_path:
                    if model_name_clean == "RealESRGAN_x2plus.pth":
                        dl_dir = os.path.join(folder_paths.models_dir, "upscale_models")
                        os.makedirs(dl_dir, exist_ok=True)
                        model_path = os.path.join(dl_dir, "RealESRGAN_x2plus.pth")
                        try:
                            urllib.request.urlretrieve("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth", model_path)
                        except Exception:
                            pbar.update_absolute(end_pct)
                            return t_img
                    else:
                        pbar.update_absolute(end_pct)
                        return t_img
                        
                try:
                    sd = load_torch_file(model_path, safe_load=True)
                    self.cached_upscaler = spandrel.ModelLoader().load_from_state_dict(sd).eval().to(device)
                    self.cached_upscaler_name = model_name_clean
                except Exception:
                    pbar.update_absolute(end_pct)
                    return t_img

            tile = max(64, ai_upscale_tile_size) if ai_upscale_tile_size > 0 else 0
            
            out_upscaled_slices = []
            for i in range(chunk_b):
                frame_slice = t_img[i:i+1].to(device)
                slice_pct_s = start_pct + (i / chunk_b) * (end_pct - start_pct)
                slice_pct_e = start_pct + ((i + 1) / chunk_b) * (end_pct - start_pct)
                
                up_slice = self.tiled_upscale(self.cached_upscaler, frame_slice, tile_size=tile, pbar=pbar, start_pct=slice_pct_s, end_pct=slice_pct_e)
                if up_slice.shape[2] != target_out_h or up_slice.shape[3] != target_out_w:
                    up_slice = F.interpolate(up_slice, size=(target_out_h, target_out_w), mode=interpolation, align_corners=False)
                out_upscaled_slices.append(up_slice.cpu())
                del frame_slice, up_slice
                mm.soft_empty_cache()
                
            up_img = torch.cat(out_upscaled_slices, dim=0).to(device)
            if t_is_mask and post_upscale_edge_soften > 0:
                up_img = self.process_alpha(up_img, 0, post_upscale_edge_soften)
            return up_img

        def execute_composite(t_fg, t_alpha, t_bg, canvas_scale_x=1.0, canvas_scale_y=1.0, chunk_offset_start=0):
            chunk_b  = t_bg.shape[0]
            canvas_h, canvas_w = t_bg.shape[2], t_bg.shape[3]
            result_img     = t_bg.clone()
            final_alpha    = torch.zeros((chunk_b, 1, canvas_h, canvas_w), device=device)
            relight_fg_out = torch.zeros_like(t_bg)
            relight_bg_out = t_bg.clone()

            pre_size_fg_w = max(8, int(t_fg.shape[3] * canvas_scale_x))
            pre_size_fg_h = max(8, int(t_fg.shape[2] * canvas_scale_y))
            t_fg    = F.interpolate(t_fg,    size=(pre_size_fg_h, pre_size_fg_w), mode=interpolation, align_corners=False)
            t_alpha = F.interpolate(t_alpha, size=(pre_size_fg_h, pre_size_fg_w), mode=interpolation, align_corners=False)

            target_fg_w = max(8, int(pre_size_fg_w * foreground_size))
            target_fg_h = max(8, int(pre_size_fg_h * foreground_size))
            t_fg    = F.interpolate(t_fg,    size=(target_fg_h, target_fg_w), mode=interpolation, align_corners=False)
            t_alpha = F.interpolate(t_alpha, size=(target_fg_h, target_fg_w), mode=interpolation, align_corners=False)

            center_x = (canvas_w // 2) - (target_fg_w // 2)
            center_y = (canvas_h // 2) - (target_fg_h // 2)
            
            scaled_lr = int(round(foreground_left_right * canvas_scale_x))
            scaled_ud = int(round(foreground_up_down * canvas_scale_y))
            final_x  = center_x + scaled_lr
            final_y  = center_y + scaled_ud

            x1, y1 = max(0, final_x), max(0, final_y)
            x2, y2 = min(canvas_w, final_x + target_fg_w), min(canvas_h, final_y + target_fg_h)
            fg_x1, fg_y1 = max(0, -final_x), max(0, -final_y)
            fg_x2, fg_y2 = fg_x1 + (x2 - x1), fg_y1 + (y2 - y1)

            for b in range(chunk_b):
                if (x2 > x1) and (y2 > y1):
                    crop_fg = t_fg[b:b+1, :, fg_y1:fg_y2, fg_x1:fg_x2]
                    crop_alpha = t_alpha[b:b+1, :, fg_y1:fg_y2, fg_x1:fg_x2]
                    bg_slice = result_img[b:b+1, :, y1:y2, x1:x2]

                    relight_fg_out[b:b+1, :, y1:y2, x1:x2] = crop_fg * crop_alpha
                    active_alpha = crop_alpha * foreground_opacity

                    if shadows and shadow_darkness > 0.0:
                        s_blur = shadow_softness if shadow_softness % 2 != 0 else shadow_softness + 1
                        full_shadow_mask = F.avg_pool2d(t_alpha[b:b+1] * foreground_opacity, kernel_size=s_blur, stride=1, padding=s_blur // 2)
                        scaled_spx = int(round(shadow_position_x * canvas_scale_x))
                        scaled_spy = int(round(shadow_position_y * canvas_scale_y))
                        sx1, sy1 = max(0, final_x + scaled_spx), max(0, final_y + scaled_spy)
                        sx2, sy2 = min(canvas_w, final_x + target_fg_w + scaled_spx), min(canvas_h, final_y + target_fg_h + scaled_spy)
                        sfg_x1, sfg_y1 = max(0, -(final_x + scaled_spx)), max(0, -(final_y + scaled_spy))
                        sfg_x2, sfg_y2 = sfg_x1 + (sx2 - sx1), sfg_y1 + (sy2 - sy1)
                        if (sx2 > sx1) and (sy2 > sy1):
                            crop_smask = full_shadow_mask[:, :, sfg_y1:sfg_y2, sfg_x1:sfg_x2]
                            s_color = torch.tensor([shadow_R, shadow_G, shadow_B], device=device).view(1, 3, 1, 1)
                            s_color = s_color.expand_as(result_img[b:b+1, :, sy1:sy2, sx1:sx2])
                            result_img[b:b+1, :, sy1:sy2, sx1:sx2] = torch.lerp(
                                result_img[b:b+1, :, sy1:sy2, sx1:sx2], s_color, crop_smask * shadow_darkness)

                    if blend_mode == "Multiply":
                        blend_math = bg_slice * crop_fg
                    elif blend_mode == "Screen":
                        blend_math = 1.0 - (1.0 - bg_slice) * (1.0 - crop_fg)
                    elif blend_mode == "Overlay":
                        mask_lt    = (bg_slice < 0.5).float()
                        blend_math = mask_lt * (2.0 * bg_slice * crop_fg) + (1.0 - mask_lt) * (1.0 - 2.0 * (1.0 - bg_slice) * (1.0 - crop_fg))
                    elif blend_mode == "Soft Light":
                        blend_math = (1.0 - 2.0 * crop_fg) * (bg_slice ** 2) + 2.0 * crop_fg * bg_slice
                    else:
                        blend_math = crop_fg

                    active_fg = torch.lerp(crop_fg, blend_math, blend_strength)
                    result_img[b:b+1, :, y1:y2, x1:x2] = torch.lerp(bg_slice, active_fg, active_alpha)
                    final_alpha[b:b+1, :, y1:y2, x1:x2] = active_alpha

            return result_img, final_alpha, relight_fg_out, relight_bg_out

        # ==========================================
        # PATH A: BACKGROUND-ONLY BYPASS
        # ==========================================
        if not merge_image:
            out_images = []
            for c_idx in range(num_chunks):
                c_start = c_idx * effective_chunk
                c_end   = min(c_start + effective_chunk, b_size)
                c_bg = bg[c_start:c_end].clone().to(device)
                
                if colors_and_filters and filter_target in ["Background", "Both"]:
                    c_bg = execute_filters(c_bg, None, c_bg)

                if ai_upscale:
                    pct_s = 50 + int(c_start / b_size * 45)
                    pct_e = 50 + int(c_end / b_size * 45)
                    c_bg = execute_upscale(c_bg, False, start_pct=pct_s, end_pct=pct_e)

                if target_resolution:
                    c_bg = self.resize_tensor(c_bg, target_width, target_height, target_resize_mode, target_crop_align, interpolation)

                out_images.append(c_bg.cpu() if multi_chunk else c_bg)
                del c_bg
                if multi_chunk: mm.soft_empty_cache()

            pbar.update_absolute(100)
            bg_final = torch.cat(out_images, dim=0)
            
            final_h, final_w = bg_final.shape[2], bg_final.shape[3]
            if has_mask:
                out_light_mask = self.generate_smart_light_gradient(m_tensor, final_h, final_w, device)
            else:
                out_light_mask = torch.zeros((b_size, final_h, final_w), dtype=torch.float32).cpu()

            empty_fg_tensor = torch.zeros((b_size, final_h, final_w, 3), dtype=torch.float32).cpu()
            return (bg_final.permute(0, 2, 3, 1).cpu(), out_light_mask, empty_fg_tensor, bg_final.permute(0, 2, 3, 1).cpu())

        # ==========================================
        # PATH B: COMPOSITING CHUNK LOOPS
        # ==========================================
        all_fi, all_fm, all_rf, all_rb = [], [], [], []

        for c_idx in range(num_chunks):
            c_start = c_idx * effective_chunk
            c_end   = min(c_start + effective_chunk, b_size)
            pct_lo  = 30 + int( c_idx      / num_chunks * 70)
            pct_hi  = 30 + int((c_idx + 1) / num_chunks * 70)

            c_bg    = bg[c_start:c_end].clone().to(device)
            c_fg    = fg[c_start:c_end].clone().to(device)
            c_alpha = alpha[c_start:c_end].clone().to(device)

            if target_resolution and pipeline_order != "upscale composite":
                c_bg = self.resize_tensor(c_bg, target_width, target_height, target_resize_mode, target_crop_align, interpolation)
            if colors_and_filters and background_blur > 0:
                c_bg = self.generate_gaussian_blur(c_bg, background_blur, device)

            if colors_and_filters and filter_target in ["Background", "Both"]:
                c_bg = execute_filters(c_bg, None, c_bg)
            if colors_and_filters and filter_target in ["Foreground", "Both"]:
                c_fg = execute_filters(c_fg, c_alpha, c_bg)

            _sx = c_bg.shape[3] / pre_transform_bg_w
            _sy = c_bg.shape[2] / pre_transform_bg_h

            if pipeline_order == "upscale foreground only":
                mid = pct_lo + (pct_hi - pct_lo) // 3
                c_fg    = execute_upscale(c_fg,    False, start_pct=pct_lo, end_pct=mid)
                c_alpha = execute_upscale(c_alpha, True,  start_pct=mid,    end_pct=pct_lo + (pct_hi - pct_lo) * 2 // 3)
                c_fi, c_fm, c_rf, c_rb = execute_composite(c_fg, c_alpha, c_bg, _sx, _sy, c_start)

            elif pipeline_order == "upscale background only":
                c_bg = execute_upscale(c_bg, False, start_pct=pct_lo, end_pct=pct_lo + (pct_hi - pct_lo) * 2 // 3)
                _sx  = c_bg.shape[3] / pre_transform_bg_w
                _sy  = c_bg.shape[2] / pre_transform_bg_h
                c_fi, c_fm, c_rf, c_rb = execute_composite(c_fg, c_alpha, c_bg, _sx, _sy, c_start)

            else:  # upscale composite
                c_fi, c_fm, c_rf, c_rb = execute_composite(c_fg, c_alpha, c_bg, _sx, _sy, c_start)
                mid = pct_lo + (pct_hi - pct_lo) // 2
                c_fi = execute_upscale(c_fi, False, start_pct=pct_lo, end_pct=mid)
                c_fm = execute_upscale(c_fm, True,  start_pct=mid,    end_pct=pct_hi)
                up_h, up_w = c_fi.shape[2], c_fi.shape[3]
                c_rf = F.interpolate(c_rf, size=(up_h, up_w), mode=interpolation, align_corners=False)
                c_rb = F.interpolate(c_rb, size=(up_h, up_w), mode=interpolation, align_corners=False)

            if target_resolution:
                c_fi = self.resize_tensor(c_fi, target_width, target_height, target_resize_mode, target_crop_align, interpolation)
                c_fm = self.resize_tensor(c_fm, target_width, target_height, target_resize_mode, target_crop_align, interpolation)
                c_rf = self.resize_tensor(c_rf, target_width, target_height, target_resize_mode, target_crop_align, interpolation)
                c_rb = self.resize_tensor(c_rb, target_width, target_height, target_resize_mode, target_crop_align, interpolation)

            all_fi.append(c_fi.cpu() if multi_chunk else c_fi)
            all_fm.append(c_fm.cpu() if multi_chunk else c_fm)
            all_rf.append(c_rf.cpu() if multi_chunk else c_rf)
            all_rb.append(c_rb.cpu() if multi_chunk else c_rb)

            del c_bg, c_fg, c_alpha, c_fi, c_fm, c_rf, c_rb
            if multi_chunk:
                mm.soft_empty_cache()

        pbar.update_absolute(100)

        final_img  = torch.cat(all_fi, dim=0)
        final_mask = torch.cat(all_fm, dim=0)
        relight_fg = torch.cat(all_rf, dim=0)
        relight_bg = torch.cat(all_rb, dim=0)
        if multi_chunk:
            final_img  = final_img.to(device)
            final_mask = final_mask.to(device)
            relight_fg = relight_fg.to(device)
            relight_bg = relight_bg.to(device)

        # ==========================================
        # LIGHT MASK & OPTION A ALPHA GENERATION
        # ==========================================
        final_h, final_w = final_img.shape[2], final_img.shape[3]

        if has_mask:
            m_resized = F.interpolate(m_tensor.float(), size=(final_h, final_w), mode='bilinear', align_corners=False)
            out_light_mask = self.generate_gaussian_blur(m_resized.to(device), 51, device).squeeze(1).cpu()
        elif has_fg and not is_real_bg:
            out_light_mask = final_mask.squeeze(1).cpu()
        else:
            out_light_mask = torch.zeros((b_size, final_h, final_w), dtype=torch.float32).cpu()

        out_images     = final_img.permute(0, 2, 3, 1).cpu()
        out_relight_fg = relight_fg.permute(0, 2, 3, 1).cpu()
        out_relight_bg = relight_bg.permute(0, 2, 3, 1).cpu()
        
        ui_msg = "[Quick Merge Engine] Auto-Position Disabled (WiP Phase Lock)"

        return {
            "ui": {"text": [ui_msg]},
            "result": (out_images, out_light_mask, out_relight_fg, out_relight_bg)
        }

NODE_CLASS_MAPPINGS = {
    "QuickMergeNode": QuickMergeNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QuickMergeNode": "🎨 Quick Merge"
}