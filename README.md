# 🎨 ComfyUI Quick Merge

A high-performance, modular image layering, blending, and batch execution engine for ComfyUI. Replace complex multi-node canvas workflows with a single, production-grade layer processing hub engineered for heavy datasets and memory safety.
<img width="1518" height="848" alt="Screenshot 2026-07-19 234816" src="https://github.com/user-attachments/assets/0933234b-079a-4b9a-aab3-40343f2797de" />
<img width="1251" height="952" alt="Screenshot 2026-07-19 234744" src="https://github.com/user-attachments/assets/43151b35-7cfc-4730-9d87-4db2e7a4bd77" />


<img width="1534" height="927" alt="Screenshot 2026-07-19 223251" src="https://github.com/user-attachments/assets/2c483cee-d4f6-42fa-b4de-6222d5e99818" />
<img width="1583" height="946" alt="Screenshot 2026-07-19 214958" src="https://github.com/user-attachments/assets/f83fced3-0579-4feb-b452-14cb8f25726c" />

## 🚀 Key Use Cases
---

* **E-Commerce Automation:** Seamlessly clip product variants, fix layer placement offsets, and overlay items across variable marketing backgrounds.
* **Concept Art Pre-Compositing:** Rapidly pair generated character assets over different landscape sheets to evaluate lighting, shadows, and composition grids.
* **Dataset Enhancement:** Batch-apply standalone background image alterations, filters, structural resolution resizing, and deep Gaussian blurs.

---

## 🔌 How To Use

Connect your assets directly to the node's native input sockets:
* **Background (Required):** The baseline background image canvas or folder canvas stream.
* **Foreground (Optional):** The product cutout, character asset, or foreground target stack.
* **Mask (Optional):** Core transparency sheets used to project custom lighting layouts down your tree.

### 🔄 Folder Sizing & Fallback Behaviors
When processing datasets, the node automatically computes matching rules across mismatched input arrays to guarantee execution stability:
* **1 Foreground Asset + N Backgrounds:** The node locks onto the single product input wire and automatically cycles it sequentially across your entire background folder.
* **N Foreground Assets + 1 Background:** The node continuously stamps your folder of product assets onto a single, stationary background wire frame.
* **M Mismatched Quantities:** If `batch_pairing_mode` is set to `Cross-Multiply`, the engine executes a full matrix pass ($N \times M$), pairing every asset variation exactly once without creating uneven duplicates.

---

## 💎 Hidden Architecture Protections

* **⚡ Sequential VRAM Isolation:** Processing heavy segment loaders (like BiRefNet) over massive batches normally triggers OOM (Out of Memory) failures. Quick Merge pins large tensor stacks safely in system RAM, feeds them to the GPU one single frame at a time, and aggressively clears the CUDA cache after every execution step.
* **🛡️ Matrix Format Intersector:** Mixing standard 3-channel `.jpg` backdrops with transparent 4-channel `.png` foreground assets breaks typical PyTorch matrix math. The loader instantly intercepts incoming files and normalizes them into uniform RGBA float32 tensors before they reach the compositor core.

---

## ⚠️ Critical Performance Warnings

* **❗ AI Supersampling VRAM Load:** The integrated AI Upscaling engine (`⚡ ai_upscale`) is computationally heavy. Running high multipliers (`>2.0x`) over high-resolution image batches simultaneously will easily exhaust your graphics memory. Keep your `📁 batch_chunk_size` set to low numbers (e.g., `1` to `4`) to prevent system crashes.
* **📐 Background Aspect Ratio Parity Rule:** For studio backgrounds containing a podium, pedestal, or table, **all images in your folder must share identical aspect ratios and resolutions**. If background sizes fluctuate, standard bounding box and letterbox resizers will cause the pedestal coordinates to shift unevenly between frames, causing product layouts to miss their structural targets.
* **🛠️ Auto-Position State:** The `auto-position (WiP)` feature is undergoing a structural logic migration and is temporarily locked down. Leave this parameter deactivated (`False`) to ensure stable rendering execution.
