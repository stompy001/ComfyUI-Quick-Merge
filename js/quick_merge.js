import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "Comfy.QuickMergeNode",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "QuickMergeNode") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }

                const node = this;
                node.setSize([380, node.computeSize()[1]]);

                setTimeout(() => {
                    const groups = {
                        "📏 initial_resize": [
                            "📏 fg_resize_mode", "📏 fg_crop_align", "📏 fg_initial_width", "📏 fg_initial_height",
                            "📏 bg_resize_mode", "📏 bg_crop_align", "📏 bg_initial_width", "📏 bg_initial_height"
                        ],
                        "✂️ merge_image": [
                            "✂️ cutout_model", "✂️ auto-position (WiP)", "✂️ foreground_size", "✂️ foreground_left_right", 
                            "✂️ foreground_up_down", "✂️ flip_horizontal", "✂️ flip_vertical", 
                            "✂️ rotation_degrees", "✂️ edge_shrink_grow", "✂️ edge_softness", 
                            "✂️ foreground_opacity", "✂️ blend_mode", "✂️ blend_strength"
                        ],
                        "⚡ ai_upscale": [
                            "⚡ pipeline_order", "⚡ choose_upscale_model", "⚡ upscale_by",
                            "⚡ tile_size (0 means off/native)", "⚡ post_upscale_edge_soften"
                        ],
                        "🎨 colors_and_filters": [
                            "🎨 filter_target", "🎨 filter_type", "🎨 filter_strength", "🎨 brightness", 
                            "🎨 contrast", "🎨 saturation", "🎨 filter_size", "🎨 background_blur"
                        ],
                        "🌗 shadows": [
                            "🌗 shadow_color_hex", "🌗 shadow_darkness", "🌗 shadow_softness", 
                            "🌗 shadow_position_x", "🌗 shadow_position_y"
                        ],
                        "🎯 target_resolution": [
                            "🎯 target_resize_mode", "🎯 target_crop_align", "🎯 target_width", "🎯 target_height"
                        ]
                    };

                    const updateVisibility = () => {
                        let isDirty = false;
                        
                        const masterBatchWidget = node.widgets.find(w => w.name === "📁 batch_processing");
                        const masterBatchActive = masterBatchWidget ? masterBatchWidget.value : false;

                        const fgBatchWidget = node.widgets.find(w => w.name === "📁 foreground_batch");
                        const bgBatchWidget = node.widgets.find(w => w.name === "📁 background_batch");

                        ["📁 foreground_batch", "📁 background_batch"].forEach(name => {
                            const w = node.widgets.find(wg => wg.name === name);
                            if (w) {
                                if (!masterBatchActive) {
                                    if (w.type !== "hidden") {
                                        w._original_type = w.type;
                                        w.type = "hidden";
                                        w.computeSize = () => [0, -4];
                                        isDirty = true;
                                    }
                                } else {
                                    if (w.type === "hidden") {
                                        w.type = w._original_type || "toggle";
                                        w.computeSize = () => [380, 20];
                                        isDirty = true;
                                    }
                                }
                            }
                        });

                        const fgFolderWidget = node.widgets.find(w => w.name === "📁 fg_batch_folder");
                        if (fgFolderWidget) {
                            const visible = masterBatchActive && fgBatchWidget && fgBatchWidget.value;
                            if (!visible && fgFolderWidget.type !== "hidden") {
                                fgFolderWidget._original_type = fgFolderWidget.type;
                                fgFolderWidget.type = "hidden";
                                fgFolderWidget.computeSize = () => [0, -4];
                                isDirty = true;
                            } else if (visible && fgFolderWidget.type === "hidden") {
                                fgFolderWidget.type = fgFolderWidget._original_type || "text";
                                fgFolderWidget.computeSize = () => [380, 20];
                                isDirty = true;
                            }
                        }

                        const bgFolderWidget = node.widgets.find(w => w.name === "📁 bg_batch_folder");
                        if (bgFolderWidget) {
                            const visible = masterBatchActive && bgBatchWidget && bgBatchWidget.value;
                            if (!visible && bgFolderWidget.type !== "hidden") {
                                bgFolderWidget._original_type = bgFolderWidget.type;
                                bgFolderWidget.type = "hidden";
                                bgFolderWidget.computeSize = () => [0, -4];
                                isDirty = true;
                            } else if (visible && bgFolderWidget.type === "hidden") {
                                bgFolderWidget.type = bgFolderWidget._original_type || "text";
                                bgFolderWidget.computeSize = () => [380, 20];
                                isDirty = true;
                            }
                        }

                        const anySubBatchActive = masterBatchActive && ((fgBatchWidget && fgBatchWidget.value) || (bgBatchWidget && bgBatchWidget.value));
                        ["📁 batch_pairing_mode", "📁 batch_chunk_size"].forEach(name => {
                            const w = node.widgets.find(wg => wg.name === name);
                            if (w) {
                                if (!anySubBatchActive) {
                                    if (w.type !== "hidden") {
                                        w._original_type = w.type;
                                        w.type = "hidden";
                                        w.computeSize = () => [0, -4];
                                        isDirty = true;
                                    }
                                } else {
                                    if (w.type === "hidden") {
                                        w.type = w._original_type || "combo";
                                        w.computeSize = () => [380, 20];
                                        isDirty = true;
                                    }
                                }
                            }
                        });

                        for (const [toggleName, widgetNames] of Object.entries(groups)) {
                            const toggleWidget = node.widgets.find(w => w.name === toggleName);
                            if (toggleWidget) {
                                const isVisible = toggleWidget.value;
                                for (const wName of widgetNames) {
                                    const w = node.widgets.find(wg => wg.name === wName);
                                    if (w) {
                                        if (!isVisible) {
                                            if (w.type !== "hidden") {
                                                w._original_type = w.type;
                                                w._original_computeSize = w.computeSize;
                                                w.type = "hidden";
                                                w.computeSize = () => [0, -4]; 
                                                isDirty = true;
                                            }
                                        } else {
                                            if (w.type === "hidden") {
                                                w.type = w._original_type || "text";
                                                w.computeSize = w._original_computeSize || (() => [380, 20]);
                                                isDirty = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        
                        if (isDirty) {
                            node.setSize([380, node.computeSize()[1]]);
                            app.graph.setDirtyCanvas(true, true);
                        }
                    };

                    const setupPlaceholder = (widgetName, defaultText) => {
                        const widget = node.widgets.find(w => w.name === widgetName);
                        if (widget) {
                            const originalMouse = widget.mouse;
                            widget.mouse = function(e, pos) {
                                if (widget.type === "hidden") return false;
                                if (e.type === "mousedown" || e.type === "pointerdown") {
                                    if (widget.value === defaultText) {
                                        widget.value = "";
                                        if (widget.callback) widget.callback(widget.value);
                                        app.graph.setDirtyCanvas(true, true);
                                    }
                                }
                                if (originalMouse) return originalMouse.apply(this, arguments);
                                return false;
                            };
                        }
                    };
                    setupPlaceholder("📁 fg_batch_folder", "📁 Type or paste folder path here...");
                    setupPlaceholder("📁 bg_batch_folder", "📁 Type or paste folder path here...");

                    const shadowHexWidget = node.widgets.find(w => w.name === "🌗 shadow_color_hex");
                    if (shadowHexWidget) {
                        const originalMouse = shadowHexWidget.mouse;
                        shadowHexWidget.mouse = function(e, pos) {
                            if (shadowHexWidget.type === "hidden") return false;
                            if (e.type === "mousedown" || e.type === "pointerdown") {
                                const colorPicker = document.createElement("input");
                                colorPicker.type = "color";
                                colorPicker.value = shadowHexWidget.value.startsWith('#') ? shadowHexWidget.value : '#' + shadowHexWidget.value;
                                colorPicker.style.position = "absolute";
                                colorPicker.style.opacity = "0";
                                colorPicker.style.pointerEvents = "none";
                                document.body.appendChild(colorPicker);

                                colorPicker.addEventListener("input", (event) => {
                                    shadowHexWidget.value = event.target.value;
                                    if (shadowHexWidget.callback) shadowHexWidget.callback(shadowHexWidget.value);
                                    app.graph.setDirtyCanvas(true, true);
                                });
                                colorPicker.addEventListener("change", () => {
                                    document.body.removeChild(colorPicker);
                                });
                                colorPicker.click();
                                return true;
                            }
                            if (originalMouse) return originalMouse.apply(this, arguments);
                            return false;
                        };
                    }

                    for (const toggleName of [...Object.keys(groups), "📁 batch_processing", "📁 foreground_batch", "📁 background_batch"]) {
                        const toggleWidget = node.widgets.find(w => w.name === toggleName);
                        if (toggleWidget) {
                            const originalCallback = toggleWidget.callback;
                            toggleWidget.callback = function () {
                                if (originalCallback) originalCallback.apply(this, arguments);
                                updateVisibility();
                            };
                        }
                    }
                    updateVisibility(); 
                }, 50); 
            };
        }
    }
});