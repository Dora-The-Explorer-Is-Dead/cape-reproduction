import os
import sys
import re
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from PIL import Image

# 1. Locate the CAPE reference repository just like explain.py does
CAPE_REPO_PATH = next((p for p in [
    os.environ.get('CAPE_REPO_PATH'),
    os.path.expanduser('~/cape-reference'),
    '/kaggle/working/cape-reference',
    '/content/cape-reference',
] if p and os.path.isdir(p)), None)

if CAPE_REPO_PATH is not None:
    sys.path.insert(0, CAPE_REPO_PATH)

# Ensure the local src directory is also accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# 2. Now you can safely import configs and models
from configs import transforms as cape_transforms
from explain import GradCAMHooks, build_model, generate_all_maps

st.set_page_config(page_title="CAPE Bird Classifier & Explainer", layout="wide")

# 3. Use 448x448 CUB transform instead of the manual Compose
_, eval_transform = cape_transforms.cub_transform(448)

@st.cache_resource
def load_checkpoint(ckpt_path):
    class Args:
        model_name = "resnet50"
        network = "resnet50"
        num_class = 200
        image_size = 448
        reload_path = ckpt_path
        checkpoint = ckpt_path
        T_kld = 2.0

    args = Args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = build_model(args, device)
    net.eval()
    return net, device


def load_classes(classes_path="classes.txt"):
    if os.path.exists(classes_path):
        with open(classes_path, "r") as f:
            lines = [line.strip().split(" ", 1) for line in f.readlines()]
            return {
                int(num) - 1: name for num, name in lines if len(num) > 0
            }
    return None

def auto_detect_class(filename, classes_map):
    """Attempts to find the true class ID from the filename."""
    if not classes_map:
        return None
    
    filename_lower = filename.lower().replace("_", " ").replace("-", " ")
    
    # 1. PRIORITY: Check for the exact bird name in the filename (Safest)
    for class_idx, class_name in classes_map.items():
        # Clean the map name (e.g., "017.Cardinal" -> "cardinal")
        clean_name = re.sub(r'^\d+\.', '', class_name).lower().replace("_", " ")
        if clean_name in filename_lower:
            return class_idx
            
    # 2. FALLBACK: Check for isolated numbers
    # We use \b to ensure we only grab standalone numbers, not parts of a hash
    numbers = re.findall(r'\b\d+\b', filename)
    for num_str in numbers:
        num = int(num_str)
        # Check exact index or 1-based index
        if num in classes_map:
            return num
        if (num - 1) in classes_map:
            return num - 1
            
    return None

# Sidebar Setup
st.sidebar.title("Configuration")
ckpt_input = st.sidebar.text_input("Checkpoint Path", "best.pth")
classes_map = load_classes("classes.txt")

st.title("CAPE: Bird Classification & Saliency Map Explanations")
st.markdown(
    """
Upload an image of a bird from the CUB-200-2011 dataset. 
The model will output the top prediction and compare the explanation heatmaps:
**CAM**, **Grad-CAM**, **Grad-CAM++**, **CAPE (PF)**, and **μ-CAPE (PF)**.
"""
)

if not os.path.exists(ckpt_input):
    st.warning(
        f"Checkpoint `{ckpt_input}` not found. Please verify the path in the sidebar."
    )
else:
    try:
        net, device = load_checkpoint(ckpt_input)
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        st.stop()

    uploaded_file = st.file_uploader(
        "Choose a bird image...", type=["jpg", "jpeg", "png"]
    )
    
    # Dropdown setup
    class_options = ["Unknown"]
    if classes_map:
        class_options += [f"{idx} - {name}" for idx, name in classes_map.items()]
    
    # Check for auto-detected class
    auto_detected_idx = None
    default_dropdown_index = 0
    
    if uploaded_file is not None:
        auto_detected_idx = auto_detect_class(uploaded_file.name, classes_map)
        if auto_detected_idx is not None:
            st.toast(f"Auto-detected true class from filename: {classes_map[auto_detected_idx]}")
            # Find the matching dropdown string to set as default
            target_str = f"{auto_detected_idx} - {classes_map[auto_detected_idx]}"
            if target_str in class_options:
                default_dropdown_index = class_options.index(target_str)

    true_class_selection = st.selectbox("Select True Class", class_options, index=default_dropdown_index)

    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file).convert("RGB")
        
        # Prepare the display image to match the 448x448 model input
        orig_image_np = np.array(raw_image.resize((448, 448)))

        col_img, col_pred = st.columns([1, 1])

        with col_img:
            st.image(raw_image.resize((448, 448)), caption="Original Image (448x448)", width=320)

        # Preprocess input tensor
        img_tensor = eval_transform(raw_image).unsqueeze(0).to(device)
        img_tensor.requires_grad_(True)

        # Model Inference
        with torch.set_grad_enabled(True):
            out = net(img_tensor)
            logits = out["orig"]["outcome"]
            probs = torch.softmax(logits, dim=1)
            conf, pred_id = probs.max(dim=1)
            class_idx = pred_id.item()

            class_label = (
                classes_map.get(class_idx, f"Class {class_idx}")
                if classes_map
                else f"Class {class_idx}"
            )

            # Generate Heatmaps
            hooks = GradCAMHooks(net.net)
            heatmaps = generate_all_maps(
                net, img_tensor.squeeze(0), class_idx, hooks
            )

        with col_pred:
            st.subheader("Model Prediction")
            predicted_str = f"{class_idx} - {class_label}"
            
            st.metric(label="Predicted Class", value=predicted_str)
            st.metric(label="Confidence", value=f"{conf.item() * 100:.2f}%")
            
            if true_class_selection != "Unknown":
                true_idx = int(true_class_selection.split(" - ")[0])
                if true_idx == class_idx:
                    st.success(f"**Correct!** The true class is {true_class_selection}")
                else:
                    st.error(f"**Incorrect.** The true class is {true_class_selection}")

        st.markdown("---")
        st.subheader("Visual Explanations Comparison")

        methods = [
            ("CAM", "CAM"),
            ("Grad-CAM", "Grad-CAM"),
            ("Grad-CAM++", "Grad-CAM++"),
            ("CAPE", "CAPE"),
            ("μ-CAPE", "mu-CAPE"),
        ]

        cols = st.columns(len(methods))

        for col, (title, key) in zip(cols, methods):
            with col:
                st.write(f"**{title}**")
                if key in heatmaps:
                    hm = heatmaps[key]
                    
                    # EXACT rendering logic from explain.py
                    h_up = np.array(Image.fromarray((hm * 255).astype(np.uint8)).resize(
                        (orig_image_np.shape[1], orig_image_np.shape[0]), Image.BILINEAR)) / 255.0
                    
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.imshow(orig_image_np)
                    ax.imshow(h_up, cmap='jet', alpha=0.5)
                    ax.axis('off')
                    
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("Map not available")