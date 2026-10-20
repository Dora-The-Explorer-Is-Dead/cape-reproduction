import os
import sys
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import streamlit as st
import cv2
from PIL import Image

# Ensure the src directory is accessible for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from explain import GradCAMHooks, build_model, generate_all_maps

st.set_page_config(page_title="CAPE Bird Classifier & Explainer", layout="wide")

# Standard CUB / ImageNet evaluation transforms
eval_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


@st.cache_resource
def load_checkpoint(ckpt_path):
    class Args:
        model_name = "resnet50"
        network = "resnet50"
        num_class = 200
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

    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file).convert("RGB")

        col_img, col_pred = st.columns([1, 1])

        with col_img:
            st.image(raw_image, caption="Original Image", width=320)

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
            st.metric(label="Predicted Class", value=class_label)
            st.metric(
                label="Confidence", value=f"{conf.item() * 100:.2f}%"
            )

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
                    hm = cv2.resize(hm, (224, 224))  # <-- Add this line to smooth the heatmap!
                    
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.imshow(raw_image.resize((224, 224)))
                    ax.imshow(hm, cmap="jet", alpha=0.5)
                    ax.axis("off")
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("Map not available")