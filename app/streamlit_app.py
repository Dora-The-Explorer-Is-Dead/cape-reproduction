"""
CAPE reproduction — deployment app.

Upload a bird image (ideally a CUB-200-2011 species, since the model was
trained on those 200 classes), get the predicted species, confidence score,
and side-by-side CAM, Grad-CAM, Grad-CAM++, CAPE, and mu-CAPE explanation
heatmaps.

Run locally:
    streamlit run app/streamlit_app.py

Requires: the checkpoint on Hugging Face (pulled automatically on first run),
and the AIML-MED/CAPE reference repo cloned locally (for model.py) — see
SETUP_RUNBOOK.md.
"""

import os
import sys

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- Locate the reference repo (same auto-detection logic as the other scripts) ---
CAPE_REPO_PATH = next((p for p in [
    os.environ.get('CAPE_REPO_PATH'),
    os.path.expanduser('~/cape-reference'),
    '/kaggle/working/cape-reference',
    '/content/cape-reference',
    './cape-reference',
    '../cape-reference',
] if p and os.path.isdir(p)), None)

if CAPE_REPO_PATH is None:
    st.error(
        "Could not find the cloned AIML-MED/CAPE reference repo. Clone it with:\n\n"
        "`git clone https://github.com/AIML-MED/CAPE.git ~/cape-reference`\n\n"
        "or set the CAPE_REPO_PATH environment variable before launching this app."
    )
    st.stop()

sys.path.insert(0, CAPE_REPO_PATH)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.model import Net  # noqa: E402
from configs import transforms as cape_transforms  # noqa: E402

# Class names: CUB-200-2011 has a classes.txt file (e.g. "1 001.Black_footed_Albatross").
# Point this at your extracted dataset's classes.txt, or fall back to numeric labels.
CLASSES_TXT_CANDIDATES = [
    os.environ.get('CUB_CLASSES_TXT'),
    os.path.expanduser('~/cape-reproduction/data/CUB_200_2011/CUB_200_2011/classes.txt'),
    os.path.expanduser('~/cape-reproduction/data/CUB_200_2011/classes.txt'),
]


@st.cache_resource
def load_class_names():
    for path in CLASSES_TXT_CANDIDATES:
        if path and os.path.isfile(path):
            names = {}
            with open(path) as f:
                for line in f:
                    idx, name = line.strip().split(' ', 1)
                    names[int(idx) - 1] = name.replace('_', ' ')
            return names
    return None  # fall back to "Class N"


@st.cache_resource
def load_model():
    from huggingface_hub import hf_hub_download
    checkpoint_path = hf_hub_download(
        repo_id=st.session_state.get('hf_repo_id', 'umamamianoor/cape-cub-pf'),
        filename='best.pth',
    )

    config = {
        'network': 'resnet50',
        'pretrained': True,
        'cls_cols_dict': {'orig': 200, 'cape': 200},
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net(config).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    net.load_state_dict(state)
    net.eval()
    return net, device


class GradCAMHooks:
    def __init__(self, backbone):
        self.features = None
        self.gradients = None
        backbone.register_forward_hook(self._save_features)
        backbone.register_full_backward_hook(self._save_gradients)

    def _save_features(self, module, input, output):
        self.features = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]


def grad_cam(features, gradients):
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    return F.relu((weights * features).sum(dim=1, keepdim=True))


def grad_cam_pp(features, gradients):
    grads_2 = gradients ** 2
    grads_3 = gradients ** 3
    sum_feat_grad3 = (features * grads_3).sum(dim=(2, 3), keepdim=True)
    denom = 2 * grads_2 + sum_feat_grad3
    denom = torch.where(denom != 0, denom, torch.ones_like(denom))
    alpha = grads_2 / denom
    weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
    return F.relu((weights * features).sum(dim=1, keepdim=True))


def normalize_map(m):
    m = m.squeeze().detach().cpu().numpy()
    m = m - m.min()
    if m.max() > 0:
        m = m / m.max()
    return m


def generate_all_maps(net, image_tensor, target_class, hooks):
    image_tensor_batched = image_tensor.unsqueeze(0)

    with torch.no_grad():
        outputs = net(image_tensor_batched)
    cam_map = normalize_map(F.relu(outputs['orig']['weighted_contribution'][:, target_class:target_class + 1]))
    cape_map = normalize_map(outputs['cape']['weighted_contribution'][:, target_class:target_class + 1])
    mu_cape_map = normalize_map(outputs['cape']['logcampe_clip0'][:, target_class:target_class + 1])

    net.zero_grad()
    image_tensor_batched = image_tensor.clone().unsqueeze(0)
    image_tensor_batched.requires_grad_(True)
    outputs_grad = net(image_tensor_batched)
    score = outputs_grad['orig']['outcome'][0, target_class]
    score.backward()

    gc_map = normalize_map(grad_cam(hooks.features, hooks.gradients))
    gcpp_map = normalize_map(grad_cam_pp(hooks.features, hooks.gradients))

    return {'CAM': cam_map, 'Grad-CAM': gc_map, 'Grad-CAM++': gcpp_map, 'CAPE': cape_map, 'mu-CAPE': mu_cape_map}


def overlay_heatmap(orig_image_np, heatmap_2d):
    h_up = np.array(Image.fromarray((heatmap_2d * 255).astype(np.uint8)).resize(
        (orig_image_np.shape[1], orig_image_np.shape[0]), Image.BILINEAR)) / 255.0
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(orig_image_np)
    ax.imshow(h_up, cmap='jet', alpha=0.5)
    ax.axis('off')
    plt.tight_layout()
    return fig


# --- App layout ---

st.set_page_config(page_title='CAPE Bird Classifier & Explainer', layout='wide')
st.title('🐦 CAPE Reproduction: Bird Classifier & Explanation Comparison')
st.markdown(
    "Upload a bird image. The model predicts its species (from the 200 CUB-200-2011 "
    "classes) and generates five explanation heatmaps — **CAM**, **Grad-CAM**, "
    "**Grad-CAM++**, **CAPE**, and **μ-CAPE** — showing which regions drove the prediction."
)

with st.sidebar:
    st.header('Settings')
    hf_repo_id = st.text_input('Hugging Face checkpoint repo', value='umamamianoor/cape-cub-pf')
    st.session_state['hf_repo_id'] = hf_repo_id
    st.markdown('---')
    st.markdown(
        '**About the methods:**\n'
        '- CAM/Grad-CAM/Grad-CAM++: relative attention, not comparable across classes\n'
        '- CAPE: sharp, class-discriminative, values sum to the confidence score\n'
        '- μ-CAPE: broader coverage, closer to CAM-family methods'
    )

uploaded_file = st.file_uploader('Upload a bird image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    with st.spinner('Loading model...'):
        net, device = load_model()
        hooks = GradCAMHooks(net.net)
        class_names = load_class_names()

    image_pil = Image.open(uploaded_file).convert('RGB')
    _, test_transform = cape_transforms.cub_transform(448)
    image_tensor = test_transform(image_pil).to(device)

    with st.spinner('Running inference...'):
        with torch.no_grad():
            out = net(image_tensor.unsqueeze(0))
            probs = out['orig']['outcome'].softmax(dim=1)
            confidence, predicted_class = probs.max(dim=1)
            predicted_class = predicted_class.item()
            confidence = confidence.item()

        heatmaps = generate_all_maps(net, image_tensor, predicted_class, hooks)

    pred_name = class_names[predicted_class] if class_names else f'Class {predicted_class}'

    col_img, col_info = st.columns([1, 2])
    with col_img:
        st.image(image_pil, caption='Uploaded image', use_container_width=True)
    with col_info:
        st.metric('Predicted species', pred_name)
        st.metric('Confidence', f'{confidence:.1%}')

    st.markdown('### Explanation heatmaps')
    raw_image_np = np.array(image_pil.resize((448, 448)))
    cols = st.columns(5)
    for col, (method, hmap) in zip(cols, heatmaps.items()):
        with col:
            st.markdown(f'**{method}**')
            fig = overlay_heatmap(raw_image_np, hmap)
            st.pyplot(fig)
            plt.close(fig)

    st.markdown(
        '---\n'
        'Notice how CAM, Grad-CAM, and Grad-CAM++ highlight broadly similar regions, '
        'CAPE concentrates on a small, sharp area, and μ-CAPE sits in between — '
        'this matches what we observed and documented in our reproduction results.'
    )
else:
    st.info('Upload an image to get started. For best results, use a bird photo — '
            'the model was trained specifically on the 200 CUB-200-2011 species.')
