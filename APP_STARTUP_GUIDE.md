# Setting Up and Running the CAPE Streamlit App

This guide walks you through the process of setting up your local environment and launching the CAPE Bird Classifier & Explainer application.

## Prerequisites

Ensure you have **Python 3.8+** and `git` installed on your system.

### Step 1: Clone the Repository

Open your command line, terminal, or bash, and clone your project repository. Then, navigate into the project folder.

```
git clone https://github.com/Dora-The-Explorer-Is-Dead/cape-reproduction/tree/stage4
cd cape-reproduction

```

### Step 2: Install Required Python Libraries

With your terminal open in the project directory, install the necessary Python packages. You can install them directly using `pip`:

```
pip install streamlit torch torchvision matplotlib opencv-python Pillow numpy huggingface_hub

```


### Step 3: Download the Model Weights (`best.pth`)

The app requires the trained model weights to run inferences. You need to download the `best.pth` file from your Hugging Face repository and place it in the root directory of your project (the same folder where `app.py` is located).

You can download it manually from the link below and move it to your folder:
🔗 [**https://huggingface.co/AyaanArif/cape-cub-pf**](https://huggingface.co/AyaanArif/cape-cub-pf)

Alternatively, since you installed `huggingface_hub`, you can download it via the command line:

```
huggingface-cli download AyaanArif/cape-cub-pf best.pth --local-dir .

```

### Step 4: Launch the Application

Once your libraries are installed and `best.pth` is in the main folder alongside `app.py`, start the Streamlit server:

```
streamlit run app.py

```

This command will initialize the app and automatically open it in your default web browser (usually at `http://localhost:8501`). From there, you can upload your CUB dataset images and view the visual explanation heatmaps! DEMO images have been added in the relevant folder for your ease.


## Understanding the CAPE Visualizations

### 1. Paper-Style Probability Grids
The app includes a sidebar toggle to **"Overlay Paper-Style Grid Values"**. This replicates the exact visualization method used in the official CAPE paper to prove mathematical rigor:
* **What it shows:** Instead of a heavily smoothed heatmap, this overlays the model's raw 14x14 spatial feature grid. The percentage in each box represents the exact probabilistic contribution of that specific image patch.
* **The Math:** CAPE is mathematically calibrated. If you set the grid threshold to `0.0`, the sum of all the boxes on the screen will perfectly equal the model's final confidence percentage! 
* **The Threshold Slider:** The slider hides low-relevance background cells (e.g., values below `0.05`). This keeps the visualization uncluttered and highlights CAPE's *sparsity*—proving the model relies on highly localized, definitive features (like a beak or eye ring) rather than the general background.

### 2. Fine-Grained Class Discrimination (Contrastive Analysis)
At the bottom of the app is the **Contrastive Analysis** tool. This demonstrates CAPE's core advantage over traditional CAM methods when dealing with visually similar sub-categories.
* **How it works:** The app generates the isolated $\mu$-CAPE map for Class A (defaulting to the top prediction) and subtracts the map for Class B (the runner-up).
* **How to read the map:** 
  * **Red Regions:** Features that strongly pushed the model toward predicting Class A.
  * **Blue Regions:** Features that pushed the model toward predicting Class B.
  * **White Regions:** Shared, generic features (like the background sky or basic bird silhouette) that cancel out because they don't help differentiate the two birds.
* **Why it matters:** If you ask standard CAM to explain two similar birds, it highlights the whole bird both times. CAPE's contrastive map explicitly isolates the precise, fine-grained physical differences that actually matter for classification.