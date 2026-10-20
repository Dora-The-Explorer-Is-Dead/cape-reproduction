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