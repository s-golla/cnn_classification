```markdown
# Run Instructions — cnn_classification

This document explains how to set up and run the `cnn_classification` project on another computer (Windows-focused). Follow the steps below to reproduce the environment, run the Streamlit app, and test single images.

## Prerequisites

- Python 3.8–3.11 installed and on `PATH`.
- Git (optional) to clone the repository.
- A trained model file named `cnn_model.keras` (place in project root).

If you don't have `cnn_model.keras`, the Streamlit app will show an error asking you to train the model first.

## Files of interest

- `app.py` — Streamlit web interface.
- `cnn_classify.py` — training script (used to create `cnn_model.keras`).
- `cnn_model.keras` — trained model file (must be provided).
- `test_single_image.py` — CLI-style test for a single image.
- `requirements.txt` — Python dependencies.

## Setup (Windows)

1. Clone or copy the project to the target machine.

2. Open PowerShell and change to the project directory:

```powershell
cd C:\path\to\cnn_classification
```

3. Create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
```

4. Upgrade pip and install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `requirements.txt` is missing or packages fail to install, you can install the main packages used by the project:

```powershell
pip install streamlit tensorflow pillow matplotlib numpy
```

5. Place the trained model file `cnn_model.keras` in the project root (same folder as `app.py`).

## Run the Streamlit app

Start the app from the activated virtual environment:

```powershell
streamlit run app.py
```

Open the URL shown by Streamlit in a browser (usually `http://localhost:8501`). Upload an MRI image using the uploader in the app.

## Test a single image from the command line

`test_single_image.py` uses the model file `cnn_model.keras` and prints predictions. You can edit the `test_image` path in the file or run a quick script:

```powershell
python test_single_image.py
```

Modify the script if you want to pass arguments; the example script reads `data/healthy/non_2.jpg` by default.

## Troubleshooting

- Model file not found: Ensure `cnn_model.keras` is in the project root.
- Dependency issues: Make sure the virtual environment is activated and Python version is compatible.
- GPU TensorFlow problems: If you don't need GPU support, the CPU build of TensorFlow is used by default. To avoid GPU errors, install a CPU-only TensorFlow wheel matching your Python version.
- Streamlit port in use: Use `streamlit run app.py --server.port 8502` to change ports.

## Notes & Disclaimer

- This project is provided for educational and research purposes only. It is not a medical diagnostic tool. Always consult a qualified medical professional for clinical decisions.
- The model performance depends on the quality and domain match of the images you supply. The app includes basic image validation but may still reject or misclassify out-of-domain images.

If you want, I can also create a simplified one-line install script (`setup_env.bat`) or add CLI arguments to `test_single_image.py`. Which would you prefer?

```
