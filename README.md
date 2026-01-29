# ⚙️ Bearing Anomaly Detection (Digital Twin Prototype)

![Python](https://img.shields.io/badge/Python-3.11-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![Status](https://img.shields.io/badge/Status-Simulation%20Phase-yellow)

An unsupervised deep learning pipeline for detecting and diagnosing mechanical faults in rotating machinery. This branch focuses on **simulation and model validation** using the SUBFv1 dataset.

> **Note:** This branch contains the software simulation only. Hardware deployment (Arduino Uno Q) is currently being developed on the `arduino` branch.

## 🔍 Overview
This project implements a **1D Convolutional Autoencoder** to act as a "Digital Twin" of a healthy motor. Instead of classifying specific faults (which requires broken motors to train), the model learns the vibration signature of a *healthy* machine.

* **Detection:** Any deviation from the healthy baseline (high reconstruction error) is flagged as an anomaly.
* **Diagnosis:** The residual signal (Error) is analyzed using FFT to identify the fault type (Inner Race vs Outer Race) based on frequency signatures.

## 📂 Project Structure
The pipeline is broken down into modular Jupyter notebooks for reproducibility:

| File | Description |
| :--- | :--- |
| `00_eda.ipynb` | Exploratory Data Analysis. Validated that **RMS (Energy)** and **Kurtosis (Spikiness)** can statistically separate healthy vs. faulty signals. |
| `01_data_preprocessing.ipynb` | Converts raw CSVs into tensors. Implements **Vector Summation** ($\sqrt{x^2+y^2+z^2}$) for rotation invariance, sliding window segmentation (128 samples), and chronological train/test splitting. |
| `02_model_training.ipynb` | Builds and trains the **1D Conv Autoencoder**. Uses purely healthy data for training (Self-Supervised). |
| `03_anomaly_testing.ipynb` | Validation logic. Establishes the anomaly threshold ($\mu + 3\sigma$) and performs **Physics-Informed Diagnosis** (FFT on residuals) to classify fault types. |
| `processed_data/` | Stores the serialized tensors (`.npy`) and the fitted `MinMaxScaler` for consistent inference. |
| `models/` | Contains the trained model weights (`autoencoder_v1.h5`). |

## 🧠 Model Architecture
* **Input:** 128-point vibration window (Vector Sum Magnitude).
* **Encoder:** 3-layer 1D Convolution (compresses signal to latent space).
* **Bottleneck:** Forces the model to learn the "shape" of healthy vibration, stripping out noise.
* **Decoder:** 3-layer Transposed Convolution (reconstructs the signal).
* **Loss Function:** Mean Squared Error (MSE).

## 📊 Results (SUBFv1)
* **Separation:** The model achieves near-perfect separation between Healthy and Faulty test samples.
* **Thresholding:** Using a **3-Sigma** cutoff on the reconstruction error effectively filters out normal operating noise.
* **Diagnosis:** FFT analysis of the residual signal successfully recovers the characteristic fault frequencies (BPFI/BPFO).

## 🚀 Usage

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download Dataset**
    Run the helper script to fetch the SUBFv1 dataset:
    ```bash
    python download_SUBFv1.py
    ```

3.  **Run Pipeline**
    Execute the notebooks in order:
    1.  `01_data_preprocessing.ipynb`
    2.  `02_model_training.ipynb`
    3.  `03_anomaly_testing.ipynb`

## 🔜 Future Roadmap
* **Hardware Integration:** Porting the inference logic to **Arduino Uno Q**.
* **Real-world Validation:** Testing on a variable-speed table fan setup.
* **Live Dashboard:** Real-time plotting of residuals via Serial communication.