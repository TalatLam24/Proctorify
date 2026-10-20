# A Unified, Multimodal Deep Learning Framework for Online Exam Proctoring and Student Attention Detection

This repository contains the complete codebase for the CSE 4098A Capstone project at the University of Liberal Arts Bangladesh (ULAB). The project develops a sophisticated system to address academic integrity and student engagement in online learning environments by analyzing behavioral biometrics in real-time.

---

## 1. Project Overview

The system utilizes a "Honeypot Exam" protocol to collect naturalistic data from users. It employs a multi-camera setup (laptop webcam + smartphone) to generate a perfectly labeled "ground truth" dataset without manual annotation. This data is then used to train a PyTorch-based LSTM sequence model.

The final AI model is capable of real-time inference, analyzing a 3-second rolling window of a user's behavior to predict their state with high accuracy.

**Key Features:**
-   **Multimodal Data Fusion:** Combines Visual (Head Pose, Gaze), Auditory (Speech), Interactional (Keystrokes, Mouse), and Contextual (Active Window) data streams.
-   **Automated Data Collection:** A robust GUI application (`honeypot_exam.py`) that guides users and records all data streams, including a secondary ground-truth camera via a Flask/Ngrok server.
-   **Automated Feature Extraction:** A processing pipeline (`feature_extractor.py`, `dataset_merger.py`) to convert raw video and logs into a machine-learning-ready dataset.
-   **Deep Learning Model:** An LSTM-based sequence model (`train_pytorch_model.py`) built in PyTorch to understand temporal patterns in user behavior.
-   **Live Inference:** A real-time dashboard (`live_proctor.py`) that uses the trained model to predict a user's state (Normal, Physical Cheating, Digital Cheating) via the webcam.

---

## 2. Running the Project

**Prerequisites:**
-   **OS:** Windows 10/11 (due to the use of `ctypes` for Active Window tracking).
-   **Python:** 3.10 or newer.
-   **Git:** [Git SCM](https://git-scm.com/downloads) installed.
-   **Ngrok:** A free account and authtoken from [ngrok.com](https://dashboard.ngrok.com/get-started/your-authtoken).

### Step-by-Step Instructions:

**1. Clone the Repository:**
Open your terminal or command prompt and run:
```bash
git clone https://github.com/Laam24/Multimodal-Proctoring-Project-data-collector.git
cd Multimodal-Proctoring-Project-data-collector
```

**2. Create a Virtual Environment:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**3. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**4. Download MediaPipe Models:**
Download the following two files and place them in the root of your project folder:
-   **Face Landmarker:** [face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)
-   **Hand Landmarker:** [hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)

**5. Configure the Server:**
-   Open the `server.py` file.
-   Find the line `ngrok.set_auth_token("YOUR_AUTHTOKEN_HERE")`.
-   Replace `YOUR_AUTHTOKEN_HERE` with your actual token from the ngrok dashboard.

**6. Run the Live Proctoring Demo:**
-   Ensure the trained model (`best_proctoring_model.pth`), `scaler.pkl`, and `label_encoder.pkl` files are in your project directory.
-   Run the live inference script:
```bash
python live_proctor.py
```
A window will open showing your webcam feed with the live AI predictions overlaid.

---

## 3. Data Collection & Training (Optional)

If you wish to collect your own data and retrain the model, follow these steps:

**1. Start the Server:**
In a dedicated terminal (with `venv` activated), run the Flask server. It will print a secure `https://...` URL.
```bash
python server.py
```

**2. Configure and Run the Exam App:**
-   Open `honeypot_exam.py`.
-   Update the `NGROK_URL` variable with the URL printed by your server.
-   In a second terminal, run the exam app:
```bash
python hone_exam.py
```
-   Follow the on-screen instructions to collect data.

**3. Process and Train:**
-   Once you have collected data in the `Honeypot_Sessions` folder, run the processing pipeline in order:
```bash
python label_standardizer.py  # Create and verify master_labels.csv
python feature_extractor.py   # Extract angles from videos
python dataset_merger.py      # Merge all data into the final dataset
python train_pytorch_model.py # Train the AI model
```
This will generate a new `best_proctoring_model.pth` file based on your custom data.