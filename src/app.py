import os
import time
import requests
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import matplotlib.pyplot as plt
import subprocess

# Set page config
st.set_page_config(
    page_title="Visual Quality Inspection System",
    page_icon="⚙️",
    layout="wide",
)

# Constants
PRED_LOG_PATH = "data/predictions_log.csv"
FEEDBACK_DIR = "data/feedback"
MODEL_PATH = "models/model.pth"
BENTO_URL = "http://localhost:3000/predict"

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs(os.path.join(FEEDBACK_DIR, "ok"), exist_ok=True)
os.makedirs(os.path.join(FEEDBACK_DIR, "defective"), exist_ok=True)

# Custom Styling
st.markdown("""
<style>
    .reportview-container {
        background: #0f1116;
    }
    .main-title {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #B0BEC5;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1A237E;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-val {
        font-size: 2rem;
        color: #ffffff;
        font-weight: bold;
    }
    .metric-lbl {
        font-size: 0.9rem;
        color: #B0BEC5;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load PyTorch model locally (fallback if BentoML is offline)
@st.cache_resource
def load_local_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Define architecture
        try:
            model = models.resnet50(weights=None)
        except AttributeError:
            model = models.resnet50()
            
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        payload = torch.load(MODEL_PATH, map_location=device)
        if isinstance(payload, dict) and "model_state_dict" in payload:
            model.load_state_dict(payload["model_state_dict"])
        else:
            model.load_state_dict(payload)
        model = model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading local model: {e}")
        return None

# Helper to run local model prediction
def predict_local(image):
    model_data = load_local_model()
    if model_data is None:
        return "N/A - Model Not Trained", 0.0, 0.0
        
    model, device = model_data
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    start_time = time.time()
    img_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, pred_idx = torch.max(probabilities, dim=0)
        
    latency = time.time() - start_time
    classes = ["defective", "ok"]
    return classes[pred_idx.item()], confidence.item(), latency

# Main App Header
st.markdown('<div class="main-title">⚙️ Visual Quality Inspection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-time Defect Detection & Active Learning Retraining Console</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Operator Console", "📋 Operator Review Queue", "🔄 Active Learning & Retraining"])

# Tab 1: Operator Console
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Component Input")
        upload_mode = st.radio("Select Image Source", ["Upload Image File", "Select Test Images from Disk"])
        
        input_image = None
        image_name = ""
        
        if upload_mode == "Upload Image File":
            uploaded_file = st.file_uploader("Upload Casting Part Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                input_image = Image.open(uploaded_file)
                image_name = uploaded_file.name
        else:
            # Look in raw test directory
            test_dirs = ["data/raw/test/ok", "data/raw/test/defective"]
            test_files = []
            for d in test_dirs:
                if os.path.exists(d):
                    for f in os.listdir(d):
                        if f.endswith((".png", ".jpg", ".jpeg")):
                            test_files.append(os.path.join(d, f))
                            
            if test_files:
                selected_file_path = st.selectbox("Select Test Image", test_files)
                if selected_file_path:
                    input_image = Image.open(selected_file_path)
                    image_name = os.path.basename(selected_file_path)
            else:
                st.warning("No test images found in data/raw/test. Run generate_data.py first.")
                
        if input_image is not None:
            st.image(input_image, caption=f"Casting Impeller: {image_name}", use_column_width=True)
            
    with col2:
        st.header("Inspection Analysis")
        if input_image is not None:
            # Trigger prediction
            with st.spinner("Analyzing casting part..."):
                # Attempt to query BentoML API
                try:
                    # Save temporary image for request
                    import io
                    img_byte_arr = io.BytesIO()
                    input_image.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    
                    response = requests.post(
                        BENTO_URL, 
                        files={"img": ("image.png", img_byte_arr, "image/png")},
                        timeout=3
                    )
                    if response.status_code == 200:
                        res_json = response.json()
                        pred_class = res_json["prediction"]
                        confidence = res_json["confidence"]
                        latency = res_json["latency_seconds"]
                        service_mode = "BentoML Server (Production)"
                    else:
                        raise Exception("BentoML returned non-200 status")
                except Exception:
                    # Fallback to local inference
                    pred_class, confidence, latency = predict_local(input_image)
                    service_mode = "Local PyTorch Engine (Fallback)"
            
            # Display results
            if pred_class == "ok":
                st.success(f"### Prediction: OK (Non-Defective)")
                color = "green"
            else:
                st.error(f"### Prediction: DEFECTIVE")
                color = "red"
                
            st.write(f"**Inference Server:** {service_mode}")
            st.metric(label="Confidence Score", value=f"{confidence*100:.2f}%")
            st.metric(label="Latency", value=f"{latency*1000:.1f} ms")
            
            # Log prediction to CSV
            if st.button("Log Prediction & Save"):
                # Save image to log directory
                log_img_dir = "data/logged_images"
                os.makedirs(log_img_dir, exist_ok=True)
                saved_img_path = os.path.join(log_img_dir, f"{int(time.time())}_{image_name}")
                input_image.save(saved_img_path)
                
                # Append to CSV
                new_row = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "image_name": image_name,
                    "image_path": saved_img_path,
                    "prediction": pred_class,
                    "confidence": confidence,
                    "latency_ms": latency * 1000,
                    "corrected_label": ""
                }
                
                if os.path.exists(PRED_LOG_PATH):
                    df = pd.read_csv(PRED_LOG_PATH)
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                else:
                    df = pd.DataFrame([new_row])
                df.to_csv(PRED_LOG_PATH, index=False)
                st.success("Inspection logged successfully!")
        else:
            st.info("Upload or select a casting part image to run visual quality inspection.")

# Tab 2: Operator Review Queue
with tab2:
    st.header("Operator Review & Label Correction (Human-in-the-Loop)")
    st.write("Correct any misclassifications identified on the shop floor. Corrected labels will be queued for the next retraining run.")
    
    if os.path.exists(PRED_LOG_PATH):
        df = pd.read_csv(PRED_LOG_PATH)
        
        if len(df) > 0:
            # Display logs
            st.dataframe(df[["timestamp", "image_name", "prediction", "confidence", "corrected_label"]])
            
            # Correction Form
            st.subheader("Correct Misclassification")
            row_idx = st.selectbox("Select Row index to correct:", range(len(df)))
            selected_row = df.iloc[row_idx]
            
            # Show image of selected row
            if os.path.exists(selected_row["image_path"]):
                col_img, col_form = st.columns([1, 1])
                with col_img:
                    st.image(Image.open(selected_row["image_path"]), caption=selected_row["image_name"], use_column_width=True)
                with col_form:
                    st.write(f"**Log Date:** {selected_row['timestamp']}")
                    st.write(f"**Model Prediction:** {selected_row['prediction']}")
                    st.write(f"**Model Confidence:** {selected_row['confidence']*100:.2f}%")
                    
                    correct_label = st.radio("Correct Label:", ["ok", "defective"])
                    
                    if st.button("Submit Label Correction"):
                            # Update CSV
                            df.at[row_idx, "corrected_label"] = correct_label
                            df.to_csv(PRED_LOG_PATH, index=False)

                            # Copy image to feedback directory for retraining
                            fb_dest = os.path.join(FEEDBACK_DIR, correct_label, selected_row["image_name"])
                            img = Image.open(selected_row["image_path"])
                            img.save(fb_dest)

                            st.success(f"Feedback saved! Image copied to: {fb_dest}")
                            st.rerun()
                st.error("Logged image file not found.")
        else:
            st.info("No logged predictions found yet.")
    else:
        st.info("No logged predictions found yet. Run inspections in the Operator Console tab.")

# Tab 3: Active Learning & Retraining
with tab3:
    st.header("Active Learning Retraining Pipeline")
    
    # Show statistics of Feedback / Active Learning dataset
    st.subheader("Dataset Statistics")
    
    # Calculate feedback counts
    fb_ok_cnt = len(os.listdir(os.path.join(FEEDBACK_DIR, "ok"))) if os.path.exists(os.path.join(FEEDBACK_DIR, "ok")) else 0
    fb_def_cnt = len(os.listdir(os.path.join(FEEDBACK_DIR, "defective"))) if os.path.exists(os.path.join(FEEDBACK_DIR, "defective")) else 0
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{fb_ok_cnt + fb_def_cnt}</div><div class="metric-lbl">Total Operator Corrections</div></div>', unsafe_allow_html=True)
    with col_stat2:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{fb_ok_cnt}</div><div class="metric-lbl">Corrected "OK" Parts</div></div>', unsafe_allow_html=True)
    with col_stat3:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{fb_def_cnt}</div><div class="metric-lbl">Corrected "Defective" Parts</div></div>', unsafe_allow_html=True)
        
    st.write("---")
    
    col_train, col_results = st.columns([1, 1])
    
    with col_train:
        st.subheader("Retraining Actions")
        st.write("Retraining triggers a PyTorch transfer learning training cycle. The pipeline will merge the baseline DVC dataset with the operator corrected labels collected above.")
        
        epochs = st.slider("Epochs", min_value=1, max_value=10, value=3)
        batch_size = st.selectbox("Batch Size", [8, 16, 32], index=1)
        lr = st.selectbox("Learning Rate", [0.01, 0.001, 0.0001], index=1)
        
        if st.button("🚀 Trigger Model Retraining"):
            st.info("Retraining initiated! Executing PyTorch training script...")
            
            # Run train.py as a subprocess
            cmd = ["python", "src/train.py", "--epochs", str(epochs), "--batch_size", str(batch_size), "--lr", str(lr)]
            
            log_area = st.empty()
            log_text = ""
            
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    shell=False,
                    cwd=os.getcwd()
                )

                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        log_text += output
                        log_area.code(log_text[-1000:])

                rc = process.poll()
                if rc == 0:
                    st.success("Model Retrained Successfully!")
                    st.cache_resource.clear()
                else:
                    st.error(f"Retraining failed with exit code: {rc}")
            except Exception as e:
                st.error(f"Error executing training process: {e}")

        st.subheader("Data Version Control (DVC) Sync")
        st.write("Save the newly compiled active-learning dataset into DVC and commit changes to Git.")
        if st.button("💾 Run DVC & Git Sync"):
            with st.spinner("Syncing dataset and model changes..."):
                try:
                    git_proc = subprocess.run(["git", "add", "data/feedback"], capture_output=True, text=True)
                    dvc_proc = subprocess.run(["python", "-m", "dvc", "commit", "-f"], capture_output=True, text=True)
                    git_dvc_proc = subprocess.run(["git", "add", "data/feedback.dvc"], capture_output=True, text=True)
                    git_commit = subprocess.run(["git", "commit", "-m", "Active learning: Operator feedback sync"], capture_output=True, text=True)

                    st.code(f"Git Add: {git_proc.stdout}\nDVC Commit: {dvc_proc.stdout}\nGit Add DVC: {git_dvc_proc.stdout}\nGit Commit: {git_commit.stdout}")
                    st.success("DVC and Git tracking synced!")
                except Exception as e:
                    st.error(f"Sync error: {e}")
                    
    with col_results:
        st.subheader("Experiment Metrics")
        st.write("MLflow Logs (simulated metrics before/after active learning loop):")
        
        # Plotting comparison of accuracy before and after retraining
        # If model.pth exists, we mock or present metrics logged
        metrics = {
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Pre-Active Learning": [0.85, 0.83, 0.86, 0.84],
            "Post-Active Learning": [0.93, 0.94, 0.92, 0.93]
        }
        metrics_df = pd.DataFrame(metrics)
        st.table(metrics_df)
        
        # Render a simple comparison bar chart
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(metrics["Metric"]))
        width = 0.35
        
        # Apply dark styling matching the dashboard
        fig.patch.set_facecolor('#0f1116')
        ax.set_facecolor('#0f1116')
        ax.tick_params(colors='white')
        ax.yaxis.label.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.title.set_color('white')
        
        rects1 = ax.bar(x - width/2, metrics["Pre-Active Learning"], width, label='Pre-Feedback', color='#ef5350')
        rects2 = ax.bar(x + width/2, metrics["Post-Active Learning"], width, label='Post-Feedback', color='#66bb6a')
        
        ax.set_ylabel('Scores')
        ax.set_title('Model Performance Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics["Metric"])
        ax.legend(facecolor='#0f1116', labelcolor='white')
        
        st.pyplot(fig)
