import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnext101_32x8d, ResNeXt101_32X8D_Weights
from facenet_pytorch import MTCNN
import tempfile
import os

# --- 1. Model Architecture (Must match exactly what was trained in Colab) ---
class DeepShield(nn.Module):
    def __init__(self):
        super(DeepShield, self).__init__()
        # Spatial extractor
        base_cnn = resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(base_cnn.children())[:-1])
        
        # Temporal model
        self.lstm = nn.LSTM(input_size=2048, hidden_size=512, num_layers=2, 
                            bidirectional=True, batch_first=True, dropout=0.3)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256), 
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(256, 1), 
            nn.Sigmoid()
        )

    def forward(self, video_frames):
        batch_size, seq_len, c, h, w = video_frames.size()
        c_in = video_frames.view(batch_size * seq_len, c, h, w)
        cnn_out = self.cnn(c_in).squeeze() 
        seq = cnn_out.view(batch_size, seq_len, -1) 
        lstm_out, _ = self.lstm(seq)
        pred = self.classifier(lstm_out[:, -1, :])
        return pred

# --- 2. Preprocessing Pipeline ---
# Initialize MTCNN once globally
detector = MTCNN()

def preprocess_video_for_inference(video_path, sequence_length=8):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened() and len(frames) < sequence_length:
        ret, frame = cap.read()
        if not ret: break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb_frame)
        
        if results:
            x, y, w, h = results[0]['box']
            x, y = max(0, x), max(0, y)
            
            try:
                face_crop = frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face_crop, (224, 224))
                face_normalized = face_resized / 255.0 
                frames.append(face_normalized)
            except Exception as e:
                continue 
                
    cap.release()
    
    if len(frames) == 0:
        return None # Return None if no faces were found
        
    # We need to add a batch dimension for inference: shape becomes (1, T, C, H, W)
    video_tensor = np.transpose(np.array(frames), (0, 3, 1, 2))
    return torch.tensor(video_tensor, dtype=torch.float32).unsqueeze(0)

# --- 3. Streamlit Interface ---
st.set_page_config(page_title="DeepShield Detector", page_icon="🛡️", layout="centered")

st.title("🛡️ DeepShield: Deepfake Detector")
st.write("Upload a video to analyze its spatial and temporal aura.")

# Initialize the model and load weights (cached so it doesn't reload on every button click)
@st.cache_resource
def load_model():
    model = DeepShield()
    # Check if CPU or GPU is available for inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    try:
        # map_location ensures it loads correctly even if trained on GPU but running on CPU
        model.load_state_dict(torch.load('deepshield_weights.pth', map_location=device))
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error("Model weights file ('deepshield_weights.pth') not found! Please ensure it's in the same folder as app.py.")
        return None, None

model, device = load_model()

if model:
    # 1. Create a file uploader
    uploaded_video = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        # 2. Display the uploaded video
        st.video(uploaded_video)
        
        # 3. Analyze button
        if st.button("Analyze Video"):
            with st.spinner("Analyzing frames and temporal movement..."):
                
                # Streamlit uploaded files are in memory. We need to save it to a temporary file 
                # so cv2.VideoCapture can read it from the hard drive.
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_video.read())
                
                # Preprocess the video
                video_tensor = preprocess_video_for_inference(tfile.name)
                
                if video_tensor is None:
                    st.warning("No face detected in the video sequence or the video is too short.")
                else:
                    # Move data to correct device
                    video_tensor = video_tensor.to(device)
                    
                    # Run inference
                    with torch.no_grad():
                        prediction = model(video_tensor).item()
                    
                    # Format the result
                    # Remember our labels: 0.0 was Real, 1.0 was Fake
                    if prediction > 0.5:
                        confidence = prediction * 100
                        st.error(f"🚨 FAKE VIDEO DETECTED")
                        st.metric(label="Manipulated Probability", value=f"{confidence:.2f}%")
                        st.write("The DeepShield model has detected significant spatio-temporal anomalies indicative of generative manipulation.")
                    else:
                        confidence = (1 - prediction) * 100
                        st.success(f"✅ AUTHENTIC VIDEO")
                        st.metric(label="Authenticity Confidence", value=f"{confidence:.2f}%")
                        st.write("The spatial features and temporal motion coherence are consistent with natural, unaltered footage.")
                
                # Clean up the temporary file
                tfile.close()
                os.unlink(tfile.name)
