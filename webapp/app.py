# app.py
import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import os
import matplotlib.pyplot as plt
from fpdf import FPDF
from collections import deque

# ============ Page config ============
st.set_page_config(page_title="Chest X-Ray Classifier", page_icon="🫁", layout="wide")
st.title("🫁 Chest X-Ray Classification")

# ============ Classes & Device ============
class_names = ['COVID-19', 'Normal', 'Pneumonia-Bacterial', 'Pneumonia-Viral']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============ Theme Toggle ============
theme = st.sidebar.radio("🌗 Theme", ["Dark", "Light"])
if theme == "Dark":
    bg_color = "#2c5364"
    text_color = "#f0f0f0"
    card_color = "rgba(255, 255, 255, 0.05)"
else:
    bg_color = "#f0f0f0"
    text_color = "#111111"
    card_color = "rgba(0, 0, 0, 0.05)"

st.markdown(f"""
    <style>
    .stApp {{
        background: {bg_color};
        color: {text_color};
    }}
    .css-18e3th9, .st-emotion-cache-18e3th9 {{
        background: {card_color};
        border-radius: 12px;
        padding: 1rem;
        color: {text_color};
    }}
    h1, h2, h3 {{
        color: {text_color} !important;
        font-weight: bold;
    }}
    .big-result {{
        font-size: 28px !important;
        font-weight: 800 !important;
        padding: 10px;
        border-radius: 8px;
    }}
    </style>
""", unsafe_allow_html=True)

# ============ Sidebar Model Info ============
st.sidebar.header("📌 Model Info")
st.sidebar.write(f"**Device:** {device}")
st.sidebar.write("### Classes:")
for c in class_names:
    st.sidebar.write(f"- {c}")
st.sidebar.write("Instructions: Upload a chest X-ray image and get prediction.")

# ============ Preprocessing ============
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

# ============ Load Model ============
@st.cache_resource
def load_model():
    import gdown

    model_path = "VitFinal30_model.pth"
    google_drive_url = "https://drive.google.com/uc?id=1N53m_EZAiDNuQMu0-8AFFacqvbhVD9wP"

    # Download the model if it doesn't exist
    if not os.path.exists(model_path):
        st.warning("Model not found locally. Downloading from Google Drive...")
        gdown.download(google_drive_url, model_path, quiet=False)

    # Load the model
    weights = models.ViT_B_16_Weights.DEFAULT
    model = models.vit_b_16(weights=weights)
    in_features = model.heads.head.in_features
    model.heads = torch.nn.Linear(in_features, len(class_names))
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    st.success("✅ Model loaded successfully!")
    return model


# ============ Prediction ============
def predict(image, model):
    transform = get_transforms()
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        idx = torch.argmax(probs, dim=1).item()
        return {
            "prediction": class_names[idx],
            "confidence": float(probs[0][idx]),
            "probabilities": {class_names[i]: float(probs[0][i]) for i in range(len(class_names))}
        }

# ============ PDF Generation ============
def generate_pdf(image: Image.Image, result: dict, filename="prediction_result.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 12, "Chest X-Ray Classification Result", ln=True, align="C")
    pdf.ln(10)
    temp_path = "temp_image.png"
    image.save(temp_path)
    pdf.image(temp_path, x=50, w=100)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 12, f"Diagnosis: {result['prediction']}", ln=True)
    pdf.cell(0, 12, f"Confidence: {result['confidence']*100:.2f}%", ln=True)
    pdf.ln(8)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, "Probabilities:", ln=True)
    for cls, val in result['probabilities'].items():
        pdf.cell(0, 8, f"- {cls}: {val*100:.2f}%", ln=True)
    # Doctor recommendation
    note = ""
    if result['prediction'] == "COVID-19":
        note = "Immediate medical attention recommended."
    elif result['prediction'].startswith("Pneumonia"):
        note = "Consult a physician for treatment guidance."
    elif result['prediction'] == "Normal":
        note = "No issues detected. Maintain regular checkups."
    if note:
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"Recommendation: {note}", ln=True)
    pdf.output(filename, "F")
    return filename

# ============ Recent Uploads ============
if 'recent_uploads' not in st.session_state:
    st.session_state.recent_uploads = deque(maxlen=3)

# ============ Upload & Analyze ============
st.header("📤 Upload X-ray Image")
st.warning("⚠️ Please upload only chest X-ray images. Non-medical images will produce incorrect results.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    # Mobile-friendly width
    width_img = 400 if st.config.get_option("server.enableCORS") else 300
    st.image(image, caption="Uploaded X-ray Image", width=width_img)

    if st.button("🔍 Analyze Image"):
        with st.spinner("Processing... Please wait..."):
            model = load_model()
            if model:
                result = predict(image, model)
                st.session_state.result = result
                st.session_state.img = image
                st.session_state.recent_uploads.appendleft(image)

# ============ Display Results ============
if 'result' in st.session_state:
    res = st.session_state.result
    pred = res["prediction"]
    conf = res["confidence"] * 100
    probs = res["probabilities"]
    color = "green" if pred == "Normal" else ("red" if pred == "COVID-19" else "orange")

    # Big diagnosis
    st.markdown(
        f"<div class='big-result' style='color:{color};'>🩺 Diagnosis: {pred}</div>",
        unsafe_allow_html=True
    )
    st.markdown(f"### 🔎 Confidence: **{conf:.2f}%**")

    # Confidence progress bar
    st.subheader("Prediction Confidence")
    st.progress(res['confidence'])
    st.write(f"{res['confidence']*100:.2f}% confident")

    # Doctor recommendation
    note = ""
    if pred == "COVID-19":
        note = "⚠️ Immediate medical attention recommended."
    elif pred.startswith("Pneumonia"):
        note = "⚠️ Consult a physician for treatment guidance."
    elif pred == "Normal":
        note = "✅ No issues detected. Maintain regular checkups."
    st.info(note)

    # ================= Probability Distribution Chart =================
    st.subheader("Probability Distribution")
    width = max(6, min(12, len(probs)*2))
    fig, ax = plt.subplots(figsize=(width, 5), dpi=120, facecolor=bg_color)
    ax.set_facecolor(bg_color)
    bars = ax.bar(probs.keys(), [v*100 for v in probs.values()],
                  color=['#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'])
    ax.set_ylabel("Probability (%)", color=text_color, fontsize=12)
    ax.set_ylim(0, 100)
    ax.tick_params(axis='x', colors=text_color, labelsize=12)
    ax.tick_params(axis='y', colors=text_color, labelsize=12)
    plt.xticks(rotation=25, ha="center")
    for i, (cls, val) in enumerate(probs.items()):
        height = val * 100
        ax.text(i, height + 2, f"{height:.1f}%", ha="center", fontsize=12, color=text_color)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True, clear_figure=True)

    # Recent uploads history
    if st.session_state.recent_uploads:
        st.subheader("Recent Uploads")
        cols = st.columns(len(st.session_state.recent_uploads))
        for col, img in zip(cols, st.session_state.recent_uploads):
            col.image(img, width=100)

    # PDF download
    pdf_file = generate_pdf(st.session_state.img, res)
    with open(pdf_file, "rb") as f:
        st.download_button("📄 Download Result as PDF", f, file_name="prediction_result.pdf", mime="application/pdf")

else:
    st.info("Upload and analyze an image to see results.")
