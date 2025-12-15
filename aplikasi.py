import streamlit as st
import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN
import pandas as pd


st.set_page_config(page_title="Facial Expression Recognition", layout="centered")
st.title("Facial Expression Recognition")
st.caption("ResNeXt-50 trained on FERPlus dataset")

device = torch.device("cpu")

emotion_labels = [
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt"
]


@st.cache_resource
def load_model():
    model = timm.create_model("resnext50_32x4d", pretrained=False)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 8),
        nn.LogSoftmax(dim=1)
    )
    checkpoint = torch.load("resnext_epoch26.pth", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

model = load_model()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

mtcnn = MTCNN(keep_all=False, device=device)


st.subheader("Input")
input_type = st.radio("Pilih input:", ["Upload Gambar", "Ambil Foto (Webcam)"])

image = None
if input_type == "Upload Gambar":
    uploaded_file = st.file_uploader("Upload gambar wajah", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
else:
    camera_img = st.camera_input("Ambil foto")
    if camera_img:
        image = Image.open(camera_img).convert("RGB")


if image:
    st.image(image, caption="Input Image", width=300)

    boxes, _ = mtcnn.detect(image)

    if boxes is None:
        st.error("Wajah tidak terdeteksi")
    else:
        box = boxes[0]
        x1, y1, x2, y2 = map(int, box)

        face = image.crop((x1, y1, x2, y2))

        input_tensor = transform(face).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.exp(output).squeeze(0).numpy()

        pred_idx = np.argmax(probs)
        pred_label = emotion_labels[pred_idx]

        # Draw bounding box
        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)

        st.subheader("Hasil Deteksi")
        st.image(img_draw, width=300)
        st.success(f"Prediksi Emosi: **{pred_label}** ({probs[pred_idx]*100:.2f}%)")

        # Table
        df = pd.DataFrame({
            "Emotion": emotion_labels,
            "Probability (%)": (probs * 100).round(2)
        }).sort_values("Probability (%)", ascending=False)

        st.dataframe(df, use_container_width=True)

        # Chart
        st.bar_chart(df.set_index("Emotion"))

