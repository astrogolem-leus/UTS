import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/best.pt")  # Model YOLO
    # Load model TFLite
    interpreter = tf.lite.Interpreter(model_path="model/model_Laporan_2.tflite")
    interpreter.allocate_tensors()
    return yolo_model, interpreter

yolo_model, interpreter = load_models()

# Dapatkan input dan output details dari TFLite model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ==========================
# UI
# ==========================
st.title("🧠 Image Classification & Object Detection App")

menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    if menu == "Deteksi Objek (YOLO)":
        # Deteksi objek dengan YOLO
        results = yolo_model(img)
        result_img = results[0].plot()
        st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

    elif menu == "Klasifikasi Gambar":
        # ==========================
        # Preprocessing untuk TFLite
        # ==========================
        img_resized = img.resize((224, 224))  # sesuaikan ukuran input model
        img_array = np.expand_dims(np.array(img_resized, dtype=np.float32) / 255.0, axis=0)

        # Set input ke model
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()

        # Ambil hasil output
        prediction = interpreter.get_tensor(output_details[0]['index'])
        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction))

        # ==========================
        # Output ke user
        # ==========================
        st.write("### Hasil Prediksi:", class_index)
        st.write("Probabilitas:", f"{confidence:.2f}")
