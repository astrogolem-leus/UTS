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

# Label kelas sesuai urutan pelatihan
class_names = ["book", "chair", "laptop", "person", "table"]

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
        try:
            results = yolo_model(img)
            result_img = results[0].plot()
            st.image(result_img, caption="Hasil Deteksi", use_container_width=True)
        except Exception:
            st.warning("⚠️ Gambar tidak bisa dideteksi oleh model YOLO. Silakan coba gambar lain.")

    elif menu == "Klasifikasi Gambar":
        try:
            # ---------------------------
            # Preprocessing
            # ---------------------------
            img_resized = img.resize((224, 224))
            img_array = np.array(img_resized, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # ---------------------------
            # Prediksi
            # ---------------------------
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])[0]

            class_index = np.argmax(output_data)
            confidence = float(np.max(output_data))

            st.markdown(f"### 🧠 Hasil Prediksi: **{class_names[class_index]}**")
            st.write(f"Probabilitas: {confidence:.2f}")

        except Exception:
            st.warning("⚠️ Gambar tidak bisa dideteksi oleh model klasifikasi. Silakan coba gambar lain.")
