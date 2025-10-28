import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ---------------------------
# Fungsi Load Model
# ---------------------------
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/best.pt")  # Model YOLO
    interpreter = tf.lite.Interpreter(model_path="model/model_Laporan_2.tflite")  # Model klasifikasi
    interpreter.allocate_tensors()
    return yolo_model, interpreter

# Muat model
yolo_model, interpreter = load_models()

# Ambil detail tensor
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Label kelas sesuai urutan pelatihan
class_names = ["book", "chair", "laptop", "person", "table"]

# ==========================
# UI
# ==========================
st.title("Image Classification & Object Detection App")

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
            # --- Ambil ukuran input dari model ---
            input_shape = input_details[0]['shape']
            target_height, target_width = input_shape[1], input_shape[2]

            # --- Preprocessing gambar ---
            img_resized = img.resize((target_width, target_height))
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # --- Prediksi dengan TFLite ---
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            class_index = np.argmax(output_data)
            probability = np.max(output_data)

            class_names = ["book", "chair", "laptop", "person", "table"]

            st.write("### 🧠 Hasil Prediksi:", class_names[class_index])
            st.write(f"📊 Probabilitas: {probability:.2f}")

        except Exception as e:
            st.error("🚨 Terjadi error saat menjalankan klasifikasi gambar:")
            st.exception(e)

