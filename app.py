import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.utils import img_to_array
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import requests
import tempfile

st.set_page_config(page_title="Indonesian Food Classifier", layout="wide")

# ===== CSS untuk tampilan =====
st.markdown("""
    <style>
    .block-container {
        padding: 2rem 3rem;
        font-family: 'Segoe UI', sans-serif;
    }
    .big-title {
        font-size: 2.3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #d17b0f;
    }
    .section-box {
        background-color: #f9f9f9;
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .center-image img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 300px;
        border-radius: 12px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🍽️ Food Recognition AI – Indonesian Cuisine</div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_url = "https://github.com/DedeUlfah/Food_classifier_streamlit/raw/main/model_food.h5"
    response = requests.get(model_url)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(response.content)
        tmp.flush()
        model = tf.keras.models.load_model(tmp.name, custom_objects={'KerasLayer': hub.KerasLayer})
    return model

def predict_image(model, image):
    image = image.resize((227, 227))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)
    return np.argmax(predictions)

def get_recipe(predicted_class, class_names):
    url = "https://raw.githubusercontent.com/DedeUlfah/Food_classifier_streamlit/main/data_resep.csv"
    df = pd.read_csv(url, delimiter=';')
    food_name = class_names[predicted_class].lower()
    result = df[df['food_name'].str.lower() == food_name].iloc[0]
    return result['ingredient'], result['step']

def get_nutrition(predicted_class, class_names):
    url = "https://raw.githubusercontent.com/DedeUlfah/Food_classifier_streamlit/main/total_nutrition_per_food.csv"
    df = pd.read_csv(url)
    food_name = class_names[predicted_class].lower()
    return df[df['food_name'].str.lower() == food_name].iloc[0]

def main():
    model = load_model()
    class_names = [
        'ayam goreng', 'ayam pop', 'gulai tambusu', 'kue ape', 'kue bika ambon',
        'kue cenil', 'kue dadar gulung', 'kue gethuk lidri', 'kue kastangel',
        'kue klepon', 'kue lapis', 'kue lumpur', 'kue nagasari', 'kue pastel',
        'kue putri salju', 'kue risoles', 'lemper', 'lumpia', 'putu ayu',
        'serabi solo', 'telur balado', 'telur dadar', 'wajik'
    ]

    uploaded_file = st.file_uploader("📷 Upload gambar makanan Indonesia", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.markdown('<div class="center-image">', unsafe_allow_html=True)
        st.image(image, use_column_width=False)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### 🔍 Sedang menganalisis gambar...")

        predicted_class = predict_image(model, image)
        food_name = class_names[predicted_class].title()

        ingredients, steps = get_recipe(predicted_class, class_names)
        nutrition = get_nutrition(predicted_class, class_names)

        st.markdown(f"<h2 style='text-align:center;'>🍰 Prediksi: <span style='color:#d17b0f;'>{food_name}</span></h2>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-box"><h4>🧾 Bahan-Bahan</h4>', unsafe_allow_html=True)
            st.write(ingredients)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-box"><h4>🥣 Langkah-Langkah</h4>', unsafe_allow_html=True)
            st.write(steps)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-box"><h4>🍎 Informasi Gizi</h4>', unsafe_allow_html=True)
            st.write(f"**Kalori:** {nutrition['calories']} kcal")
            st.write(f"**Karbohidrat:** {nutrition['carbohydrate']} g")
            st.write(f"**Protein:** {nutrition['proteins']} g")
            st.write(f"**Lemak:** {nutrition['fat']} g")
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()