import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image




# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="RiceCare AI",
    page_icon="assets/logoriceai.png",
    layout="centered"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

.main {
    background-color: #F5F7FA;
}

.title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: #2E8B57;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: gray;
    margin-bottom: 30px;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 50px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model(
    "saved_models/rice_leaf_disease_model.h5"
)

# =========================
# CLASS NAMES
# =========================
class_names = [
    "Bacterial leaf blight",
    "Brown spot",
    "Leaf smut"
]

# =========================
# DISEASE INFORMATION
# =========================
disease_info = {

    "Brown spot":
    "Brown Spot is a fungal disease that causes brown lesions on rice leaves.",

    "Leaf smut":
    "Leaf Smut is caused by fungal infection resulting in black smut spores.",

    "Bacterial leaf blight":
    "Bacterial Leaf Blight is a serious bacterial disease affecting rice crops."
}

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🌾 RiceCare AI")

st.sidebar.image(
    "assets/logoriceai.png",
    width=180
)


st.sidebar.info(
    """
    AI-powered agricultural disease detection platform.

    Upload rice leaf images and instantly detect diseases using Deep Learning.

    """
)

st.sidebar.markdown("### Diseases Detected")

st.sidebar.success("✔ Brown Spot")
st.sidebar.success("✔ Leaf Smut")
st.sidebar.success("✔ Bacterial Leaf Blight")

# =========================
# MAIN TITLE
# =========================
st.markdown(
    '<p class="title">RiceCare AI 🌾</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Deep Learning Based Rice Leaf Disease Detection System</p>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="
        background: linear-gradient(
            90deg,
            #e8f5e9,
            #f1f8e9
        );
        padding:20px;
        border-radius:20px;
        text-align:center;
        margin-bottom:25px;
    ">

    <h2 style="color:#2E8B57;">
        🌾 AI for Smart Agriculture
    </h2>

    <p style="font-size:18px; color:#555;">
        Detect rice leaf diseases instantly with Deep Learning technology.
    </p>

    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader(
    " Upload Rice Leaf Image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PREDICTION SECTION
# =========================
if uploaded_file is not None:

    # Open Image
    image = Image.open(uploaded_file)

    # Show Image
    st.image(
        image,
        caption="Uploaded Rice Leaf Image",
        use_container_width=True
    )

    # Resize Image
    img = image.resize((224, 224))

    # Convert to Array
    img_array = np.array(img) / 255.0

    # Expand Dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]

    confidence = float(np.max(prediction) * 100)

    # Separator
    st.markdown("---")

    # Success Message
    st.success(" Prediction Completed Successfully")

    # Prediction Card
    st.markdown(
        f"""
        <h3 style="
    color:#1b5e20;
    font-size:30px;
">
    🌿 Disease Detected:
</h3>

<h2 style="
    color:#2E8B57;
    font-weight:bold;
">
    {predicted_class}
</h2>

<h3 style="
    color:#0d47a1;
">
     Confidence Score:
    {confidence:.2f}%
</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Confidence Bar
    st.progress(int(confidence))

    # Disease Info
    st.markdown("###  Disease Information")

    st.info(disease_info[predicted_class])

# =========================
# FOOTER
# =========================
st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:gray;
        padding:20px;
    ">

    🌾 <b>RiceCare AI</b><br>

    AI-Powered Rice Disease Detection Platform<br><br>

    Developed by <b>Sayed Atif Hosen</b> 

    </div>
    """,
    unsafe_allow_html=True
)