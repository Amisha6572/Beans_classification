# import streamlit as st
# import numpy as np
# import pickle
# from PIL import Image

# # ===============================
# # PAGE CONFIG
# # ===============================
# st.set_page_config(
#     page_title="Dry Bean Classifier",
#     page_icon="🌱",
#     layout="wide"
# )

# # ===============================
# # LOAD MODEL
# # ===============================
# model = pickle.load(open("model.pkl", "rb"))
# scaler = pickle.load(open("scaler.pkl", "rb"))

# # ===============================
# # CUSTOM CSS (COLOR THEME)
# # ===============================
# st.markdown("""
#     <style>
#     .main {
#         background-color: #f5f7fa;
#     }
#     .title {
#         text-align: center;
#         color: #2c7be5;
#         font-size: 40px;
#         font-weight: bold;
#     }
#     .subtitle {
#         text-align: center;
#         font-size: 18px;
#         color: #555;
#     }
#     .stButton>button {
#         background-color: #2c7be5;
#         color: white;
#         font-size: 18px;
#         border-radius: 10px;
#         height: 3em;
#         width: 100%;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # ===============================
# # HEADER
# # ===============================
# st.markdown('<div class="title">🌱 Dry Bean Classification App</div>', unsafe_allow_html=True)
# st.markdown('<div class="subtitle">AI-powered bean type prediction system</div>', unsafe_allow_html=True)

# # ===============================
# # IMAGE (Add your image file)
# # ===============================
# try:
#     image = Image.open("beans.jpg")  # put image in same folder
#     st.image(image, use_container_width=True)
# except:
#     st.warning("Add 'beans.jpg' image to project folder for better UI")

# # ===============================
# # INPUT SECTION
# # ===============================
# st.markdown("### 📥 Enter Bean Features")

# col1, col2 = st.columns(2)

# with col1:
#     area = st.number_input("Area")
#     perimeter = st.number_input("Perimeter")
#     majoraxislength = st.number_input("Major Axis Length")
#     minoraxislength = st.number_input("Minor Axis Length")
#     aspectration = st.number_input("Aspect Ratio")
#     eccentricity = st.number_input("Eccentricity")
#     convexarea = st.number_input("Convex Area")
#     equivdiameter = st.number_input("Equivalent Diameter")

# with col2:
#     extent = st.number_input("Extent")
#     solidity = st.number_input("Solidity")
#     roundness = st.number_input("Roundness")
#     compactness = st.number_input("Compactness")
#     shapefactor1 = st.number_input("Shape Factor 1")
#     shapefactor2 = st.number_input("Shape Factor 2")
#     shapefactor3 = st.number_input("Shape Factor 3")
#     shapefactor4 = st.number_input("Shape Factor 4")

# # ===============================
# # PREDICTION
# # ===============================
# if st.button("🔍 Predict Bean Type"):

#     input_data = np.array([[area, perimeter, majoraxislength, minoraxislength,
#                             aspectration, eccentricity, convexarea, equivdiameter,
#                             extent, solidity, roundness, compactness,
#                             shapefactor1, shapefactor2, shapefactor3, shapefactor4]])

#     input_scaled = scaler.transform(input_data)
#     prediction = model.predict(input_scaled)

#     st.success(f"🌟 Predicted Bean Type: {prediction[0]}")








import streamlit as st
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from PIL import Image
import base64

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Dry Bean Classifier",
    page_icon="🌱",
    layout="wide"
)

# ===============================
# BACKGROUND IMAGE
# ===============================
def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
    """, unsafe_allow_html=True)

set_bg("beans.jpg")

# ===============================
# CUSTOM CSS (GLASS EFFECT)
# ===============================
st.markdown("""
<style>
.block-container {
    background: rgba(255, 255, 255, 0.85);
    padding: 2rem;
    border-radius: 15px;
}

h1, h2, h3 {
    color: #2c3e50;
    text-align: center;
}

.stButton>button {
    background-color: #27ae60;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL
# ===============================
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ===============================
# HEADER
# ===============================
st.title("🌱 Dry Bean Classification System")
st.markdown("### AI-powered classification with explainability")

# ===============================
# LAYOUT
# ===============================
col1, col2 = st.columns([1, 1])

# ===============================
# INPUT PANEL
# ===============================
with col1:
    st.subheader("📥 Input Bean Features")

    area = st.number_input("Area")
    perimeter = st.number_input("Perimeter")
    majoraxislength = st.number_input("Major Axis Length")
    minoraxislength = st.number_input("Minor Axis Length")
    aspectration = st.number_input("Aspect Ratio")
    eccentricity = st.number_input("Eccentricity")
    convexarea = st.number_input("Convex Area")
    equivdiameter = st.number_input("Equivalent Diameter")
    extent = st.number_input("Extent")
    solidity = st.number_input("Solidity")
    roundness = st.number_input("Roundness")
    compactness = st.number_input("Compactness")
    shapefactor1 = st.number_input("Shape Factor 1")
    shapefactor2 = st.number_input("Shape Factor 2")
    shapefactor3 = st.number_input("Shape Factor 3")
    shapefactor4 = st.number_input("Shape Factor 4")

    predict_btn = st.button("🔍 Predict Bean Type")

# ===============================
# RESULT PANEL
# ===============================
with col2:
    st.subheader("📊 Prediction & Explanation")

    if predict_btn:

        input_data = np.array([[area, perimeter, majoraxislength, minoraxislength,
                                aspectration, eccentricity, convexarea, equivdiameter,
                                extent, solidity, roundness, compactness,
                                shapefactor1, shapefactor2, shapefactor3, shapefactor4]])

        # Scale
        input_scaled = scaler.transform(input_data)

        # Prediction
        prediction = model.predict(input_scaled)

        st.success(f"🌟 Predicted Bean Type: **{prediction[0]}**")

        # ===============================
        # SHAP EXPLANATION
        # ===============================
    st.markdown("### 🧠 Model Explanation (SHAP)")

try:
    import shap

    explainer = shap.Explainer(model)
    shap_values = explainer(input_scaled)

    # Get predicted class index
    pred_class = model.predict(input_scaled)[0]
    class_index = list(model.classes_).index(pred_class)

    fig, ax = plt.subplots()

    shap.plots.waterfall(
        shap_values[0, :, class_index],   # ✅ correct indexing
        max_display=10,
        show=False
    )

    st.pyplot(fig)

except Exception as e:
    st.error(f"SHAP error: {e}")
# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("ℹ️ About Project")
st.sidebar.info("""
This application classifies dry beans into different categories using Machine Learning.

Models used:
- Random Forest
- SVM
- KNN

Features include:
- Shape
- Area
- Compactness

Includes SHAP for explainability.
""")