import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="Butterfly Identifier", layout="centered")

st.title("🦋 Butterfly Species Identifier")

model_path = 'butterfly_final_eff.keras'

if not os.path.exists(model_path):
    st.error(f"Model file not found! Make sure '{model_path}' is in the same folder as this script.")
else:
    st.info("Loading AI Model... please wait.")
    
@st.cache_resource
def load_my_model():
    try:
        # The fix: compile=False prevents Keras from trying to reconstruct 
        # the training gradients, which is where the BatchNormalization error lives.
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

    model = load_my_model()

    if model:
        st.success("Model loaded successfully!")
        
        # 3. Species List
        class_names = [
            'ADONIS', 'AFRICAN GIANT SWALLOWTAIL', 'AMERICAN SNOOT', 'AN 88', 'APPOLLO', 'ATALA', 
            'BANDED ORANGE HELICONIAN', 'BANDED PEACOCK', 'BECKERS WHITE', 'BLACK HAIRSTREAK', 
            'BLUE MORPHO', 'BLUE SPOTTED CROW', 'BROWN SIPROETA', 'CABBAGE WHITE', 'CAIRNS BIRDWING', 
            'CHECQUERED SKIPPER', 'CHESTNUT', 'CLEOPATRA', 'CLODIUS PARNASSIAN', 'CLOUDED SULPHUR', 
            'COMMON BANDED AWL', 'COMMON WOOD-NYMPH', 'COPPER TAIL', 'CRECENT', 'CRIMSON PATCH', 
            'DANAID EGGFLY', 'EASTERN COMA', 'EASTERN DAPPLE WHITE', 'EASTERN PINE ELFIN', 
            'ELBOWED PIERROT', 'GOLD BANDED', 'GREAT EGGFLY', 'GREAT JAY', 'GREEN CELLED CATTLEHEART', 
            'GREY HAIRSTREAK', 'INDRA SWALLOW', 'IPHICLUS SISTER', 'JULIA', 'LARGE MARBLE', 
            'MALACHITE', 'MANGROVE SKIPPER', 'MESTRA', 'METALMARK', 'MILBERTS TORTOISESHELL', 
            'MONARCH', 'MOURNING CLOAK', 'ORANGE OAKLEAF', 'ORANGE TIP', 'ORCHARD SWALLOW', 
            'PAINTED LADY', 'PAPER KITE', 'PEACOCK', 'PINE WHITE', 'PIPEVINE SWALLOW', 'POPINJAY', 
            'PURPLE HAIRSTREAK', 'PURPLISH COPPER', 'QUESTION MARK', 'RED ADMIRAL', 'RED CRACKER', 
            'RED POSTMAN', 'RED SPOTTED PURPLE', 'SCARCE SWALLOW', 'SILVER SPOT SKIPPER', 
            'SLEEPY ORANGE', 'SOOTYWING', 'SOUTHERN DOGFACE', 'STRAITED QUEEN', 'TROPICAL LEAFWING', 
            'TWO BARRED FLASHER', 'ULYSES', 'VICEROY', 'WOOD SATYR', 'YELLOW SWALLOW TAIL', 
            'ZEBRA LONG WING'
        ]

        uploaded_file = st.file_uploader("Upload a butterfly photo", type=["jpg", "png", "jpeg"])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption='Target Image', use_column_width=True)
            
            # Preprocessing
            with st.spinner('Analyzing...'):
                img = image.resize((224, 224))
                img_array = np.array(img)
                
                img_array = img_array / 255.0 
                img_array = np.expand_dims(img_array, axis=0)
                
                # Predict
                predictions = model.predict(img_array)
                # For EfficientNet usually we use Softmax if not in model
                score = tf.nn.softmax(predictions[0])
                
                label = class_names[np.argmax(score)]
                conf = 100 * np.max(score)
                
                st.subheader(f"Result: {label}")

                st.write(f"Confidence: {conf:.2f}%")
