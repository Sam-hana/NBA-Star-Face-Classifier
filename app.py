import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from mtcnn import MTCNN
import cv2

# Simple web page
st.set_page_config(page_title="NBA Star Classifier", page_icon="🏀")
st.title("🏀 NBA Star AI Classifier")
st.write("V3 Face-Only Edition: We detect the face first, then classify!")

#  Load the model and MTCNN
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('nba_star_classifier.keras')

@st.cache_resource
def load_face_detector():
    return MTCNN()

model = load_model()
detector = load_face_detector()
class_names = ['LeBron_James', 'Luka_Doncic', 'Shai_Gilgeous_Alexander', 'Unknown']

# Upload photo box
uploaded_file = st.file_uploader("Upload an image of a player...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Let the photo be RGB format
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Original Uploaded Image', use_container_width=True)
    
    st.write("🔍 MTCNN is looking for a face...")
    
    # Detect and crop out the face
    img_array_cv = np.array(image)
    faces = detector.detect_faces(img_array_cv)
    
    # It no face is detected
    if not faces:
        st.error("❌ No face detected! Please upload a clearer photo with a visible face.")
    else:
        # Catch the coordinate of face and add the padding
        x, y, width, height = faces[0]['box']
        pad_w = int(width * 0.3)
        pad_h = int(height * 0.3)
        
        y1 = max(0, y - pad_h)
        y2 = min(img_array_cv.shape[0], y + height + pad_h)
        x1 = max(0, x - pad_w)
        x2 = min(img_array_cv.shape[1], x + width + pad_w)
        cropped_face = img_array_cv[y1:y2, x1:x2]
        
        # Show the cropped version of photo
        st.image(cropped_face, caption='Cropped Face (Fed to AI)', width=250)
        st.write("We are analyzing the face features...")
        
        # change to 224x224 and let it do prediction
        face_img = Image.fromarray(cropped_face).resize((224, 224))
        final_array = tf.keras.preprocessing.image.img_to_array(face_img)
        final_array = np.expand_dims(final_array, axis=0)
        predictions = model.predict(final_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = class_names[predicted_class_index]
        confidence = 100 * np.max(predictions[0])

        # Set the confidence threshold, if lower than this, predict as unknown
        THRESHOLD = 70.0
        st.divider()
        if predicted_class_name == 'Unknown' or confidence < THRESHOLD:
            st.warning(f"⚠️ Prediction: **Unknown** (Not in our target list)")
            if predicted_class_name != 'Unknown':
                st.write(f"*(AI thought it might be {predicted_class_name} with {confidence:.2f}%, but it's too low to be sure.)*")
        else:
            st.success(f"🎉 Prediction: **{predicted_class_name}**")
            st.info(f"📊 Confidence Level: {confidence:.2f}%")