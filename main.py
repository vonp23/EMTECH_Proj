import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input

# Load the pre-trained model
model = tf.keras.models.load_model("mdl_wt.hdf5")

# Mapping of class indices to labels
map_dict = {
    0: 'dog',
    1: 'horse',
    2: 'elephant',
    3: 'butterfly',
    4: 'chicken',
    5: 'cat',
    6: 'cow',
    7: 'sheep',
    8: 'spider',
    9: 'squirrel'
}

def main():
    st.title("Image Classification with Streamlit")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg"])

    if uploaded_file is not None:
        # Read and preprocess the uploaded image
        opencv_image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(opencv_image, 1)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(opencv_image, (224, 224))
        resized_image = mobilenet_v2_preprocess_input(resized_image)
        img_reshape = resized_image[np.newaxis, ...]

        # Display the uploaded image
        st.image(opencv_image, caption='Uploaded Image', use_column_width=True)

        # Button to generate prediction
        if st.button("Generate Prediction"):
            # Make prediction
            prediction = model.predict(img_reshape).argmax()
            predicted_label = map_dict.get(prediction, "Unknown")
            st.success(f"Predicted Label: {predicted_label}")

if __name__ == "__main__":
    main()
