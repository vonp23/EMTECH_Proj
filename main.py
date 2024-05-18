import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input

# Disable TensorFlow INFO and WARNING logs
tf.get_logger().setLevel('ERROR')

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model("mdl_wt.hdf5", custom_objects={'BatchNormalization': tf.keras.layers.BatchNormalization})

model = load_model()

map_dict = {0: 'dog', 1: 'horse', 2: 'elephant', 3: 'butterfly', 4: 'chicken', 
            5: 'cat', 6: 'cow', 7: 'sheep', 8: 'spider', 9: 'squirrel'}

# Define function to make predictions
def predict(image_data):
    resized = cv2.resize(image_data, (224, 224))
    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis, ...]
    prediction = model.predict(img_reshape).argmax()
    return map_dict[prediction]

# Main Streamlit app
def main():
    st.title("Image Classification with TensorFlow and Streamlit")

    uploaded_file = st.file_uploader("Choose an image file", type="jpg")

    if uploaded_file is not None:
        image_data = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(image_data, 1)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

        st.image(opencv_image, caption='Uploaded Image', use_column_width=True)

        if st.button('Classify'):
            prediction = predict(opencv_image)
            st.write(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()

