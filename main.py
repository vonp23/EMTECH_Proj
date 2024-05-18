import cv2
import streamlit as st
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

model = tf.keras.models.load_model("mdl_wt.hdf5")

uploaded_file = st.file_uploader("Choose a image file", type="jpg")




# Define a function to handle WebSocket messages
async def handle_message(message):
    # Process the message here
    pass

# Define the main Streamlit app
def main():
    # Create a WebSocket route
    @st.experimental_streamlit_ws(route='/stream')
    async def streamlit_websocket(ws):
        while True:
            message = await ws.receive_text()
            await handle_message(message)

    # Run the Streamlit app
    if __name__ == '__main__':
        main()


map_dict = {0: 'dog',
            1: 'horse',
            2: 'elephant',
            3: 'butterfly',
            4: 'chicken',
            5: 'cat',
            6: 'cow',
            7: 'sheep',
            8: 'spider',
            9: 'squirrel'}

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))

    st.image(opencv_image, channels="RGB")

    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Generate Prediction")
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        st.title("Predicted Label for the image is {}".format(map_dict[prediction]))

