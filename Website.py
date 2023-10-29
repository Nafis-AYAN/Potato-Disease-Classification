import streamlit as st
from PIL import Image
import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
import base64
from streamlit_lottie import st_lottie
import json



model = load_model("E:/Assignment 4.1/CSE 410/Potato_Disease_Project/potatoes.h5")


background_image = 'E:/Assignment 4.1/CSE 410/Potato_Disease_Project/web image/field.jpg'  # Replace 'path/to/your/image.jpg' with the actual path to your image

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url('data:image/jpeg;base64,{base64.b64encode(open(background_image, "rb").read()).decode()}');
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

  
lottie_hello = load_lottiefile("E:/Assignment 4.1/CSE 410/Potato_Disease_Project/web image/animation_loafcvjg.json")  # replace link to local lottie file
#lottie_hello = load_lottieurl("https://lottie.host/b866fdb1-91a1-4e00-8db8-d844b13227e2/4D1sPptfxm.json")

for layer in lottie_hello['layers']:
    if 'w' in layer:
        layer['w'] = 'Tr'

st_lottie(
    lottie_hello,
    speed=1,
    reverse=False,
    loop=True,
    quality="low",
    height=400,
    width=700,
    key=None,
)                        

#img = Image.open("E:/Assignment 4.1/CSE 410/Potato_Disease_Project/web image/Header.jpg")
#st.image(img)

CLASS_NAMES = ('Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy')

# Setting Title of App
st.title("Potato Disease Detection")
st.markdown("Upload an image of the potato leaf")

# Uploading the dog image
plant_image = st.file_uploader("Choose an image...", type = "jpg")
submit = st.button('predict Disease')

# On predict button click
if submit:
    if plant_image is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray (bytearray(plant_image.read()), dtype = np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        
        # Displaying the image
        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)
        
        # Resizing the image
        opencv_image = cv2.resize(opencv_image, (256, 256))
        
        # Convert image to 4 Dimension
        opencv_image.shape = (1, 256, 256, 3)
        
        #Make Prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]
        
        st.subheader(str("This is "+result.split('__')[0]+ " leaf with " +  result.split('__')[1]))


        if result=="Potato___Early_blight":
            
            st.subheader("Treatment:")
            st.markdown("Early blight in potatoes can be treated by applying fungicides containing copper or chlorothalonil and practicing crop rotation to reduce disease pressure.")
            st.subheader("Prevention:")
            st.markdown("Prevent early blight in potatoes by practicing crop rotation, ensuring proper plant spacing, and applying fungicides when necessary.")
            
        if result=="Potato___Late_blight":
           
            st.subheader("Treatment:")
            st.markdown("Potato late blight can be treated with fungicides containing active ingredients such as copper-based compounds or systemic fungicides like azoxystrobin applied preventatively on a regular schedule, combined with good cultural practices and crop rotation to manage the disease.")
            st.subheader("Prevention:")
            st.markdown("Implementing crop rotation, applying fungicides, and ensuring proper drainage can help prevent potato late blight.")
            
        
        
                            
                            
                                            