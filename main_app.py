#import libraries
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model


#loading the model
model = load_model('dog_breed.h5')

# selecting the first 3 breeds
names = [ "scottish_deerhound", "maltese_dog", "afghan_hound"]

# setiing the title of app
st.title("Dog Breed Prediction")
st.markdown("Upload an image of the dog")

#Uploading the dog image
dog_image = st.file_uploader("Choose an image...", type="png")
submit = st.button("Predict")

if submit:
    if dog_image is not None:
        #convert the file to an open cv image
        file_bytes = np.asanyarray(bytearray(dog_image.read()),dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        
        
        
        #display the image
        st.image(opencv_image, channels="BGR")
        
        #resize the image
        opencv_image = cv2.resize(opencv_image, (224,224))
        #convert image to 4 dimension
        opencv_image.shape = (1,224,224,3)
        #make prediction
        Y_pred = model.predict(opencv_image)
        
        st.title(str("The Dog Breed is " + names[np.argmax(Y_pred)]))