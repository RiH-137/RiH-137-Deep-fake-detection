import streamlit as st
import cv2
import tempfile
import os
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
import pickle   

## loading the trained model
model = tf.keras.models.load_model('xception_deepfake_image.h5')

## loading the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

## image preprocessing
def preprocess_image(image):
    image = image.resize((299, 299))  ## resize the image to match the input size expected by the model
    image = np.array(image)
    image = image.astype('float32') / 255.0  ## normalize the image to [0, 1]
    image = np.expand_dims(image, axis=0)  ## add a batch dimension
    return image

# main function to predict if an image is deep fake or real
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction[0][0]  ## returning probability


def detect_faces(image):
    img= np.array(image)
    gray= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces= face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces


def draw_bounding_boxes(image, faces, predictions):
    draw = ImageDraw.Draw(image)
    for (x, y, w, h), pred in zip(faces, predictions):
        if pred > 0.4:  
            draw.rectangle([x, y, x + w, y + h], outline="red", width=5)
    return image

## function to extract frames from video
def extract_frames(video_path, interval=30):
    frames= []
    video= cv2.VideoCapture(video_path)
    frame_count= int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    ## progress bar
    progress_bar= st.progress(0)
    
    for i in range(0, frame_count, interval):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame= video.read()
        if ret:
            frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            break
       
        progress_bar.progress(int((i / frame_count) * 100))
    
    video.release()
    progress_bar.empty()
    return frames

## chatbot response function
def chatbot_response(user_input):
    user_input = user_input.lower()
    
    if "hello" in user_input:
        return "Hi there! How can I assist you today?"
    
    elif "deep fake" or "deepfake" in user_input:
        return "Deep fakes are AI-generated media that can manipulate real video or audio to mislead viewers. I can help you detect them in images or videos."
    
    elif "ethics" in user_input:
        return ("The ethics of AI involve concerns about privacy, fairness, transparency, and accountability. "
                "For example, using deep fake technology to deceive or harm others is unethical and can have serious consequences.")
    
    elif "legal" in user_input:
        return ("The legal use of AI is a complex area. It's essential to ensure AI is used in a way that complies with existing laws, "
                "such as data protection, intellectual property rights, and avoiding harm to individuals or society.")
    
    elif "false information" in user_input or "misinformation" in user_input:
        return ("Deep fake videos can be used to spread false information, leading to significant harm. "
                "It's crucial to develop tools and regulations to combat this misuse of technology.")
    
    elif "how to upload" in user_input:
        return ("To upload an image or video, simply click on the 'Choose an image...' or 'Choose a video...' button depending on your selection. "
                "You can upload images in jpg, jpeg, or png formats, and videos in mp4, avi, or mov formats.")
    
    elif "size of video" in user_input or "size of image" in user_input:
        return ("The recommended image size is under 10MB, and videos should be under 100MB to ensure smooth processing. "
                "Larger files may take longer to process and could result in timeouts.")
    
    elif "how this project works" in user_input or "how does this work" in user_input:
        return ("This project detects deep fake images and videos using a machine learning model trained on the Xception network. "
                "For images, the uploaded image is preprocessed and passed through the model to predict whether it is real or fake. "
                "For videos, the video is split into frames, and each frame is analyzed individually. A summary of the results is provided based on the analysis of all frames.")
    
    elif "goodbye" in user_input or "bye" in user_input:
        return "Goodbye! If you have more questions in the future, feel free to ask."
    
    else:
        return "I'm not sure how to respond to that. Could you ask something else?"

## Streamlit application
st.set_page_config(page_title="Deep Fake Detector Tool", page_icon="🕵️‍♂️")
st.write("Tenet Presents")
## banner
st.markdown("""
    <style>
        .banner {
            background-color: #f8f9fa;
            padding: 10px;
            text-align: center;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .banner h1 {
            color: #343a40;
            margin: 0;
        }
    </style>
    <div class="banner">
        <h1>Deep Fake Detector Tool</h1>
    </div>
""", unsafe_allow_html=True)
if st.button('Go to Deepfake-image-distroyed'):
    st.write("[Click here to go to Deepfake-image-distroyed](https://www.google.com)")

st.sidebar.title('Chatbot Support')
st.sidebar.write("Ask me anything related to deep fake detection, AI ethics, the legal use of AI, or how this project works!")
user_input = st.sidebar.text_input("You:", key="user_input")
if user_input:
    response = chatbot_response(user_input)
    st.sidebar.text_area("Chatbot:", value=response, height=150)


st.title("Video Deep Fake Detection")

## upload video
uploaded_video= st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

if uploaded_video is not None:

    temp_video= tempfile.NamedTemporaryFile(delete=False)
    temp_video.write(uploaded_video.read())
    temp_video.close()
    
    st.write("Extracting frames from the video...")
    frames = extract_frames(temp_video.name)

    if frames:
        fake_count= 0
        real_count= 0

        st.write(f"Total frames extracted: {len(frames)}")

        for i, frame in enumerate(frames):
            st.image(frame, caption=f'Frame {i+1}', use_column_width=True)
            
            pil_frame= Image.fromarray(frame)
            faces= detect_faces(pil_frame)
            predictions= []

            for (x, y, w, h) in faces:
                face_image= pil_frame.crop((x, y, x + w, y + h))
                prediction= predict(face_image)
                predictions.append(prediction)
            
            pil_frame_with_boxes= draw_bounding_boxes(pil_frame, faces, predictions)

            st.image(pil_frame_with_boxes, caption=f'Frame {i+1} with Deep Fake Highlights', use_column_width=True)

            for j, (x, y, w, h) in enumerate(faces):
                if predictions[j] >0.4:
                    st.write(f'Frame {i+1}, Face {j+1}: **DEEP FAKE**')
                    fake_count+=1
                else:
                    st.write(f'Frame {i+1}, Face {j+1}: **Real Image**')
                    real_count+= 1
            if len(faces)== 0:
                st.write(f'Frame {i+1}: No face detected', color='orange')

        st.write(f"\nSummary:")
        st.write(f"Deep Fake Frames: {fake_count}", color='red')
        st.write(f"Real Frames: {real_count}", color='green')
        if fake_count > real_count:
            st.write("The video is likely to be a **DEEP FAKE**.", color='red')
        else:
            st.write("The video is likely to be a **Real Video**.", color='green')
    
    os.remove(temp_video.name) 
else:
    st.write("Upload a video to start the detection.")
