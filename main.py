import streamlit as st
from groq import Groq
import cv2
import mediapipe as mp
import numpy as np
import base64
import os
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

# Models
llama3_model = "llama3-70b-8192"  

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Function to detect faces and draw bounding boxes
def detect_faces(image):
    """Detects faces using MediaPipe and returns annotated image + face count."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detector.process(image_rgb)
    annotated_image = image.copy()
    face_count = 0

    if results.detections:
        for detection in results.detections:
            # Draw bounding box
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            x, y = int(bbox.xmin * w), int(bbox.ymin * h)
            width, height = int(bbox.width * w), int(bbox.height * h)
            cv2.rectangle(annotated_image, (x, y), (x + width, y + height), (0, 255, 0), 2)
            face_count += 1

    return annotated_image, face_count

# Function to encode image to base64
def encode_image(image):
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")

# Function to describe facial expressions using Groq
def describe_expression(client, base64_image, face_count):
    prompt = f"""
    Analyze this image with {face_count} detected face(s). Describe:
    1. The person's likely facial expression (happy, sad, angry, etc.)
    2. Key visual cues (smile, frown, raised eyebrows)
    3. A creative interpretation of their mood
    """
    
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    }
                ]
            }
        ],
        model=llama3_model,
        temperature=0.7
    )
    return response.choices[0].message.content

# Streamlit UI
st.title("Facial Expression Analyzer")
st.write("Upload an image to detect faces and analyze expressions using Groq's LLM")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read and convert image
    image = np.array(Image.open(uploaded_file))
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Detect faces
    annotated_image, face_count = detect_faces(image_bgr)
    st.image(annotated_image, caption=f"Detected {face_count} face(s)", use_container_width=True)
    
    if face_count > 0:
        # Encode image and get description
        base64_img = encode_image(annotated_image)
        with st.spinner("Analyzing expressions..."):
            description = describe_expression(client, base64_img, face_count)
        
        st.subheader("Expression Analysis")
        st.write(description)
    else:
        st.warning("No faces detected!")
