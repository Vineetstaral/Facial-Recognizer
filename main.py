import streamlit as st
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace
from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Emotion configuration
EMOTION_CONFIG = {
    "happy": {"color": (0, 255, 0)},
    "sad": {"color": (255, 0, 0)},
    "angry": {"color": (0, 0, 255)},
    "neutral": {"color": (0,0,0)},
    "surprise": {"color": (255, 255, 0)},
    "fear": {"color": (255, 0, 255)},
    "disgust": {"color": (0, 255, 255)}
}

# Predefined minimum confidence threshold
MIN_CONFIDENCE = 50

def get_reliable_emotion(analysis):
    """Handle ambiguous emotion predictions"""
    emotions = analysis[0]['emotion']
    dominant = max(emotions, key=emotions.get)
    confidence = emotions[dominant]
    
    if confidence < 45:  # Fallback to neutral if unclear
        return "neutral", confidence
    return dominant, confidence

def get_llm_insight(emotion, confidence):
    """Get concise scientific explanation"""
    prompt = f"Provide a 1-sentence scientific explanation for a '{emotion}' facial expression"
    
    response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192",
        temperature=0.3
    )
    return response.choices[0].message.content

def analyze_faces(image):
    """Face analysis with predefined confidence threshold"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    ).detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    
    results = []
    for i, (x, y, w, h) in enumerate(faces, 1):
        face_img = image[y:y+h, x:x+w]
        
        try:
            analysis = DeepFace.analyze(face_img, actions=['emotion'], silent=True)
            emotion, confidence = get_reliable_emotion(analysis)
            
            insight = get_llm_insight(emotion, confidence) if confidence > MIN_CONFIDENCE else "Low confidence detection"
            
            results.append({
                'id': i,
                'box': (x, y, w, h),
                'emotion': emotion,
                'confidence': confidence,
                'insight': insight,
                'color': EMOTION_CONFIG.get(emotion, {}).get("color", (0, 255, 0))
            })
        except:
            continue
    
    return image, results

# Streamlit UI
st.title("AI Emotion Detector")
uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = np.array(Image.open(uploaded_file))
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    with st.spinner("Analyzing facial expressions..."):
        processed_img, faces = analyze_faces(image_bgr)
    
    # Display image with annotations
    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(image, (x, y), (x+w, y+h), face['color'], 3)
        label = f"{face['id']}: {face['emotion'].upper()} ({face['confidence']:.0f}%)"
        cv2.putText(image, label, (x+10, y-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, face['color'], 2)
    
    st.image(image, caption="Detected Expressions", use_column_width=True)
    
    # Analysis report below image
    if faces:
        st.subheader("Analysis Report")
        for face in faces:
            st.markdown(f"""
            **Face {face['id']}**  
            **Emotion**: {face['emotion'].upper()}  
            **Confidence**: {face['confidence']:.0f}%  
            **Insight**: {face['insight']}  
            """)
            st.divider()
    else:
        st.warning("No faces detected with sufficient confidence")
