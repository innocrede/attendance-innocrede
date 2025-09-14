import streamlit as st
import face_recognition
import numpy as np
import pickle
import pandas as pd
from datetime import datetime

# Load or create face encodings
try:
    with open('encodings.pickle', 'rb') as f:
        known_faces = pickle.load(f)
except:
    known_faces = {}

# Load or create attendance CSV
attendance_file = 'attendance.csv'
try:
    df = pd.read_csv(attendance_file)
except:
    df = pd.DataFrame(columns=['Name','Date','Time'])

st.title("Face Recognition Attendance (Browser Webcam)")

menu = st.sidebar.selectbox("Menu", ["Register Face", "Mark Attendance", "Download CSV"])

if menu == "Register Face":
    st.subheader("Register Face")
    name = st.text_input("Enter Name")
    img_file = st.camera_input("Take a picture")
    
    if img_file and name:
        # Convert image to numpy array
        img_array = np.array(bytearray(img_file.read()), dtype=np.uint8)
        image = face_recognition.load_image_file(img_file)
        
        face_locations = face_recognition.face_locations(image)
        if len(face_locations) != 1:
            st.warning("Ensure only one face is visible")
        else:
            face_encoding = face_recognition.face_encodings(image, face_locations)[0]
            known_faces[name] = face_encoding
            with open('encodings.pickle','wb') as f:
                pickle.dump(known_faces,f)
            st.success(f"Face of {name} registered successfully!")

elif menu == "Mark Attendance":
    st.subheader("Mark Attendance")
    img_file = st.camera_input("Scan your face")
    
    if img_file:
        image = face_recognition.load_image_file(img_file)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        if len(face_encodings) == 0:
            st.warning("No face detected")
        else:
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(list(known_faces.values()), face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(list(known_faces.values()), face_encoding)
                best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None
                if best_match_index is not None and matches[best_match_index]:
                    name = list(known_faces.keys())[best_match_index]
                    now = datetime.now()
                    date_str = now.strftime("%Y-%m-%d")
                    time_str = now.strftime("%H:%M:%S")
                    if not ((df['Name']==name) & (df['Date']==date_str)).any():
                        df = pd.concat([df, pd.DataFrame([[name, date_str, time_str]], columns=['Name','Date','Time'])], ignore_index=True)
                        df.to_csv(attendance_file, index=False)
                        st.success(f"Attendance marked for {name} at {time_str}")
                st.write(f"Detected: {name}")

elif menu == "Download CSV":
    st.subheader("Download Attendance CSV")
    st.download_button("Download CSV", attendance_file, file_name="attendance.csv")
