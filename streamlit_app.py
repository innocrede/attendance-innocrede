

import streamlit as st
import cv2
import face_recognition
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
from tempfile import NamedTemporaryFile

# Load or create encodings
try:
    with open('encodings.pickle','rb') as f:
        known_faces = pickle.load(f)
except:
    known_faces = {}

# Load or create attendance CSV
attendance_file = 'attendance.csv'
try:
    df = pd.read_csv(attendance_file)
except:
    df = pd.DataFrame(columns=['Name','Date','Time'])

st.title("Face Recognition Attendance Web App")

menu = st.sidebar.selectbox("Menu", ["Register Face", "Mark Attendance", "Download CSV"])

if menu == "Register Face":
    st.subheader("Register Face")
    name = st.text_input("Enter Name")
    run = st.button("Start Camera")
    if run and name:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        st.warning("Press 'q' to capture your face")
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, channels="RGB")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                face_locations = face_recognition.face_locations(frame)
                if len(face_locations) != 1:
                    st.warning("Ensure only one face visible")
                    continue
                face_encoding = face_recognition.face_encodings(frame, face_locations)[0]
                known_faces[name] = face_encoding
                with open('encodings.pickle','wb') as f:
                    pickle.dump(known_faces,f)
                st.success(f"Face of {name} registered!")
                break
        cap.release()
        cv2.destroyAllWindows()

elif menu == "Mark Attendance":
    st.subheader("Mark Attendance")
    run = st.button("Start Camera")
    if run:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        st.warning("Press 'q' to stop")
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small)
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(list(known_faces.values()), face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(list(known_faces.values()), face_encoding)
                best_match_index = np.argmin(face_distances) if len(face_distances)>0 else None
                if best_match_index is not None and matches[best_match_index]:
                    name = list(known_faces.keys())[best_match_index]
                    now = datetime.now()
                    date_str = now.strftime("%Y-%m-%d")
                    time_str = now.strftime("%H:%M:%S")
                    if not ((df['Name']==name) & (df['Date']==date_str)).any():
                        df = pd.concat([df, pd.DataFrame([[name, date_str, time_str]], columns=['Name','Date','Time'])], ignore_index=True)
                        df.to_csv(attendance_file, index=False)
                        st.success(f"Attendance marked for {name} at {time_str}")
                top, right, bottom, left = [v*4 for v in face_location]
                cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
                cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, channels="RGB")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

elif menu == "Download CSV":
    st.subheader("Download Attendance CSV")
    tmp_download_link = NamedTemporaryFile(delete=False)
    df.to_csv(tmp_download_link.name, index=False)
    st.download_button("Download CSV", tmp_download_link.name, file_name="attendance.csv")
