import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import streamlit as st

# Disable oneDNN warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Attendance log file
ATTENDANCE_FILE = "attendance.csv"
MODEL_FILE = "hcr_model.h5"

# Create attendance file if not exists
if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=["Name", "Time"]).to_csv(ATTENDANCE_FILE, index=False)

# Function to train model
def train_model():
    st.info("📊 Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
    model.save(MODEL_FILE)
    st.success("✅ Model trained & saved!")
    return model

# Function to load or train model
def get_model():
    if os.path.exists(MODEL_FILE):
        return tf.keras.models.load_model(MODEL_FILE)
    else:
        st.warning("⚠️ Model not found. Training a new one...")
        return train_model()

# Function to mark attendance
def mark_attendance(name):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.read_csv(ATTENDANCE_FILE)
    df.loc[len(df)] = [name, now]
    df.to_csv(ATTENDANCE_FILE, index=False)

# Predict digit from image
def predict_image(image):
    model = get_model()
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28,28))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))

    pred = np.argmax(model.predict(img))
    name_map = {0:"Alice", 1:"Bob", 2:"Charlie", 3:"David", 4:"Eva",
                5:"Frank", 6:"Grace", 7:"Hannah", 8:"Ivan", 9:"Julia"}
    return pred, name_map[pred]

# Streamlit UI
st.title("📒 Attendance Management System")

menu = st.sidebar.radio("Menu", ["Train Model", "Recognize & Mark Attendance", "View Attendance Log"])

if menu == "Train Model":
    if st.button("Start Training"):
        train_model()

elif menu == "Recognize & Mark Attendance":
    uploaded_file = st.file_uploader("Upload digit image", type=["png", "jpg", "jpeg"])
    manual_num = st.text_input("Or enter digit (0-9) manually")

    if uploaded_file is not None:
        pred_digit, name = predict_image(uploaded_file)
        st.write(f"🔍 Predicted Digit: **{pred_digit}** → Name: **{name}**")
        mark_attendance(name)
        st.success(f"Attendance marked for {name}")

    elif manual_num.isdigit() and 0 <= int(manual_num) <= 9:
        name_map = {0:"Alice", 1:"Bob", 2:"Charlie", 3:"David", 4:"Eva",
                    5:"Frank", 6:"Grace", 7:"Hannah", 8:"Ivan", 9:"Julia"}
        name = name_map[int(manual_num)]
        mark_attendance(name)
        st.success(f"Attendance marked for {name}")

elif menu == "View Attendance Log":
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
        st.dataframe(df)
        st.download_button("Download CSV", df.to_csv(index=False), file_name="attendance.csv")
        st.download_button("Download HTML", df.to_html(index=False), file_name="attendance.html")
