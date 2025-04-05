import streamlit as st

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #77dd77 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

class Player:
    def __init__(self, name):
        self.name = name.strip()
        self.level = 1
        self.xp = 0

    def xp_needed_for_next_level(self):
        return 100 * self.level

    def add_xp(self, amount):
        self.xp += amount
        self.check_level_up()

    def check_level_up(self):
        while self.xp >= self.xp_needed_for_next_level():
            self.xp -= self.xp_needed_for_next_level()
            self.level += 1

    def __repr__(self):
        return f"{self.name} | Level {self.level} | XP {self.xp}"

class Leaderboard:
    def __init__(self):
        self.players = {}

    def add_player(self, name):
        normalized_name = name.strip().lower()  # Normalize name to lower case and remove extra spaces
        if normalized_name not in self.players:
            self.players[normalized_name] = Player(name)  # Store the original name for display purposes
            return True
        return False

    def remove_player(self, name):
        normalized_name = name.strip().lower()
        if normalized_name in self.players:
            del self.players[normalized_name]
            return True
        return False

    def add_xp_to_player(self, name, xp):
        normalized_name = name.strip().lower()  # Normalize name to lower case and remove extra spaces
        if normalized_name in self.players:
            self.players[normalized_name].add_xp(xp)
            return True
        return False

    def get_leaderboard(self, sort_by='level'):
        if sort_by == 'level':
            sorted_players = sorted(self.players.values(), key=lambda p: (p.level, p.xp), reverse=True)
        elif sort_by == 'xp':
            sorted_players = sorted(self.players.values(), key=lambda p: p.xp, reverse=True)
        return sorted_players

if 'leaderboard' not in st.session_state:
    st.session_state.leaderboard = Leaderboard()

board = st.session_state.leaderboard

st.markdown("<h1 style='text-align: center;'>Frevation</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>1 Level = 100 XP</h3>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;'>Add Player</h2>", unsafe_allow_html=True)
player_name = st.text_input("Enter player's name:")
if st.button("Add Player"):
    if player_name.strip():  # Ensure name is not empty after strip
        if board.add_player(player_name):
            st.success(f"✅ Player '{player_name.strip()}' added!")
        else:
            st.warning(f"⚠️ Player '{player_name.strip()}' already exists.")
    else:
        st.warning("❌ Please enter a valid name.")

st.markdown("<h2 style='text-align: center;'>Remove Player</h2>", unsafe_allow_html=True)
remove_player_name = st.selectbox("Select player to remove:", list(board.players.keys()))
if st.button("Remove Player"):
    if remove_player_name.strip():
        if board.remove_player(remove_player_name):
            st.success(f"✅ Player '{remove_player_name}' removed!")
        else:
            st.warning(f"⚠️ Player '{remove_player_name}' not found.")
    else:
        st.warning("❌ Please select a valid player.")

st.markdown("<h2 style='text-align: center;'>Add or Remove XP</h2>", unsafe_allow_html=True)
xp_name = st.text_input("Enter player's name to add/remove XP:")
xp_amount = st.number_input("Enter amount of XP to add/remove:", min_value=1)
if st.button("Add XP"):
    if xp_name.strip() and xp_amount > 0:  # Ensure name is not empty after strip
        if board.add_xp_to_player(xp_name, xp_amount):
            st.success(f"✨ Added {xp_amount} XP to {xp_name}.")
        else:
            st.warning(f"❌ Player '{xp_name}' not found.")
    else:
        st.warning("❌ Please enter a valid name and XP amount.")

st.markdown("<h2 style='text-align: center;'>View Leaderboard</h2>", unsafe_allow_html=True)
sort_option = st.radio("Sort leaderboard by:", ("Level", "XP"))

if st.button("Show Leaderboard"):
    leaderboard = board.get_leaderboard(sort_by=sort_option.lower())
    st.subheader(f"Leaderboard (sorted by {sort_option}):")
    for i, player in enumerate(leaderboard, start=1):
        st.write(f"{i}. {player}")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    ai_button = st.button("Your Daily UV Safety Check", use_container_width=True)
    calendar_button = st.button("Go to Calendar", use_container_width=True)

if ai_button:
    import os
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from tensorflow.keras import Sequential, Input, Model
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import gradio as gr

    project = "histology"
    prefix = "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Towards%20Precision%20Medicine/"
    os.system(f'curl -O "{prefix}images.npy"')
    os.system(f'curl -O "{prefix}labels.npy"')
    images = np.load("images.npy")
    labels = np.load("labels.npy")
    os.remove("images.npy")
    os.remove("labels.npy")

    images = images.astype('float32') / 255.0

    label_names = np.unique(labels)
    labels_ohe = np.array(pd.get_dummies(labels))

    X_train, X_test, y_train, y_test = train_test_split(images, labels_ohe, test_size=0.2, random_state=42)

    datagen = ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X_train)

    base_model = MobileNetV2(input_shape=X_train.shape[1:], include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze base

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(y_train.shape[1], activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)
    ]

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_test, y_test),
        epochs=25,
        callbacks=callbacks
    )

    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {acc:.4f}")

    # Get true labels and predictions
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    print(classification_report(y_true, y_pred, target_names=label_names))

    def predict_image(img):
        img = img.resize((X_train.shape[1], X_train.shape[2]))
        img = np.array(img).astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)[0]
        return {label_names[i]: float(prediction[i]) for i in range(len(label_names))}

    interface = gr.Interface(
        fn=predict_image,
        inputs=gr.Image(type="pil"),
        outputs=gr.Label(num_top_classes=3),
        title="Histology Image Classifier",
        description="Upload a histology image to classify its category using MobileNetV2."
    )

    interface.launch()

if calendar_button:
    import time
    import tkinter as tk
    from tkinter import messagebox
    from datetime import datetime, timedelta

    upcoming_workouts = []

    def send_workout_notification(event_time):
        while True:
            current_time = datetime.now()

            if current_time >= event_time:
                # Show a simple notification
                messagebox.showinfo("Workout Reminder", "It's time for your workout!")
                break  # Stop checking once the notification is sent

            time.sleep(1)

    def schedule_workout():
        try:
            workout_time = entry_time.get()

            event_time = datetime.strptime(workout_time, '%Y-%m-%d %H:%M:%S')

            upcoming_workouts.append(event_time)

            send_workout_notification(event_time)

            messagebox.showinfo("Workout Scheduled", f"Workout scheduled for {event_time.strftime('%Y-%m-%d %H:%M:%S')}")

            update_workout_list()

        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter the time in the format YYYY-MM-DD HH:MM:SS")

    def update_workout_list():
        workout_text.delete(1.0, tk.END)  # Clear the existing text
        if upcoming_workouts:
            workout_text.insert(tk.END, "Upcoming Workouts:\n")
            for workout in upcoming_workouts:
                workout_text.insert(tk.END, f"- {workout.strftime('%Y-%m-%d %H:%M:%S')}\n")
        else:
            workout_text.insert(tk.END, "No upcoming workouts scheduled.")

    window = tk.Tk()
    window.title("Workout Reminder")

    frame_schedule = tk.Frame(window)
    frame_schedule.pack(pady=20)

    label = tk.Label(frame_schedule, text="Enter workout time (YYYY-MM-DD HH:MM:SS):")
    label.pack(pady=10)

    entry_time = tk.Entry(frame_schedule)
    entry_time.pack(pady=10)

    button_schedule = tk.Button(frame_schedule, text="Schedule Workout", command=schedule_workout)
    button_schedule.pack(pady=20)

    workout_text = tk.Text(window, height=10, width=50)
    workout_text.pack(pady=20)

    update_workout_list()

    window.mainloop()

