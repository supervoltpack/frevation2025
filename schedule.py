import time
import tkinter as tk
from tkinter import messagebox
from datetime import datetime, timedelta

# List to store upcoming workout times
upcoming_workouts = []

# Function to send notification (simplified for this example)
def send_workout_notification(event_time):
    while True:
        current_time = datetime.now()

        # Check if the current time has reached the event time
        if current_time >= event_time:
            # Show a simple notification
            messagebox.showinfo("Workout Reminder", "It's time for your workout!")
            break  # Stop checking once the notification is sent

        # Wait for a while before checking again
        time.sleep(1)

# Function to trigger workout scheduling
def schedule_workout():
    try:
        # Get the user input time for the workout
        workout_time = entry_time.get()

        # Parse the time input
        event_time = datetime.strptime(workout_time, '%Y-%m-%d %H:%M:%S')

        # Add the workout to the upcoming workouts list
        upcoming_workouts.append(event_time)

        # Schedule the workout notification
        send_workout_notification(event_time)

        # Notify the user
        messagebox.showinfo("Workout Scheduled", f"Workout scheduled for {event_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Update the text box with the upcoming workouts
        update_workout_list()

    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter the time in the format YYYY-MM-DD HH:MM:SS")

# Function to update the upcoming workouts list in the Text widget
def update_workout_list():
    workout_text.delete(1.0, tk.END)  # Clear the existing text
    if upcoming_workouts:
        workout_text.insert(tk.END, "Upcoming Workouts:\n")
        for workout in upcoming_workouts:
            workout_text.insert(tk.END, f"- {workout.strftime('%Y-%m-%d %H:%M:%S')}\n")
    else:
        workout_text.insert(tk.END, "No upcoming workouts scheduled.")

# Create the main window
window = tk.Tk()
window.title("Workout Reminder")

# Create a frame for scheduling workouts
frame_schedule = tk.Frame(window)
frame_schedule.pack(pady=20)

# Create a label and entry for the user to input the workout time
label = tk.Label(frame_schedule, text="Enter workout time (YYYY-MM-DD HH:MM:SS):")
label.pack(pady=10)

entry_time = tk.Entry(frame_schedule)
entry_time.pack(pady=10)

# Create a button to trigger workout scheduling
button_schedule = tk.Button(frame_schedule, text="Schedule Workout", command=schedule_workout)
button_schedule.pack(pady=20)

# Create a Text widget to display the upcoming workouts
workout_text = tk.Text(window, height=10, width=50)
workout_text.pack(pady=20)

# Initial display of upcoming workouts
update_workout_list()

# Run the Tkinter event loop
window.mainloop()
