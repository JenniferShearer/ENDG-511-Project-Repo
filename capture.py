import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import os
import time
import numpy as np
from datetime import datetime

class ImageCaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vision Dataset Capture")
        self.root.geometry("700x520")   # smaller window
        self.root.resizable(False, False)

        self.base_folder = "dataset"
        os.makedirs(self.base_folder, exist_ok=True)

        # Open webcam
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            self.root.destroy()
            return

        # Stable webcam settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        time.sleep(2)

        # GUI Title
        self.title_label = tk.Label(root, text="Vision Dataset Capture", font=("Arial", 16, "bold"))
        self.title_label.pack(pady=8)

        # Label input row
        self.label_frame = tk.Frame(root)
        self.label_frame.pack(pady=5)

        self.label_text = tk.Label(self.label_frame, text="Label:", font=("Arial", 11))
        self.label_text.pack(side=tk.LEFT, padx=5)

        self.label_entry = tk.Entry(self.label_frame, font=("Arial", 11), width=20)
        self.label_entry.pack(side=tk.LEFT, padx=5)

        # Video preview
        self.video_label = tk.Label(root)
        self.video_label.pack(pady=8)

        # Buttons
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(pady=8)

        self.capture_button = tk.Button(
            self.button_frame,
            text="Take Photo",
            font=("Arial", 11),
            bg="green",
            fg="white",
            width=12,
            command=self.capture_image
        )
        self.capture_button.pack(side=tk.LEFT, padx=8)

        self.quit_button = tk.Button(
            self.button_frame,
            text="Quit",
            font=("Arial", 11),
            bg="red",
            fg="white",
            width=12,
            command=self.close_app
        )
        self.quit_button.pack(side=tk.LEFT, padx=8)

        # Status label
        self.status_label = tk.Label(root, text="Ready", font=("Arial", 10), fg="blue")
        self.status_label.pack(pady=5)

        self.current_frame = None

        # Spacebar capture shortcut
        self.root.bind("<space>", lambda event: self.capture_image())

        self.update_video()
        self.root.protocol("WM_DELETE_WINDOW", self.close_app)

    def sanitize_label(self, label):
        label = label.strip().lower()
        label = label.replace(" ", "_")
        allowed = "abcdefghijklmnopqrstuvwxyz0123456789_-"
        label = "".join(c for c in label if c in allowed)
        return label

    def is_black_frame(self, frame, threshold=15):
        if frame is None:
            return True
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray) < threshold

    def capture_image(self):
        label = self.label_entry.get().strip()

        if not label:
            messagebox.showwarning("Missing Label", "Please enter a label/class name first.")
            return

        label = self.sanitize_label(label)

        if not label:
            messagebox.showwarning("Invalid Label", "Please enter a valid label.")
            return

        if self.current_frame is None or self.is_black_frame(self.current_frame):
            messagebox.showwarning("Camera Issue", "No valid camera frame available.")
            return

        label_folder = os.path.join(self.base_folder, label)
        os.makedirs(label_folder, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{label}_{timestamp}.jpg"
        filepath = os.path.join(label_folder, filename)

        cv2.imwrite(filepath, self.current_frame)

        self.status_label.config(text=f"Saved to {label}/", fg="green")
        print(f"Saved image to: {filepath}")

    def update_video(self):
        ret, frame = self.cap.read()

        if ret and frame is not None:
            if not self.is_black_frame(frame):
                self.current_frame = frame.copy()

            preview_frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)

            # Smaller preview
            rgb_frame = cv2.resize(rgb_frame, (560, 315))

            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(30, self.update_video)

    def close_app(self):
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCaptureApp(root)
    root.mainloop()