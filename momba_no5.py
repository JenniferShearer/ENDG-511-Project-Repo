import os
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import csv
import tkinter as tk
from tkinter import font, filedialog, messagebox
import threading
import queue
import time
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_FPS = 10
FRAME_INTERVAL_S = 1.0 / TARGET_FPS
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 480

ALL_CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
LETTER_CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

CLASS_NAMES = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z'
]

# ---------------------------------------------------------------------------
# Camera thread
# ---------------------------------------------------------------------------
class CaptureThread(threading.Thread):
    def __init__(self, frame_queue, camera_index=0):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.camera_index = camera_index
        self._stop_event = threading.Event()
        self.error = None

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.error = f"Could not open camera at index {self.camera_index}."
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, PREVIEW_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREVIEW_HEIGHT)

        next_tick = time.monotonic()

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                self.error = "Failed to read frame from camera."
                break

            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass

            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass

            next_tick += FRAME_INTERVAL_S
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()

        cap.release()

    def stop(self):
        self._stop_event.set()

# ---------------------------------------------------------------------------
# Model thread
# ---------------------------------------------------------------------------
class ModelThread(threading.Thread):
    def __init__(self, callback, in_q, out_q):
        super().__init__(daemon=True)
        self.callback = callback
        self.in_q = in_q
        self.out_q = out_q
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            try:
                frame = self.in_q.get(timeout=0.5)
            except queue.Empty:
                continue

            while True:
                try:
                    frame = self.in_q.get_nowait()
                except queue.Empty:
                    break

            try:
                result = self.callback(frame)

                while True:
                    try:
                        self.out_q.get_nowait()
                    except queue.Empty:
                        break

                self.out_q.put_nowait(result)

            except Exception as exc:
                while True:
                    try:
                        self.out_q.get_nowait()
                    except queue.Empty:
                        break

                self.out_q.put_nowait({
                    "text": f"[Model error] {exc}",
                    "prediction_vector": None
                })

    def stop(self):
        self._stop_event.set()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
class CameraApp:
    def __init__(self, model_callback, camera_index=0):
        self.model_callback = model_callback
        self.camera_index = camera_index

        self._frame_q = queue.Queue(maxsize=1)
        self._model_in_q = queue.Queue(maxsize=1)
        self._result_q = queue.Queue(maxsize=1)

        self._capture_thread = None
        self._model_thread = None

        self._paused = False
        self._frame_count = 0
        self._fps_last_time = time.monotonic()

        # Saved rows for export:
        # {"selected_char": "A", "vector": [26 values]}
        self._saved_rows = []

        # Most recent model result
        self._last_result = None

        self._build_ui()

    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("AI Camera")
        self.root.configure(bg="#1e1e2e")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._target_char = tk.StringVar(value="A")

        title_font = font.Font(family="Helvetica", size=14, weight="bold")
        small_font = font.Font(family="Helvetica", size=10)
        mono_font = font.Font(family="Courier", size=11)
        button_font = font.Font(family="Helvetica", size=10, weight="bold")

        top = tk.Frame(self.root, bg="#1e1e2e", pady=8)
        top.pack(fill="x", padx=16)

        tk.Label(
            top,
            text="📷 AI Camera Viewer",
            font=title_font,
            bg="#1e1e2e",
            fg="#cdd6f4"
        ).pack(side="left")

        self._fps_label = tk.Label(
            top,
            text="FPS: --",
            font=small_font,
            bg="#1e1e2e",
            fg="#a6e3a1"
        )
        self._fps_label.pack(side="right")

        preview_frame = tk.Frame(self.root, bg="#313244", bd=2, relief="flat")
        preview_frame.pack(padx=16, pady=(0, 8))

        self.canvas = tk.Canvas(
            preview_frame,
            width=PREVIEW_WIDTH,
            height=PREVIEW_HEIGHT,
            bg="#181825",
            highlightthickness=0
        )
        self.canvas.pack()

        self._canvas_image_id = self.canvas.create_image(0, 0, anchor="nw")
        self._placeholder = self.canvas.create_text(
            PREVIEW_WIDTH // 2,
            PREVIEW_HEIGHT // 2,
            text="Starting camera…",
            fill="#6c7086",
            font=font.Font(family="Helvetica", size=16)
        )

        control_frame = tk.Frame(self.root, bg="#1e1e2e")
        control_frame.pack(fill="x", padx=16, pady=(0, 8))

        selector_row = tk.Frame(control_frame, bg="#1e1e2e")
        selector_row.pack(anchor="w")

        tk.Label(
            selector_row,
            text="Selected Character:",
            font=small_font,
            bg="#1e1e2e",
            fg="#a6adc8"
        ).pack(side="left")

        char_menu = tk.OptionMenu(
            selector_row,
            self._target_char,
            *ALL_CHARS,
            command=self._on_char_changed
        )
        char_menu.config(
            bg="#313244",
            fg="#cdd6f4",
            font=small_font,
            relief="flat",
            highlightthickness=0,
            width=3
        )
        char_menu["menu"].config(
            bg="#313244",
            fg="#cdd6f4",
            font=small_font
        )
        char_menu.pack(side="left", padx=(6, 10))

        tk.Button(
            selector_row,
            text="💾 Save Row",
            command=self._save_current_prediction_row,
            bg="#a6e3a1",
            fg="#1e1e2e",
            font=button_font,
            relief="flat",
            padx=10,
            pady=4,
            cursor="hand2"
        ).pack(side="left", padx=(0, 8))

        tk.Button(
            selector_row,
            text="📄 Export CSV",
            command=self._export_csv,
            bg="#89b4fa",
            fg="#1e1e2e",
            font=button_font,
            relief="flat",
            padx=10,
            pady=4,
            cursor="hand2"
        ).pack(side="left")

        self._selected_label_var = tk.StringVar(value="Selected character: A")
        tk.Label(
            control_frame,
            textvariable=self._selected_label_var,
            font=small_font,
            bg="#1e1e2e",
            fg="#89b4fa"
        ).pack(anchor="w", pady=(6, 0))

        self._status = tk.StringVar(value="")
        tk.Label(
            control_frame,
            textvariable=self._status,
            font=small_font,
            bg="#1e1e2e",
            fg="#a6e3a1"
        ).pack(anchor="w", pady=(4, 0))

        output_outer = tk.Frame(self.root, bg="#1e1e2e")
        output_outer.pack(fill="x", padx=16, pady=(0, 8))

        tk.Label(
            output_outer,
            text="Model output",
            font=small_font,
            bg="#1e1e2e",
            fg="#89b4fa"
        ).pack(anchor="w")

        self._output = tk.StringVar(value="Waiting...")
        tk.Label(
            output_outer,
            textvariable=self._output,
            font=mono_font,
            bg="#313244",
            fg="#cdd6f4",
            anchor="w",
            justify="left",
            wraplength=PREVIEW_WIDTH - 16,
            padx=10,
            pady=8
        ).pack(fill="x")

        bottom_controls = tk.Frame(self.root, bg="#1e1e2e", pady=8)
        bottom_controls.pack(padx=16, pady=(0, 12))

        self._pause_btn = tk.Button(
            bottom_controls,
            text="⏸ Pause",
            command=self._toggle_pause,
            bg="#f9e2af",
            fg="#1e1e2e",
            font=button_font,
            relief="flat",
            padx=14,
            pady=5,
            cursor="hand2"
        )
        self._pause_btn.pack(side="left", padx=(0, 8))

        tk.Button(
            bottom_controls,
            text="✕ Quit",
            command=self._on_close,
            bg="#f38ba8",
            fg="#1e1e2e",
            font=button_font,
            relief="flat",
            padx=14,
            pady=5,
            cursor="hand2"
        ).pack(side="left")

    def _on_char_changed(self, selected_value=None):
        if selected_value is None:
            selected_value = self._target_char.get()
        self._selected_label_var.set(f"Selected character: {selected_value}")

    def _save_current_prediction_row(self):
        char = self._target_char.get()

        if self._last_result is None or self._last_result.get("prediction_vector") is None:
            self._status.set("No prediction vector available yet.")
            self.root.after(3000, lambda: self._status.set(""))
            return

        self._saved_rows.append({
            "selected_char": char,
            "vector": list(self._last_result["prediction_vector"])
        })

        self._status.set(f"Saved one row under {char}.")
        self.root.after(3000, lambda: self._status.set(""))

    def _export_csv(self):
        if not self._saved_rows:
            messagebox.showwarning("No Data", "There is no saved prediction data to export.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Save labelled prediction data"
        )

        if not file_path:
            return

        try:
            with open(file_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                writer.writerow([""] + LETTER_CLASS_NAMES)

                for row in self._saved_rows:
                    writer.writerow([row["selected_char"]] + row["vector"])

            messagebox.showinfo("Export Complete", f"CSV saved successfully:\n{file_path}")
            self._status.set("CSV export complete.")
            self.root.after(3000, lambda: self._status.set(""))

        except Exception as exc:
            messagebox.showerror("Export Error", f"Could not export CSV:\n{exc}")

    def _poll(self):
        if not self._paused:
            try:
                frame = self._frame_q.get_nowait()
            except queue.Empty:
                frame = None

            if frame is not None:
                self._show_frame(frame)

                if self.model_callback is not None:
                    try:
                        self._model_in_q.get_nowait()
                    except queue.Empty:
                        pass

                    try:
                        self._model_in_q.put_nowait(frame.copy())
                    except queue.Full:
                        pass

                self._frame_count += 1
                now = time.monotonic()
                elapsed = now - self._fps_last_time
                if elapsed >= 1.0:
                    fps = self._frame_count / elapsed
                    self._frame_count = 0
                    self._fps_last_time = now
                    self._fps_label.config(text=f"FPS: {fps:.1f}")

            if self._capture_thread and self._capture_thread.error:
                self._output.set(f"⚠ {self._capture_thread.error}")

        latest = None
        try:
            while True:
                latest = self._result_q.get_nowait()
        except queue.Empty:
            pass

        if latest is not None:
            if isinstance(latest, dict):
                self._last_result = latest
                self._output.set(latest.get("text", ""))
            else:
                self._output.set(str(latest))

        self.root.after(33, self._poll)

    def _show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((PREVIEW_WIDTH, PREVIEW_HEIGHT), Image.BILINEAR)
        self._tk_img = ImageTk.PhotoImage(img)
        self.canvas.itemconfig(self._canvas_image_id, image=self._tk_img)
        self.canvas.itemconfig(self._placeholder, state="hidden")

    def _toggle_pause(self):
        self._paused = not self._paused
        if self._paused:
            self._pause_btn.config(text="▶ Resume", bg="#a6e3a1")
        else:
            self._pause_btn.config(text="⏸ Pause", bg="#f9e2af")

    def _on_close(self):
        if self._capture_thread:
            self._capture_thread.stop()
        if self._model_thread:
            self._model_thread.stop()
        self.root.destroy()

    def run(self):
        self._capture_thread = CaptureThread(self._frame_q, self.camera_index)
        self._capture_thread.start()

        if self.model_callback is not None:
            self._model_thread = ModelThread(
                self.model_callback,
                self._model_in_q,
                self._result_q
            )
            self._model_thread.start()
        else:
            self._output.set("No model callback provided.")

        self.root.after(100, self._poll)
        self.root.mainloop()

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_PATH = r"C:\Users\scarv\OneDrive\Desktop\Ucalgary\ENDG511\Final Project\model_12.keras"

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded.")

# Smoothing history for display only
prediction_history = [deque(maxlen=10), deque(maxlen=10), deque(maxlen=10)]

def model_callback(frame):
    img = cv2.resize(frame, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32")
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    outputs = model(img, training=False)

    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    # -------------------------------------------------------
    # Bottom text output with smoothing added back
    # -------------------------------------------------------
    results = []
    usable_heads = min(len(outputs), len(prediction_history))

    for i in range(usable_heads):
        pred_np = outputs[i].numpy()[0]
        idx = int(np.argmax(pred_np))
        label = CLASS_NAMES[idx]
        raw_conf = float(pred_np[idx])

        prediction_history[i].append((label, raw_conf))

        scores = {}
        for lbl, score in prediction_history[i]:
            scores[lbl] = scores.get(lbl, 0.0) + score

        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]
        total_score = sum(scores.values())
        smooth_conf = best_score / total_score if total_score > 0 else 0.0

        results.append(
            f"Head {i+1}: {label} ({raw_conf*100:.1f}%)"
            f"  |  Smoothed: {best_label} ({smooth_conf*100:.1f}%)"
        )

    # If model has more heads than history slots, still show raw output for them
    for i in range(usable_heads, len(outputs)):
        pred_np = outputs[i].numpy()[0]
        idx = int(np.argmax(pred_np))
        label = CLASS_NAMES[idx]
        raw_conf = float(pred_np[idx])
        results.append(f"Head {i+1}: {label} ({raw_conf*100:.1f}%)")

    display_text = "\n".join(results)

    # -------------------------------------------------------
    # Save/export uses only A-Z from the FINAL head
    # -------------------------------------------------------
    final_pred_np = outputs[-1].numpy()[0]
    letter_indices = [CLASS_NAMES.index(ch) for ch in LETTER_CLASS_NAMES]
    prediction_vector = [float(final_pred_np[i]) for i in letter_indices]

    return {
        "text": display_text,
        "prediction_vector": prediction_vector
    }

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = CameraApp(model_callback, camera_index=0)
    app.run()