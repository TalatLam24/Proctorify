import tkinter as tk
from tkinter import messagebox
import cv2
import pyaudio
import wave
import time
import csv
import os
import threading
import ctypes
import requests
import socket
from pynput import keyboard, mouse
from PIL import Image, ImageTk
from datetime import datetime
from collections import deque
import urllib3

# Disable insecure request warnings for local HTTPS
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    import qrcode
except ImportError:
    print("WARNING: 'qrcode' library not found. Run 'pip install qrcode'")

# --- CONFIGURATION ---
BASE_OUTPUT_DIR = "Honeypot_Sessions"
RESOLUTION = (640, 480)
AUDIO_CHUNK = 1024
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 1
AUDIO_RATE = 44100
user32 = ctypes.windll.user32
EXAM_DURATION_MINUTES = 10

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

LOCAL_IP = get_local_ip()
MOBILE_URL = f"https://{LOCAL_IP}:5000"
LOCAL_API_URL = f"https://127.0.0.1:5000"

QUESTIONS =[
    "1. Explain HTML, CSS and JavaScript.",
    "2. Write the differences betweem GET and POST.",
    "3. Explain ER model and ER diagram.",
    "4. Explain Supervised, Unsupervised and Reinforcement Learning.",
    "5. What is the difference between overfitting and underfitting?"
]

class LightweightDataRecorder:
    def __init__(self, session_dir):
        self.session_dir = session_dir
        os.makedirs(self.session_dir, exist_ok=True)
        self.is_recording = True
        
        self.start_time_absolute = time.time()
        
        self.telemetry_file = open(os.path.join(self.session_dir, 'frame_telemetry.csv'), 'w', newline='', encoding='utf-8')
        self.telemetry_writer = csv.writer(self.telemetry_file)
        self.telemetry_writer.writerow(['Frame_ID', 'Timestamp', 'Interactions_Last_2s', 'Active_Window'])
        
        self.raw_input_file = open(os.path.join(self.session_dir, 'raw_inputs.csv'), 'w', newline='', encoding='utf-8')
        self.raw_input_writer = csv.writer(self.raw_input_file)
        self.raw_input_writer.writerow(['Timestamp', 'Event_Type', 'Key_Button', 'Active_Window'])
        
        self.event_file = open(os.path.join(self.session_dir, 'exam_events.csv'), 'w', newline='', encoding='utf-8')
        self.event_writer = csv.writer(self.event_file)
        self.event_writer.writerow(['Timestamp', 'Event', 'Details'])
        
        self.audio_frames = []
        self.audio_thread = threading.Thread(target=self._record_audio)
        self.audio_thread.start()
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.temp_video_path = os.path.join(self.session_dir, 'temp_webcam.mp4')
        self.video_out = cv2.VideoWriter(self.temp_video_path, fourcc, 30.0, RESOLUTION)
        
        self.input_timestamps = deque() 
        self.raw_input_queue = deque()
        self.input_lock = threading.Lock()
        self.frame_id = 0
        
        self.k_listener = keyboard.Listener(on_press=self._on_key)
        self.m_listener = mouse.Listener(on_click=self._on_click)
        self.k_listener.start()
        self.m_listener.start()
        
        self.log_event("SESSION_START", "")

    def _get_active_window(self):
        try:
            hwnd = user32.GetForegroundWindow()
            length = user32.GetWindowTextLengthW(hwnd)
            if length == 0: return ""
            buff = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buff, length + 1)
            return buff.value
        except: return ""

    def _on_key(self, key):
        if not self.is_recording: return
        t = time.time()
        try: k = key.char
        except AttributeError: k = str(key)
        with self.input_lock:
            self.input_timestamps.append(t)
            self.raw_input_queue.append((t, "Keyboard", k))

    def _on_click(self, x, y, button, pressed):
        if not self.is_recording or not pressed: return
        t = time.time()
        with self.input_lock:
            self.input_timestamps.append(t)
            self.raw_input_queue.append((t, "Mouse", str(button)))

    def log_frame_data(self, frame, exact_timestamp):
        if not self.is_recording: return
        active_window = self._get_active_window()
        with self.input_lock:
            while self.raw_input_queue:
                t, event_type, key_button = self.raw_input_queue.popleft()
                self.raw_input_writer.writerow([t, event_type, key_button, active_window])
            while self.input_timestamps and self.input_timestamps[0] < exact_timestamp - 2.0:
                self.input_timestamps.popleft()
            interactions_count = len(self.input_timestamps)

        self.telemetry_writer.writerow([self.frame_id, exact_timestamp, interactions_count, active_window])
        if self.video_out: self.video_out.write(frame)
        self.frame_id += 1

    def log_event(self, event, details):
        if self.is_recording: 
            self.event_writer.writerow([time.time(), event, details])
            self.event_file.flush()

    def _record_audio(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=AUDIO_FORMAT, channels=AUDIO_CHANNELS, rate=AUDIO_RATE, input=True, frames_per_buffer=AUDIO_CHUNK)
        while self.is_recording: self.audio_frames.append(stream.read(AUDIO_CHUNK))
        stream.close()
        p.terminate()
        with wave.open(os.path.join(self.session_dir, 'audio.wav'), 'wb') as wf:
            wf.setnchannels(AUDIO_CHANNELS)
            wf.setsampwidth(p.get_sample_size(AUDIO_FORMAT))
            wf.setframerate(AUDIO_RATE)
            wf.writeframes(b''.join(self.audio_frames))

    def stop(self):
        self.is_recording = False
        self.k_listener.stop()
        self.m_listener.stop()
        self.audio_thread.join()
        
        if self.video_out: 
            self.video_out.release()
            
        self.telemetry_file.close()
        self.raw_input_file.close()
        self.event_file.close()
        
        total_time = time.time() - self.start_time_absolute
        actual_fps = self.frame_id / total_time
        
        print(f"\n[INFO] Real duration: {total_time:.2f}s. Captured {self.frame_id} frames.")
        print(f"[INFO] Fixing video speed to {actual_fps:.2f} FPS. Please wait...")
        
        final_video_path = os.path.join(self.session_dir, 'webcam_video.mp4')
        cap = cv2.VideoCapture(self.temp_video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(final_video_path, fourcc, actual_fps, RESOLUTION)
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            out.write(frame)
            
        cap.release()
        out.release()
        os.remove(self.temp_video_path) 
        print("[INFO] Video speed fixed successfully!")
        
        return self.session_dir

class ExamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Honeypot Exam Collector v2.4")
        self.root.geometry("1000x800")
        self.recorder = None
        self.exam_active = False
        self.current_question = -1
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
        
        self.setup_frame = tk.Frame(root)
        self.exam_frame = tk.Frame(root)
        self.setup_frame.pack(expand=True, fill=tk.BOTH)
        
        tk.Label(self.setup_frame, text="Honeypot Exam Protocol", font=("Arial", 24, "bold")).pack(pady=20)
        tk.Label(self.setup_frame, text="Subject ID:", font=("Arial", 12)).pack()
        self.entry_id = tk.Entry(self.setup_frame, font=("Arial", 14), width=20, justify="center")
        self.entry_id.insert(0, "User_01")
        self.entry_id.pack(pady=10)
        
        try:
            qr_img = qrcode.make(MOBILE_URL).resize((200, 200))
            self.qr_photo = ImageTk.PhotoImage(qr_img)
            tk.Label(self.setup_frame, text="Scan with phone to start workspace camera:", font=("Arial", 12)).pack(pady=10)
            tk.Label(self.setup_frame, image=self.qr_photo).pack()
            tk.Label(self.setup_frame, text=f"URL: {MOBILE_URL}", font=("Arial", 10)).pack()
        except:
            tk.Label(self.setup_frame, text="(QR Code failed to load)", fg="red").pack(pady=10)
        
        tk.Button(self.setup_frame, text="START EXAM", font=("Arial", 16), bg="green", fg="white", command=self.start_exam).pack(pady=30)
        
        top_bar = tk.Frame(self.exam_frame)
        top_bar.pack(fill=tk.X, pady=10, padx=20)
        self.lbl_timer = tk.Label(top_bar, text="10:00", font=("Consolas", 24), fg="red")
        self.lbl_timer.pack(side=tk.LEFT)
        self.lbl_mini_vid = tk.Label(top_bar, text="Camera Loading...")
        self.lbl_mini_vid.pack(side=tk.RIGHT)

        self.lbl_question = tk.Label(self.exam_frame, text="", font=("Arial", 16, "bold"), wraplength=800, justify="left")
        self.lbl_question.pack(pady=30, padx=20, fill=tk.X)
        self.answer_text = tk.Text(self.exam_frame, height=12, width=80, font=("Arial", 12))
        self.answer_text.pack(pady=10)
        self.btn_next = tk.Button(self.exam_frame, text="Next Question", font=("Arial", 14), bg="blue", fg="white", command=self.next_question)
        self.btn_next.pack(pady=20)
        
        self.update_feed()

    def start_exam(self):
        subject_id = self.entry_id.get().strip()
        if not subject_id: return
        session_dir = os.path.join(BASE_OUTPUT_DIR, f"{subject_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        try:
            requests.post(f"{LOCAL_API_URL}/start_session", json={"session_dir": session_dir}, timeout=3, verify=False)
            time.sleep(1.0) 
        except Exception as e:
            messagebox.showerror("Error", f"Server unreachable.\n{e}")
            return

        self.recorder = LightweightDataRecorder(session_dir)
        self.exam_active = True
        
        self.setup_frame.pack_forget()
        self.exam_frame.pack(expand=True, fill=tk.BOTH)
        self.root.update() 
        
        orig_bg = self.root.cget("bg")
        self.root.configure(bg="white")
        self.exam_frame.configure(bg="white")
        self.lbl_timer.configure(bg="white")
        self.lbl_question.configure(bg="white")
        self.root.update() 
        time.sleep(0.5) 
        self.root.configure(bg=orig_bg)
        self.exam_frame.configure(bg=orig_bg)
        self.lbl_timer.configure(bg=orig_bg)
        self.lbl_question.configure(bg=orig_bg)
        self.root.update()
        
        self.exam_end_time = time.time() + (EXAM_DURATION_MINUTES * 60)
        self.update_exam_timer()
        self.next_question()

    def next_question(self):
        if self.current_question >= 0:
            answer = self.answer_text.get("1.0", tk.END).strip()
            with open(os.path.join(self.recorder.session_dir, f"answer_q{self.current_question+1}.txt"), 'w', encoding='utf-8') as f:
                f.write(answer)
            self.recorder.log_event("QUESTION_END", f"Q{self.current_question+1}")
        
        self.current_question += 1
        self.answer_text.delete("1.0", tk.END)
        
        if self.current_question >= len(QUESTIONS): 
            self.finish_exam()
            return
            
        self.lbl_question.config(text=QUESTIONS[self.current_question])
        self.recorder.log_event("QUESTION_START", f"Q{self.current_question+1}")
        if self.current_question == len(QUESTIONS) - 1: 
            self.btn_next.config(text="FINISH EXAM", bg="red")

    def update_exam_timer(self):
        if self.exam_active:
            remaining = max(0, int(self.exam_end_time - time.time()))
            mins, secs = divmod(remaining, 60)
            self.lbl_timer.config(text=f"{mins:02d}:{secs:02d}")
            if remaining == 0: self.finish_exam()
            self.root.after(1000, self.update_exam_timer)

    def finish_exam(self):
        if not self.exam_active: return
        self.exam_active = False
        
        self.exam_frame.pack_forget()
        tk.Label(self.root, text="Processing Final Video...\nPlease wait.", font=("Arial", 20)).pack(expand=True)
        self.root.update()
        
        # --- THE FIX: STOP THE PHONE FIRST ---
        try: requests.post(f"{LOCAL_API_URL}/stop_session", timeout=5, verify=False)
        except: pass
        
        if self.recorder:
            self.recorder.log_event("SESSION_END", "")
            saved_path = self.recorder.stop() # This now happens AFTER the phone stops
            self.recorder = None
            
        messagebox.showinfo("Exam Finished", f"Session saved to:\n{saved_path}")
        self.root.destroy()
        
    def update_feed(self):
        ret, frame = self.cap.read()
        capture_time = time.time() 
        if ret:
            if self.exam_active and self.recorder:
                self.recorder.log_frame_data(frame, capture_time)
            preview = cv2.resize(frame, (160, 120))
            preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(preview_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.lbl_mini_vid.imgtk = imgtk
            self.lbl_mini_vid.configure(image=imgtk)
        self.root.after(10, self.update_feed)

    def on_closing(self):
        if self.exam_active:
            if messagebox.askokcancel("Quit", "Quit exam? Progress will be lost."):
                if self.recorder: self.recorder.stop()
                try: requests.post(f"{LOCAL_API_URL}/stop_session", timeout=2, verify=False)
                except: pass
                self.root.destroy()
        else:
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ExamApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()