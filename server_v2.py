from flask import Flask, render_template, request, jsonify
import os
import threading
import time
import socket
import cv2

app = Flask(__name__)
print("Flask App Initialized.")

BASE_OUTPUT_DIR = "Honeypot_Sessions"

current_session_data = {
    "is_recording": False,
    "session_dir": "",
    "start_time": 0.0,
    "lock": threading.Lock() 
}

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

@app.route('/')
def index():
    return render_template('camera.html')

@app.route('/status', methods=['GET'])
def get_status():
    with current_session_data["lock"]:
        is_rec = current_session_data["is_recording"]
    return jsonify({"is_recording": is_rec})

@app.route('/start_session', methods=['POST'])
def start_session():
    data = request.get_json()
    session_dir = data.get("session_dir")
    if not session_dir: return jsonify({"status": "error", "message": "session_dir not provided"}), 400
    
    with current_session_data["lock"]:
        current_session_data["session_dir"] = session_dir
        current_session_data["is_recording"] = True
        current_session_data["start_time"] = time.time()
    
    print(f"INFO: Phone camera recording started for session: {session_dir}")
    return jsonify({"status": "success", "message": "Recording started"})

@app.route('/stop_session', methods=['POST'])
def stop_session():
    with current_session_data["lock"]:
        current_session_data["is_recording"] = False
    print("INFO: is_recording set to FALSE. Waiting for phone to upload video...")
    return jsonify({"status": "success", "message": "Stop command received."})

@app.route('/upload_video', methods=['POST'])
def upload_video():
    with current_session_data["lock"]:
        session_dir = current_session_data["session_dir"]
    
    if not session_dir: return jsonify({"error": "No active session"}), 400
    if 'video' not in request.files: return jsonify({"error": "No video file part"}), 400
        
    file = request.files['video']
    filename = file.filename # Will be phone_video.mp4 or phone_video.webm
    temp_path = os.path.join(session_dir, filename)
    final_mp4_path = os.path.join(session_dir, "phone_video.mp4")
    
    try:
        os.makedirs(session_dir, exist_ok=True)
        file.save(temp_path)
        
        # --- NEW: INSTANT SERVER-SIDE CONVERSION ---
        if filename.endswith(".webm"):
            print(f"🔄 Android WebM detected. Converting to MP4...")
            cap = cv2.VideoCapture(temp_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or fps != fps: fps = 30.0 # Fallback
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(final_mp4_path, fourcc, fps, (width, height))
            
            while True:
                ret, frame = cap.read()
                if not ret: break
                out.write(frame)
                
            cap.release()
            out.release()
            os.remove(temp_path) # Delete the webm
            print(f"✅ SUCCESS: Converted and saved as {final_mp4_path}")
        else:
            print(f"✅ SUCCESS: Native iOS MP4 saved to {final_mp4_path}")
            
    except Exception as e:
        print(f"❌ ERROR saving/converting video: {e}")
        
    return jsonify({"status": "success"})

if __name__ == '__main__':
    local_ip = get_local_ip()
    print("\n" + "="*50)
    print(f"🚀 SERVER RUNNING ON LOCAL WI-FI 🚀")
    print(f"📱 Connect Phone To: https://{local_ip}:5000")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, ssl_context='adhoc', use_reloader=False)