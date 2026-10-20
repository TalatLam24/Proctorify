import pandas as pd
import cv2
import os
import numpy as np

# --- CONFIGURATION ---
SESSIONS_DIR = "Honeypot_Sessions"

# Playback drift correction settings
CORRECTION_INTERVAL = 15   # Check drift every N frames (~0.5s at 30fps)
DRIFT_THRESHOLD_MS = 50    # Correct phone if drift exceeds this many ms

def run_labeler(session_id):
    session_path = os.path.join(SESSIONS_DIR, session_id)
    webcam_path = os.path.join(session_path, "webcam_video.mp4")
    phone_path = os.path.join(session_path, "phone_video.mp4")
    csv_path = os.path.join(session_path, "ml_features.csv")
    telemetry_path = os.path.join(session_path, "frame_telemetry.csv")
    output_path = os.path.join(session_path, "final_training_dataset.csv")

    if not (os.path.exists(webcam_path) and os.path.exists(phone_path) 
            and os.path.exists(csv_path) and os.path.exists(telemetry_path)):
        print(f"Error: Missing files in {session_path}")
        return

    df = pd.read_csv(csv_path)
    df_telemetry = pd.read_csv(telemetry_path)
    
    if 'Target_Label' not in df.columns:
        df['Target_Label'] = 'NORMAL'
        df.loc[df['Unauthorized_Window'] == 1.0, 'Target_Label'] = 'CHEAT_DIGITAL'

    cap_webcam = cv2.VideoCapture(webcam_path)
    cap_phone = cv2.VideoCapture(phone_path)
    
    total_frames = len(df)
    frame_idx = 0
    is_playing = False
    need_seek = True
    
    frame_web = None
    frame_phone = np.zeros((360, 480, 3), dtype=np.uint8)

    # --- TIME-BASED CALIBRATION STATE ---
    # Sync points store: (webcam_real_timestamp, phone_video_time_msec)
    sync_A = None
    sync_B = None
    sync_A_webcam_frame = None
    calibrated = False
    
    phone_total_frames = int(cap_phone.get(cv2.CAP_PROP_FRAME_COUNT))
    phone_fps = cap_phone.get(cv2.CAP_PROP_FPS)
    
    # Pre-load real timestamps from telemetry
    real_timestamps = df_telemetry['Timestamp'].values
    
    # Uncalibrated: manual phone frame offset
    phone_offset_frames = 0
    
    # Playback drift correction counter
    frames_since_correction = 0
    paint_mode = 'NORMAL'

    # ======== HELPER FUNCTIONS ========

    def compute_phone_time_msec(real_time):
        """Map a real-world timestamp to a phone video time in milliseconds
        using linear interpolation between the two sync points."""
        if calibrated and sync_A and sync_B:
            rt_A, pt_A = sync_A
            rt_B, pt_B = sync_B
            if rt_B != rt_A:
                return pt_A + (real_time - rt_A) * (pt_B - pt_A) / (rt_B - rt_A)
            return pt_A
        return None

    def get_target_phone_msec():
        """Get the target phone video time for the current frame_idx."""
        if calibrated:
            return compute_phone_time_msec(real_timestamps[frame_idx])
        return None

    def seek_webcam():
        """Seek webcam to frame_idx and read."""
        nonlocal frame_web
        cap_webcam.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, f = cap_webcam.read()
        if ret:
            frame_web = f

    def seek_phone_by_time(msec):
        """Seek phone video to a specific time in milliseconds and read."""
        nonlocal frame_phone
        if msec < 0:
            msec = 0
        cap_phone.set(cv2.CAP_PROP_POS_MSEC, msec)
        ret, f = cap_phone.read()
        if ret:
            frame_phone = f
        else:
            frame_phone = np.zeros((360, 480, 3), dtype=np.uint8)

    def seek_phone_by_frame(phone_frame):
        """Seek phone video to a specific frame number and read (used only when uncalibrated)."""
        nonlocal frame_phone
        phone_frame = max(0, min(phone_frame, phone_total_frames - 1))
        cap_phone.set(cv2.CAP_PROP_POS_FRAMES, phone_frame)
        ret, f = cap_phone.read()
        if ret:
            frame_phone = f
        else:
            frame_phone = np.zeros((360, 480, 3), dtype=np.uint8)

    def seek_both():
        """Seek both videos to their correct positions for the current frame_idx."""
        seek_webcam()
        if calibrated:
            target_msec = get_target_phone_msec()
            if target_msec is not None:
                seek_phone_by_time(target_msec)
        else:
            seek_phone_by_frame(frame_idx + phone_offset_frames)

    # ======== INSTRUCTIONS ========

    print("\n" + "="*60)
    print("FRAME ANNOTATOR (TIME-BASED SYNC)")
    print("")
    print("  [SPACE] : Play / Pause")
    print("  [ S ]   : Jump to Start")
    print("  [ E ]   : Jump to End")
    print("  [ A ]   : Jump to Sync Point A")
    print("  [,] / [.] : Step 1 Frame Back / Forward")
    print("  [ [ ] / [ ] ] : Shift Mobile 1 Frame (Uncalibrated)")
    print("")
    print("--- HOW TO SYNC (Do this before labeling) ---")
    print("  1. Press 'S'. Find the white flash on both videos.")
    print("     Use [ ] to align the phone. Press '1'.")
    print("  2. Press 'E'. Find a matching event at the end.")
    print("     Use [ ] to align the phone. Press '2'.")
    print("  3. Press '3' to apply Time-Based Sync.")
    print("  4. Press 'A' to jump back and begin labeling!")
    print("  0. Press '0' to clear calibration if you made a mistake.")
    print("")
    print("--- LABELING ---")
    print("  [ C ] : CHEAT_PHYSICAL")
    print("  [ N ] : NORMAL")
    print("  [ D ] : CHEAT_DIGITAL")
    print("  [ Q ] : Quit & Save")
    print("="*60 + "\n")

    # ======== MAIN LOOP ========

    while True:
        
        # --- FRAME READING ---
        if need_seek or not is_playing:
            # PAUSED / STEPPING / AFTER JUMP: seek both to exact positions
            seek_both()
            need_seek = False
            frames_since_correction = 0
            
        else:
            # PLAYING: sequential reads for speed
            ret_web, f_web = cap_webcam.read()
            if not ret_web:
                is_playing = False
                need_seek = True
                continue
            frame_web = f_web
            
            ret_phone, f_phone = cap_phone.read()
            if ret_phone:
                frame_phone = f_phone
            
            # Periodic drift correction
            frames_since_correction += 1
            if frames_since_correction >= CORRECTION_INTERVAL and calibrated:
                target_msec = get_target_phone_msec()
                if target_msec is not None:
                    actual_msec = cap_phone.get(cv2.CAP_PROP_POS_MSEC)
                    if abs(actual_msec - target_msec) > DRIFT_THRESHOLD_MS:
                        seek_phone_by_time(target_msec)
                frames_since_correction = 0
        
        # --- DATA HANDLING ---
        current_label = df.iloc[frame_idx]['Target_Label']
        current_time = real_timestamps[frame_idx]
        try: active_window = df_telemetry.iloc[frame_idx]['Active_Window']
        except IndexError: active_window = "Unknown"
        
        if current_label == 'CHEAT_PHYSICAL': color = (0, 0, 255)
        elif current_label == 'CHEAT_DIGITAL': color = (0, 165, 255)
        else: color = (0, 255, 0)

        # --- UI RENDERING ---
        display_web = frame_web.copy() if frame_web is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(display_web, (0, 0), (display_web.shape[1], 100), (0,0,0), -1)
        cv2.putText(display_web, f"Frame: {frame_idx}/{total_frames-1} | Time: {current_time:.2f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_web, f"LABEL: {current_label}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        cv2.putText(display_web, f"Window: {str(active_window)[:40]}", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        display_phone = cv2.resize(frame_phone.copy(), (480, 360))
        cv2.rectangle(display_phone, (0, 0), (480, 50), (0,0,0), -1)
        
        if calibrated:
            phone_msec = get_target_phone_msec()
            phone_time_str = f"{phone_msec/1000:.2f}s" if phone_msec else "N/A"
            cv2.putText(display_phone, "TIME-SYNCED", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_phone, f"Phone Time: {phone_time_str}", (10, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(display_phone, "UNCALIBRATED (Press 1, 2, then 3)", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(display_phone, f"A:{'Set' if sync_A else 'No'} | B:{'Set' if sync_B else 'No'}", (10, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        if not is_playing:
            cv2.putText(display_phone, "PAUSED", (380, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Webcam (Primary Data)", display_web)
        cv2.imshow("Mobile (Ground Truth)", display_phone)

        # --- CONTROLS ---
        wait_time = 33 if is_playing else 0
        key = cv2.waitKey(wait_time) & 0xFF

        if key == ord('q'): 
            break
            
        elif key == ord(' '):
            is_playing = not is_playing
            need_seek = True
            
        elif key == ord('c'): 
            df.at[frame_idx, 'Target_Label'] = 'CHEAT_PHYSICAL'
            paint_mode = 'CHEAT_PHYSICAL'
        elif key == ord('n'): 
            df.at[frame_idx, 'Target_Label'] = 'NORMAL'
            paint_mode = 'NORMAL'
        elif key == ord('d'): 
            df.at[frame_idx, 'Target_Label'] = 'CHEAT_DIGITAL'
            paint_mode = 'CHEAT_DIGITAL'
            
        elif key == ord('s'):
            frame_idx = 0
            need_seek = True
        elif key == ord('e'):
            frame_idx = max(0, total_frames - 10)
            need_seek = True
        elif key == ord('a'):
            if sync_A_webcam_frame is not None:
                frame_idx = sync_A_webcam_frame
                need_seek = True
                print(f"Jumped to Sync Point A (Frame {frame_idx}).")
            else:
                print("Sync Point A not set yet.")

        elif key == ord(',') and not is_playing:
            frame_idx = max(0, frame_idx - 1)
            need_seek = True
        elif key == ord('.') and not is_playing:
            current_label_before_step = df.iloc[frame_idx]['Target_Label']
            frame_idx = min(total_frames - 1, frame_idx + 1)
            if current_label_before_step == paint_mode:
                df.at[frame_idx, 'Target_Label'] = paint_mode
            need_seek = True
            
        elif key == ord('[') and not calibrated and not is_playing:
            phone_offset_frames -= 1
            need_seek = True
        elif key == ord(']') and not calibrated and not is_playing:
            phone_offset_frames += 1
            need_seek = True
            
        elif key == ord('1'):
            # Set sync point A: capture webcam real time + phone video time
            webcam_real_time = real_timestamps[frame_idx]
            phone_msec = cap_phone.get(cv2.CAP_PROP_POS_MSEC)
            sync_A = (webcam_real_time, phone_msec)
            sync_A_webcam_frame = frame_idx
            print(f"Point A: Frame {frame_idx} (real t={webcam_real_time:.3f}) -> Phone {phone_msec:.0f}ms ({phone_msec/1000:.2f}s)")
            
        elif key == ord('2'):
            # Set sync point B
            webcam_real_time = real_timestamps[frame_idx]
            phone_msec = cap_phone.get(cv2.CAP_PROP_POS_MSEC)
            sync_B = (webcam_real_time, phone_msec)
            print(f"Point B: Frame {frame_idx} (real t={webcam_real_time:.3f}) -> Phone {phone_msec:.0f}ms ({phone_msec/1000:.2f}s)")
            
        elif key == ord('3'):
            if sync_A and sync_B and sync_A[0] != sync_B[0]:
                calibrated = True
                need_seek = True
                rt_A, pt_A = sync_A
                rt_B, pt_B = sync_B
                print("Time-Based Calibration Applied!")
                print(f"  Real time span: {rt_B - rt_A:.2f}s")
                print(f"  Phone time span: {(pt_B - pt_A)/1000:.2f}s")
                print(f"  Speed ratio: {(pt_B - pt_A)/1000 / (rt_B - rt_A):.6f}")
            else:
                print("Cannot calibrate. Set both points at DIFFERENT times first.")
                
        elif key == ord('0'):
            calibrated = False
            sync_A = None
            sync_B = None
            sync_A_webcam_frame = None
            need_seek = True
            print("Cleared Calibration.")

        # --- ADVANCE FRAME WHEN PLAYING ---
        if is_playing:
            if frame_idx < total_frames - 1:
                frame_idx += 1
                df.at[frame_idx, 'Target_Label'] = paint_mode
            else:
                is_playing = False
                need_seek = True

    cap_webcam.release()
    cap_phone.release()
    cv2.destroyAllWindows()
    
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

def main():
    sessions = [d for d in os.listdir(SESSIONS_DIR) 
                if os.path.isdir(os.path.join(SESSIONS_DIR, d))]
    while True:
        print("\nAvailable Sessions:")
        for i, s in enumerate(sessions):
            print(f"  [{i}] {s}")
        choice = input("Enter number (or 'q' to quit): ")
        if choice.lower() == 'q': break
        try:
            run_labeler(sessions[int(choice)])
        except (ValueError, IndexError):
            print("Invalid input.")

if __name__ == "__main__":
    main()