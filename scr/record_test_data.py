import cv2
import os
import time

def record_video(filename, duration=5, fps=15):
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    print(f"Recording {filename} for {duration} seconds...")
    print("Press 'q' to stop early.")
    
    start_time = time.time()
    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        if not ret: break
        
        out.write(frame)
        
        cv2.putText(frame, f"Recording: {int(time.time() - start_time)}/{duration}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Recording', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Saved {filename}")

if __name__ == "__main__":
    os.makedirs("data/test_videos", exist_ok=True)
    
    scenarios = [
        ("normal_authorized.mp4", "Normal driving (authorized user)"),
        ("unauthorized.mp4", "Unauthorized person sitting in front of camera"),
        ("occluded.mp4", "Camera covered by hand or object")
    ]
    
    print("=== Test Data Recording Script ===")
    print("This will record 5-second clips for system evaluation.")
    
    for filename, desc in scenarios:
        input(f"\nPrepare for: {desc}\nPress ENTER to start recording {filename}...")
        record_video(f"data/test_videos/{filename}")
        
    print("\nAll done! You can now run evaluate_system.py.")
