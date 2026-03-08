import cv2
from ultralytics import YOLO
import pygame
import time
import threading
import winsound

# Initialize pygame mixer for audio
pygame.mixer.init()

# Try loading sounds, create dummy if not exists
try:
    warning_sound = pygame.mixer.Sound('assets/warning.wav')
    success_sound = pygame.mixer.Sound('assets/success.wav')
except Exception as e:
    print(f"Warning: Could not load sound files from assets/. Error: {e}")
    warning_sound = None
    success_sound = None

def play_sound(sound_obj):
    if sound_obj:
        sound_obj.play()

def play_warning_beep():
    # Play multi-beep to indicate missing PPE
    for _ in range(3):
        winsound.Beep(2500, 500)
        time.sleep(0.1)

# States
STATE_WAITING = "WAITING"
STATE_SCANNING = "SCANNING"
STATE_PASSED = "PASSED"
STATE_ALARM = "ALARM"

# Load YOLO model (Using yolov8n as a placeholder. In a real scenario, use a specific PPE trained model e.g., best.pt)
model = YOLO('yolov8n.pt') 

# Define classes. We will assume the following classes exist in your custom PPE model
# For a generic yolov8n, 'person' is class 0, but it won't detect helmet/vest out of the box.
# For this script we will assume we have a model that detects: 
# 0: person, 1: helmet, 2: vest, 3: gloves, 4: shoes
CLASS_PERSON = 0
CLASS_HELMET = 1
CLASS_VEST = 2
CLASS_GLOVES = 3
CLASS_SHOES = 4

# Map class names for display
CLASS_NAMES = {
    CLASS_PERSON: "Person",
    CLASS_HELMET: "Helmet",
    CLASS_VEST: "Vest",
    CLASS_GLOVES: "Gloves",
    CLASS_SHOES: "Shoes"
}

def main():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or video path
    
    current_state = STATE_WAITING
    scan_start_time = 0
    SCAN_DURATION = 10 # seconds
    
    # Track detected items during the scanning window
    items_detected_in_window = {
        'helmet': False,
        'vest': False,
        'gloves': False,
        'shoes': False
    }

    print("Starting PPE Detection System...")
    print("Press SPACEBAR to 'Scan Next' when in ALARM state.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        results = model(frame, verbose=False)
        
        # Parse detections
        person_detected = False
        current_frame_items = {
            'helmet': False,
            'vest': False,
            'gloves': False,
            'shoes': False
        }

        # Draw bounding boxes and track detections
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if conf > 0.5: # Confidence threshold
                    if cls_id == CLASS_PERSON:
                        person_detected = True
                    elif cls_id == CLASS_HELMET:
                        current_frame_items['helmet'] = True
                        items_detected_in_window['helmet'] = True
                    elif cls_id == CLASS_VEST:
                        current_frame_items['vest'] = True
                        items_detected_in_window['vest'] = True
                    elif cls_id == CLASS_GLOVES:
                        current_frame_items['gloves'] = True
                        items_detected_in_window['gloves'] = True
                    elif cls_id == CLASS_SHOES:
                        current_frame_items['shoes'] = True
                        items_detected_in_window['shoes'] = True
                    
                    # Draw box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = f"{CLASS_NAMES.get(cls_id, 'Unknown')} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # State Machine Logic
        if current_state == STATE_WAITING:
            if person_detected:
                # Transition to SCANNING
                current_state = STATE_SCANNING
                scan_start_time = time.time()
                # Reset tracking window
                items_detected_in_window = {k: False for k in items_detected_in_window}

        elif current_state == STATE_SCANNING:
            elapsed_time = time.time() - scan_start_time
            
            # Check if all items are found
            if all(items_detected_in_window.values()):
                current_state = STATE_PASSED
                threading.Thread(target=play_sound, args=(success_sound,), daemon=True).start()
                passed_start_time = time.time()
                
            # Check timeout
            elif elapsed_time >= SCAN_DURATION:
                current_state = STATE_ALARM
                threading.Thread(target=play_sound, args=(warning_sound,), daemon=True).start()
                threading.Thread(target=play_warning_beep, daemon=True).start()

        elif current_state == STATE_PASSED:
            # Display passed and reset after 3 seconds
            if time.time() - passed_start_time > 3:
                current_state = STATE_WAITING

        elif current_state == STATE_ALARM:
            # Wait for manual click (Spacebar) handled in key event
            pass

        # Overlay UI based on State
        overlay = frame.copy()
        if current_state == STATE_WAITING:
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 50), (128, 128, 128), -1)
            cv2.putText(overlay, "WAITING FOR PERSON", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        elif current_state == STATE_SCANNING:
            elapsed = time.time() - scan_start_time
            remaining = max(0, int(SCAN_DURATION - elapsed))
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 165, 255), -1)
            cv2.putText(overlay, f"SCANNING... Time Left: {remaining}s", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show required items status
            status_text = f"H: {'[X]' if items_detected_in_window['helmet'] else '[ ]'} " \
                          f"V: {'[X]' if items_detected_in_window['vest'] else '[ ]'} " \
                          f"G: {'[X]' if items_detected_in_window['gloves'] else '[ ]'} " \
                          f"S: {'[X]' if items_detected_in_window['shoes'] else '[ ]'}"
            cv2.putText(overlay, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
        elif current_state == STATE_PASSED:
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 50), (0, 255, 0), -1)
            cv2.putText(overlay, "ALL PPE DETECTED! (GREEN LIGHT)", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        elif current_state == STATE_ALARM:
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (40, 40, 40), -1) # Dark background
            # Flash effect
            if int(time.time() * 2) % 2 == 0:
                cv2.circle(overlay, (frame.shape[1] - 60, 50), 35, (0, 0, 255), -1) # Red light ON
                cv2.putText(overlay, "PPE MISSING! (RED LIGHT)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                cv2.circle(overlay, (frame.shape[1] - 60, 50), 35, (0, 0, 50), -1) # Red light OFF
                cv2.putText(overlay, "PPE MISSING! (RED LIGHT)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 3)
            cv2.putText(overlay, "PRESS SPACE TO SCAN NEXT", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Blend overlay
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.imshow('PPE Detection System', frame)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32: # Spacebar
            if current_state == STATE_ALARM:
                current_state = STATE_WAITING
                pygame.mixer.stop() # Stop alarm sound

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
