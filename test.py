import os
import cv2
import mediapipe as mp
import numpy as np
import threading
import speech_recognition as sr
from collections import deque
import time

class NeonGestureDrawingApp:
    def __init__(self):
        # Voice detection improvements
        self.voice_energy = 0.0
        self.command_queue = deque(maxlen=5)
        self.last_command_time = time.time()
        self.is_listening = True
        self.voice_status = "Listening..."
        
        # Visual settings
        self.use_white_bg = False
        self.neon_mode = True
        self.draw_color = (139, 73, 105)  # Soft Rose (BGR)
        self.thickness = 10  # Initial brush size 10px
        self.draw_thickness_multiplier = 1.0
        self.canvas_image = None
        self.last_voice_command = ""
        
        # Color palette with subdued colors
        self.neon_colors = {
            "pink": (139, 73, 105),     # Soft Rose
            "cyan": (209, 178, 128),    # Soft Cyan
            "green": (76, 153, 76),     # Forest Green
            "yellow": (102, 204, 204),  # Muted Yellow
            "orange": (71, 117, 185),   # Soft Orange
            "purple": (147, 61, 113),   # Mauve Purple
            "blue": (178, 104, 71),     # Navy Blue
            "red": (71, 71, 191),       # Deep Red
        }
        
        # Interactive features
        self.show_tutorial = True
        self.tutorial_timer = 0
        self.gesture_detected = ""
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Particle effects for neon mode
        self.particles = []
        self.max_particles = 50

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.85,
            min_tracking_confidence=0.85
        )
        self.segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

        camera_index = self.find_working_camera()
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)

        # Improved voice thread
        self.voice_thread = threading.Thread(target=self.listen_voice_commands_continuous, daemon=True)
        self.voice_thread.start()

        # Track previous smoothed points
        self.prev_points = {"Right": None}
        self.fingertip_buffer = []

    def find_working_camera(self):
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    cap.release()
                    return i
            cap.release()
        raise RuntimeError("No working camera found")

    def is_finger_extended(self, lm, tip_id, pip_id, mcp_id):
        tip = np.array([lm[tip_id].x, lm[tip_id].y])
        pip = np.array([lm[pip_id].x, lm[pip_id].y])
        mcp = np.array([lm[mcp_id].x, lm[mcp_id].y])

        v1 = pip - mcp
        v2 = tip - pip
        angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
        angle_deg = np.degrees(angle)

        return angle_deg < 45

    def add_particle(self, pos, color):
        """Add particle effect for neon drawing"""
        if len(self.particles) < self.max_particles:
            self.particles.append({
                'pos': np.array(pos, dtype=np.float32),
                'vel': np.random.randn(2) * 2,
                'life': 20,
                'color': color
            })

    def update_particles(self):
        """Update and remove dead particles"""
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def draw_particles(self, frame):
        """Draw particle effects"""
        for p in self.particles:
            pos = tuple(p['pos'].astype(int))
            alpha = p['life'] / 20.0
            color = tuple(int(c * alpha) for c in p['color'])
            cv2.circle(frame, pos, 3, color, -1)

    def listen_voice_commands_continuous(self):
        """Improved continuous voice recognition with better interval handling"""
        recognizer = sr.Recognizer()
        # Adjust these parameters for better detection
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        recognizer.dynamic_energy_adjustment_damping = 0.15
        recognizer.dynamic_energy_ratio = 1.5
        recognizer.pause_threshold = 0.5  # Reduced pause threshold
        recognizer.phrase_threshold = 0.3
        recognizer.non_speaking_duration = 0.3
        
        mic = sr.Microphone()
        
        with mic as source:
            print("Calibrating microphone for ambient noise... Please wait.")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            print(f"Energy threshold set to: {recognizer.energy_threshold}")
            print("Voice recognition ready! Listening continuously...")
            
            while self.is_listening:
                try:
                    # Listen with shorter timeout for better responsiveness
                    audio = recognizer.listen(source, timeout=1, phrase_time_limit=3)
                    
                    # Calculate audio energy
                    audio_data = np.frombuffer(audio.get_raw_data(), np.int16)
                    energy = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))
                    self.voice_energy = energy
                    
                    # Visual feedback
                    bar_len = int(min(energy / 1000, 1.0) * 30)
                    bar = "[" + "#" * bar_len + " " * (30 - bar_len) + "]"
                    print(f"Voice Energy: {bar} ({energy:.1f})")
                    
                    self.voice_status = "Processing..."
                    
                    # Try to recognize with faster processing
                    try:
                        command = recognizer.recognize_google(audio, language='en-US').lower()
                        print(f"✓ Recognized: '{command}'")
                        self.process_voice_command(command)
                        self.voice_status = f"✓ {command}"
                        self.last_command_time = time.time()
                        
                    except sr.UnknownValueError:
                        print("• Could not understand audio")
                        self.voice_status = "Not understood"
                        
                except sr.WaitTimeoutError:
                    # Timeout is normal, just continue listening
                    self.voice_status = "Listening..."
                    continue
                    
                except sr.RequestError as e:
                    print(f"✗ Service error: {e}")
                    self.voice_status = "Service error"
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"✗ Error: {e}")
                    time.sleep(0.1)

    def process_voice_command(self, command):
        """Process voice commands with expanded vocabulary"""
        self.last_voice_command = command
        self.command_queue.append(command)
        
        # Color commands
        for color_name, color_value in self.neon_colors.items():
            if color_name in command:
                self.draw_color = color_value
                print(f"→ Color changed to {color_name}")
                return
        
        # Canvas commands
        if "clear" in command or "clean" in command or "erase all" in command:
            self.canvas_image = np.zeros((480, 640, 3), dtype=np.uint8)
            print("→ Canvas cleared")
            
        elif "background" in command:
            if "white" in command or "add" in command:
                self.use_white_bg = True
                print("→ White background enabled")
            elif "remove" in command or "no" in command:
                self.use_white_bg = False
                print("→ Background removed")
        
        # Thickness commands
        elif "thick" in command or "bigger" in command or "large" in command:
            self.thickness = min(self.thickness + 10, 100)
            print(f"→ Thickness: {self.thickness}")
            
        elif "thin" in command or "smaller" in command or "small" in command:
            self.thickness = max(self.thickness - 10, 5)
            print(f"→ Thickness: {self.thickness}")
        
        # Mode commands
        elif "neon" in command:
            if "off" in command or "disable" in command:
                self.neon_mode = False
                print("→ Neon mode OFF")
            else:
                self.neon_mode = True
                print("→ Neon mode ON")
        
        elif "reset" in command or "restart" in command:
            self.canvas_image = np.zeros((480, 640, 3), dtype=np.uint8)
            self.use_white_bg = False
            self.draw_color = (255, 0, 255)
            self.thickness = 10
            print("→ Everything reset")
        
        elif "help" in command or "tutorial" in command:
            self.show_tutorial = True
            self.tutorial_timer = time.time()
            print("→ Showing tutorial")

    def draw_neon_glow(self, canvas, point, radius, color):
        """Create smooth brush strokes with anti-aliasing and soft edges"""
        x, y = point
        
        # Create a temporary overlay for the brush stroke
        overlay = np.zeros_like(canvas, dtype=np.float32)
        
        # Draw multiple layers with gaussian blur for smooth edges
        core_radius = max(int(radius * 0.7), 1)
        glow_radius = radius
        
        # Draw the core
        cv2.circle(overlay, (x, y), core_radius, (1.0, 1.0, 1.0), -1, lineType=cv2.LINE_AA)
        
        # Apply gaussian blur for smooth edges
        sigma = radius * 0.3
        kernel_size = max(int(sigma * 4) | 1, 3)  # Must be odd number
        overlay = cv2.GaussianBlur(overlay, (kernel_size, kernel_size), sigma)
        
        # Convert color to float32 for smooth blending
        color_f32 = np.array(color, dtype=np.float32) / 255.0
        
        # Apply color with smooth transition
        for c in range(3):
            overlay[:,:,c] *= color_f32[c]
        
        # Blend with existing canvas
        cv2.addWeighted(canvas, 1.0, (overlay * 255).astype(np.uint8), 1.0, 0, canvas)

    def draw_tutorial_overlay(self, frame):
        """Interactive tutorial overlay"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Semi-transparent background
        cv2.rectangle(overlay, (w//2 - 250, 50), (w//2 + 250, 400), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title with neon effect
        title = "AI GESTURE DRAWING - TUTORIAL"
        cv2.putText(frame, title, (w//2 - 230, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 3, lineType=cv2.LINE_AA)
        cv2.putText(frame, title, (w//2 - 230, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        
        # Instructions
        instructions = [
            ("RIGHT HAND:", (255, 0, 255)),
            ("  - Open fingers = DRAW", (255, 255, 255)),
            ("  - Close fist = STOP drawing", (255, 255, 255)),
            ("", (0, 0, 0)),
            ("LEFT HAND:", (0, 255, 255)),
            ("  - Open palm = ERASE", (255, 255, 255)),
            ("", (0, 0, 0)),
            ("VOICE COMMANDS:", (0, 255, 0)),
            ("  'pink/cyan/green/yellow'", (255, 255, 255)),
            ("  'clear' 'thicker' 'thinner'", (255, 255, 255)),
            ("  'white background'", (255, 255, 255)),
            ("", (0, 0, 0)),
            ("KEYBOARD:", (255, 165, 0)),
            ("  Q=Quit  C=Clear  W=Background", (255, 255, 255)),
            ("  T=Tutorial  N=Neon Mode", (255, 255, 255)),
        ]
        
        y_offset = 130
        for text, color in instructions:
            cv2.putText(frame, text, (w//2 - 220, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, lineType=cv2.LINE_AA)
            y_offset += 25
        
        # Auto-hide after 10 seconds
        if time.time() - self.tutorial_timer > 10:
            self.show_tutorial = False

    def draw_interactive_ui(self, frame):
        """Enhanced UI with neon styling"""
        h, w = frame.shape[:2]
        
        # Company branding with neon glow
        cv2.putText(frame, "TECHNOLOGY GARAGE - TRICHY", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, lineType=cv2.LINE_AA)
        cv2.putText(frame, "TECHNOLOGY GARAGE - TRICHY", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        
        # FPS counter
        cv2.putText(frame, f"FPS: {int(self.fps)}", (w - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 55, 0), 2, lineType=cv2.LINE_AA)
        
        # Voice energy bar with neon effect
        voice_bar_len = int(min(self.voice_energy / 1000, 1.0) * 200)
        cv2.rectangle(frame, (10, h - 60), (210, h - 40), (50, 50, 50), -1)
        
        if voice_bar_len > 0:
            # Gradient effect for voice bar
            for i in range(voice_bar_len):
                color_intensity = int(255 * (i / 200))
                cv2.line(frame, (10 + i, h - 60), (10 + i, h - 40), 
                        (0, 255 - color_intensity, color_intensity), 2)
        
        cv2.putText(frame, "VOICE", (15, h - 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        cv2.putText(frame, self.voice_status, (220, h - 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, lineType=cv2.LINE_AA)
        
        # Last command with neon background
        if self.last_voice_command and time.time() - self.last_command_time < 3:
            cmd_text = f"Command: {self.last_voice_command}"
            text_size = cv2.getTextSize(cmd_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (w - text_size[0] - 20, h - 35), 
                         (w - 5, h - 5), (50, 50, 50), -1)
            cv2.putText(frame, cmd_text, (w - text_size[0] - 15, h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, lineType=cv2.LINE_AA)
        
        # Current color indicator
        cv2.circle(frame, (w - 40, 70), 25, (0, 0, 0), -1)
        cv2.circle(frame, (w - 40, 70), 20, self.draw_color, -1)
        cv2.circle(frame, (w - 40, 70), 20, (255, 255, 255), 2)
        
        # Thickness indicator
        cv2.putText(frame, f"Brush: {self.thickness}px", (w - 120, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        
        # Gesture detected
        if self.gesture_detected:
            cv2.putText(frame, f"Gesture: {self.gesture_detected}", (10, h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, lineType=cv2.LINE_AA)
        
        # Neon mode indicator
        if self.neon_mode:
            cv2.putText(frame, "Ai Art", (w//2 - 60, h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, lineType=cv2.LINE_AA)

    def run(self):
        print("\n" + "="*60)
        print("NEON GESTURE DRAWING APP - STARTING")
        print("="*60)
        print("\n✓ Camera initialized")
        print("✓ Hand tracking ready")
        print("✓ Voice recognition active")
        print("\nPress 'T' to show tutorial\n")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # FPS calculation
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                self.fps = 30 / (time.time() - self.start_time)
                self.start_time = time.time()

            # Initialize canvas
            if self.canvas_image is None:
                self.canvas_image = np.zeros_like(frame)

            # Background segmentation
            if self.use_white_bg:
                seg = self.segmentation.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).segmentation_mask
                seg = cv2.medianBlur((seg * 255).astype(np.uint8), 7)
                seg = seg.astype(np.float32) / 255.0
                seg = np.clip(seg, 0.1, 0.9)
                seg_3d = np.repeat(seg[:, :, np.newaxis], 3, axis=2)
                white_bg = np.ones_like(frame, dtype=np.uint8) * 255
                frame = (frame.astype(np.float32) * seg_3d +
                        white_bg.astype(np.float32) * (1 - seg_3d)).astype(np.uint8)

            # Hand detection and processing
            results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.gesture_detected = ""
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label
                    score = handedness.classification[0].score
                    
                    if score < 0.85:
                        continue

                    # Create a list of finger states for the current hand
                    lm = hand_landmarks.landmark
                    hand_points = np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in range(21)], dtype=np.int32)
                    fingers_open = [
                        self.is_finger_extended(lm, 8, 6, 5),   # Index
                        self.is_finger_extended(lm, 12, 10, 9), # Middle
                        self.is_finger_extended(lm, 16, 14, 13),# Ring
                        self.is_finger_extended(lm, 20, 18, 17),# Pinky
                        self.is_finger_extended(lm, 4, 3, 2)    # Thumb
                    ]
                    
                    # Draw hand landmarks with neon effect
                    for idx, landmark in enumerate(lm):
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        if self.neon_mode:
                            cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)
                            cv2.circle(frame, (cx, cy), 2, self.draw_color, -1)
                        else:
                            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
                    
                    fingers_open = [
                        self.is_finger_extended(lm, 8, 6, 5),
                        self.is_finger_extended(lm, 12, 10, 9),
                        self.is_finger_extended(lm, 16, 14, 13),
                        self.is_finger_extended(lm, 20, 18, 17),
                        self.is_finger_extended(lm, 4, 3, 2)
                    ]
                    
                    # RIGHT HAND - Drawing
                    if label == "Right":
                        fingertip = np.array([lm[8].x * w, lm[8].y * h])
                        self.fingertip_buffer.append(fingertip)
                        if len(self.fingertip_buffer) > 5:
                            self.fingertip_buffer.pop(0)
                        smooth_tip = np.mean(self.fingertip_buffer, axis=0)
                        
                        # Check if index and middle fingers are extended (minimum 2 fingers)
                        drawing_fingers = sum(fingers_open[:2])  # Check first two fingers
                        if drawing_fingers >= 2:  # Only draw if at least 2 fingers are up
                            self.gesture_detected = "Drawing"
                            
                            if self.prev_points["Right"] is not None:
                                p1 = self.prev_points["Right"]
                                p2 = smooth_tip
                                distance = np.linalg.norm(p2 - p1)
                                steps = max(int(distance // 2), 1)
                                
                                for i in range(steps + 1):
                                    t = i / steps
                                    interp_point = (1 - t) * p1 + t * p2
                                    point_int = tuple(np.int32(interp_point))
                                    
                                    if self.neon_mode:
                                        self.draw_neon_glow(self.canvas_image, point_int, 
                                                          int(self.thickness * self.draw_thickness_multiplier),
                                                          self.draw_color)
                                        # Add particles
                                        if i % 2 == 0:
                                            self.add_particle(point_int, self.draw_color)
                                    else:
                                        cv2.circle(self.canvas_image, point_int,
                                                 int(self.thickness * self.draw_thickness_multiplier),
                                                 self.draw_color, -1)
                            
                            self.prev_points["Right"] = smooth_tip
                        else:
                            self.gesture_detected = "Hand Closed"
                            self.prev_points["Right"] = None
                            self.fingertip_buffer = []  # Clear the buffer when hand is closed

                    # LEFT HAND - Erasing
                    elif label == "Left":
                        palm_open = sum(fingers_open) >= 4  # Check if most fingers are open
                        if palm_open:
                            self.gesture_detected = "Erasing"
                            # Create eraser mask using hand contour
                            hull = cv2.convexHull(hand_points)
                            eraser_mask = np.zeros((h, w), dtype=np.uint8)
                            cv2.fillConvexPoly(eraser_mask, hull, 255)
                            
                            # Smooth the eraser edges
                            eraser_mask = cv2.GaussianBlur(eraser_mask, (31, 31), 10)
                            eraser_mask = cv2.threshold(eraser_mask, 50, 255, cv2.THRESH_BINARY)[1]
                            eraser_mask = cv2.merge([eraser_mask] * 3)
                            
                            # Apply the eraser mask to the canvas
                            self.canvas_image = cv2.bitwise_and(self.canvas_image, cv2.bitwise_not(eraser_mask))
                        else:
                            self.gesture_detected = "Left Hand Ready"

            # Update and draw particles
            self.update_particles()
            self.draw_particles(frame)

            # Overlay canvas on frame
            mask = cv2.cvtColor(self.canvas_image, cv2.COLOR_BGR2GRAY)
            _, bin_mask = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)
            inv_mask = cv2.bitwise_not(bin_mask)
            if inv_mask.shape[:2] == frame.shape[:2] and bin_mask.shape[:2] == frame.shape[:2]:
                bg = cv2.bitwise_and(frame, frame, mask=inv_mask)
                fg = cv2.bitwise_and(self.canvas_image, self.canvas_image, mask=bin_mask)
                output = cv2.add(bg, fg)
            else:
                output = frame.copy()

            # Draw overlays
            self.draw_interactive_ui(output)
            if self.show_tutorial:
                self.draw_tutorial_overlay(output)

            cv2.namedWindow("Neon Gesture Drawing", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Neon Gesture Drawing", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Neon Gesture Drawing", output)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.canvas_image = np.zeros_like(frame)
            elif key == ord('w'):
                self.use_white_bg = not self.use_white_bg
            elif key == ord('t'):
                self.show_tutorial = True
                self.tutorial_timer = time.time()
            elif key == ord('n'):
                self.neon_mode = not self.neon_mode

        self.cap.release()
        cv2.destroyAllWindows()

# Main entry point
if __name__ == "__main__":
    app = NeonGestureDrawingApp()
    app.run()


