import os
import cv2
import mediapipe as mp
import numpy as np
import threading
import speech_recognition as sr

class GestureDrawingAppCV:
    def __init__(self):
        self.voice_energy = 0.0
        self.use_white_bg = False
        self.draw_color = (0, 0, 255)  # Red in BGR
        self.thickness = 15
        self.draw_thickness_multiplier = 1.0  # Allows dynamic brush scaling
        self.canvas_image = None
        self.last_voice_command = ""

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.85,
            min_tracking_confidence=0.85
        )
        self.segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

        camera_index = self.find_working_camera()
        self.cap = cv2.VideoCapture(camera_index)

        self.voice_thread = threading.Thread(target=self.listen_voice_commands, daemon=True)
        self.voice_thread.start()

        # Track previous smoothed points for smoothing right hand index
        self.prev_points = {"Right": None}
        # Buffer for moving average smoothing of fingertip
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

    def pick_color(self):
        import matplotlib.pyplot as plt
        import matplotlib.widgets as widgets

        fig, ax = plt.subplots()
        fig.canvas.manager.set_window_title("Pick a color")
        plt.subplots_adjust(left=0.25, bottom=0.4)

        r_slider = widgets.Slider(ax=plt.axes([0.25, 0.3, 0.65, 0.03]), label='Red', valmin=0, valmax=255, valinit=255)
        g_slider = widgets.Slider(ax=plt.axes([0.25, 0.2, 0.65, 0.03]), label='Green', valmin=0, valmax=255, valinit=0)
        b_slider = widgets.Slider(ax=plt.axes([0.25, 0.1, 0.65, 0.03]), label='Blue', valmin=0, valmax=255, valinit=0)

        def update(val):
            r, g, b = int(r_slider.val), int(g_slider.val), int(b_slider.val)
            ax.set_facecolor((r / 255, g / 255, b / 255))
            fig.canvas.draw()

        r_slider.on_changed(update)
        g_slider.on_changed(update)
        b_slider.on_changed(update)

        def on_close(event):
            self.draw_color = (
                int(b_slider.val),
                int(g_slider.val),
                int(r_slider.val)
            )

        fig.canvas.mpl_connect('close_event', on_close)
        update(None)
        plt.show()

    def simple_hand_gesture_color_picker(self, frame, lm):
        # Use index fingertip position to pick color from frame
        h, w, _ = frame.shape
        x, y = int(lm[8].x * w), int(lm[8].y * h)
        if 0 <= x < w and 0 <= y < h:
            b, g, r = frame[y, x]
            self.draw_color = (int(b), int(g), int(r))

    def listen_voice_commands(self):
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            recognizer.energy_threshold = 200
            print("Listening for voice command... (no timeout)")
            while True:
                try:
                    audio = recognizer.listen(source, phrase_time_limit=5)
                    energy = np.sqrt(np.mean(np.square(np.frombuffer(audio.get_raw_data(), np.int16))))
                    self.voice_energy = energy
                    bar_len = int(min(energy / 1000, 1.0) * 30)
                    bar = "[" + "#" * bar_len + " " * (30 - bar_len) + "]"
                    print(f"Voice Energy: {bar} ({energy:.1f})")

                    command = recognizer.recognize_google(audio).lower()
                    print(f"Voice command recognized: {command}")
                    self.last_voice_command = command

                    if "clear" in command:
                        self.canvas_image = np.zeros((480, 640, 3), dtype=np.uint8)
                    elif "red" in command:
                        self.draw_color = (0, 0, 255)
                    elif "blue" in command:
                        self.draw_color = (255, 0, 0)
                    elif "green" in command:
                        self.draw_color = (0, 255, 0)
                    elif "pink" in command:
                        self.draw_color = (203, 192, 255)
                    elif "white background" in command:
                        self.use_white_bg = True
                    elif "remove background" in command:
                        self.use_white_bg = False
                    elif "yellow" in command:
                        self.draw_color = (0, 255, 255)
                    elif "black" in command:
                        self.draw_color = (1, 1, 1)
                    elif "cyan" in command:
                        self.draw_color = (255, 255, 0)
                    elif "magenta" in command:
                        self.draw_color = (255, 0, 255)
                    elif "gray" in command or "grey" in command:
                        self.draw_color = (128, 128, 128)
                    elif "orange" in command:
                        self.draw_color = (0, 165, 255)
                    elif "purple" in command:
                        self.draw_color = (128, 0, 128)
                    elif "brown" in command:
                        self.draw_color = (19, 69, 139)
                    elif "thicker" in command or "thick" in command:
                        self.thickness = min(self.thickness + 10, 100)
                        print(f"Brush thickness increased to {self.thickness}")
                    elif "thinner" in command or "thin" in command:
                        self.thickness = max(self.thickness - 10, 5)
                        print(f"Brush thickness decreased to {self.thickness}")
                    elif "reset" in command:
                        self.canvas_image = np.zeros((480, 640, 3), dtype=np.uint8)
                        self.use_white_bg = False
                        print("Canvas and background reset")
                except sr.UnknownValueError:
                    print("Could not understand audio")
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")
                except Exception as e:
                    print(f"Voice thread error: {e}")

    def run(self):
        while True:
            try:
                # Set camera properties for faster processing and lower lag
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                while True:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Failed to read frame from camera.")
                        cv2.waitKey(1)  # Allow buffer clear
                        break

                    frame = cv2.flip(frame, 1)
                    if self.canvas_image is not None and self.canvas_image.shape[:2] != frame.shape[:2]:
                        self.canvas_image = cv2.resize(self.canvas_image, (frame.shape[1], frame.shape[0]))
                    h, w, _ = frame.shape
                    # Avoid double color conversion, use BGR2RGB only when calling process
                    # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    if self.use_white_bg:
                        seg = self.segmentation.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).segmentation_mask
                        seg = cv2.medianBlur((seg * 255).astype(np.uint8), 7)
                        seg = seg.astype(np.float32) / 255.0
                        seg = np.clip(seg, 0.1, 0.9)
                        seg_3d = np.repeat(seg[:, :, np.newaxis], 3, axis=2)
                        white_bg = np.ones_like(frame, dtype=np.uint8) * 255
                        frame = (frame.astype(np.float32) * seg_3d +
                                 white_bg.astype(np.float32) * (1 - seg_3d)).astype(np.uint8)

                    if self.canvas_image is None:
                        self.canvas_image = np.zeros_like(frame)

                    results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if results.multi_hand_landmarks and results.multi_handedness:
                        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                            label = handedness.classification[0].label
                            score = handedness.classification[0].score
                            if score < 0.85:
                                continue  # skip low-confidence handedness
                            lm = hand_landmarks.landmark
                            # Smoothing logic for right hand index fingertip using moving average buffer
                            if label == "Right" and score >= 0.85:
                                new_tip = np.array([lm[8].x * w, lm[8].y * h])
                                self.fingertip_buffer.append(new_tip)
                                if len(self.fingertip_buffer) > 5:
                                    self.fingertip_buffer.pop(0)
                                fingertip = np.mean(self.fingertip_buffer, axis=0)

                                fingers_open = [
                                    self.is_finger_extended(lm, 8, 6, 5),
                                    self.is_finger_extended(lm, 12, 10, 9),
                                    self.is_finger_extended(lm, 16, 14, 13),
                                    self.is_finger_extended(lm, 20, 18, 17),
                                    self.is_finger_extended(lm, 4, 3, 2)
                                ]
                            for idx, landmark in enumerate(lm):
                                cx, cy = int(landmark.x * w), int(landmark.y * h)
                                cv2.circle(frame, (cx, cy), 2, (0, 0, 0), -1)
                                if idx == 0:
                                    label_text = f"{label}: ({cx}, {cy})"
                                    cv2.putText(frame, label_text, (cx + 10, cy),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, lineType=cv2.LINE_AA)

                            hand_points = np.array([
                                [int(lm[i].x * w), int(lm[i].y * h)]
                                for i in range(21)
                            ], dtype=np.int32)
                            hull = cv2.convexHull(hand_points)
                            palm_mask = np.zeros((h, w), dtype=np.uint8)
                            cv2.fillConvexPoly(palm_mask, hull, 255)
                            palm_mask = cv2.GaussianBlur(palm_mask, (self.thickness//2*2+1, self.thickness//2*2+1), 0)

                            if label == "Right" and score >= 0.85:
                                if sum(fingers_open) > 0:
                                    if self.prev_points["Right"] is not None:
                                        p1 = self.prev_points["Right"]
                                        p2 = fingertip
                                        distance = np.linalg.norm(p2 - p1)
                                        steps = int(distance // 1)
                                        for i in range(steps + 1):
                                            t = i / max(steps, 1)
                                            interp_point = (1 - t) * p1 + t * p2
                                            cv2.circle(self.canvas_image, tuple(np.int32(interp_point)), int(self.thickness * self.draw_thickness_multiplier), self.draw_color, -1)
                                    self.prev_points["Right"] = fingertip
                                else:
                                    self.prev_points["Right"] = None
                                    self.fingertip_buffer = []

                            elif label == "Left":
                                fingers_open = [
                                    self.is_finger_extended(lm, 8, 6, 5),
                                    self.is_finger_extended(lm, 12, 10, 9),
                                    self.is_finger_extended(lm, 16, 14, 13),
                                    self.is_finger_extended(lm, 20, 18, 17),
                                    self.is_finger_extended(lm, 4, 3, 2)
                                ]
                                palm_open = sum(fingers_open) >= 4
                                if palm_open:
                                    erase_mask = cv2.merge([palm_mask] * 3)
                                    if erase_mask.shape != self.canvas_image.shape:
                                        erase_mask = cv2.resize(erase_mask, (self.canvas_image.shape[1], self.canvas_image.shape[0]))
                                    self.canvas_image = cv2.bitwise_and(self.canvas_image, cv2.bitwise_not(erase_mask))

                    try:
                        mask = cv2.cvtColor(self.canvas_image, cv2.COLOR_BGR2GRAY)
                        _, bin_mask = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)
                        inv_mask = cv2.bitwise_not(bin_mask)

                        if inv_mask.dtype != np.uint8:
                            inv_mask = inv_mask.astype(np.uint8)
                        if bin_mask.dtype != np.uint8:
                            bin_mask = bin_mask.astype(np.uint8)

                        if inv_mask.shape[:2] == frame.shape[:2] and bin_mask.shape[:2] == frame.shape[:2]:
                            bg = cv2.bitwise_and(frame, frame, mask=inv_mask)
                            fg = cv2.bitwise_and(self.canvas_image, self.canvas_image, mask=bin_mask)
                            output = cv2.add(bg, fg)
                        else:
                            print("Warning: mask shape mismatch, skipping blending.")
                            output = frame.copy()
                    except Exception as e:
                        print(f"Error in mask blending: {e}")
                        output = frame.copy()

                    cv2.putText(output, "[q]=Quit  [w]=ToggleBG  [c]=Clear ", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, lineType=cv2.LINE_AA)

                    voice_bar_len = int(min(self.voice_energy / 1000, 1.0) * w)
                    cv2.rectangle(output, (10, h - 30), (10 + voice_bar_len, h - 10), (0, 255, 0), -1)
                    cv2.putText(output, "Voice", (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)

                    if self.last_voice_command:
                        cv2.putText(output, f"Command: {self.last_voice_command}", (w - 250, h - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)

                    cv2.namedWindow("Gesture Drawing", cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty("Gesture Drawing", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    cv2.imshow("Gesture Drawing", output)

                    key = cv2.waitKey(5) & 0xFF

                    if key == ord('q'):
                        break
                    elif key == ord('w'):
                        self.use_white_bg = not self.use_white_bg
                    elif key == ord('c'):
                        self.canvas_image = np.zeros_like(frame)
                    elif key == ord('p'):
                        self.pick_color()
                    elif key == ord('h'):
                        self.canvas_image = np.zeros_like(frame)
                        self.use_white_bg = False
                        self.pick_color()

                self.cap.release()
                cv2.destroyAllWindows()
                break
            except Exception as e:
                print(f"Error occurred: {e}, restarting...")
                self.cap.release()
                cv2.destroyAllWindows()
                self.cap = cv2.VideoCapture(0)

if __name__ == "__main__":
    app = GestureDrawingAppCV()
    app.run()