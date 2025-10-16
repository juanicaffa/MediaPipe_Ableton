import cv2
import mediapipe as mp
from pythonosc import udp_client
import time
import math

# OSC
OSC_IP = "127.0.0.1"
OSC_PORT = 4567
client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)

def scale(value, in_min, in_max, out_min, out_max):
    if in_min == in_max:
        return out_min
    scaled = (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    return max(min(scaled, out_max), out_min)

def distance(p1, p2):
    return math.dist((p1.x, p1.y), (p2.x, p2.y))

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara.")

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Guardar info por mano usando 'Left' / 'Right' como claves
    hand_map = {}

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # 'Left' o 'Right'
            thumb = hand_landmarks.landmark[4]
            index = hand_landmarks.landmark[8]
            wrist = hand_landmarks.landmark[0]
            middle = hand_landmarks.landmark[12]

            pinch_dist = distance(thumb, index)
            # Punto de interacción: punto medio entre pulgar e índice
            mid_x = (thumb.x + index.x) / 2.0
            mid_y = (thumb.y + index.y) / 2.0

            hand_map[label] = {
                "landmarks": hand_landmarks,
                "pinch_dist": pinch_dist,
                "mid": (mid_x, mid_y),
                "wrist": wrist,
                "middle": middle,
            }

            # Dibujar landmarks y puntos
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cx_t, cy_t = int(thumb.x * frame.shape[1]), int(thumb.y * frame.shape[0])
            cx_i, cy_i = int(index.x * frame.shape[1]), int(index.y * frame.shape[0])
            cx_m, cy_m = int(mid_x * frame.shape[1]), int(mid_y * frame.shape[0])
            cv2.circle(frame, (cx_t, cy_t), 6, (0, 0, 255), -1)
            cv2.circle(frame, (cx_i, cy_i), 6, (0, 255, 0), -1)
            cv2.circle(frame, (cx_m, cy_m), 6, (255, 0, 0), -1)

    # Solo cuando haya ambas manos 'Left' y 'Right'
    if "Left" in hand_map and "Right" in hand_map:
        left = hand_map["Left"]
        right = hand_map["Right"]

        # 1) DISTANCIA MUTUA entre los puntos medios (midpoints) de cada mano
        mutual_dist = math.dist(left["mid"], right["mid"])
        # Ajusta in_min/in_max según tu cámara y distancia de manos
        scaled_distance = scale(mutual_dist, 0.02, 0.6, 0, 1)
        client.send_message("/hand/distance", scaled_distance)

        # 2) APRETAR (pinch) por mano (1 = apretado, 0 = suelto)
        press_threshold = 0.04
        press_left = 1 if left["pinch_dist"] < press_threshold else 0
        press_right = 1 if right["pinch_dist"] < press_threshold else 0
        client.send_message("/hand/press1", press_left)  # mano izquierda
        client.send_message("/hand/press2", press_right)  # mano derecha

        # 3) ROTACIÓN (ej: mano derecha)
       
        dx = right["middle"].x - right["wrist"].x
        dy = right["middle"].y - right["wrist"].y
        angle = math.atan2(dy, dx)  # ángulo en radianes

        # Invertimos el valor para que mirando arriba sea 1 y mirando abajo 0
        rotation_scaled = 1 - scale(angle, -math.pi/2, math.pi/2, 0, 1)
        client.send_message("/hand/rotation", rotation_scaled)


        # Mostrar en pantalla
        cv2.putText(frame, f"Distance: {scaled_distance:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Press L:{press_left}  R:{press_right}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame, f"Rotation: {rotation_scaled:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    else:
        # Opcional: enviar ceros si querés valores siempre presentes
        # client.send_message("/hand/distance", 0)
        # client.send_message("/hand/press1", 0)
        # client.send_message("/hand/press2", 0)
        # client.send_message("/hand/rotation", 0)
        cv2.putText(frame, "Se requieren ambas manos (Left + Right).", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    cv2.imshow("MediaPipe OSC - mutual distance", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
