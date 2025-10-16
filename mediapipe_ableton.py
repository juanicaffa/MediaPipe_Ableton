import cv2
import mediapipe as mp
from pythonosc import udp_client
import time
import math

# ========= CONFIGURACIÓN OSC =========
IP = "127.0.0.1"   # localhost
PUERTO = 4567      # mismo puerto que en Ableton
client = udp_client.SimpleUDPClient(IP, PUERTO)

def scale(val, in_min, in_max, out_min, out_max):
    """Escala un valor de un rango a otro con límites"""
    return max(min((val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min, out_max), out_min)

# ========= CONFIGURACIÓN MEDIAPIPE =========
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,  # Ahora soportamos dos manos
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ========= OPEN CV CAMARA =========
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW evita problemas en Windows

# Suavizado independiente para cada mano
smooth = {
    0: {"x": 0, "dist": 0},  # Mano 1
    1: {"x": 0, "dist": 0}   # Mano 2
}
alpha = 0.25  # Suavizado (0.1 = más suave, 0.5 = más rápido)

# ========= FPS =========
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ No se pudo capturar la cámara")
        break
    cv2.imshow("Frame crudo", frame)  # <--- Esto muestra la imagen antes de MediaPipe
    
    # Convertir BGR -> RGB para MediaPipe
    # Para MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # OJO: el frame que mostramos debe seguir en BGR
    cv2.imshow("MediaPipe OSC - 2 Hands", frame)

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if i > 1:
                break  # Solo procesar 2 manos

            # Dibujar landmarks en la cámara
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # ===== Coordenadas dedo índice =====
            x = hand_landmarks.landmark[8].x  # Dedo índice (horizontal)

            # ===== Distancia pulgar-índice =====
            thumb = hand_landmarks.landmark[4]
            index = hand_landmarks.landmark[8]
            dist = math.sqrt((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2)

            # === Suavizado ===
            smooth[i]["x"] = smooth[i]["x"] * (1 - alpha) + x * alpha
            smooth[i]["dist"] = smooth[i]["dist"] * (1 - alpha) + dist * alpha

            # Escalar valores a rango 0-127
            x_scaled = scale(smooth[i]["x"], 0.2, 0.8, 0, 127)
            dist_scaled = scale(smooth[i]["dist"], 0.02, 0.15, 0, 127)

            # ===== Enviar OSC =====
            if i == 0:  # Mano 1
                client.send_message("/hand/distance", dist_scaled)
                client.send_message("/hand/press1", x_scaled)
            elif i == 1:  # Mano 2
                client.send_message("/hand/press2", dist_scaled)
                client.send_message("/hand/rotation", x_scaled)

            # Mostrar datos en pantalla
            cv2.putText(frame,
                        f"Hand {i+1} | X:{x_scaled:.0f} Dist:{dist_scaled:.0f}",
                        (10, 30 + i * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # ===== FPS =====
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Mostrar cámara
    cv2.imshow("MediaPipe OSC - 2 Hands", frame)

    # Salir con ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
