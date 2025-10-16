import cv2
import mediapipe as mp
from pythonosc import udp_client
import time
import math

# ==========================================================
# CONFIGURACIÓN OSC
# ==========================================================
OSC_IP = "127.0.0.1"     # Dirección IP destino (localhost)
OSC_PORT = 4567          # Puerto OSC (debe coincidir con Ableton o bridge)
client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)


# ==========================================================
# FUNCIONES AUXILIARES
# ==========================================================
def scale(value, in_min, in_max, out_min, out_max):
    """
    Escala un valor de un rango de entrada a un rango de salida,
    con saturación en los límites.
    """
    if in_min == in_max:
        return out_min
    scaled = (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    return max(min(scaled, out_max), out_min)


# ==========================================================
# CONFIGURACIÓN MEDIAPIPE
# ==========================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# ==========================================================
# CONFIGURACIÓN OPEN CV
# ==========================================================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW evita bloqueos en Windows

if not cap.isOpened():
    raise RuntimeError("❌ No se pudo abrir la cámara. Verifica que esté conectada o en uso.")

# Suavizado independiente por mano
smooth = {0: {"x": 0, "dist": 0}, 1: {"x": 0, "dist": 0}}
alpha = 0.25  # Factor de suavizado (0.1 = más lento, 0.5 = más rápido)

prev_time = 0  # Para FPS


# ==========================================================
# LOOP PRINCIPAL
# ==========================================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ No se pudo leer frame de la cámara.")
        break

    # Procesar imagen con MediaPipe (RGB)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Si detecta manos
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if i > 1:
                break  # Solo procesar hasta 2 manos

            # Dibujar landmarks sobre el frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # === Dedos clave ===
            thumb = hand_landmarks.landmark[4]   # Pulgar
            index = hand_landmarks.landmark[8]   # Índice
            middle = hand_landmarks.landmark[12] # Medio
            palm_center = hand_landmarks.landmark[0]  # Centro de la palma

            # === Distancias ===
            dist_thumb_index = math.dist((thumb.x, thumb.y), (index.x, index.y))
            dist_palm_index = math.dist((palm_center.x, palm_center.y), (index.x, index.y))

            # === Suavizado ===
            smooth[i]["dist"] = smooth[i]["dist"] * (1 - alpha) + dist_thumb_index * alpha
            smooth[i]["x"] = smooth[i]["x"] * (1 - alpha) + dist_palm_index * alpha

            # === Escalado a rango MIDI (0–127) ===
            thumb_index_scaled = scale(smooth[i]["dist"], 0.02, 0.15, 0, 1)
            palm_index_scaled = scale(smooth[i]["x"], 0.05, 0.3, 0, 1)

            # === Envío OSC ===
            if i == 0: # Mano 1 
                client.send_message("/hand/distance", thumb_index_scaled) 
                client.send_message("/hand/press1", palm_index_scaled) 
            elif i == 1: # Mano 2 
                client.send_message("/hand/press2", thumb_index_scaled) 
                client.send_message("/hand/rotation", palm_index_scaled)


            # === Overlay en pantalla ===
            cv2.putText(frame,
                        f"Hand {i+1} | Thumb-Idx:{thumb_index_scaled:.0f} Palm-Idx:{palm_index_scaled:.0f}",
                        (10, 30 + i * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Círculos debug (pulgar, índice, palma)
            for point in [thumb, index, palm_center]:
                cx, cy = int(point.x * frame.shape[1]), int(point.y * frame.shape[0])
                cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)

    # === FPS ===
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Mostrar frame con todo
    cv2.imshow("MediaPipe OSC - 2 Hands", frame)

    # Salir con tecla ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break


# ==========================================================
# LIBERAR RECURSOS
# ==========================================================
cap.release()
cv2.destroyAllWindows()
hands.close()
