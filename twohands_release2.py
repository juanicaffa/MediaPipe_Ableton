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
    """Escala un valor de un rango de entrada a un rango de salida con saturación."""
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
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("❌ No se pudo abrir la cámara. Verifica que esté conectada o en uso.")

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

    # Si detecta exactamente dos manos
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        hand1 = results.multi_hand_landmarks[0]
        hand2 = results.multi_hand_landmarks[1]

        # === Puntos clave ===
        thumb1 = hand1.landmark[4]   # Pulgar mano 1
        index1 = hand1.landmark[8]   # Índice mano 1

        thumb2 = hand2.landmark[4]   # Pulgar mano 2
        index2 = hand2.landmark[8]   # Índice mano 2

        # === Distancia entre ambos pulgares e índices ===
        thumb_distance = math.dist((thumb1.x, thumb1.y), (thumb2.x, thumb2.y))
        index_distance = math.dist((index1.x, index1.y), (index2.x, index2.y))

        # === Rotación relativa ===
        # Diferencia en el eje X entre pulgares e índices
        thumb_rotation = abs(thumb1.x - thumb2.x)
        index_rotation = abs(index1.x - index2.x)

        # === Escalado (ajusta rangos según tu cámara y movimiento) ===
        thumb_distance_scaled = scale(thumb_distance, 0.05, 0.4, 0, 1)
        index_distance_scaled = scale(index_distance, 0.05, 0.4, 0, 1)

        thumb_rotation_scaled = scale(thumb_rotation, 0.0, 0.5, 0, 1)
        index_rotation_scaled = scale(index_rotation, 0.0, 0.5, 0, 1)

        # === Envío OSC ===
        client.send_message("/hand/distance1", thumb_distance_scaled)
        client.send_message("/hand/rotation1", thumb_rotation_scaled)
        client.send_message("/hand/distance2", index_distance_scaled)
        client.send_message("/hand/rotation2", index_rotation_scaled)

        # === Overlay en pantalla ===
        cv2.putText(frame,
                    f"Thumb Dist:{thumb_distance_scaled:.2f} | Thumb Rot:{thumb_rotation_scaled:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame,
                    f"Index Dist:{index_distance_scaled:.2f} | Index Rot:{index_rotation_scaled:.2f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Dibujar landmarks y puntos clave
        mp_drawing.draw_landmarks(frame, hand1, mp_hands.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, hand2, mp_hands.HAND_CONNECTIONS)

        for point in [thumb1, index1, thumb2, index2]:
            cx, cy = int(point.x * frame.shape[1]), int(point.y * frame.shape[0])
            cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)

    # === FPS ===
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Mostrar frame
    cv2.imshow("MediaPipe OSC - Two Hands Interaction", frame)

    # Salir con ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ==========================================================
# LIBERAR RECURSOS
# ==========================================================
cap.release()
cv2.destroyAllWindows()
hands.close()
