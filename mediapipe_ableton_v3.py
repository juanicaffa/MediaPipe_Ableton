import cv2
import mediapipe as mp
from pythonosc import udp_client
import time
import math

# ==========================================================
# CONFIGURACIÓN OSC
# ==========================================================
OSC_IP = "127.0.0.1"  # Dirección IP destino
OSC_PORT = 4567       # Puerto OSC (igual que en el bridge MIDI)
client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)

# ==========================================================
# FUNCIONES AUXILIARES
# ==========================================================
def scale(value, in_min, in_max, out_min, out_max):
    """Escala un valor de un rango a otro, con saturación."""
    if in_min == in_max:
        return out_min
    scaled = (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    return max(min(scaled, out_max), out_min)

def distance(p1, p2):
    """Distancia Euclidiana 2D entre dos puntos normalizados de MediaPipe."""
    return math.dist((p1.x, p1.y), (p2.x, p2.y))

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
    raise RuntimeError("❌ No se pudo abrir la cámara.")

prev_time = 0  # Para FPS

# ==========================================================
# LOOP PRINCIPAL
# ==========================================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ No se pudo leer frame de la cámara.")
        break

    # Procesar imagen
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Datos temporales
    hand_data = []

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if i > 1:
                break  # Solo procesar 2 manos

            # === Puntos clave ===
            thumb = hand_landmarks.landmark[4]   # Pulgar
            index = hand_landmarks.landmark[8]   # Índice
            wrist = hand_landmarks.landmark[0]   # Muñeca
            middle = hand_landmarks.landmark[12] # Dedo medio

            # === Distancia pulgar - índice (de esta mano) ===
            pinch_dist = distance(thumb, index)
            hand_data.append({
                "pinch_dist": pinch_dist,
                "wrist": wrist,
                "middle": middle,
                "landmarks": hand_landmarks
            })

            # Dibujar landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.circle(frame, (int(thumb.x * frame.shape[1]), int(thumb.y * frame.shape[0])), 8, (0, 0, 255), -1)
            cv2.circle(frame, (int(index.x * frame.shape[1]), int(index.y * frame.shape[0])), 8, (0, 255, 0), -1)

    # ==========================================================
    # PROCESAR DATOS CUANDO HAY 2 MANOS
    # ==========================================================
    if len(hand_data) == 2:
        # 1) DISTANCIA MUTUA ENTRE AMBAS MANOS
        avg_pinch = (hand_data[0]["pinch_dist"] + hand_data[1]["pinch_dist"]) / 2
        scaled_distance = scale(avg_pinch, 0.02, 0.15, 0, 1)
        client.send_message("/hand/distance", scaled_distance)

        # 2) DETECTAR "APRETAR" EN CADA MANO
        press_threshold = 0.04
        press1 = 1 if hand_data[0]["pinch_dist"] < press_threshold else 0
        press2 = 1 if hand_data[1]["pinch_dist"] < press_threshold else 0
        client.send_message("/hand/press1", press1)
        client.send_message("/hand/press2", press2)

        # 3) ROTACIÓN DE UNA MANO (usamos la mano derecha por ejemplo)
        right_hand = hand_data[1]  # Segunda mano detectada
        dx = right_hand["middle"].x - right_hand["wrist"].x
        dy = right_hand["middle"].y - right_hand["wrist"].y
        angle = math.atan2(dy, dx)  # Ángulo en radianes
        rotation_scaled = scale(angle, -math.pi/2, math.pi/2, 0, 1)
        client.send_message("/hand/rotation", rotation_scaled)

        # Mostrar valores en pantalla
        cv2.putText(frame, f"Distance: {scaled_distance:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Press1: {press1}  Press2: {press2}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Rotation: {rotation_scaled:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Mostrar frame
    cv2.imshow("MediaPipe OSC", frame)

    # Salir con ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ==========================================================
# LIBERAR
# ==========================================================
cap.release()
cv2.destroyAllWindows()
hands.close()
