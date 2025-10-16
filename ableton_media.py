import cv2
import mediapipe as mp
from pythonosc import udp_client
import time

# ========= CONFIGURACIÓN OSC =========
IP = "127.0.0.1"   # localhost
PUERTO = 4567      # mismo puerto que en Ableton
client = udp_client.SimpleUDPClient(IP, PUERTO)

def scale(val, in_min, in_max, out_min, out_max):
    return (val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

# ========= CONFIGURACIÓN MEDIAPIPE =========
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ========= OPEN CV CAMARA =========
# En Windows CAP_DSHOW evita problemas de congelado
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
smooth_x, smooth_y, smooth_dist = 0, 0, 0
alpha = 0.2  # 0.1 = muy suave, 0.5 = más rápido
# ========= FPS =========
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ No se pudo capturar la cámara")
        break
    print("Frame capturado:", frame.shape)


    # Convertir BGR -> RGB para MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibujar landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Coordenadas dedo índice
            x = hand_landmarks.landmark[8].x
            y = hand_landmarks.landmark[8].y

            # Distancia pulgar-índice
            thumb = hand_landmarks.landmark[4]
            index = hand_landmarks.landmark[8]
            dist = ((thumb.x - index.x)**2 + (thumb.y - index.y)**2) ** 0.5

            # === Aplicar suavizado ===
            smooth_x = smooth_x * (1 - alpha) + x * alpha
            smooth_y = smooth_y * (1 - alpha) + y * alpha
            smooth_dist = smooth_dist * (1 - alpha) + dist * alpha

            # Escalar valores suavizados
            x_scaled = scale(smooth_x, 0.2, 0.8, 0, 127)
            y_scaled = scale(smooth_y, 0.2, 0.8, 0, 127)
            dist_scaled = scale(smooth_dist, 0, 0.5, 0, 127)


            # Enviar mensajes OSC
            client.send_message("/hand/distance1", dist)
            client.send_message("/hand/distance2", x)
            client.send_message("/hand/distance3", y)



            # Mostrar valores en pantalla
            cv2.putText(
                frame,
                f"X:{x_scaled:.0f} Y:{y_scaled:.0f} Pinch:{dist_scaled:.0f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    # ===== FPS =====
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Mostrar cámara
    cv2.imshow("MediaPipe OSC", frame)

    # Salir con ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
