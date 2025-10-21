import cv2
import mediapipe as mp
from pythonosc import udp_client
import time
import math

# --- Configuración de Constantes (Principio DRY) ---

# Configuración OSC
OSC_IP = "127.0.0.1"
OSC_PORT = 4567

# Configuración de Captura de Video
VIDEO_SOURCE = 0
CV_API = cv2.CAP_DSHOW

# Configuración de MediaPipe
MAX_HANDS = 2
MIN_DETECT_CONFIDENCE = 0.7
MIN_TRACK_CONFIDENCE = 0.7

# Rangos de Escalado (basados en la lógica original)
DIST_IN_MIN = 0.05
DIST_IN_MAX = 0.4
DIST_OUT_MIN = 0.0
DIST_OUT_MAX = 1.0

# Configuración de Visualización
CIRCLE_RADIUS = 8
CIRCLE_COLOR = (0, 0, 255)  # BGR para OpenCV
TEXT_COLOR = (0, 255, 0)
FPS_COLOR = (255, 0, 0)


def scale(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    """
    Escala un valor de un rango de entrada a un rango de salida
    con saturación (clamping).
    """
    if in_min == in_max:
        return out_min
    
    scaled = (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    
    # Aplicar clamping
    return max(min(scaled, out_max), out_min)


def main():
    """Función principal que encapsula la lógica de la aplicación."""
    
    # --- Inicialización ---
    try:
        client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
        print(f"Cliente OSC inicializado en {OSC_IP}:{OSC_PORT}")
    except Exception as e:
        print(f"Error fatal: No se pudo inicializar el cliente OSC. {e}")
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        max_num_hands=MAX_HANDS,
        min_detection_confidence=MIN_DETECT_CONFIDENCE,
        min_tracking_confidence=MIN_TRACK_CONFIDENCE
    )

    cap = cv2.VideoCapture(VIDEO_SOURCE, CV_API)
    if not cap.isOpened():
        print(f"Error fatal: No se pudo abrir la cámara (Fuente: {VIDEO_SOURCE}).")
        return

    prev_time = time.time()  # Inicializar para el primer cálculo de FPS

    # --- Bucle Principal de Procesamiento ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Advertencia: No se pudo leer frame de la cámara. Finalizando.")
            break

        # Voltear el frame horizontalmente para efecto espejo
        frame = cv2.flip(frame, 1)

        # Optimización de rendimiento:
        # Marcar el frame como no escribible para pasarlo por referencia.
        frame.flags.writeable = False
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        frame.flags.writeable = True

        landmarks_left = None
        landmarks_right = None

        # --- Identificación Robusta de Manos ---
        if results.multi_hand_landmarks and results.multi_handedness:
            # Iterar para identificar inequívocamente 'Left' y 'Right'
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                
                # Dibujar todas las manos detectadas
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                hand_label = handedness.classification[0].label
                if hand_label == 'Left':
                    landmarks_left = hand_landmarks
                elif hand_label == 'Right':
                    landmarks_right = hand_landmarks

        # --- Lógica de Procesamiento (Solo si AMBAS manos están presentes) ---
        if landmarks_left and landmarks_right:
            
            # Usar enums de MediaPipe para legibilidad (KISS)
            thumb_left = landmarks_left.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_left = landmarks_left.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            thumb_right = landmarks_right.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_right = landmarks_right.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Calcular distancias 2D
            thumb_distance = math.dist((thumb_left.x, thumb_left.y), (thumb_right.x, thumb_right.y))
            index_distance = math.dist((index_left.x, index_left.y), (index_right.x, index_right.y))

            # Escalar valores
            thumb_scaled = scale(thumb_distance, DIST_IN_MIN, DIST_IN_MAX, DIST_OUT_MIN, DIST_OUT_MAX)
            index_scaled = scale(index_distance, DIST_IN_MIN, DIST_IN_MAX, DIST_OUT_MIN, DIST_OUT_MAX)

            # --- Envío OSC ---
            try:
                client.send_message("/hand/thumb_distance", thumb_scaled)
                client.send_message("/hand/index_distance", index_scaled)
            except Exception as e:
                # Evitar que un error de red (ej. receptor cerrado) detenga el script
                print(f"Advertencia: No se pudo enviar mensaje OSC. {e}")

            # --- Visualización de Datos y Puntos Clave ---
            cv2.putText(frame,
                        f"ThumbDist: {thumb_scaled:.2f} | IndexDist: {index_scaled:.2f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)

            # Dibujar círculos en los puntos de interés
            frame_height, frame_width, _ = frame.shape
            for point in [thumb_left, index_left, thumb_right, index_right]:
                cx = int(point.x * frame_width)
                cy = int(point.y * frame_height)
                cv2.circle(frame, (cx, cy), CIRCLE_RADIUS, CIRCLE_COLOR, -1)

        # --- Cálculo y Visualización de FPS ---
        curr_time = time.time()
        delta_time = curr_time - prev_time
        prev_time = curr_time
        
        fps = 1.0 / delta_time if delta_time > 0 else 0.0
        
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, FPS_COLOR, 2)

        # --- Mostrar Ventana Principal ---
        cv2.imshow("MediaPipe OSC - Two Hands Distance", frame)

        # Salir con 'ESC' (waitKey(5) es más eficiente que waitKey(1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

    # --- Limpieza de Recursos ---
    print("Finalizando aplicación. Liberando recursos...")
    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()