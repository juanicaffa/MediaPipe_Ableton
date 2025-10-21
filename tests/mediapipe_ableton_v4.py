import cv2
import mediapipe as mp
from pythonosc import udp_client
import time
import math
from typing import NamedTuple, Dict, Any, Tuple

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

# Parámetros de Lógica de Control
PRESS_THRESHOLD = 0.04
MUTUAL_DIST_IN_MIN = 0.02
MUTUAL_DIST_IN_MAX = 0.6
ANGLE_IN_MIN = -math.pi / 2
ANGLE_IN_MAX = math.pi / 2
SCALE_OUT_MIN = 0.0
SCALE_OUT_MAX = 1.0

# Configuración de Visualización
CIRCLE_COLOR_THUMB = (0, 0, 255)  # BGR
CIRCLE_COLOR_INDEX = (0, 255, 0)
CIRCLE_COLOR_MID = (255, 0, 0)
TEXT_COLOR_1 = (0, 255, 0)
TEXT_COLOR_2 = (0, 255, 255)
TEXT_COLOR_3 = (255, 0, 0)
TEXT_COLOR_WARN = (0, 0, 255)
FPS_COLOR = (255, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_MAIN = 0.7
FONT_SCALE_WARN = 0.6
FONT_THICKNESS = 2


class HandMetrics(NamedTuple):
    """Estructura de datos para almacenar los cálculos de una mano."""
    landmarks: Any
    pinch_dist: float
    mid_point: Tuple[float, float]
    angle_rad: float


def scale(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    """Escala un valor de un rango a otro, con saturación."""
    if in_min == in_max:
        return out_min
    scaled = (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    return max(min(scaled, out_max), out_min)

def distance(p1: Any, p2: Any) -> float:
    """Distancia Euclidiana 2D entre dos puntos normalizados de MediaPipe."""
    return math.dist((p1.x, p1.y), (p2.x, p2.y))

def calculate_hand_metrics(hand_landmarks: Any, mp_hands: Any) -> HandMetrics:
    """Extrae landmarks clave y calcula las métricas de una mano."""
    
    # Usar enums para legibilidad y robustez (KISS)
    thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    middle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

    # Métrica 1: Distancia de "pinch"
    pinch_dist = distance(thumb, index)
    
    # Métrica 2: Punto medio entre pulgar e índice
    mid_x = (thumb.x + index.x) / 2.0
    mid_y = (thumb.y + index.y) / 2.0
    
    # Métrica 3: Ángulo de rotación de la mano
    dx = middle.x - wrist.x
    dy = middle.y - wrist.y
    angle_rad = math.atan2(dy, dx)
    
    return HandMetrics(
        landmarks=hand_landmarks,
        pinch_dist=pinch_dist,
        mid_point=(mid_x, mid_y),
        angle_rad=angle_rad
    )

def main():
    """Función principal que encapsula la inicialización y el bucle de la aplicación."""
    
    # --- Inicialización de Servicios ---
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

    # --- Inicialización de Estado del Bucle ---
    prev_time = time.time()
    frame_width, frame_height = 0, 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Advertencia: No se pudo leer frame de la cámara. Finalizando.")
                break

            # Inicializar dimensiones del frame en la primera iteración
            if frame_height == 0:
                frame_height, frame_width, _ = frame.shape

            # Voltear para efecto espejo (UX)
            frame = cv2.flip(frame, 1)

            # Optimización de rendimiento: Pasar por referencia
            frame.flags.writeable = False
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            frame.flags.writeable = True

            hand_map: Dict[str, HandMetrics] = {}

            # --- Procesamiento y Visualización de Manos ---
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label
                    
                    # Calcular todas las métricas para esta mano
                    metrics = calculate_hand_metrics(hand_landmarks, mp_hands)
                    hand_map[label] = metrics

                    # Dibujar esqueleto
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Dibujar puntos clave (requiere re-extraer landmarks)
                    thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    mid_x, mid_y = metrics.mid_point

                    cx_t, cy_t = int(thumb.x * frame_width), int(thumb.y * frame_height)
                    cx_i, cy_i = int(index.x * frame_width), int(index.y * frame_height)
                    cx_m, cy_m = int(mid_x * frame_width), int(mid_y * frame_height)

                    cv2.circle(frame, (cx_t, cy_t), 6, CIRCLE_COLOR_THUMB, -1)
                    cv2.circle(frame, (cx_i, cy_i), 6, CIRCLE_COLOR_INDEX, -1)
                    cv2.circle(frame, (cx_m, cy_m), 6, CIRCLE_COLOR_MID, -1)

            # --- Lógica de Control OSC (Solo con ambas manos) ---
            if "Left" in hand_map and "Right" in hand_map:
                left = hand_map["Left"]
                right = hand_map["Right"]

                # 1) Distancia mutua
                mutual_dist = math.dist(left.mid_point, right.mid_point)
                scaled_distance = scale(mutual_dist, MUTUAL_DIST_IN_MIN, MUTUAL_DIST_IN_MAX, SCALE_OUT_MIN, SCALE_OUT_MAX)
                
                # 2) "Apretar" (Pinch)
                press_left = 1 if left.pinch_dist < PRESS_THRESHOLD else 0
                press_right = 1 if right.pinch_dist < PRESS_THRESHOLD else 0
                
                # 3) Rotación (Mano derecha)
                # Usamos la métrica precalculada (right.angle_rad)
                rotation_scaled = 1.0 - scale(right.angle_rad, ANGLE_IN_MIN, ANGLE_IN_MAX, SCALE_OUT_MIN, SCALE_OUT_MAX)

                # Envío OSC (Protegido)
                try:
                    client.send_message("/hand/distance", scaled_distance)
                    client.send_message("/hand/press1", press_left)
                    client.send_message("/hand/press2", press_right)
                    client.send_message("/hand/rotation", rotation_scaled)
                except Exception as e:
                    print(f"Advertencia: No se pudo enviar mensaje OSC. {e}")

                # Visualización de Datos
                cv2.putText(frame, f"Distance: {scaled_distance:.2f}", (10, 30), FONT, FONT_SCALE_MAIN, TEXT_COLOR_1, FONT_THICKNESS)
                cv2.putText(frame, f"Press L:{press_left}  R:{press_right}", (10, 60), FONT, FONT_SCALE_MAIN, TEXT_COLOR_2, FONT_THICKNESS)
                cv2.putText(frame, f"Rotation: {rotation_scaled:.2f}", (10, 90), FONT, FONT_SCALE_MAIN, TEXT_COLOR_3, FONT_THICKNESS)
            
            else:
                # Mensaje de advertencia si no se detectan ambas manos
                cv2.putText(frame, "Se requieren ambas manos (Left + Right).", (10, 30), FONT, FONT_SCALE_WARN, TEXT_COLOR_WARN, FONT_THICKNESS)

            # --- Visualización de FPS ---
            curr_time = time.time()
            delta_time = curr_time - prev_time
            prev_time = curr_time
            fps = 1.0 / delta_time if delta_time > 0 else 0.0
            
            # Posicionar FPS en la esquina inferior izquierda
            cv2.putText(frame, f"FPS: {int(fps)}", (10, frame_height - 20), FONT, FONT_SCALE_WARN, FPS_COLOR, FONT_THICKNESS)

            cv2.imshow("MediaPipe OSC - Refactored", frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    finally:
        # --- Limpieza de Recursos (Garantizada) ---
        print("Finalizando aplicación. Liberando recursos...")
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()