import cv2
import mediapipe as mp
from pythonosc import udp_client
import time
import math
from typing import NamedTuple, Optional, Dict, Any

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
PINCH_SCALE_IN_MIN = 0.02
PINCH_SCALE_IN_MAX = 0.15
ANGLE_SCALE_IN_MIN = -math.pi / 2
ANGLE_SCALE_IN_MAX = math.pi / 2
SCALE_OUT_MIN = 0.0
SCALE_OUT_MAX = 1.0

# Configuración de Visualización
CIRCLE_COLOR_THUMB = (0, 0, 255)  # BGR
CIRCLE_COLOR_INDEX = (0, 255, 0)
TEXT_COLOR_1 = (0, 255, 0)
TEXT_COLOR_2 = (0, 255, 255)
TEXT_COLOR_3 = (255, 0, 0)
FPS_COLOR = (255, 255, 0)


class HandCalculations(NamedTuple):
    """Estructura de datos para almacenar los cálculos de una mano."""
    pinch_dist: float
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

def calculate_hand_data(hand_landmarks: Any, mp_hands: Any) -> HandCalculations:
    """Calcula el pinch y el ángulo para un conjunto de landmarks."""
    
    # Usar enums para legibilidad (KISS)
    thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    middle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

    # 1. Distancia pulgar - índice
    pinch_dist = distance(thumb, index)
    
    # 2. Ángulo de rotación de la mano
    dx = middle.x - wrist.x
    dy = middle.y - wrist.y
    angle_rad = math.atan2(dy, dx)
    
    return HandCalculations(pinch_dist=pinch_dist, angle_rad=angle_rad)

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

    prev_time = time.time()
    frame_height, frame_width = 0, 0 # Inicializar dimensiones

    # --- Bucle Principal de Procesamiento ---
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Advertencia: No se pudo leer frame de la cámara. Finalizando.")
                break
            
            if frame_height == 0:
                frame_height, frame_width, _ = frame.shape

            # Voltear para efecto espejo (UX)
            frame = cv2.flip(frame, 1)

            # Optimización de rendimiento: Pasar por referencia
            frame.flags.writeable = False
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            frame.flags.writeable = True

            # Usar un diccionario para mapeo estable 'Left'/'Right'
            hand_results: Dict[str, HandCalculations] = {}

            # --- Identificación Robusta de Manos ---
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    
                    hand_label = handedness.classification[0].label  # 'Left' o 'Right'
                    
                    # Realizar cálculos
                    hand_data = calculate_hand_data(hand_landmarks, mp_hands)
                    
                    # Almacenar datos por etiqueta (L/R)
                    hand_results[hand_label] = hand_data

                    # --- Visualización ---
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Dibujar puntos clave
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    cv2.circle(frame, (int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height)), 8, CIRCLE_COLOR_THUMB, -1)
                    cv2.circle(frame, (int(index_tip.x * frame_width), int(index_tip.y * frame_height)), 8, CIRCLE_COLOR_INDEX, -1)

            # --- Lógica de Procesamiento (Solo si AMBAS manos están presentes) ---
            if 'Left' in hand_results and 'Right' in hand_results:
                
                # Acceder a los datos de forma estable
                left_data = hand_results['Left']
                right_data = hand_results['Right']

                # 1) Distancia (Promedio de Pinch)
                avg_pinch = (left_data.pinch_dist + right_data.pinch_dist) / 2
                scaled_distance = scale(avg_pinch, PINCH_SCALE_IN_MIN, PINCH_SCALE_IN_MAX, SCALE_OUT_MIN, SCALE_OUT_MAX)

                # 2) "Apretar" (Binario estable)
                press_left = 1 if left_data.pinch_dist < PRESS_THRESHOLD else 0
                press_right = 1 if right_data.pinch_dist < PRESS_THRESHOLD else 0

                # 3) Rotación (Estable de la mano derecha)
                rotation_scaled = scale(right_data.angle_rad, ANGLE_SCALE_IN_MIN, ANGLE_SCALE_IN_MAX, SCALE_OUT_MIN, SCALE_OUT_MAX)

                # --- Envío OSC (Protegido) ---
                try:
                    client.send_message("/hand/distance", scaled_distance)
                    client.send_message("/hand/press1", press_left)
                    client.send_message("/hand/press2", press_right)
                    client.send_message("/hand/rotation", rotation_scaled)
                except Exception as e:
                    print(f"Advertencia: No se pudo enviar mensaje OSC. {e}")

                # --- Visualización de Datos ---
                cv2.putText(frame, f"Distance (Avg Pinch): {scaled_distance:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR_1, 2)
                cv2.putText(frame, f"Press L:{press_left}  Press R:{press_right}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR_2, 2)
                cv2.putText(frame, f"Rotation (Right Hand): {rotation_scaled:.2f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR_3, 2)

            # --- Cálculo y Visualización de FPS ---
            curr_time = time.time()
            delta_time = curr_time - prev_time
            prev_time = curr_time
            fps = 1.0 / delta_time if delta_time > 0 else 0.0
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, FPS_COLOR, 2)

            # --- Mostrar Ventana Principal ---
            cv2.imshow("MediaPipe OSC - Hybrid Control", frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break
    
    finally:
        # --- Limpieza de Recursos ---
        print("Finalizando aplicación. Liberando recursos...")
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()