import cv2
import mediapipe as mp
from pythonosc import udp_client
import time
import math

# Configuración OSC
OSC_IP = "127.0.0.1"
OSC_PORT = 4567

# Direcciones OSC (Asignaciones de control)
OSC_HAND_RIGHT_DIST = "/hand/right/distance"
OSC_HAND_RIGHT_PRESS = "/hand/right/press"
OSC_HAND_LEFT_DIST = "/hand/left/distance"
OSC_HAND_LEFT_ROT = "/hand/left/rotation"

# Configuración de Suavizado
ALPHA = 0.25  # Factor de suavizado (EMA)

# Rangos de Normalización (Calibrar según sea necesario)
DIST_THUMB_INDEX_MIN = 0.02
DIST_THUMB_INDEX_MAX = 0.15
DIST_PALM_INDEX_MIN = 0.05
DIST_PALM_INDEX_MAX = 0.3


def scale(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    """
    Escala un valor de un rango de entrada a un rango de salida,
    con saturación (clamping) en los límites.
    """
    if in_min == in_max:
        return out_min
    
    # Normalizar (0-1)
    normalized = (value - in_min) / (in_max - in_min)
    
    # Escalar a rango de salida y aplicar clamp
    scaled = normalized * (out_max - out_min) + out_min
    return max(min(scaled, out_max), out_min)


def main():
    """
    Función principal que ejecuta la captura de cámara,
    procesamiento de MediaPipe y envío de OSC.
    """
    client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)

    # Estado de suavizado, clave por etiqueta de mano ('Left', 'Right')
    smooth_state = {
        'Left': {"dist_thumb_index": 0.0, "dist_palm_index": 0.0},
        'Right': {"dist_thumb_index": 0.0, "dist_palm_index": 0.0}
    }

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara. Verifique la conexión.")

    prev_time = time.time()

    try:
        with mp.solutions.hands.Hands(
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7) as hands:

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("No se pudo leer frame de la cámara. Terminando.")
                    break

                h, w, _ = frame.shape
                
                # Convertir a RGB y marcar como no escribible para optimizar MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = hands.process(rgb_frame)
                rgb_frame.flags.writeable = True

                # Cálculo de FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                prev_time = curr_time
                cv2.putText(frame, f"FPS: {int(fps)}", (w - 100, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                if results.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, 
                                                          results.multi_handedness):
                        
                        # Identificación robusta de la mano
                        hand_label = handedness.classification[0].label
                        
                        # Dibujar esqueleto
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                        # --- Obtener landmarks clave ---
                        thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
                        index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                        wrist = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]

                        # --- Calcular distancias 2D ---
                        dist_thumb_index = math.dist((thumb_tip.x, thumb_tip.y), (index_tip.x, index_tip.y))
                        dist_palm_index = math.dist((wrist.x, wrist.y), (index_tip.x, index_tip.y))

                        # --- Suavizado (Filtro Pasa-Bajos / EMA) ---
                        current_smooth = smooth_state[hand_label]
                        current_smooth["dist_thumb_index"] = (current_smooth["dist_thumb_index"] * (1 - ALPHA) + 
                                                              dist_thumb_index * ALPHA)
                        current_smooth["dist_palm_index"] = (current_smooth["dist_palm_index"] * (1 - ALPHA) + 
                                                             dist_palm_index * ALPHA)

                        # --- Escalado a rango 0-1 ---
                        thumb_index_scaled = scale(current_smooth["dist_thumb_index"],
                                                   DIST_THUMB_INDEX_MIN, DIST_THUMB_INDEX_MAX, 0.0, 1.0)
                        palm_index_scaled = scale(current_smooth["dist_palm_index"],
                                                  DIST_PALM_INDEX_MIN, DIST_PALM_INDEX_MAX, 0.0, 1.0)

                        # --- Envío OSC basado en etiqueta de mano ---
                        if hand_label == 'Right':
                            client.send_message(OSC_HAND_RIGHT_DIST, thumb_index_scaled)
                            client.send_message(OSC_HAND_RIGHT_PRESS, palm_index_scaled)
                        elif hand_label == 'Left':
                            client.send_message(OSC_HAND_LEFT_DIST, thumb_index_scaled)
                            client.send_message(OSC_HAND_LEFT_ROT, palm_index_scaled)

                        # --- Información en pantalla ---
                        info_text = f"{hand_label} | Dist:{thumb_index_scaled:.2f} Palm:{palm_index_scaled:.2f}"
                        y_pos = 30 if hand_label == 'Right' else 70
                        cv2.putText(frame, info_text, (10, y_pos),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # --- Círculos de depuración ---
                        for point in [thumb_tip, index_tip, wrist]:
                            cx, cy = int(point.x * w), int(point.y * h)
                            cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)

                cv2.imshow("MediaPipe OSC", frame)

                if cv2.waitKey(1) & 0xFF == 27:  # Salir con ESC
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Recursos liberados. Saliendo.")


if __name__ == "__main__":
    main()