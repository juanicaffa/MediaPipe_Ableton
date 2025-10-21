import cv2
import mediapipe as mp
from pythonosc import udp_client
import time
import math

# --- Configuración de Constantes ---
# Mover los "números mágicos" a constantes nombradas mejora la
# legibilidad y mantenibilidad (principio DRY).

# Configuración OSC
OSC_IP = "127.0.0.1"
OSC_PORT = 4567

# Configuración de Captura de Video
VIDEO_SOURCE = 0
CV_API = cv2.CAP_DSHOW  # API específica para Windows, mejora la compatibilidad

# Configuración de MediaPipe
MAX_HANDS = 2
MIN_DETECT_CONFIDENCE = 0.7
MIN_TRACK_CONFIDENCE = 0.7

# Parámetros de Procesamiento
SMOOTHING_ALPHA = 0.25  # 0.1 = más suave, 0.9 = más reactivo

# Rangos de Escalado (basados en observación empírica)
SCALE_X_IN_MIN = 0.2
SCALE_X_IN_MAX = 0.8
SCALE_DIST_IN_MIN = 0.02
SCALE_DIST_IN_MAX = 0.15

# Rango de Salida (ej. MIDI 0-127)
SCALE_OUT_MIN = 0
SCALE_OUT_MAX = 127


def scale(val: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    """
    Escala un valor de un rango de entrada a un rango de salida,
    aplicando "clamping" (límites) para asegurar que el valor
    resultante no exceda el rango de salida.
    """
    # Evitar división por cero si el rango de entrada es nulo
    if in_max == in_min:
        return out_min
    
    scaled_val = (val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    
    # Aplicar clamping
    return max(min(scaled_val, out_max), out_min)


def main():
    """Función principal que encapsula la lógica de la aplicación."""
    
    # --- Inicialización ---
    try:
        client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
        print(f"Cliente OSC conectado a {OSC_IP}:{OSC_PORT}")
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
        print(f"Error fatal: No se pudo abrir la fuente de video {VIDEO_SOURCE}.")
        return

    # Estructura de suavizado robusta basada en etiquetas 'Left'/'Right'
    # Inicializado en valores medios para evitar saltos desde 0.
    smooth_data = {
        'Left': {"x": 0.5, "dist": 0.1},
        'Right': {"x": 0.5, "dist": 0.1}
    }
    
    prev_time = time.time() # Inicializar para el primer cálculo de FPS

    # --- Bucle Principal de Procesamiento ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Advertencia: No se pudo capturar frame. Finalizando stream.")
            break

        # Voltear el frame horizontalmente para efecto espejo
        frame = cv2.flip(frame, 1)

        # Optimización de rendimiento:
        # Marcar el frame como no escribible pasa la imagen por referencia
        # a MediaPipe, evitando una copia interna.
        frame.flags.writeable = False
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        frame.flags.writeable = True

        # --- Procesamiento de Manos Detectadas ---
        if results.multi_hand_landmarks and results.multi_handedness:
            # Iterar simultáneamente sobre los landmarks y la lateralidad (handedness)
            for i, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                
                # Identificar si la mano es 'Left' o 'Right'
                hand_label = handedness.classification[0].label
                
                # Dibujar los landmarks en el frame BGR
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # --- Cálculos de Geometría ---
                # Usar enums de MediaPipe para legibilidad (DRY/KISS)
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Coordenada X del índice
                x_pos = index_tip.x
                
                # Distancia 2D entre pulgar e índice.
                # math.hypot es más limpio y numéricamente estable que sqrt(dx^2 + dy^2)
                distance = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)

                # --- Suavizado EMA (Exponential Moving Average) ---
                # Se aplica el suavizado al estado específico de 'Left' o 'Right'
                smooth_data[hand_label]["x"] = (smooth_data[hand_label]["x"] * (1 - SMOOTHING_ALPHA)) + (x_pos * SMOOTHING_ALPHA)
                smooth_data[hand_label]["dist"] = (smooth_data[hand_label]["dist"] * (1 - SMOOTHING_ALPHA)) + (distance * SMOOTHING_ALPHA)
                
                smooth_x = smooth_data[hand_label]["x"]
                smooth_dist = smooth_data[hand_label]["dist"]

                # --- Escalado de Valores ---
                x_scaled = scale(smooth_x, SCALE_X_IN_MIN, SCALE_X_IN_MAX, SCALE_OUT_MIN, SCALE_OUT_MAX)
                dist_scaled = scale(smooth_dist, SCALE_DIST_IN_MIN, SCALE_DIST_IN_MAX, SCALE_OUT_MIN, SCALE_OUT_MAX)

                # --- Envío de Mensajes OSC ---
                try:
                    # Enviar mensajes OSC basados en la etiqueta de la mano
                    if hand_label == 'Left':
                        client.send_message("/hand/distance", dist_scaled)
                        client.send_message("/hand/press1", x_scaled)
                    elif hand_label == 'Right':
                        # Preservar la lógica original: dist -> press2, x -> rotation
                        client.send_message("/hand/press2", dist_scaled)
                        client.send_message("/hand/rotation", x_scaled)
                except Exception as e:
                    # Evitar que un error de red detenga el bucle de video
                    print(f"Advertencia: No se pudo enviar mensaje OSC. {e}")

                # --- Mostrar Datos en Pantalla ---
                text_y_pos = 30 + i * 40
                cv2.putText(frame,
                            f"{hand_label} | X:{x_scaled:.0f} Dist:{dist_scaled:.0f}",
                            (10, text_y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # --- Cálculo y Visualización de FPS ---
        curr_time = time.time()
        delta_time = curr_time - prev_time
        prev_time = curr_time
        
        # Prevenir división por cero en el primer frame o si delta_time es 0
        fps = 1.0 / delta_time if delta_time > 0 else 0.0
        
        cv2.putText(frame, f"FPS: {int(fps)}", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # --- Mostrar Ventana Principal ---
        # Se elimina la ventana "Frame crudo" y la llamada a imshow duplicada
        # para reducir la sobrecarga y la confusión.
        cv2.imshow("Controlador de Mano OSC", frame)

        # Salir con 'ESC'. waitKey(5) es un buen balance entre
        # reactividad y eficiencia de CPU (en lugar de 1ms).
        if cv2.waitKey(5) & 0xFF == 27:
            break

    # --- Limpieza de Recursos ---
    print("Finalizando aplicación. Liberando recursos...")
    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()