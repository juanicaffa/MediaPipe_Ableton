import cv2
import mediapipe as mp
import time
import math
from pythonosc import udp_client
from typing import Optional, Tuple, NamedTuple

# --- Constantes de Configuración ---

# Configuración OSC
OSC_IP = "127.0.0.1"
OSC_PORT = 4567

# Configuración de Cámara
CAMERA_INDEX = 0
USE_CAP_DSHOW = True  # Usar cv2.CAP_DSHOW en Windows para evitar congelamiento

# Configuración de MediaPipe
MAX_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

# Configuración de Procesamiento
SMOOTHING_ALPHA = 0.2  # Factor de suavizado (0.1 = muy suave, 0.9 = muy reactivo)

# Mapeo de Landmarks
INDEX_FINGER_TIP = mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP
THUMB_TIP = mp.solutions.hands.HandLandmark.THUMB_TIP

# Rangos de Escalado (Entrada: rango normalizado de MediaPipe, Salida: rango OSC/MIDI)
# Se ajustan empíricamente para un control cómodo
INPUT_X_RANGE = (0.2, 0.8)
INPUT_Y_RANGE = (0.2, 0.8)
INPUT_DIST_RANGE = (0.0, 0.25)  # Distancia normalizada entre pulgar e índice

OUTPUT_RANGE = (0, 127)


class HandTrackingData(NamedTuple):
    """Estructura de datos para almacenar los resultados del tracking."""
    x: float
    y: float
    pinch_distance: float


class ValueSmoother:
    """Aplica un filtro de paso bajo (suavizado exponencial) a un valor."""

    def __init__(self, alpha: float, initial_value: float = 0.0):
        self.alpha = alpha
        self.smooth_value = initial_value

    def update(self, new_value: float) -> float:
        """Actualiza el valor suavizado."""
        self.smooth_value = (self.smooth_value * (1 - self.alpha)) + (new_value * self.alpha)
        return self.smooth_value


class HandTracker:
    """Encapsula la lógica de detección de manos de MediaPipe."""

    def __init__(self, max_hands: int, min_detect_conf: float, min_track_conf: float):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=min_detect_conf,
            min_tracking_confidence=min_track_conf
        )

    def process_frame(self, frame_rgb: cv2.typing.MatLike) -> \
            Tuple[Optional[HandTrackingData], Optional[mp.framework.formats.landmark_pb2.NormalizedLandmarkList]]:
        """Procesa un frame RGB y retorna los datos de tracking y los landmarks."""
        results = self.hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return None, None

        # Tomar solo la primera mano detectada
        hand_landmarks = results.multi_hand_landmarks[0]

        try:
            index_tip = hand_landmarks.landmark[INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[THUMB_TIP]

            # Calcular distancia euclidiana 2D
            distance = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)

            data = HandTrackingData(
                x=index_tip.x,
                y=index_tip.y,
                pinch_distance=distance
            )
            return data, hand_landmarks

        except IndexError:
            # En caso de que un landmark específico no esté disponible (poco probable)
            return None, None

    def draw_landmarks(self, frame: cv2.typing.MatLike, landmarks: mp.framework.formats.landmark_pb2.NormalizedLandmarkList) -> None:
        """Dibuja los landmarks de la mano en el frame."""
        self.mp_drawing.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)

    def close(self) -> None:
        """Libera los recursos de MediaPipe."""
        self.hands.close()


class OSCClient:
    """Encapsula la lógica del cliente UDP de OSC."""

    def __init__(self, ip: str, port: int):
        self.client = udp_client.SimpleUDPClient(ip, port)

    def send_hand_data(self, x: int, y: int, distance: int) -> None:
        """Envía los datos procesados de la mano a direcciones OSC específicas."""
        try:
            self.client.send_message("/hand/x", x)
            self.client.send_message("/hand/y", y)
            self.client.send_message("/hand/pinch", distance)
        except Exception as e:
            # Evita que un error de red (ej. receptor no disponible) detenga el script
            print(f"[ERROR OSC] No se pudo enviar el mensaje: {e}")


class CVUtility:
    """Funciones de utilidad para escalado, limitación y dibujo en OpenCV."""

    @staticmethod
    def scale_and_clamp(val: float, in_min: float, in_max: float, out_min: int, out_max: int) -> int:
        """Escala un valor de un rango a otro y lo limita (clamping)."""
        # Escalar
        scaled_val = (val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
        
        # Limitar (Clamping)
        clamped_val = max(out_min, min(scaled_val, out_max))
        
        return int(clamped_val)

    @staticmethod
    def put_text(frame: cv2.typing.MatLike, text: str, pos: Tuple[int, int], color: Tuple[int, int, int]) -> None:
        """Dibuja texto estandarizado en el frame."""
        cv2.putText(
            frame, text, pos,
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )


class HandOSCController:
    """Clase principal que orquesta la aplicación."""

    def __init__(self):
        self.osc_client = OSCClient(OSC_IP, OSC_PORT)
        self.tracker = HandTracker(MAX_HANDS, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE)
        
        self.smoother_x = ValueSmoother(SMOOTHING_ALPHA, (INPUT_X_RANGE[0] + INPUT_X_RANGE[1]) / 2)
        self.smoother_y = ValueSmoother(SMOOTHING_ALPHA, (INPUT_Y_RANGE[0] + INPUT_Y_RANGE[1]) / 2)
        self.smoother_dist = ValueSmoother(SMOOTHING_ALPHA, INPUT_DIST_RANGE[0])
        
        self.prev_time = 0.0

        api_preference = cv2.CAP_DSHOW if USE_CAP_DSHOW else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(CAMERA_INDEX, api_preference)
        if not self.cap.isOpened():
            raise IOError(f"No se puede abrir la cámara (índice {CAMERA_INDEX})")

    def run(self) -> None:
        """Inicia el bucle principal de procesamiento."""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("[ERROR] No se pudo capturar el frame de la cámara.")
                    break

                # 1. Procesamiento de MediaPipe (sobre copia RGB)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False  # Optimización: marcar como no escribible
                
                hand_data, landmarks = self.tracker.process_frame(rgb_frame)
                
                rgb_frame.flags.writeable = True # Permitir escritura de nuevo (aunque no se usa)

                if hand_data:
                    # 2. Suavizado de Valores
                    smooth_x = self.smoother_x.update(hand_data.x)
                    smooth_y = self.smoother_y.update(hand_data.y)
                    smooth_dist = self.smoother_dist.update(hand_data.pinch_distance)

                    # 3. Escalado y Limitación (Clamping)
                    x_scaled = CVUtility.scale_and_clamp(
                        smooth_x, INPUT_X_RANGE[0], INPUT_X_RANGE[1], OUTPUT_RANGE[0], OUTPUT_RANGE[1])
                    y_scaled = CVUtility.scale_and_clamp(
                        smooth_y, INPUT_Y_RANGE[0], INPUT_Y_RANGE[1], OUTPUT_RANGE[0], OUTPUT_RANGE[1])
                    dist_scaled = CVUtility.scale_and_clamp(
                        smooth_dist, INPUT_DIST_RANGE[0], INPUT_DIST_RANGE[1], OUTPUT_RANGE[0], OUTPUT_RANGE[1])

                    # 4. Envío OSC (Datos procesados)
                    self.osc_client.send_hand_data(x_scaled, y_scaled, dist_scaled)

                    # 5. Visualización (Dibujar en el frame BGR original)
                    if landmarks:
                        self.tracker.draw_landmarks(frame, landmarks)
                    
                    CVUtility.put_text(frame, f"X: {x_scaled}  Y: {y_scaled}", (10, 30), (0, 255, 0))
                    CVUtility.put_text(frame, f"Pinch: {dist_scaled}", (10, 60), (0, 255, 0))

                # Cálculo de FPS
                self._draw_fps(frame)

                # Mostrar resultado
                cv2.imshow("MediaPipe OSC Controller", frame)

                if cv2.waitKey(1) & 0xFF == 27:  # Salir con ESC
                    break
        finally:
            self.cleanup()

    def _draw_fps(self, frame: cv2.typing.MatLike) -> None:
        """Calcula y dibuja los FPS en el frame."""
        curr_time = time.time()
        delta = curr_time - self.prev_time
        self.prev_time = curr_time
        
        if delta > 0:
            fps = 1 / delta
            CVUtility.put_text(frame, f"FPS: {int(fps)}", (10, 90), (255, 0, 0))

    def cleanup(self) -> None:
        """Libera todos los recursos."""
        print("Cerrando aplicación y liberando recursos...")
        self.cap.release()
        self.tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        app = HandOSCController()
        app.run()
    except Exception as e:
        print(f"[ERROR CRÍTICO] La aplicación falló: {e}")