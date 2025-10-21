import cv2
import mediapipe as mp
from pythonosc import udp_client
import time
import math
from typing import NamedTuple, Optional, Dict, Tuple, Any
from dataclasses import dataclass

# --- Configuración Basada en Clases (DRY, SRP) ---

@dataclass(frozen=True)
class OscConfig:
    """Encapsula la configuración del cliente OSC."""
    IP: str = "127.0.0.1"
    PORT: int = 4567

@dataclass(frozen=True)
class CameraConfig:
    """Encapsula la configuración de la captura de video."""
    SOURCE: int = 0
    API: Any = cv2.CAP_DSHOW

@dataclass(frozen=True)
class MediaPipeConfig:
    """Encapsula la configuración de MediaPipe Hands."""
    MAX_HANDS: int = 2
    MIN_DETECT_CONFIDENCE: float = 0.7
    MIN_TRACK_CONFIDENCE: float = 0.7

@dataclass(frozen=True)
class ControlConfig:
    """Encapsula los parámetros de la lógica de control."""
    PRESS_THRESHOLD: float = 0.04
    PINCH_SCALE_IN_MIN: float = 0.02
    PINCH_SCALE_IN_MAX: float = 0.15
    ANGLE_SCALE_IN_MIN: float = -math.pi / 2
    ANGLE_SCALE_IN_MAX: float = math.pi / 2
    SCALE_OUT_MIN: float = 0.0
    SCALE_OUT_MAX: float = 1.0

@dataclass(frozen=True)
class DrawingConfig:
    """Encapsula la configuración de visualización."""
    CIRCLE_COLOR_THUMB: Tuple[int, int, int] = (0, 0, 255)
    CIRCLE_COLOR_INDEX: Tuple[int, int, int] = (0, 255, 0)
    TEXT_COLOR_1: Tuple[int, int, int] = (0, 255, 0)
    TEXT_COLOR_2: Tuple[int, int, int] = (0, 255, 255)
    TEXT_COLOR_3: Tuple[int, int, int] = (255, 0, 0)
    FPS_COLOR: Tuple[int, int, int] = (255, 255, 0)
    FONT: int = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE: float = 0.7
    FONT_THICKNESS: int = 2


# --- Estructuras de Datos ---

class HandCalculations(NamedTuple):
    """Estructura de datos para almacenar los cálculos de una mano."""
    pinch_dist: float
    angle_rad: float

@dataclass
class OscData:
    """Estructura de datos para los valores OSC a enviar."""
    distance: float
    press_left: int
    press_right: int
    rotation: float


# --- Funciones Utilitarias (Puras) ---

def scale(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    """Escala un valor de un rango a otro, con saturación (clamping)."""
    if in_min == in_max:
        return out_min
    scaled = (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    return max(min(scaled, out_max), out_min)

def distance(p1: Any, p2: Any) -> float:
    """Distancia Euclidiana 2D entre dos puntos (landmarks) de MediaPipe."""
    return math.dist((p1.x, p1.y), (p2.x, p2.y))

def calculate_hand_metrics(thumb_tip: Any, index_tip: Any, wrist: Any, middle_tip: Any) -> HandCalculations:
    """
    Calcula el pinch y el ángulo. Es una función pura y no depende de módulos externos.
    """
    # 1. Distancia pulgar - índice
    pinch_dist = distance(thumb_tip, index_tip)
    
    # 2. Ángulo de rotación de la mano
    dx = middle_tip.x - wrist.x
    dy = middle_tip.y - wrist.y
    angle_rad = math.atan2(dy, dx)
    
    return HandCalculations(pinch_dist=pinch_dist, angle_rad=angle_rad)


# --- Clases de Servicio (SRP) ---

class FPSCounter:
    """Calcula y mantiene el estado de los FPS."""
    def __init__(self):
        self._prev_time = time.time()
        self._current_fps = 0.0

    def tick(self) -> float:
        curr_time = time.time()
        delta_time = curr_time - self._prev_time
        self._prev_time = curr_time
        self._current_fps = 1.0 / delta_time if delta_time > 0 else 0.0
        return self._current_fps

    @property
    def fps(self) -> int:
        return int(self._current_fps)

class OscClient:
    """Encapsula la lógica de inicialización y envío de mensajes OSC."""
    def __init__(self, config: OscConfig):
        self.client = None
        try:
            self.client = udp_client.SimpleUDPClient(config.IP, config.PORT)
            print(f"Cliente OSC inicializado en {config.IP}:{config.PORT}")
        except Exception as e:
            print(f"Error: No se pudo inicializar el cliente OSC. {e}")
            raise

    def send_data(self, data: OscData):
        if not self.client:
            return
        
        try:
            self.client.send_message("/hand/distance", data.distance)
            self.client.send_message("/hand/press1", data.press_left)
            self.client.send_message("/hand/press2", data.press_right)
            self.client.send_message("/hand/rotation", data.rotation)
        except Exception as e:
            print(f"Advertencia: No se pudo enviar mensaje OSC. {e}")


# --- Clase Principal de Orquestación ---

class HandOscController:
    """
    Orquesta la captura de video, procesamiento de MediaPipe,
    cálculo de métricas y envío de OSC.
    """
    def __init__(self, cam_cfg: CameraConfig, mp_cfg: MediaPipeConfig, 
                 osc_cfg: OscConfig, ctrl_cfg: ControlConfig, draw_cfg: DrawingConfig):
        
        # Almacenamiento de configuraciones
        self.mp_cfg = mp_cfg
        self.ctrl_cfg = ctrl_cfg
        self.draw_cfg = draw_cfg

        # Módulos de MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        # Inicialización de servicios
        self.osc_client = OscClient(osc_cfg)
        self.fps_counter = FPSCounter()
        
        self.hands = self.mp_hands.Hands(
            max_num_hands=self.mp_cfg.MAX_HANDS,
            min_detection_confidence=self.mp_cfg.MIN_DETECT_CONFIDENCE,
            min_tracking_confidence=self.mp_cfg.MIN_TRACK_CONFIDENCE
        )
        
        self.cap = cv2.VideoCapture(cam_cfg.SOURCE, cam_cfg.API)
        if not self.cap.isOpened():
            raise IOError(f"Error fatal: No se pudo abrir la cámara (Fuente: {cam_cfg.SOURCE}).")

        # Inicialización de estado
        self.frame_height = 0
        self.frame_width = 0

    def run(self):
        """Bucle principal de la aplicación."""
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("Advertencia: No se pudo leer frame de la cámara. Finalizando.")
                    break
                
                # Obtener dimensiones en el primer frame
                if self.frame_height == 0:
                    self.frame_height, self.frame_width, _ = frame.shape
                
                # Procesamiento de imagen
                frame, results = self._process_frame(frame)
                
                # Cálculo de métricas
                hand_metrics = self._get_hand_metrics(results)

                # Cómputo de datos OSC (si ambas manos están presentes)
                osc_data = self._compute_osc_data(hand_metrics)
                
                if osc_data:
                    self.osc_client.send_data(osc_data)
                
                # Actualizar FPS
                self.fps_counter.tick()

                # Visualización
                self._draw_overlay(frame, results, hand_metrics, osc_data)
                
                cv2.imshow("MediaPipe OSC - Hybrid Control", frame)

                if cv2.waitKey(5) & 0xFF == 27:
                    break
        finally:
            self._cleanup()

    def _process_frame(self, frame: Any) -> Tuple[Any, Any]:
        """Prepara y procesa un frame con MediaPipe."""
        frame = cv2.flip(frame, 1)
        
        frame.flags.writeable = False
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        frame.flags.writeable = True
        
        return frame, results

    def _get_hand_metrics(self, results: Any) -> Dict[str, HandCalculations]:
        """Extrae landmarks y calcula métricas para las manos detectadas."""
        hand_results = {}
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label # 'Left' o 'Right'
                
                # Extracción de landmarks
                lm = hand_landmarks.landmark
                thumb = lm[self.mp_hands.HandLandmark.THUMB_TIP]
                index = lm[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                wrist = lm[self.mp_hands.HandLandmark.WRIST]
                middle = lm[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                # Cálculo
                hand_data = calculate_hand_metrics(thumb, index, wrist, middle)
                hand_results[hand_label] = hand_data
        
        return hand_results

    def _compute_osc_data(self, hand_metrics: Dict[str, HandCalculations]) -> Optional[OscData]:
        """Calcula los valores finales de OSC si ambas manos están presentes."""
        if 'Left' not in hand_metrics or 'Right' not in hand_metrics:
            return None

        left_data = hand_metrics['Left']
        right_data = hand_metrics['Right']

        # 1) Distancia (Promedio de Pinch)
        avg_pinch = (left_data.pinch_dist + right_data.pinch_dist) / 2
        scaled_distance = scale(avg_pinch, 
                                self.ctrl_cfg.PINCH_SCALE_IN_MIN, 
                                self.ctrl_cfg.PINCH_SCALE_IN_MAX, 
                                self.ctrl_cfg.SCALE_OUT_MIN, 
                                self.ctrl_cfg.SCALE_OUT_MAX)

        # 2) "Apretar" (Binario estable)
        press_left = 1 if left_data.pinch_dist < self.ctrl_cfg.PRESS_THRESHOLD else 0
        press_right = 1 if right_data.pinch_dist < self.ctrl_cfg.PRESS_THRESHOLD else 0

        # 3) Rotación (Estable de la mano derecha)
        rotation_scaled = scale(right_data.angle_rad, 
                                self.ctrl_cfg.ANGLE_SCALE_IN_MIN, 
                                self.ctrl_cfg.ANGLE_SCALE_IN_MAX, 
                                self.ctrl_cfg.SCALE_OUT_MIN, 
                                self.ctrl_cfg.SCALE_OUT_MAX)
        
        return OscData(
            distance=scaled_distance,
            press_left=press_left,
            press_right=press_right,
            rotation=rotation_scaled
        )

    def _draw_overlay(self, frame: Any, results: Any, 
                      hand_metrics: Dict[str, HandCalculations], 
                      osc_data: Optional[OscData]):
        """Dibuja todos los elementos de visualización en el frame."""
        
        # Dibujar esqueleto y círculos de landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                cv2.circle(frame, (int(thumb_tip.x * self.frame_width), int(thumb_tip.y * self.frame_height)), 
                           8, self.draw_cfg.CIRCLE_COLOR_THUMB, -1)
                cv2.circle(frame, (int(index_tip.x * self.frame_width), int(index_tip.y * self.frame_height)), 
                           8, self.draw_cfg.CIRCLE_COLOR_INDEX, -1)

        # Dibujar datos de OSC (si existen)
        if osc_data:
            cv2.putText(frame, f"Distance (Avg Pinch): {osc_data.distance:.2f}", (10, 30),
                        self.draw_cfg.FONT, self.draw_cfg.FONT_SCALE, self.draw_cfg.TEXT_COLOR_1, self.draw_cfg.FONT_THICKNESS)
            cv2.putText(frame, f"Press L:{osc_data.press_left}  Press R:{osc_data.press_right}", (10, 60),
                        self.draw_cfg.FONT, self.draw_cfg.FONT_SCALE, self.draw_cfg.TEXT_COLOR_2, self.draw_cfg.FONT_THICKNESS)
            cv2.putText(frame, f"Rotation (Right Hand): {osc_data.rotation:.2f}", (10, 90),
                        self.draw_cfg.FONT, self.draw_cfg.FONT_SCALE, self.draw_cfg.TEXT_COLOR_3, self.draw_cfg.FONT_THICKNESS)

        # Dibujar FPS
        cv2.putText(frame, f"FPS: {self.fps_counter.fps}", (10, 120),
                    self.draw_cfg.FONT, self.draw_cfg.FONT_SCALE, self.draw_cfg.FPS_COLOR, self.draw_cfg.FONT_THICKNESS)

    def _cleanup(self):
        """Libera todos los recursos."""
        print("Finalizando aplicación. Liberando recursos...")
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        if self.hands:
            self.hands.close()

def main():
    """Punto de entrada principal de la aplicación."""
    try:
        # Inicializar configuraciones
        cam_cfg = CameraConfig()
        mp_cfg = MediaPipeConfig()
        osc_cfg = OscConfig()
        ctrl_cfg = ControlConfig()
        draw_cfg = DrawingConfig()
        
        # Crear e iniciar el controlador
        controller = HandOscController(cam_cfg, mp_cfg, osc_cfg, ctrl_cfg, draw_cfg)
        controller.run()
        
    except (IOError, Exception) as e:
        print(f"Error fatal durante la inicialización o ejecución: {e}")

if __name__ == "__main__":
    main()