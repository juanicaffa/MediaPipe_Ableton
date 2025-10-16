Este código Python implementa un sistema para controlar música o efectos en tiempo real (típicamente en un DAW como Ableton Live) mediante el movimiento de tus manos, capturado por una cámara web. 
Utiliza MediaPipe para el seguimiento de manos y Open Sound Control (OSC) para la comunicación.

Transforma las posiciones y distancias de los dedos detectados por la cámara en mensajes OSC escalados, permitiendo controlar parámetros en cualquier aplicación compatible (como Ableton Live, Sonic Pi, TouchDesigner, etc.)

Librerias utilizadas: https://docs.google.com/spreadsheets/d/1XI-NycKVXZZHhmrqolcT3TdjLXnHpSrmq_ga8YA0RTI/edit?gid=138328323#gid=138328323

INSTALACION: 

1. Clonar el Repositorio

git clone https://github.com/Juanicaffa/mediapipe-osc-controller.git
cd mediapipe-osc-controller

2. Crear y Activar Entorno Virtual (Recomendado)
python -m venv venv
#Windows:
.\venv\Scripts\activate

#macOS/Linux:
source venv/bin/activate

4. Instalar Dependencias
Instala las librerías requeridas utilizando pip.

pip install opencv-python mediapipe python-osc


Configuración y Funcionalidad Clave

1. Configuración OSC
Se establece la conexión para enviar datos.

OSC_IP = "127.0.0.1": Indica que los mensajes se enviarán al mismo equipo (localhost).

OSC_PORT = 4567: El puerto que debe estar configurado en la aplicación de destino (ej. un bridge OSC-MIDI o directamente en el DAW).

Se inicializa un cliente (SimpleUDPClient) para el envío de datos.

2. Función de Escalado (scale)
Una función vital para la asignación MIDI/OSC.

Propósito: Transforma un valor crudo de entrada (como la distancia entre dos dedos, que varía de 0 a 1) en un valor dentro de un rango de salida (típicamente de 0 a 1 para control de parámetros flotantes, o de 0 a 127 para valores MIDI).

Incluye saturación (max(min(...))) para asegurar que el valor escalado siempre se mantenga dentro de los límites definidos (out_min y out_max).

3. Seguimiento con MediaPipe Hands
Se configura el modelo de manos (mp_hands.Hands) para detectar hasta dos manos (max_num_hands=2) y con una alta fiabilidad de detección y seguimiento (min_detection_confidence=0.7, min_tracking_confidence=0.7).


Bucle Principal

El corazón del script :

1. Captura y Preprocesamiento
Se lee un frame de la cámara (cap.read()).

La imagen se convierte de BGR a RGB, ya que MediaPipe requiere el formato RGB para su procesamiento.

2. Detección y Extracción de Gestos
El script itera sobre las manos detectadas (hasta dos):

Puntos Clave: Se extraen las coordenadas de los landmarks esenciales:

thumb (Pulgar, punto 4)

index (Índice, punto 8)

palm_center (Centro de la palma, punto 0)

Cálculo de Distancias:

dist_thumb_index: Distancia crucial para medir el apretón o pinza.

dist_palm_index: Distancia desde el centro de la palma hasta el índice, útil para medir la extensión del dedo o la posición general.

3. Suavizado (Smoothing)
Se aplica un filtro de baja frecuencia (Low-Pass Filter) a los valores de posición y distancia utilizando el factor alpha = 0.25.

Esto es esencial para evitar el nerviosismo o jittering de los datos de la cámara, resultando en un control de audio mucho más musical y estable.

4. Escalado y Envío OSC
Los valores suavizados se escalan de su rango crudo a un rango de 0 a 1 y se envían a través de OSC, mapeados a diferentes direcciones según la mano:

ESCALADO Y ENVIO: https://docs.google.com/spreadsheets/d/1XI-NycKVXZZHhmrqolcT3TdjLXnHpSrmq_ga8YA0RTI/edit?gid=138328323#gid=138328323

5. Visualización (Debugging)
Dibujo: Se dibujan los landmarks y las conexiones sobre el frame.

Overlay: Se superponen los valores escalados (thumb_index_scaled, palm_index_scaled) y el cálculo de FPS (Frames por Segundo) para debugging en tiempo real.

El loop finaliza al presionar la tecla ESC

