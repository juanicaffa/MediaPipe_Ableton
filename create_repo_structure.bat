@echo off
REM -------------------------------------------------------
REM create_repo_structure.bat
REM Ejecutar desde la raíz del repo (Windows CMD)
REM -------------------------------------------------------

REM 1) Crear carpetas
mkdir src
mkdir examples
mkdir tests
mkdir docs
mkdir assets
mkdir assets\screenshots

REM 2) Mover archivos .py existentes a src (si hay .py en root)
REM Si no querés mover alguno en particular, no ejecutes esta línea.
move /Y "*.py" src >nul 2>&1

REM 3) Crear .gitignore
(
echo # Python
echo __pycache__/
echo *.py[cod]
echo *.pyo
echo venv/
echo env/
echo .venv/
echo pip-log.txt
echo pip-delete-this-directory.txt
echo htmlcov/
echo .pytest_cache/
echo .cache/
echo .mypy_cache/
echo .vscode/
echo .idea/
echo .DS_Store
echo *.egg-info/
echo dist/
echo build/
echo *.log
) > .gitignore

REM 4) Crear requirements.txt con dependencias sugeridas
(
echo opencv-python>=4.6.0
echo mediapipe>=0.10.0
echo python-osc>=1.8.2
echo numpy>=1.23.0
) > requirements.txt

REM 5) Crear README.md básico
(
echo # MediaPipe_Ableton
echo.
echo Sistema para controlar Ableton Live (u otras aplicaciones) utilizando detecci^on de manos con MediaPipe y enviando mensajes OSC.
echo.
echo ## Instalaci^on
echo 1. Clonar el repositorio
echo ^```bat
echo git clone https://github.com/tu_usuario/MediaPipe_Ableton.git
echo cd MediaPipe_Ableton
echo ^```
echo.
echo 2. Crear y activar entorno virtual
echo ^```bat
echo python -m venv venv
echo venv\Scripts\activate
echo pip install -r requirements.txt
echo ^```
echo.
echo ## Uso
echo Ejecutar el script principal:
echo ^```bat
echo python src\main.py
echo ^```
echo.
echo ## Estructura
echo ^```
echo /src
echo /examples
echo /tests
echo /docs
echo /assets
echo README.md
echo LICENSE
echo CHANGELOG.md
echo CONTRIBUTING.md
echo ^```
echo.
echo ## Licencia
echo Revisar el archivo LICENSE.
) > README.md

REM 6) Crear LICENSE (MIT) - REEMPLAZAR NOMBRE_AQUI y AÑO_AQUI
(
echo MIT License
echo.
echo Copyright (c) AÑO_AQUI NOMBRE_AQUI
echo.
echo Permission is hereby granted, free of charge, to any person obtaining a copy
echo of this software and associated documentation files (the "Software"), to deal
echo in the Software without restriction, including without limitation the rights
echo to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
echo copies of the Software, and to permit persons to whom the Software is
echo furnished to do so, subject to the following conditions:
echo.
echo THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
echo IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
echo FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
echo AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
echo LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
echo OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
echo SOFTWARE.
) > LICENSE

REM 7) Crear CONTRIBUTING.md básico
(
echo # Contributing
echo Gracias por querer contribuir. Por favor:
echo 1. Haz fork del repo.
echo 2. Crea una rama: feature/nombre-o bugfix/descripcion.
echo 3. Abre un Pull Request con una descripci^on clara.
echo 4. Sigue los tests y el estilo (PEP8).
) > CONTRIBUTING.md

REM 8) Crear CHANGELOG.md inicial
(
echo # Changelog
echo Todas las changes importantes en este repo seguir^an el versionado sem^antico.
echo.
echo ## [Unreleased]
echo - Inicial
) > CHANGELOG.md

REM 9) Crear archivo de ejemplo principal en src (main.py) si no existe
if not exist src\main.py (
  (
  echo # -*- coding: utf-8 -*-
  echo "Ejecutar este script para iniciar el envio OSC desde MediaPipe"
  echo
  echo if __name__ == "__main__":
  echo     print("Coloca aqu'i tu flujo principal. Ej: importa mediapipe, procesa webcam y envia OSC.")
  ) > src\main.py
)

REM 10) Crear un ejemplo en examples
(
echo # Ejemplo: simple_control.py
echo # Ejecutar como ejemplo: python examples\simple_control.py
) > examples\simple_control.py

REM 11) Crear tests placeholder
(
echo import unittest
echo
echo class TestPlaceholder(unittest.TestCase):
echo     def test_true(self):
echo         self.assertTrue(True)
echo
echo if __name__ == "__main__":
echo     unittest.main()
) > tests\test_placeholder.py

REM 12) Crear docs/usage_guide.md
(
echo # Guía de uso
echo Documentación de c^omo configurar OSC, puertos y ejemplos.
) > docs\usage_guide.md

REM 13) Git init / primer commit (si no hay repo)
if not exist .git (
  git init
  git add .
  git commit -m "chore: init repo structure + base files"
  git tag v1.0.0
) else (
  echo .git ya existe, saltando git init/commit.
)

echo.
echo ESTRUCTURA CREADA!
echo Revisar LICENSE y reemplazar AÑO_AQUI y NOMBRE_AQUI por tus datos.
echo Si no queres que se muevan los .py existentes, revive el archivo y elimina la linea de move.
pause
