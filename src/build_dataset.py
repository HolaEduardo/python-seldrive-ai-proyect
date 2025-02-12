""" MIT License

Copyright (c) 2025 HackConEdu - Eduardo Jara

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

import pyautogui
import os
import csv
import cv2
import time
import numpy as np
from mss import mss
from pynput import keyboard

# Crear directorio de salida
output_dir = "./dataset"
os.makedirs(output_dir, exist_ok=True)

# Crear y abrir archivo CSV donde se registran las acciones
csv_filename = os.path.join(output_dir, "actions.csv")
csv_file = open(csv_filename, "w", newline="")
csv_writer = csv.writer(csv_file)

# Escribir encabezados en el CSV
csv_writer.writerow(["timestamp", "action_type", "aceleration", "direction", "screenshot_path"])

# Teclas de control
ACCELERATION_KEYS = {"w", "s"}  # Adelante y atrás
DIRECTION_KEYS = {"a", "d"}  # Izquierda y derecha
current_pressed = set()

# Altura de la línea de cruce
LINE_Y = 300

# Obtenemos
screen_width, screen_height = pyautogui.size()
monitor = {"top": 0, "left": 0, "width": screen_width, "height": screen_height}
sct = mss()
# Función para capturar frames
def on_press(key_pressed):
    try:
        key_str = key_pressed.char.lower()
        if key_str in ACCELERATION_KEYS or key_str in DIRECTION_KEYS:
            current_pressed.add(key_str)
    except AttributeError:
        pass

# Capturar teclas liberadas, función que se ejecuta al soltar una tecla
def on_release(key_dismiss):
    try:
        key_str = key_dismiss.char.lower()
        if key_str in ACCELERATION_KEYS or key_str in DIRECTION_KEYS:
            current_pressed.discard(key_str)
    except AttributeError:
        pass

# Dibujar carriles estáticos
def draw_static_lanes(frame):
    cv2.line(frame, (-100, 1000), (910, 0), (0, 255, 0), 8)
    cv2.line(frame, (2000, 1000), (910, -100), (0, 255, 0), 8)

    h, w = frame.shape[:2]
    cv2.line(frame, (0, LINE_Y), (w, LINE_Y), (255, 0, 0), 2)
    return frame

# Iniciar captura de teclas
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# Funcion para realizar toma de datos.
def main():
    # Definimos variables globales
    global COUNTER, acceleration_message, direction_message
    try:
        # Mientras no detengamos o presionemos "q", el bucle se seguirá ejecutando
        while True:
            # Obtener el tiempo actual
            timestamp = time.time()

            # Ruta de la captura
            screenshot_path = os.path.join(output_dir, f"frame_{timestamp}.png")

            # Capturar pantalla
            screenshot = sct.grab(monitor)

            # Convertir captura a un array de numpy
            screenshot_array = np.array(screenshot)

            # Convertir captura a RGB
            screenshot_rgb = cv2.cvtColor(screenshot_array, cv2.COLOR_BGRA2BGR)


            acceleration = "none"
            direction = "none"

            # Verificar las teclas presionadas
            for key in current_pressed:
                if key in ACCELERATION_KEYS:
                    acceleration = key  # Guarda 'w' o 's'
                elif key in DIRECTION_KEYS:
                    direction = key  # Guarda 'a' o 'd'


            annotated_frame = draw_static_lanes(screenshot_rgb)

            # Mostrar la captura con las detecciones
            cv2.imshow("Captura", annotated_frame)

            # Guardar la captura con la acción
            cv2.imwrite(output_dir, annotated_frame)

            # Guardar la acción en el archivo CSV
            csv_writer.writerow([timestamp, "keys", acceleration, direction, screenshot_path])
            csv_file.flush()

            # Aumentar contador de capturas
            if acceleration == 'w':
                acceleration_message = "Acelerando"
            elif acceleration == 's':
                acceleration_message = "Frenando"
            elif acceleration == 'none':
                acceleration_message = "Sin aceleración"

            if direction == 'none':
                direction_message = "Sin dirección"
            if direction == 'a':
                direction_message = "Izquierda"

            elif direction == 'd':
                direction_message = "Derecha"

            COUNTER += 1

            print(f"Captura {COUNTER}, aceleración: {acceleration_message}, dirección: {direction_message}")

            # Función para terminar la captura utilizando la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q') or COUNTER == 8000:
                break

        # Liberar recursos de OpenCV
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        print("Captura detenida por el usuario.")
    finally:
        # Cerrar el archivo CSV y detener el listener
        csv_file.close()
        listener.stop()

        # Imprimir mensaje de finalización
        print("Se guardaron" + str(COUNTER) + " capturas en " + output_dir)

if __name__ == "__main__":
    main()