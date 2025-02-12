# Proyecto AI Car Command

Este proyecto tiene como objetivo desarrollar un modelo de inteligencia artificial para controlar un coche en el juego [Mr. Racer - Car Racing](https://www.1001juegos.com/juego/mr-racer---car-racing).

## Pasos para Ejecutar el Proyecto

### Paso 1: Instalar Dependencias Necesarias
Asegúrate de tener Python y pip instalados. Luego, instala los paquetes necesarios ejecutando:
```bash
pip install -r requirements.txt
```

### Paso 2: Grabar Datos durante 40 Minutos
Ejecuta el script de captura de datos para construir el conjunto de datos:
```bash
python data_capture.py
```
Este script registrará tus acciones y capturas de pantalla mientras juegas.

### Paso 3: Entrenar el Modelo
Ejecuta el script de entrenamiento para entrenar el modelo utilizando el conjunto de datos registrado:
```bash
python training.py
```
Esto realizará una validación cruzada y guardará el mejor modelo.

### Paso 4: Iniciar una Carrera en el Juego
Abre el juego y comienza una nueva carrera.

### Paso 5: Ejecutar el Script de Inferencia
Usa el modelo entrenado para controlar el coche ejecutando el script de inferencia:
```bash
python inference.py
```

Siguiendo estos pasos, podrás entrenar el modelo y entender el proceso.

## Estructura del Proyecto
- `data_capture.py`: Script para capturar datos del juego.
- `training.py`: Script para entrenar el modelo de IA.
- `inference.py`: Script para usar el modelo entrenado para controlar el coche.
- `requirements.txt`: Lista de dependencias.
- `dataset/`: Directorio donde se almacenan los datos capturados.
- `README.md`: Documentación del proyecto.

## Licencia
Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.

# AI Car Command Project

This project aims to develop an artificial intelligence model to control a car in the game [Mr. Racer - Car Racing](https://www.1001juegos.com/juego/mr-racer---car-racing).

## Steps to Run the Project

### Step 1: Install Necessary Dependencies
Ensure you have Python and pip installed. Then, install the required packages by running:
```bash
pip install -r requirements.txt
```

### Step 2: Record Data for 40 Minutes
Run the data capture script to build the dataset:
```bash
python data_capture.py
```
This script will record your actions and screenshots while you play the game.

### Step 3: Train the Model
Execute the training script to train the model using the recorded dataset:
```bash
python training.py
```
This will perform cross-validation and save the best model.

### Step 4: Start a Race in the Game
Open the game and start a new race.

### Step 5: Run the Inference Script
Use the trained model to control the car by running the inference script:
```bash
python inference.py
```

By following these steps, you will be able to train the model and understand the process.

## Project Structure
- `data_capture.py`: Script to capture gameplay data.
- `training.py`: Script to train the AI model.
- `inference.py`: Script to use the trained model for controlling the car.
- `requirements.txt`: List of dependencies.
- `dataset/`: Directory where the captured data is stored.
- `README.md`: Project documentation.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
```