# Sistema Generativo Interactivo de Transformación de Espacios con Feedback Humano y Agentes de IA

Este repositorio contiene la implementación del **Proyecto #2** para el curso **Planificación y Toma de Decisiones en IA** en la **Universidad de Ingeniería y Tecnología (UTEC)**.

El sistema es una plataforma generativa interactiva que transforma habitaciones no optimizadas (como dormitorios o salas de estar) en oficinas funcionales y estéticas (estudios), permitiendo al usuario evaluar las alternativas en tiempo real y generando explicaciones automáticas sobre sus decisiones mediante un Agente de Inteligencia Artificial (LLM).

---

## 📋 1. Definición del Problema y Solución

### El Problema
Muchos usuarios carecen de criterio estético o técnico para adaptar sus habitaciones residenciales en oficinas de trabajo funcionales. Esto genera retos en:
* **Distribución Espacial (Layout):** Ubicación ineficiente del mobiliario (ej. colocar el escritorio bloqueando el tránsito).
* **Iluminación:** Mal aprovechamiento de la luz natural o esquemas de luz inadecuados para el trabajo de escritorio.
* **Composición:** Saturación visual y falta de coherencia en el estilo.

### Nuestra Solución (Enfoque Híbrido Humano-IA)
Diseñamos un **sistema interactivo con humano en el bucle (Human-in-the-loop)** que consta de tres pilares integrados:
1. **Módulo Generativo Controlado (AI):** Usa **Stable Diffusion v1.5** condicionado por dos ControlNets en paralelo:
   * **ControlNet Depth:** Extrae la geometría 3D y límites físicos de la habitación (para no alterar el tamaño del cuarto).
   * **ControlNet Segmentation (Seg):** Toma el mapa semántico original, localiza la cama/sofá (ID de clase 4) y la reemplaza por un escritorio (ID de clase 14) para generar una nueva distribución visual.
2. **Interfaz Gráfica Interactiva (Humano):** El usuario evalúa **5 variaciones visuales** diferentes generadas por el sistema (Nórdico, Industrial, Minimalista, Acogedor de Noche y Ejecutivo), votando por ellas y dejando comentarios cualitativos.
3. **Agente de Explicabilidad (AI Agent):** Un modelo de lenguaje (LLM) que analiza las decisiones estructuradas y comentarios del usuario para sintetizar un **Reporte de Perfil Estético y Técnico** que justifica y documenta las elecciones del usuario.

---

## 🛠️ 2. Estructura del Código

El proyecto está organizado de manera modular para garantizar la reproducibilidad y escalabilidad del sistema:

* 📄 **`app.py`**: Interfaz gráfica de usuario desarrollada en **Gradio**. Administra las interacciones, pestañas de navegación y renderizado de resultados.
* 📄 **`pipeline.py`**: Pipeline de generación de imágenes. Soporta generación real por GPU usando `diffusers` de Hugging Face y un modo *fallback/simulado* en CPU que lee imágenes de oficina del dataset.
* 📄 **`agent.py`**: Agente explicador. Admite APIs de LLMs comerciales (OpenAI) y cuenta con un motor local de plantillas analíticas en caso de ejecución offline.
* 📄 **`unzip_all.py`**: Script de descompresión paralela de alta velocidad para preparar las escenas de InteriorNet.
* 📄 **`download_manager.py`**: Gestor de descarga asíncrona con soporte de reanudación y filtrado por patrones (`Study`, `Bedroom`, `Living_room`).
* 📁 **`extracted_temp/`**: Directorio donde se almacenan las escenas descomprimidas listas para la aplicación.
* 📄 **`user_decisions_log.json`**: Registro estructurado donde se graban las decisiones de aceptación/rechazo y feedback del usuario.

---

## 🚀 3. Instrucciones de Instalación y Uso

### Requisitos Previos
El sistema está diseñado para correr en sistemas Linux/WSL2 con soporte de GPU Nvidia (CUDA disponible). Para instalar todas las dependencias necesarias de PyTorch, Hugging Face e Interfaz, ejecuta:

```bash
# 1. Instalar PyTorch y Torchvision con soporte CUDA GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Instalar el resto de dependencias de IA e Interfaz
pip install diffusers transformers accelerate timm gradio Pillow opencv-python
```

### Ejecutar el Sistema

1. **Preparar el Dataset (Si usas imágenes del dataset):**
   Asegúrate de tener descargados los archivos ZIP de InteriorNet en tu carpeta `HD7/` y descomprímelos ejecutando:
   ```bash
   python unzip_all.py
   ```

2. **Iniciar la Interfaz Gráfica:**
   Corre la aplicación web de Gradio:
   ```bash
   python app.py
   ```

3. **Interactuar en el Navegador:**
   Abre el enlace que se imprima en tu consola (generalmente `http://localhost:7860`).

---

## 📖 4. Modos de Uso de la Interfaz

La aplicación web cuenta con dos pestañas de funcionamiento para cubrir diferentes escenarios de hardware y pruebas:

| Modo de Uso | Entrada del Usuario | Operación Generativa | Hardware Requerido |
| :--- | :--- | :--- | :--- |
| **Modo 1: Escenas del Dataset** | Selección por lista desplegable (ej. `3FO4IEMODOYX`) | Carga mapas de control del dataset y realiza Multi-ControlNet (GPU) o simula mostrando las imágenes de oficinas correspondientes (CPU). | CPU o GPU |
| **Modo 2: Subir Mi Propia Imagen** | Carga una foto personal (`.png` / `.jpg`) | Estima el mapa de profundidad en caliente con el modelo `Intel/dpt-hybrid-midas` y realiza la transformación generativa con `ControlNet Img2Img` con `strength=0.6`. | GPU Nvidia (Recomendado) |

---

## 📈 5. Evaluación y Métricas de Calidad

El proyecto evalúa la calidad y estabilidad del sistema a través de las siguientes dimensiones:
1. **Coherencia Espacial:** Evaluada mediante la alineación del mapa de profundidad (`depth0`). El uso de ControlNet Depth asegura que las esquinas y límites físicos no sufran distorsiones mayores a un umbral geométrico aceptable.
2. **Diversidad Estética:** El sistema produce exactamente 5 variaciones que abarcan espectros de luz diurna, nocturna, materiales rústicos, modernos y minimalistas, asegurando que el espacio de búsqueda de diseño para el usuario sea amplio.
3. **Fidelidad del Agente LLM:** Validación mediante pruebas de consistencia lógica entre el JSON de entrada (`user_decisions_log.json`) y el resumen descriptivo generado por el LLM.

---

## ⚖️ 6. Consideraciones Éticas

* **Privacidad de Datos:** Al permitir que los usuarios suban fotos personales de sus habitaciones (Modo 2), el sistema procesa las imágenes localmente. No se almacenan fotos en servidores externos de terceros, protegiendo la privacidad del espacio doméstico del usuario.
* **Sesgo de Mobiliario:** Los modelos generativos comerciales tienen un sesgo inherente hacia estilos de diseño occidentales y de alta gama. El reporte final del Agente debe considerar críticamente las limitaciones de viabilidad económica de las recomendaciones físicas generadas.