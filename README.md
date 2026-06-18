# SYNTH_LAB — Sistema Generativo Interactivo

Proyecto #2 · Planificación y Toma de Decisiones en IA · UTEC.

Transforma una habitación (dormitorio/sala) en 5 variaciones de oficina usando **Stable Diffusion 1.5 + Multi-ControlNet**, deja que el usuario acepte/rechace cada una, y un **agente LLM** genera el reporte de decisiones.

## Estructura

```
app.py              entrypoint del servidor Flask
metrics.py          evaluación (diversidad, nitidez, coherencia)
webapp/
  __init__.py       create_app() + Swagger
  routes.py         API REST
  datasets.py       descarga + descompresión de escenas (Google Drive)
  pipeline.py       generación con SD 1.5 + ControlNet
  agent.py          agente LLM (OpenAI o fallback local)
  templates/        UI
  static/           JS
docs/arquitectura.md  diagrama del sistema
```

## Uso local

```powershell
uv pip install -r requirements.txt
python app.py
```

- App: http://localhost:5000
- Swagger: http://localhost:5000/apidocs/

En la UI: elige dispositivo (CPU/GPU/AUTO), descarga una escena del repositorio, genera, evalúa las 5 y pulsa **Generar Reporte**.

## Uso con Docker

```bash
docker build -t synth-lab .
docker run -p 5000:5000 -e OPENAI_API_KEY=sk-... synth-lab
```

La imagen corre en **CPU** (lento). Para GPU usa una base `nvidia/cuda` y `--gpus all`.

## Configuración

- `OPENAI_API_KEY` — opcional; sin ella el agente usa un generador local.
- `PLANIFICA_DEVICE` — `cpu` / `gpu` / vacío (auto).
- `PLANIFICA_CPU_THREADS` — nº de hilos en CPU.

## Evaluación

```powershell
python metrics.py ./generations            # con CLIP
python metrics.py ./generations --no-clip   # solo pixeles, sin descargas
```
