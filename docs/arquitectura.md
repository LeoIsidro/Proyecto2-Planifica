# Arquitectura del sistema

Diagrama de flujo de información y decisiones. Pegar en https://mermaid.live para exportar PNG/SVG.

```mermaid
flowchart TD
    U([Usuario]) -->|1. Elige escena o sube imagen| UI[Interfaz Web<br/>Flask + HTML/JS]
    UI -->|2a. dataset: scene_id| GEN[/POST /api/generate/]
    UI -->|2b. upload: imagen| GEN
    UI -->|0. GET escena del repo| DS[(Repositorio remoto<br/>Google Drive)]
    DS -->|descarga + descompresión<br/>webapp/datasets.py| LOCAL[(extracted_temp/)]
    LOCAL --> GEN

    GEN --> PIPE[pipeline.py<br/>Módulo generativo]
    PIPE -->|CPU / GPU / AUTO| DEV{Dispositivo}
    PIPE --> CN[Stable Diffusion v1.5<br/>+ Multi-ControlNet<br/>Depth + Segmentation]
    CN -->|stream NDJSON<br/>1 imagen por evento| UI

    UI -->|3. 5 variaciones| EVAL[Evaluación humana<br/>aceptar / rechazar + comentario]
    EVAL -->|4. POST /api/report| LOG[(user_decisions_log.json<br/>registro estructurado)]
    LOG --> AGENT[agent.py<br/>Agente LLM]
    AGENT -->|OpenAI o plantilla local| REPORT[Reporte de síntesis<br/>aceptadas / rechazadas]
    REPORT --> UI

    EVAL -.->|métricas offline| MET[metrics.py<br/>diversidad · nitidez · coherencia]
```

## Componentes

- **Interfaz Web (`webapp/`)**: Flask sirve la SPA; API REST documentada con Swagger en `/apidocs/`.
- **Repositorio de datos (`webapp/datasets.py`)**: lista escenas de InteriorNet desde Drive, descarga y descomprime bajo demanda.
- **Módulo generativo (`pipeline.py`)**: SD 1.5 con dos ControlNet (profundidad + segmentación) en GPU/CPU; emite cada variación conforme se genera.
- **Registro de decisiones**: cada aceptación/rechazo + comentario se persiste en `user_decisions_log.json`.
- **Agente de explicabilidad (`agent.py`)**: sintetiza un reporte coherente con las decisiones (OpenAI o fallback local).
- **Evaluación (`metrics.py`)**: diversidad perceptual, nitidez y coherencia decisión↔reporte.
