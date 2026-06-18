import os
import json
import base64
import tempfile

from flask import Blueprint, request, jsonify, Response, stream_with_context, send_file

from . import pipeline
from . import datasets
from .agent import ExplainerAgent

bp = Blueprint("api", __name__, url_prefix="/api")
OUTPUT_DIR = "./generations"
DECISIONS_LOG = "user_decisions_log.json"


def _set_device(device):
    device = (device or "auto").lower()
    if device == "auto":
        os.environ.pop("PLANIFICA_DEVICE", None)
    else:
        os.environ["PLANIFICA_DEVICE"] = device


def _ndjson(events):
    body = (json.dumps(e) + "\n" for e in events)
    return Response(stream_with_context(body), mimetype="application/x-ndjson")


def _image_events(gen, styles):
    try:
        for idx, path in gen:
            with open(path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            yield {"idx": idx, "name": styles[idx], "image": f"data:image/png;base64,{b64}"}
        yield {"done": True}
    except Exception as exc:
        yield {"error": str(exc)}


@bp.get("/scenes")
def scenes():
    """Escenas disponibles localmente para el Modo 1.
    ---
    responses:
      200:
        description: Lista de IDs de escena descargados.
    """
    return jsonify(datasets.local_scenes())


@bp.get("/scenes/<scene_id>/preview")
def scene_preview(scene_id):
    """Imagen de entrada (dormitorio) de una escena descargada.
    ---
    parameters:
      - {in: path, name: scene_id, type: string, required: true}
    responses:
      200:
        description: PNG de la escena.
      404:
        description: Escena no descargada.
    """
    path = os.path.join(
        pipeline.EXTRACTED_DIR, f"{scene_id}_Bedroom", f"{scene_id}_Bedroom", "cam0", "data", "0.png"
    )
    if not os.path.exists(path):
        return jsonify({"error": "preview no disponible"}), 404
    return send_file(path, mimetype="image/png")


@bp.get("/datasets")
def list_datasets():
    """Escenas del repositorio remoto y si ya estan descargadas.
    ---
    parameters:
      - {in: query, name: refresh, type: boolean, required: false}
    responses:
      200:
        description: Lista de escenas remotas.
      502:
        description: Error consultando el repositorio.
    """
    try:
        refresh = request.args.get("refresh") == "true"
        return jsonify(datasets.list_datasets(refresh=refresh))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502


@bp.post("/datasets/<scene_id>/download")
def download_dataset(scene_id):
    """Descarga y descomprime una escena. Emite progreso NDJSON.
    ---
    parameters:
      - {in: path, name: scene_id, type: string, required: true}
    responses:
      200:
        description: Stream de progreso.
    """
    return _ndjson(datasets.download_scene(scene_id))


@bp.post("/generate")
def generate():
    """Genera 5 variaciones. Emite una imagen por evento NDJSON.
    ---
    consumes: [multipart/form-data]
    parameters:
      - {in: formData, name: mode, type: string, description: "dataset | upload"}
      - {in: formData, name: scene_id, type: string, required: false}
      - {in: formData, name: device, type: string, description: "auto | gpu | cpu"}
      - {in: formData, name: image, type: file, required: false}
    responses:
      200:
        description: Stream de imagenes generadas.
    """
    mode = request.form.get("mode", "dataset")
    _set_device(request.form.get("device", "auto"))
    styles = [s["name"] for s in pipeline.VARIATION_STYLES]

    if mode == "upload":
        f = request.files.get("image")
        if not f:
            return _ndjson([{"error": "No se envio ninguna imagen."}])
        tmp = os.path.join(tempfile.gettempdir(), f.filename or "upload.png")
        f.save(tmp)
        gen = pipeline.generate_custom_variations(tmp, OUTPUT_DIR)
    else:
        scene_id = request.form.get("scene_id")
        if not scene_id:
            return _ndjson([{"error": "Falta seleccionar una escena."}])
        scene_dir = os.path.join(pipeline.EXTRACTED_DIR, f"{scene_id}_Bedroom", f"{scene_id}_Bedroom")
        input_img = os.path.join(scene_dir, "cam0/data/0.png")
        depth = os.path.join(scene_dir, "depth0/data/0.png")
        label = os.path.join(scene_dir, "label0/data/0_nyu.png")
        depth = depth if os.path.exists(depth) else None
        label = label if os.path.exists(label) else None
        gen = pipeline.generate_variations(input_img, depth, label, OUTPUT_DIR, scene_id)

    return _ndjson(_image_events(gen, styles))


@bp.post("/report")
def report():
    """Genera el reporte del agente LLM a partir de las decisiones del usuario.
    ---
    parameters:
      - in: body
        name: body
        schema:
          type: object
          properties:
            decisions:
              type: array
              items: {type: object}
    responses:
      200:
        description: Explicacion generada.
    """
    decisions = (request.get_json(force=True) or {}).get("decisions", [])
    if len(decisions) != 5 or any(d.get("status") not in ("accepted", "rejected") for d in decisions):
        return jsonify({"error": "Debes aceptar o rechazar las 5 variaciones antes de generar el reporte."}), 400

    with open(DECISIONS_LOG, "w", encoding="utf-8") as f:
        json.dump(decisions, f, indent=2, ensure_ascii=False)

    explanation = ExplainerAgent().generate_explanation(json.dumps(decisions))
    return jsonify({"explanation": explanation})
