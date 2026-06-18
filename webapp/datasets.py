import os
import json
import zipfile

import gdown

from .pipeline import EXTRACTED_DIR

DRIVE_FOLDER_ID = "1bAyIR3KvCbkhinDRHX2xeWgT3F_VSLAq"
_BASE = os.path.dirname(EXTRACTED_DIR)
ZIP_DIR = os.path.join(_BASE, "HD7")
CACHE_FILE = os.path.join(_BASE, "drive_scenes.json")
KINDS = ("Bedroom", "Study", "Living_room")


def _drive_index(refresh=False):
    if not refresh and os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)

    files = gdown.download_folder(id=DRIVE_FOLDER_ID, skip_download=True, quiet=True)
    scenes = {}
    for item in files:
        name = os.path.basename(item.path)
        for kind in KINDS:
            suffix = f"_{kind}.zip"
            if name.endswith(suffix):
                sid = name[: -len(suffix)]
                scenes.setdefault(sid, {})[kind] = item.id

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(scenes, f, indent=2)
    return scenes


def local_scenes():
    if not os.path.isdir(EXTRACTED_DIR):
        return []
    ids = set()
    for name in os.listdir(EXTRACTED_DIR):
        for kind in ("_Bedroom", "_Study"):
            if name.endswith(kind):
                ids.add(name[: -len(kind)])
    return sorted(
        sid
        for sid in ids
        if os.path.isdir(os.path.join(EXTRACTED_DIR, f"{sid}_Bedroom"))
        and os.path.isdir(os.path.join(EXTRACTED_DIR, f"{sid}_Study"))
    )


def list_datasets(refresh=False):
    index = _drive_index(refresh=refresh)
    local = set(local_scenes())
    out = []
    for sid, parts in sorted(index.items()):
        if "Bedroom" in parts and "Study" in parts:
            out.append({"id": sid, "downloaded": sid in local})
    return out


def download_scene(scene_id):
    index = _drive_index()
    parts = index.get(scene_id)
    if not parts:
        yield {"error": f"La escena {scene_id} no existe en el repositorio remoto."}
        return

    os.makedirs(ZIP_DIR, exist_ok=True)
    os.makedirs(EXTRACTED_DIR, exist_ok=True)

    for kind in ("Bedroom", "Study"):
        if kind not in parts:
            continue
        name = f"{scene_id}_{kind}"
        target = os.path.join(EXTRACTED_DIR, name)
        if os.path.isdir(target) and os.listdir(target):
            yield {"step": f"{name}: ya disponible localmente."}
            continue

        zip_path = os.path.join(ZIP_DIR, f"{name}.zip")
        yield {"step": f"{name}: descargando..."}
        gdown.download(id=parts[kind], output=zip_path, quiet=True)

        yield {"step": f"{name}: descomprimiendo..."}
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target)

    yield {"done": True, "scene_id": scene_id}
