import os
import sys
import json
import itertools

import numpy as np
from PIL import Image


def _load(path, size=256):
    return np.asarray(Image.open(path).convert("RGB").resize((size, size)), dtype=np.float32)


def list_generations(folder):
    files = [f for f in sorted(os.listdir(folder)) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    return [os.path.join(folder, f) for f in files if f.startswith("variation")]


def sharpness(paths):
    try:
        import cv2
    except ImportError:
        return None
    out = {}
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        out[os.path.basename(p)] = float(cv2.Laplacian(img, cv2.CV_64F).var())
    return out


def _clip_embeddings(paths):
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    imgs = [Image.open(p).convert("RGB") for p in paths]
    inputs = proc(images=imgs, return_tensors="pt")
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()


def diversity(paths, use_clip=True):
    if use_clip:
        try:
            emb = _clip_embeddings(paths)
            dists = [1.0 - float(np.dot(a, b)) for a, b in itertools.combinations(emb, 2)]
            return {"method": "clip_cosine", "mean_pairwise_distance": float(np.mean(dists)),
                    "min": float(np.min(dists)), "max": float(np.max(dists))}
        except Exception as exc:
            print(f"[metrics] CLIP no disponible ({exc}); usando distancia de pixeles.")
    vecs = [_load(p).flatten() for p in paths]
    vecs = [v / (np.linalg.norm(v) + 1e-8) for v in vecs]
    dists = [float(np.linalg.norm(a - b)) for a, b in itertools.combinations(vecs, 2)]
    return {"method": "pixel_l2", "mean_pairwise_distance": float(np.mean(dists)),
            "min": float(np.min(dists)), "max": float(np.max(dists))}


def coherence(decisions, report_text):
    report = (report_text or "").lower()
    total = len(decisions)
    accepted = [d for d in decisions if d.get("status") == "accepted"]
    rejected = [d for d in decisions if d.get("status") == "rejected"]
    named = sum(1 for d in decisions if d.get("style_name", "").lower() in report)
    counts_ok = str(len(accepted)) in report and str(len(rejected)) in report
    return {
        "accepted": len(accepted),
        "rejected": len(rejected),
        "styles_named_in_report": named,
        "styles_named_ratio": named / total if total else 0.0,
        "counts_match_report": bool(counts_ok),
    }


def evaluate(folder="./generations", decisions_path="user_decisions_log.json", report_text=None, use_clip=True):
    paths = list_generations(folder)
    result = {"n_images": len(paths), "diversity": diversity(paths, use_clip) if paths else None,
              "sharpness": sharpness(paths)}
    if os.path.exists(decisions_path) and report_text is not None:
        with open(decisions_path, encoding="utf-8") as f:
            result["coherence"] = coherence(json.load(f), report_text)
    return result


def _selfcheck():
    import tempfile
    d = tempfile.mkdtemp()
    base = Image.new("RGB", (64, 64), "white")
    paths = []
    for i in range(3):
        p = os.path.join(d, f"variation_{i}.png")
        base.save(p)
        paths.append(p)
    div = diversity(paths, use_clip=False)
    assert div["mean_pairwise_distance"] < 1e-3, "imagenes identicas => diversidad ~0"
    decs = [{"style_name": "Nordico", "status": "accepted"}, {"style_name": "Industrial", "status": "rejected"}]
    coh = coherence(decs, "Se acepto Nordico y se rechazo Industrial. Aceptadas: 1, Rechazadas: 1.")
    assert coh["styles_named_ratio"] == 1.0 and coh["counts_match_report"]
    print("selfcheck OK")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selfcheck":
        _selfcheck()
    else:
        folder = sys.argv[1] if len(sys.argv) > 1 else "./generations"
        print(json.dumps(evaluate(folder, use_clip="--no-clip" not in sys.argv), indent=2, ensure_ascii=False))
