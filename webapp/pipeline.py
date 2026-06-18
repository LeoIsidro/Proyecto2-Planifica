import os
import shutil
import random

try:
    import torch
    import numpy as np
    from PIL import Image
    from diffusers import (
        StableDiffusionControlNetPipeline,
        StableDiffusionControlNetImg2ImgPipeline,
        ControlNetModel,
        UniPCMultistepScheduler,
    )
    from diffusers.utils import load_image
    from transformers import pipeline as tf_pipeline

    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACTED_DIR = os.environ.get(
    "PLANIFICA_DATASET_DIR",
    os.path.join(_PROJECT_ROOT, "extracted_temp"),
)

DATASET_STUDY_DIR = os.path.join(
    EXTRACTED_DIR, "3FO4IEMODOYX_Study", "3FO4IEMODOYX_Study", "cam0", "data"
)

VARIATION_STYLES = [
    {
        "name": "Estilo Nórdico (Día)",
        "prompt": "a modern Nordic style office, white walls, light oak wood desk, bright natural sunlight from window, green plants",
        "negative": "bed, bedroom, cozy blankets, dark, low quality, bad lighting",
    },
    {
        "name": "Estilo Industrial (Cálido)",
        "prompt": "an industrial loft style office, brick wall, dark metal and rustic wood desk, warm vintage Edison light bulbs, professional layout",
        "negative": "bed, bedroom, cozy blankets, bright white light, low quality",
    },
    {
        "name": "Estilo Minimalista (Futurista)",
        "prompt": "a minimalist high-tech office study, sleek glass desk, black ergonomic task chair, clean white LED lighting, organized workspace",
        "negative": "bed, bedroom, cozy blankets, colorful, cluttered, low quality",
    },
    {
        "name": "Estilo Acogedor (Noche)",
        "prompt": "a cozy home office at night, dark wood desk, glowing yellow desk lamp, warm ambient lighting, peaceful atmosphere",
        "negative": "bed, bedroom, daylight, bright sun, low quality",
    },
    {
        "name": "Estilo Ejecutivo (Corporativo)",
        "prompt": "a professional executive study room, premium leather chair, large mahogany desk, bookshelves in the background, elegant office lighting",
        "negative": "bed, bedroom, cheap furniture, toys, low quality",
    },
]


def _resolve_device():
    if not DEEP_LEARNING_AVAILABLE:
        return None
    dev = os.environ.get("PLANIFICA_DEVICE", "").lower()
    if dev == "cpu":
        device = "cpu"
    elif dev == "gpu":
        device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(
            os.environ.get("PLANIFICA_CPU_THREADS")
            and int(os.environ["PLANIFICA_CPU_THREADS"])
            or os.cpu_count() - 1
        )
    return device


def generate_mock_variations(output_dir, scene_id=None):
    os.makedirs(output_dir, exist_ok=True)
    generated_paths = []

    study_dir = DATASET_STUDY_DIR
    if scene_id:
        study_dir = os.path.join(
            EXTRACTED_DIR, f"{scene_id}_Study", f"{scene_id}_Study", "cam0", "data"
        )

    if os.path.exists(study_dir):
        study_images = sorted([f for f in os.listdir(study_dir) if f.endswith(".png")])
        for i in range(5):
            if i < len(study_images):
                src = os.path.join(study_dir, study_images[i])
                dest = os.path.join(output_dir, f"variation_{i}.png")
                shutil.copy(src, dest)
                generated_paths.append(dest)
    else:
        print("Muestra de Study no encontrada. Generando imágenes de color de prueba.")
        from PIL import Image, ImageDraw

        colors = ["#4A90E2", "#50E3C2", "#F5A623", "#D0021B", "#9013FE"]
        for i in range(5):
            img = Image.new("RGB", (512, 512), color=colors[i])
            draw = ImageDraw.Draw(img)
            draw.text((150, 250), f"Variacion {i+1}\n{VARIATION_STYLES[i]['name']}", fill="white")
            dest = os.path.join(output_dir, f"variation_{i}.png")
            img.save(dest)
            generated_paths.append(dest)

    return generated_paths


def generate_variations(input_image_path, depth_map_path, label_map_path, output_dir, scene_id=None):
    device = _resolve_device()

    if device is None:
        print("Sin PyTorch/diffusers: modo fallback (copia vistas del dataset).")
        for i, p in enumerate(generate_mock_variations(output_dir, scene_id)):
            yield i, p
        return

    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Generación real con ControlNet en {device.upper()}.")
    if device == "cpu":
        print("CPU: esto es lento (varios minutos por imagen).")
    from PIL import Image
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    temp_seg_path = os.path.join(output_dir, "temp_seg.png")

    label_img = Image.open(label_map_path)
    label_arr = np.array(label_img)
    h, w = label_arr.shape[:2]

    seg_ade = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            val = label_arr[y, x]
            if val == 4:
                seg_ade[y, x] = [120, 120, 80]
            elif val == 1:
                seg_ade[y, x] = [120, 120, 120]
            elif val == 2:
                seg_ade[y, x] = [80, 50, 50]
            elif val == 5:
                seg_ade[y, x] = [0, 255, 255]
            elif val == 9:
                seg_ade[y, x] = [255, 0, 0]
            else:
                seg_ade[y, x] = [0, 0, 0]

    Image.fromarray(seg_ade).save(temp_seg_path)

    depth_image = load_image(depth_map_path).resize((512, 512))
    seg_image = load_image(temp_seg_path).resize((512, 512))

    controlnet_depth = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth", torch_dtype=dtype
    )
    controlnet_seg = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_seg", torch_dtype=dtype
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=[controlnet_depth, controlnet_seg],
        torch_dtype=dtype,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cpu")

    base_seed = random.randint(0, 2**31 - 1)
    for idx, style in enumerate(VARIATION_STYLES):
        print(f"Generando variación {idx+1}: {style['name']}")
        generator = torch.manual_seed(base_seed + idx)

        result = pipe(
            prompt=style["prompt"],
            negative_prompt=style["negative"],
            image=[depth_image, seg_image],
            num_inference_steps=25,
            controlnet_conditioning_scale=[0.7, 0.9],
            generator=generator,
        ).images[0]

        dest = os.path.join(output_dir, f"variation_{idx}.png")
        result.save(dest)
        yield idx, dest


def generate_custom_variations(input_image_path, output_dir):
    device = _resolve_device()
    if device is None:
        raise Exception("Se requieren PyTorch/diffusers instalados para procesar imágenes nuevas.")
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Procesando imagen personalizada en {device.upper()}.")
    if device == "cpu":
        print("CPU: esto es lento (varios minutos por imagen).")

    os.makedirs(output_dir, exist_ok=True)

    device_id = 0 if device == "cuda" else -1
    depth_estimator = tf_pipeline("depth-estimation", model="Intel/dpt-hybrid-midas", device=device_id)

    raw_img = Image.open(input_image_path).convert("RGB").resize((512, 512))
    depth_map = depth_estimator(raw_img)["depth"]

    temp_depth_path = os.path.join(output_dir, "temp_custom_depth.png")
    depth_map.save(temp_depth_path)

    controlnet_depth = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth", torch_dtype=dtype
    )

    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet_depth, torch_dtype=dtype
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cpu")

    base_seed = random.randint(0, 2**31 - 1)
    for idx, style in enumerate(VARIATION_STYLES):
        print(f"Generando variación personalizada {idx+1}: {style['name']}")
        generator = torch.manual_seed(base_seed + idx)

        result = pipe(
            prompt=style["prompt"],
            negative_prompt=style["negative"],
            image=raw_img,
            control_image=depth_map,
            strength=0.6,
            controlnet_conditioning_scale=0.8,
            num_inference_steps=25,
            generator=generator,
        ).images[0]

        dest = os.path.join(output_dir, f"variation_custom_{idx}.png")
        result.save(dest)
        yield idx, dest
