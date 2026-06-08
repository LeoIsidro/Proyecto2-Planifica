import os
import shutil
import random

# Intentar importar librerías de Deep Learning
try:
    import torch
    import numpy as np
    from PIL import Image
    from diffusers import (
        StableDiffusionControlNetPipeline, 
        StableDiffusionControlNetImg2ImgPipeline, 
        ControlNetModel, 
        UniPCMultistepScheduler
    )
    from diffusers.utils import load_image
    from transformers import pipeline as tf_pipeline
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False

# Rutas predefinidas del dataset local
DATASET_STUDY_DIR = "/home/leoisidro/CICLOS/X/PLANIFICA/proyecto_2/extracted_temp/3FO4IEMODOYX_Study/3FO4IEMODOYX_Study/cam0/data"

# Estilos predefinidos para las 5 variaciones
VARIATION_STYLES = [
    {
        "name": "Estilo Nórdico (Día)",
        "prompt": "a modern Nordic style office, white walls, light oak wood desk, bright natural sunlight from window, green plants",
        "negative": "bed, bedroom, cozy blankets, dark, low quality, bad lighting"
    },
    {
        "name": "Estilo Industrial (Cálido)",
        "prompt": "an industrial loft style office, brick wall, dark metal and rustic wood desk, warm vintage Edison light bulbs, professional layout",
        "negative": "bed, bedroom, cozy blankets, bright white light, low quality"
    },
    {
        "name": "Estilo Minimalista (Futurista)",
        "prompt": "a minimalist high-tech office study, sleek glass desk, black ergonomic task chair, clean white LED lighting, organized workspace",
        "negative": "bed, bedroom, cozy blankets, colorful, cluttered, low quality"
    },
    {
        "name": "Estilo Acogedor (Noche)",
        "prompt": "a cozy home office at night, dark wood desk, glowing yellow desk lamp, warm ambient lighting, peaceful atmosphere",
        "negative": "bed, bedroom, daylight, bright sun, low quality"
    },
    {
        "name": "Estilo Ejecutivo (Corporativo)",
        "prompt": "a professional executive study room, premium leather chair, large mahogany desk, bookshelves in the background, elegant office lighting",
        "negative": "bed, bedroom, cheap furniture, toys, low quality"
    }
]

def generate_mock_variations(output_dir, scene_id=None):
    """
    Función de fallback (CPU o sin dependencias):
    Copia 5 imágenes del dataset real de Study para simular las 5 variaciones generadas.
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_paths = []
    
    study_dir = DATASET_STUDY_DIR
    if scene_id:
        study_dir = f"/home/leoisidro/CICLOS/X/PLANIFICA/proyecto_2/extracted_temp/{scene_id}_Study/{scene_id}_Study/cam0/data"
    
    if os.path.exists(study_dir):
        # Tomamos 5 imágenes del dataset real de oficinas
        study_images = sorted([f for f in os.listdir(study_dir) if f.endswith(".png")])
        for i in range(5):
            if i < len(study_images):
                src = os.path.join(study_dir, study_images[i])
                dest = os.path.join(output_dir, f"variation_{i}.png")
                shutil.copy(src, dest)
                generated_paths.append(dest)
    else:
        # Si no existe el dataset extraído, creamos imágenes de prueba simples de colores
        print("⚠️ Muestra de Study no encontrada. Generando imágenes de color de prueba.")
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
    """
    Genera 5 variaciones visuales utilizando Stable Diffusion + ControlNet para imágenes del dataset.
    Si CUDA no está disponible o faltan dependencias, usa el modo fallback/mock.
    """
    use_gpu = DEEP_LEARNING_AVAILABLE and torch.cuda.is_available()
    
    if not use_gpu:
        print("🤖 Ejecutando en MODO FALLBACK (Copia vistas de oficina del dataset original)...")
        return generate_mock_variations(output_dir, scene_id)
        
    print("🚀 Ejecutando en MODO GPU (Generación real con ControlNet)...")
    from PIL import Image
    import numpy as np
    
    os.makedirs(output_dir, exist_ok=True)
    generated_paths = []
    
    # 1. Modificar la máscara semántica (cama -> escritorio)
    temp_seg_path = os.path.join(output_dir, "temp_seg.png")
    
    label_img = Image.open(label_map_path)
    label_arr = np.array(label_img)
    h, w = label_arr.shape[:2]
    
    # ADE20K colores: desk = [120, 120, 80], chair = [0, 255, 255]
    seg_ade = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            val = label_arr[y, x]
            if val == 4: # cama -> escritorio
                seg_ade[y, x] = [120, 120, 80]
            elif val == 1: # pared
                seg_ade[y, x] = [120, 120, 120]
            elif val == 2: # piso
                seg_ade[y, x] = [80, 50, 50]
            elif val == 5: # silla
                seg_ade[y, x] = [0, 255, 255]
            elif val == 9: # ventana
                seg_ade[y, x] = [255, 0, 0]
            else:
                seg_ade[y, x] = [0, 0, 0]
                
    Image.fromarray(seg_ade).save(temp_seg_path)
    
    # 2. Cargar imágenes guía para ControlNet
    depth_image = load_image(depth_map_path).resize((512, 512))
    seg_image = load_image(temp_seg_path).resize((512, 512))
    
    # 3. Inicializar ControlNets y Pipeline
    controlnet_depth = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16
    )
    controlnet_seg = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_seg", torch_dtype=torch.float16
    )
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=[controlnet_depth, controlnet_seg],
        torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    
    # 4. Generar las 5 variaciones
    for idx, style in enumerate(VARIATION_STYLES):
        print(f"\n🎨 Generando Variación {idx+1}: {style['name']}...")
        generator = torch.manual_seed(42 + idx)
        
        result = pipe(
            prompt=style["prompt"],
            negative_prompt=style["negative"],
            image=[depth_image, seg_image],
            num_inference_steps=25,
            controlnet_conditioning_scale=[0.7, 0.9],
            generator=generator
        ).images[0]
        
        dest = os.path.join(output_dir, f"variation_{idx}.png")
        result.save(dest)
        generated_paths.append(dest)
        
    return generated_paths

def generate_custom_variations(input_image_path, output_dir):
    """
    Genera 5 variaciones de una imagen propia cargada por el usuario (en caliente usando GPU).
    Usa estimación de profundidad al vuelo y StableDiffusionControlNetImg2ImgPipeline.
    """
    use_gpu = DEEP_LEARNING_AVAILABLE and torch.cuda.is_available()
    if not use_gpu:
        raise Exception("❌ Se requiere GPU Nvidia con CUDA y dependencias instaladas para procesar imágenes nuevas en caliente.")
        
    print(f"\n--- PROCESANDO IMAGEN PERSONALIZADA EN GPU ---")
    print(f"Cargando imagen: {input_image_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    generated_paths = []
    
    # 1. Estimar Mapa de Profundidad al vuelo
    device_id = 0 # GPU principal
    print("Estimando profundidad con Intel/dpt-hybrid-midas...")
    depth_estimator = tf_pipeline("depth-estimation", model="Intel/dpt-hybrid-midas", device=device_id)
    
    raw_img = Image.open(input_image_path).convert("RGB").resize((512, 512))
    depth_res = depth_estimator(raw_img)
    depth_map = depth_res["depth"]
    
    # Guardar mapa de profundidad temporal
    temp_depth_path = os.path.join(output_dir, "temp_custom_depth.png")
    depth_map.save(temp_depth_path)
    
    # 2. Cargar ControlNet Depth y Pipeline de Inferencia Img2Img
    print("Cargando pipeline generativo ControlNet Img2Img...")
    controlnet_depth = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16
    )
    
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet_depth,
        torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    
    # 3. Generar las 5 variaciones usando Img2Img guiado por la profundidad
    # El valor de strength (0.6) permite rediseñar y reemplazar muebles (cama -> escritorio)
    # manteniendo la estructura tridimensional del cuarto (paredes, ventanas)
    for idx, style in enumerate(VARIATION_STYLES):
        print(f"\n🎨 Generando Variación Personalizada {idx+1}: {style['name']}...")
        generator = torch.manual_seed(42 + idx)
        
        result = pipe(
            prompt=style["prompt"],
            negative_prompt=style["negative"],
            image=raw_img,
            control_image=depth_map,
            strength=0.6,  # 0.6 es el balance perfecto para rediseñar muebles
            controlnet_conditioning_scale=0.8,
            num_inference_steps=25,
            generator=generator
        ).images[0]
        
        dest = os.path.join(output_dir, f"variation_custom_{idx}.png")
        result.save(dest)
        generated_paths.append(dest)
        
    return generated_paths
