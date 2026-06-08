import os
import sys
import json
import zipfile
import subprocess

# Instrucciones de dependencias
REQUIRED_PACKAGES = [
    "torch", "torchvision", "diffusers", "transformers", "accelerate", "numpy", "Pillow", "opencv-python"
]

def check_dependencies():
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            if pkg == "Pillow":
                __import__("PIL")
            elif pkg == "opencv-python":
                __import__("cv2")
            else:
                __import__(pkg.replace("-", "_").lower())
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print("⚠️  Faltan algunas dependencias necesarias para ejecutar el pipeline:")
        print("   " + ", ".join(missing))
        print("\n👉 Puedes instalarlas todas juntas ejecutando el siguiente comando:")
        print(f"   pip install {' '.join(missing)}")
        print("\nUna vez instaladas, vuelve a ejecutar este script.")
        return False
    return True

def edit_segmentation_mask(label_path, output_path):
    """
    Lee la máscara de segmentación semántica de InteriorNet, mapea los IDs
    a los colores RGB del formato ADE20K (que espera ControlNet Seg)
    y reemplaza la categoría de 'Cama' por 'Escritorio'.
    """
    from PIL import Image
    import numpy as np

    print(f"Cargando mapa de segmentación semántica desde: {label_path}")
    label_img = Image.open(label_path)
    label_arr = np.array(label_img)

    # Definir paleta de colores ADE20K para ControlNet Seg
    # ControlNet Seg de lllyasviel fue entrenado con el dataset ADE20K.
    # Colores RGB ADE20K estándar:
    ADE_COLORS = {
        "wall": [120, 120, 120],       # ID 1 en InteriorNet
        "floor": [80, 50, 50],         # ID 2 en InteriorNet
        "chair": [0, 255, 255],        # ID 5 en InteriorNet
        "sofa": [0, 0, 255],           # ID 6 en InteriorNet
        "table": [255, 224, 32],       # ID 7 en InteriorNet
        "window": [255, 0, 0],         # ID 9 en InteriorNet
        "bookshelf": [0, 150, 150],    # ID 10 en InteriorNet
        "bed": [0, 0, 180],            # ID 4 en InteriorNet (Dormitorio original)
        "desk": [120, 120, 80],        # ID 14 en InteriorNet (Nuestra meta)
        "door": [255, 255, 255],       # ID 8 en InteriorNet
        "background": [0, 0, 0]        # Otros
    }

    # Crear una imagen RGB vacía para el mapa de ControlNet Seg
    h, w = label_arr.shape[:2]
    seg_ade = np.zeros((h, w, 3), dtype=np.uint8)

    # Mapeo simple de IDs de InteriorNet a colores ADE20K
    # Nota: InteriorNet guarda el ID de categoría directamente en el pixel
    # Categoría 4 = Cama (Bed), Categoría 14 = Escritorio (Desk)
    for y in range(h):
        for x in range(w):
            val = label_arr[y, x]
            
            # --- ESTRATEGIA DE DISTRIBUCIÓN (LAYOUT EDIT) ---
            # Si el pixel pertenece a una cama (ID 4), lo transformamos en escritorio (ID 14)
            if val == 4:
                seg_ade[y, x] = ADE_COLORS["desk"]  # Cambiar cama por escritorio
            elif val == 1:
                seg_ade[y, x] = ADE_COLORS["wall"]
            elif val == 2:
                seg_ade[y, x] = ADE_COLORS["floor"]
            elif val == 5:
                seg_ade[y, x] = ADE_COLORS["chair"]
            elif val == 6:
                seg_ade[y, x] = ADE_COLORS["sofa"]
            elif val == 7:
                seg_ade[y, x] = ADE_COLORS["table"]
            elif val == 9:
                seg_ade[y, x] = ADE_COLORS["window"]
            elif val == 10:
                seg_ade[y, x] = ADE_COLORS["bookshelf"]
            elif val == 14:
                seg_ade[y, x] = ADE_COLORS["desk"]
            elif val == 8:
                seg_ade[y, x] = ADE_COLORS["door"]
            else:
                seg_ade[y, x] = ADE_COLORS["background"]

    # Guardar máscara modificada
    Image.fromarray(seg_ade).save(output_path)
    print(f"✅ Máscara ADE20K editada guardada exitosamente en: {output_path}")
    print("   (Se reemplazaron todos los pixeles de 'Cama' por 'Escritorio')")

def run_diffusion_pipeline(depth_path, seg_path, output_image_path):
    """
    Inicializa el pipeline de Stable Diffusion con Multi-ControlNet (Depth + Segmentation)
    y genera la imagen final de la habitación transformada.
    """
    import torch
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
    from diffusers.utils import load_image
    
    print("\n--- INICIANDO PIPELINE GENERATIVO CONTROLADO ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Ejecutando en: {device.upper()}")
    if device == "cpu":
        print("⚠️  Advertencia: Ejecutar Stable Diffusion en CPU puede tardar varios minutos por imagen.")

    # 1. Cargar las imágenes guía
    depth_image = load_image(depth_path).resize((512, 512))
    seg_image = load_image(seg_path).resize((512, 512))

    # 2. Inicializar los modelos de ControlNet
    print("Descargando/Cargando pesos de ControlNet (Depth y Segmentation)...")
    controlnet_depth = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float32 if device == "cpu" else torch.float16
    )
    controlnet_seg = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_seg", torch_dtype=torch.float32 if device == "cpu" else torch.float16
    )

    # 3. Inicializar el pipeline de Stable Diffusion
    print("Cargando modelo base Stable Diffusion v1.5...")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=[controlnet_depth, controlnet_seg],
        torch_dtype=torch.float32 if device == "cpu" else torch.float16
    )
    
    # Optimizar scheduler y memoria
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if device == "cuda":
        pipe.enable_model_cpu_offload()
        # pipe.to("cuda") # Descomentar si tienes suficiente VRAM (>8GB)
    
    # 4. Prompting y generación
    prompt = "a modern professional office, a clean wooden desk, comfortable ergonomic office chair, windows with bright natural daylight, highly detailed, photorealistic render"
    negative_prompt = "bed, bedroom, cozy blankets, messy room, low quality, blurry, distorted furniture, bad lighting"
    
    print(f"Generando transformación con prompt: '{prompt}'...")
    
    # Ejecutar inferencia
    # Enviamos ambas imágenes de control. La escala de fuerza (controlnet_conditioning_scale)
    # equilibra cuánto obedecer a la profundidad (estructura del cuarto) y a la segmentación (muebles).
    generator = torch.manual_seed(42)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=[depth_image, seg_image],
        num_inference_steps=30,
        controlnet_conditioning_scale=[0.7, 0.9], # Damos más peso al seg para cambiar muebles
        generator=generator
    ).images[0]

    # Guardar resultado
    result.save(output_image_path)
    print(f"🎉 ¡Transformación exitosa! Imagen guardada en: {output_image_path}")

def main():
    print("=== PIPELINE DE TRANSFORMACIÓN DE HABITACIÓN A OFICINA ===")
    
    if not check_dependencies():
        return
        
    # Definir rutas locales de la muestra extraída
    # (Usando la escena 3FO4IEMODOYX que ya se encuentra descomprimida en extracted_temp)
    scene_dir = "/home/leoisidro/CICLOS/X/PLANIFICA/proyecto_2/extracted_temp/3FO4IEMODOYX_Bedroom/3FO4IEMODOYX_Bedroom"
    
    if not os.path.exists(scene_dir):
        print(f"❌ Error: No se encontró la carpeta de escena descomprimida en: {scene_dir}")
        print("   Por favor ejecuta primero 'python explore_dataset.py' para extraer una escena de prueba.")
        return
        
    # Muestras de entrada
    original_rgb = os.path.join(scene_dir, "cam0/data/0.png")
    depth_map = os.path.join(scene_dir, "depth0/data/0.png")
    label_map = os.path.join(scene_dir, "label0/data/0_nyu.png")
    
    # Archivos temporales y de salida
    edited_seg_path = "./temp_edited_seg.png"
    output_result_path = "./resultado_oficina_transformada.png"
    
    # Paso 1: Modificar la máscara semántica (Paso de Layout/Distribución)
    print("\n--- PASO 1: PLANIFICACIÓN DE LAYOUT (EDICIÓN SEMÁNTICA) ---")
    edit_segmentation_mask(label_map, edited_seg_path)
    
    # Paso 2: Ejecutar Stable Diffusion con ControlNet
    print("\n--- PASO 2: INFERENCIA GENERATIVA ---")
    try:
        run_diffusion_pipeline(depth_map, edited_seg_path, output_result_path)
        print("\n✨ Proceso completo. Puedes visualizar el resultado abriendo: './resultado_oficina_transformada.png'")
    except Exception as e:
        print(f"\n❌ Error durante la inferencia generativa: {e}")
        print("Asegúrate de tener conexión a internet para descargar los modelos pre-entrenados de Hugging Face.")

if __name__ == "__main__":
    main()
