import os
import zipfile
import json

base_dir = "/home/leoisidro/CICLOS/X/PLANIFICA/proyecto_2/HD7"
scene_id = "3FO4IEMODOYX"

bedroom_zip = os.path.join(base_dir, f"{scene_id}_Bedroom.zip")
study_zip = os.path.join(base_dir, f"{scene_id}_Study.zip")

extract_dir = "/home/leoisidro/CICLOS/X/PLANIFICA/proyecto_2/extracted_temp"
b_extract = os.path.join(extract_dir, f"{scene_id}_Bedroom")
s_extract = os.path.join(extract_dir, f"{scene_id}_Study")

def main():
    print("=== EXPLORADOR DEL DATASET INTERIORNET ===")
    
    # Check if zip files exist
    if not os.path.exists(bedroom_zip) or not os.path.exists(study_zip):
        print(f"Error: No se encontraron los archivos ZIP para la escena {scene_id} en {base_dir}")
        print("Por favor, asegúrate de tener descargados:")
        print(f" - {scene_id}_Bedroom.zip")
        print(f" - {scene_id}_Study.zip")
        return
        
    print(f"Descomprimiendo muestras en: {extract_dir}...")
    os.makedirs(extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(bedroom_zip, 'r') as z:
        z.extractall(b_extract)
    with zipfile.ZipFile(study_zip, 'r') as z:
        z.extractall(s_extract)
        
    print("Descompresión completada.\n")
    
    # Find JSON file paths
    b_json = None
    for root, dirs, files in os.walk(b_extract):
        if "cocolabel.json" in files:
            b_json = os.path.join(root, "cocolabel.json")
            break
            
    s_json = None
    for root, dirs, files in os.walk(s_extract):
        if "cocolabel.json" in files:
            s_json = os.path.join(root, "cocolabel.json")
            break

    if b_json and s_json:
        with open(b_json, "r") as f:
            b_data = json.load(f)
        with open(s_json, "r") as f:
            s_data = json.load(f)
            
        print("--- INFORMACIÓN DE LA HABITACIÓN (Dormitorio) ---")
        print(f"Imágenes totales (vistas): {len(b_data.get('images', []))}")
        print(f"Anotaciones de objetos: {len(b_data.get('annotations', []))}")
        
        print("\n--- INFORMACIÓN DE LA OFICINA (Study) ---")
        print(f"Imágenes totales (vistas): {len(s_data.get('images', []))}")
        print(f"Anotaciones de objetos: {len(s_data.get('annotations', []))}")
        
        # Get category mappings
        categories = b_data.get("categories", [])
        print("\n--- CATEGORÍAS SEMÁNTICAS DISPONIBLES EN EL DATASET ---")
        cats_list = [f"{c['id']}: {c['name']}" for c in categories]
        print(", ".join(cats_list[:20]) + " ...")
        
        # Let's inspect some of the specific categories related to offices and bedrooms
        office_keywords = ["desk", "chair", "table", "bookshelf", "bed", "sofa", "wardrobe"]
        print("\nIdentificación de IDs de interés para transformación:")
        for c in categories:
            name = c["name"].lower()
            if any(kw in name for kw in office_keywords):
                print(f"  * Mueble: '{c['name']}' -> ID de Categoría Semántica: {c['id']}")
                
        # Let's see some sample file paths
        img0 = b_data["images"][0]["file_name"]
        print("\n--- EJEMPLOS DE RUTAS DE ARCHIVOS EN LA ESCENA DESCOMPRIMIDA ---")
        scene_subdir = os.path.relpath(os.path.dirname(b_json), extract_dir)
        print(f"Imagen RGB original:      {os.path.join(extract_dir, scene_subdir, img0)}")
        print(f"Mapa de Profundidad:      {os.path.join(extract_dir, scene_subdir, img0.replace('cam0', 'depth0'))}")
        print(f"Mapa de Normales:        {os.path.join(extract_dir, scene_subdir, img0.replace('cam0', 'normal0'))}")
        print(f"Segmentación Semántica:   {os.path.join(extract_dir, scene_subdir, img0.replace('cam0', 'label0').replace('.png', '_nyu.png'))}")
        print(f"Imagen con luz aleatoria: {os.path.join(extract_dir, scene_subdir, img0.replace('cam0', 'random_lighting_cam0'))}")
        
        print("\n💡 Tip: Puedes abrir estas imágenes en tu editor de código o cargarlas en un notebook de Jupyter")
        print("   usando PIL o OpenCV para explorar visualmente cómo se alinean los mapas geométricos y semánticos.")
        
    else:
        print("No se encontró cocolabel.json en las carpetas descomprimidas.")

if __name__ == "__main__":
    main()
