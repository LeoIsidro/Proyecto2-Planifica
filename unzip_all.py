import os
import zipfile
import glob
from concurrent.futures import ThreadPoolExecutor

zip_dir = "/home/leoisidro/CICLOS/X/PLANIFICA/proyecto_2/HD7"
dest_dir = "/home/leoisidro/CICLOS/X/PLANIFICA/proyecto_2/extracted_temp"

def unzip_file(zip_path):
    filename = os.path.basename(zip_path)
    name_without_ext = os.path.splitext(filename)[0]
    target_extract_path = os.path.join(dest_dir, name_without_ext)
    
    # Check if already extracted to avoid redundant work
    if os.path.exists(target_extract_path):
        # Let's check if it has files
        if len(os.listdir(target_extract_path)) > 0:
            print(f"  [SKIPPED] Already extracted: {filename}")
            return True
            
    print(f"  [START] Unzipping {filename}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(target_extract_path)
        print(f"  [SUCCESS] Finished unzipping {filename}")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to unzip {filename}: {e}")
        return False

def main():
    print("=== DESCOMPRESOR MULTI-ESCENAS DE INTERIORNET ===")
    zips = glob.glob(os.path.join(zip_dir, "*.zip"))
    print(f"Se encontraron {len(zips)} archivos ZIP en {zip_dir}")
    
    os.makedirs(dest_dir, exist_ok=True)
    
    # We can use parallel unzipping to speed it up
    print("Descomprimiendo en paralelo...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(unzip_file, zips))
        
    success_count = sum(1 for r in results if r)
    print(f"\nProceso finalizado: {success_count}/{len(zips)} archivos descomprimidos con éxito.")

if __name__ == "__main__":
    main()
