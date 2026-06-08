import os
import json
import shutil
import subprocess

# Intentar importar Gradio
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

from pipeline import (
    generate_variations, 
    generate_custom_variations, 
    VARIATION_STYLES, 
    DEEP_LEARNING_AVAILABLE
)
from agent import ExplainerAgent

# Rutas de prueba por defecto
DEFAULT_SCENE_DIR = "/home/leoisidro/CICLOS/X/PLANIFICA/proyecto_2/extracted_temp/3FO4IEMODOYX_Bedroom/3FO4IEMODOYX_Bedroom"
OUTPUT_DIR = "./generations"

def get_available_scenes():
    extracted_dir = "/home/leoisidro/CICLOS/X/PLANIFICA/proyecto_2/extracted_temp"
    if not os.path.exists(extracted_dir):
        return []
    
    folders = os.listdir(extracted_dir)
    scenes = set()
    for f in folders:
        if f.endswith("_Bedroom") or f.endswith("_Study") or f.endswith("_Living_room"):
            scene_id = f.split("_")[0]
            scenes.add(scene_id)
            
    valid_scenes = []
    for s in sorted(scenes):
        bed_exists = os.path.exists(os.path.join(extracted_dir, f"{s}_Bedroom"))
        study_exists = os.path.exists(os.path.join(extracted_dir, f"{s}_Study"))
        if bed_exists and study_exists:
            valid_scenes.append(s)
    return valid_scenes

def update_scene_preview(scene_id):
    if not scene_id:
        return None
    scene_dir = f"/home/leoisidro/CICLOS/X/PLANIFICA/proyecto_2/extracted_temp/{scene_id}_Bedroom/{scene_id}_Bedroom"
    return os.path.join(scene_dir, "cam0/data/0.png")

def process_and_generate(scene_id):
    """
    Ejecuta el pipeline de generación de 5 variaciones para la escena seleccionada del dataset.
    """
    if not scene_id:
        return [None]*5 + ["⚠️ Por favor, selecciona una escena primero."]
        
    print(f"🎨 Iniciando la generación de 5 variaciones para la escena {scene_id}...")
    
    scene_dir = f"/home/leoisidro/CICLOS/X/PLANIFICA/proyecto_2/extracted_temp/{scene_id}_Bedroom/{scene_id}_Bedroom"
    input_img = os.path.join(scene_dir, "cam0/data/0.png")
    depth_map = os.path.join(scene_dir, "depth0/data/0.png")
    label_map = os.path.join(scene_dir, "label0/data/0_nyu.png")
    
    if not os.path.exists(depth_map):
        depth_map = None
    if not os.path.exists(label_map):
        label_map = None
        
    try:
        paths = generate_variations(
            input_image_path=input_img,
            depth_map_path=depth_map,
            label_map_path=label_map,
            output_dir=OUTPUT_DIR,
            scene_id=scene_id
        )
        return paths[0], paths[1], paths[2], paths[3], paths[4], f"✅ Se generaron 5 variaciones con éxito para {scene_id}. Evalúa cada una a continuación."
    except Exception as e:
        return [None]*5 + [f"❌ Error en la generación: {e}"]

def process_and_generate_custom(input_img):
    """
    Ejecuta el pipeline de transformación en caliente en la GPU para una imagen subida por el usuario.
    """
    if input_img is None:
        return [None]*5 + ["⚠️ Por favor, sube una imagen de tu habitación primero."]
        
    print("🎨 Iniciando generación en caliente en GPU para imagen personalizada...")
    try:
        paths = generate_custom_variations(
            input_image_path=input_img,
            output_dir=OUTPUT_DIR
        )
        return paths[0], paths[1], paths[2], paths[3], paths[4], "✅ Se generaron 5 variaciones en caliente sobre tu imagen usando GPU. Evalúa cada una a continuación."
    except Exception as e:
        return [None]*5 + [f"❌ Error en la generación en GPU: {e}"]

def submit_feedback(*args):
    """
    Recibe los votos y comentarios de las 5 opciones y los envía al Agente de IA.
    """
    # args contiene alternadamente: [voto1, comentario1, voto2, comentario2, ...]
    decisions = []
    for i in range(5):
        vote = args[i*2]
        comment = args[i*2 + 1]
        
        status = "pending"
        if vote == "Aceptar 👍":
            status = "accepted"
        elif vote == "Rechazar 👎":
            status = "rejected"
            
        decisions.append({
            "option": i + 1,
            "style_name": VARIATION_STYLES[i]["name"],
            "status": status,
            "comment": comment
        })
        
    # Guardar estructurado de decisiones localmente
    decisions_json = json.dumps(decisions, indent=2)
    with open("user_decisions_log.json", "w", encoding="utf-8") as f:
        f.write(decisions_json)
        
    print("💾 Decisiones del usuario guardadas en user_decisions_log.json")
    
    # Invocar al Agente LLM
    agent = ExplainerAgent()
    explanation = agent.generate_explanation(decisions_json)
    return explanation

def build_gui():
    if not GRADIO_AVAILABLE:
        print("⚠️ Gradio no está instalado en el entorno de Python.")
        print("   Por favor corre: pip install gradio")
        return
        
    theme = gr.themes.Soft(
        primary_hue="cyan",
        secondary_hue="slate",
    )
    
    with gr.Blocks(theme=theme, title="Rediseñador de Espacios Generativo e Interactivo") as demo:
        gr.Markdown(
            """
            # 🏠 Sistema Generativo Interactivo de Transformación de Espacios
            ### Curso: Planificación y Toma de Decisiones en IA - Proyecto #2
            Sube una foto propia o selecciona una habitación del dataset, genera 5 variaciones de oficinas con GPU, evalúalas y obtén el reporte técnico de nuestro Agente de IA.
            """
        )
        
        # Definición de outputs comunes para las variaciones (se usarán en ambas pestañas)
        variation_imgs = []
        votes = []
        comments = []
        
        with gr.Tabs():
            # PESTAÑA 1: DATASET
            with gr.Tab("📖 Modo 1: Escenas del Dataset (Rápido / Fallback)"):
                scenes = get_available_scenes()
                default_scene = scenes[0] if scenes else None
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📸 Selección de Escena")
                        scene_dropdown = gr.Dropdown(
                            choices=scenes,
                            value=default_scene,
                            label="ID de Escena del Dataset"
                        )
                        input_image_ds = gr.Image(
                            value=update_scene_preview(default_scene) if default_scene else None,
                            type="filepath",
                            label="Vista Previa de Entrada (Dormitorio)"
                        )
                        btn_generate_ds = gr.Button("🎨 Generar 5 Variaciones (Dataset)", variant="primary")
                        
            # PESTAÑA 2: SUBIR IMAGEN PROPIA
            with gr.Tab("📸 Modo 2: Subir Mi Propia Imagen (En Caliente / Requiere GPU)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📤 Sube tu Foto")
                        input_image_custom = gr.Image(
                            type="filepath",
                            label="Cargar foto de tu dormitorio / sala de estar (PNG/JPG)"
                        )
                        gr.Markdown(
                            "⚠️ *Requisito:* Esta función estimará la profundidad en tiempo real de tu imagen "
                            "y usará el pipeline de control ControlNet Img2Img en caliente sobre la **GPU**."
                        )
                        btn_generate_custom = gr.Button("🚀 Generar 5 Variaciones en Caliente (GPU)", variant="primary")
        
        # Estado del sistema común
        status_box = gr.Textbox(label="Estado del Generador", value="Listo. Elige un modo y presiona Generar.", interactive=False)
        
        gr.Markdown("---")
        gr.Markdown("### 📐 2. Variaciones Generadas y Evaluación (Humano-IA)")
        
        # Grid de visualización y votación de variaciones
        with gr.Row():
            for i in range(5):
                style_name = VARIATION_STYLES[i]["name"]
                with gr.Column():
                    gr.Markdown(f"#### {i+1}. {style_name}")
                    var_img = gr.Image(label=style_name, interactive=False, show_label=False)
                    variation_imgs.append(var_img)
                    
                    vote_radio = gr.Radio(
                        choices=["Aceptar 👍", "Rechazar 👎"], 
                        label="Tu Decisión",
                        value="Rechazar 👎"
                    )
                    votes.append(vote_radio)
                    
                    comment_txt = gr.Textbox(
                        placeholder="Escribe por qué te gusta/disgusta...", 
                        label="Comentario / Feedback"
                    )
                    comments.append(comment_txt)
                    
        gr.Markdown("---")
        gr.Markdown("### 🧠 3. Agente de Explicabilidad e Informe Final")
        
        btn_report = gr.Button("📋 Generar Reporte de Explicabilidad (Agente LLM)", variant="secondary")
        report_output = gr.Markdown(value="El informe aparecerá aquí una vez presiones el botón de arriba.")
        
        # --- CONEXIÓN DE COMPONENTES Y LÓGICA ---
        
        # Cambio de previsualización en Dropdown
        scene_dropdown.change(
            fn=update_scene_preview,
            inputs=[scene_dropdown],
            outputs=[input_image_ds]
        )
        
        # Generar para Dataset
        btn_generate_ds.click(
            fn=process_and_generate,
            inputs=[scene_dropdown],
            outputs=variation_imgs + [status_box]
        )
        
        # Generar para Imagen Personalizada
        btn_generate_custom.click(
            fn=process_and_generate_custom,
            inputs=[input_image_custom],
            outputs=variation_imgs + [status_box]
        )
        
        # Lógica del reporte del Agente
        feedback_inputs = []
        for i in range(5):
            feedback_inputs.append(votes[i])
            feedback_inputs.append(comments[i])
            
        btn_report.click(
            fn=submit_feedback,
            inputs=feedback_inputs,
            outputs=[report_output]
        )
        
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

if __name__ == "__main__":
    if not GRADIO_AVAILABLE:
        print("⚠️ Gradio no está disponible. Instalando dependencias de interfaz...")
        subprocess.run(["pip", "install", "gradio"])
        import gradio as gr
        GRADIO_AVAILABLE = True
    build_gui()
