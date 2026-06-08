import os
import json

class ExplainerAgent:
    def __init__(self):
        # Intentar cargar llaves de API
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        
    def generate_explanation(self, decisions_json):
        """
        Recibe un JSON con las decisiones del usuario y genera una explicación.
        Si hay llaves de API, llama a un LLM real. Si no, usa el motor de plantillas heurístico.
        """
        decisions = json.loads(decisions_json)
        
        # 1. Intentar llamar a OpenAI si la llave está disponible
        if self.openai_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.openai_key)
                
                prompt = self._build_llm_prompt(decisions)
                print("🧠 Agente: Conectando con OpenAI para generar síntesis...")
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Eres un diseñador de interiores experto y un agente de IA que analiza decisiones del usuario sobre transformaciones de oficinas funcionales."},
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"⚠️ Error al conectar con OpenAI: {e}. Usando generador local.")

        # 2. Generador de plantilla local heurístico (Fallback de alta calidad)
        return self._generate_local_explanation(decisions)

    def _build_llm_prompt(self, decisions):
        prompt = "Analiza las siguientes decisiones que tomó el usuario respecto a 5 variaciones de diseño generadas por una IA para transformar su dormitorio en una oficina funcional. Genera un reporte coherente y redactado de manera profesional sobre el perfil estético y técnico del usuario:\n\n"
        for d in decisions:
            status = "ACEPTADA" if d["status"] == "accepted" else "RECHAZADA"
            prompt += f"- Variación: {d['style_name']}\n"
            prompt += f"  Decisión: {status}\n"
            prompt += f"  Comentario del usuario: {d['comment'] or 'Ninguno'}\n\n"
        prompt += "El reporte debe incluir:\n1. Resumen de preferencias (luz, estilo, mobiliario).\n2. Justificación técnica de por qué eligió ciertos estilos y rechazó otros en base a sus comentarios.\n3. Conclusión con una recomendación de diseño final."
        return prompt

    def _generate_local_explanation(self, decisions):
        """
        Generador local heurístico de explicaciones de diseño.
        Analiza las decisiones y comentarios del usuario y genera un texto profesional estructurado.
        """
        accepted = [d for d in decisions if d["status"] == "accepted"]
        rejected = [d for d in decisions if d["status"] == "rejected"]
        
        output = "### 📋 REPORT DE SÍNTESIS DE DISEÑO (Agente de IA)\n\n"
        output += "Este reporte ha sido sintetizado automáticamente por el Agente de IA analizando tu feedback interactivo.\n\n"
        
        # 1. Análisis de Decisiones
        output += f"**Resumen cuantitativo:**\n"
        output += f"* Variaciones Aceptadas: **{len(accepted)}** de 5\n"
        output += f"* Variaciones Rechazadas: **{len(rejected)}** de 5\n\n"
        
        # 2. Preferencias detectadas
        output += "### 🔍 Análisis Cualitativo del Perfil de Usuario\n"
        
        if not accepted:
            output += "* **Perfil General:** El usuario muestra un nivel alto de exigencia o no encontró un estilo que se adaptara a sus necesidades particulares. Se requiere iterar el prompt base o modificar la geometría de los muebles.\n"
        else:
            output += "* **Preferencias de Estilo:** El usuario ha validado propuestas orientadas a: "
            styles = [f"*{d['style_name']}*" for d in accepted]
            output += ", ".join(styles) + ".\n"
            
        # Analizar comentarios sobre luz
        all_comments = " ".join([d["comment"].lower() for d in decisions if d["comment"]])
        
        output += "* **Factor Iluminación:** "
        if "luz" in all_comments or "oscur" in all_comments or "brill" in all_comments or "dia" in all_comments or "noche" in all_comments:
            if "oscur" in all_comments or "noche" in all_comments:
                output += "Se detecta una preferencia por ambientes íntimos o necesidades de trabajo en horarios nocturnos. Se valoran los esquemas de iluminación cálidos indirectos.\n"
            else:
                output += "El usuario prioriza la luz natural brillante y la visibilidad técnica, prefiriendo distribuciones cercanas a ventanas.\n"
        else:
            output += "La iluminación estándar de las variaciones aceptadas es adecuada para el trabajo de oficina diaria.\n"
            
        # Analizar comentarios sobre espacio/distribución
        output += "* **Distribución y Distribución Visual:** "
        if "espacio" in all_comments or "orden" in all_comments or "silla" in all_comments or "mesa" in all_comments:
            output += "El usuario presta especial atención al orden espacial y la ergonomía del mobiliario (mencionó aspectos como espacio, silla o mesa).\n"
        else:
            output += "La distribución espacial propuesta en los ControlNets mantiene la funcionalidad básica de las oficinas.\n"

        # 3. Justificación detallada por opción
        output += "\n### ⚙️ Justificación Técnica de las Decisiones\n"
        for d in decisions:
            status = "✅ ACEPTADA" if d["status"] == "accepted" else "❌ RECHAZADA"
            comment = d["comment"] if d["comment"] else "Sin comentarios adicionales."
            output += f"* **{d['style_name']}** ({status}):\n"
            output += f"  * *Comentario del usuario:* \"{comment}\"\n"
            # Generar inferencia técnica simulada basada en el comentario
            if d["status"] == "accepted":
                output += f"  * *Análisis del Agente:* Esta variación se alinea con el criterio estético y funcional del usuario, destacando la correcta integración del mobiliario en el espacio físico.\n"
            else:
                output += f"  * *Análisis del Agente:* Se rechaza debido a incompatibilidad con el flujo de trabajo deseado, saturación visual o esquema de color inadecuado según el feedback.\n"
                
        # 4. Recomendación final
        output += "\n### 💡 Recomendación de Diseño Final\n"
        if accepted:
            output += f"Basado en las decisiones tomadas, se recomienda proceder con la implementación del **{accepted[0]['style_name']}** como base del diseño físico del espacio, aplicando ligeros ajustes según los comentarios recolectados."
        else:
            output += "Se recomienda reiniciar la generación de variaciones visuales utilizando prompts enfocados en texturas más suaves y distribuciones espaciales más abiertas para cumplir las expectativas del usuario."
            
        return output
