from flask import Flask, request, jsonify
from flask_cors import CORS
from firebase_admin import credentials, auth, initialize_app
import os
import json
from pinecone import Pinecone
from google import genai

app = Flask(__name__)
CORS(app)

credenciales_firebase = os.environ.get("FIREBASE_JSON")
if credenciales_firebase:
    cred_dict = json.loads(credenciales_firebase)
    cred = credentials.Certificate(cred_dict)
    initialize_app(cred)
else:
    initialize_app()

cliente_gemini = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
indice = pc.Index("manual-mantenimiento")

@app.route('/api/v1/consultar-manual', methods=['POST'])
def consultar_manual():
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({"error": "No autorizado"}), 401

    try:
        auth.verify_id_token(token)
    except Exception:
        return jsonify({"error": "Token inválido"}), 401

    data = request.get_json()
    pregunta = data.get('pregunta')

    prompt_enrutador = f"""
    Clasifica la siguiente pregunta en una de dos categorías:
    1. 'GENERAL': Saludos, preguntas sobre qué puedes hacer, o preguntas abstractas.
    2. 'TECNICA': Búsqueda de datos, procedimientos, equipos o pasos de mantenimiento.
    
    Responde ÚNICAMENTE con la palabra GENERAL o TECNICA.
    Pregunta: "{pregunta}"
    """

    respuesta_ruta = cliente_gemini.models.generate_content(
        model='gemini-3.5-flash',
        contents=prompt_enrutador
    ).text.strip().upper()

    if "GENERAL" in respuesta_ruta:
        prompt_final = f"""
        Eres un asistente virtual experto en Automatización y Control.
        Responde de manera profesional a esta pregunta general. Indica que estás diseñado para consultar manuales técnicos y que puedes ayudar con procedimientos, verificación de fugas, calibraciones y repuestos de instrumentación.
        Pregunta del usuario: {pregunta}
        """
        
        respuesta_final = cliente_gemini.models.generate_content(
            model='gemini-3.5-flash',
            contents=prompt_final
        ).text
    else:
        embedding_pregunta = cliente_gemini.models.embed_content(
            model='text-embedding-004',
            contents=pregunta
        ).embeddings[0].values

        resultados = indice.query(
            vector=embedding_pregunta,
            top_k=3,
            include_metadata=True
        )

        contexto = " ".join([match['metadata']['texto'] for match in resultados['matches']]) if resultados['matches'] else ""
        
        prompt_final = f"""
        Eres un asistente técnico de mantenimiento industrial. 
        Responde la pregunta usando la información del contexto.
        
        REGLA ESTRICTA: Sé conciso y directo. Si estás explicando un procedimiento, resúmelo únicamente en los pasos más críticos utilizando viñetas. No generes respuestas de más de 3 párrafos.
        
        Contexto: {contexto}
        Pregunta: {pregunta}
        """
        
        respuesta_final = cliente_gemini.models.generate_content(
            model='gemini-3.5-flash',
            contents=prompt_final
        ).text

    return jsonify({"respuesta": respuesta_final})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)