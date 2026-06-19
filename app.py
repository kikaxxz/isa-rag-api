from flask import Flask, request, jsonify
from flask_cors import CORS
from firebase_admin import credentials, auth, initialize_app
import os
import json
import chromadb
import google.generativeai as genai

app = Flask(__name__)
CORS(app)


credenciales_firebase = os.environ.get("FIREBASE_JSON")
if credenciales_firebase:

    cred_dict = json.loads(credenciales_firebase)
    cred = credentials.Certificate(cred_dict)
    initialize_app(cred)
else:
  
    initialize_app()

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
modelo = genai.GenerativeModel('gemini-3.5-flash')

cliente_chroma = chromadb.PersistentClient(path="./bd_vectorial")
coleccion = cliente_chroma.get_collection(name="manual_mantenimiento")

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

    resultados = coleccion.query(
        query_texts=[pregunta],
        n_results=3
    )

    contexto = " ".join(resultados['documents'][0])
    
    prompt = f"""
    Eres un asistente técnico de mantenimiento industrial. 
    Responde la siguiente pregunta usando SOLAMENTE la información en este contexto.
    Si no está en el contexto, indica que no tienes la información.
    
    Contexto:
    {contexto}
    
    Pregunta:
    {pregunta}
    """

    respuesta_gemini = modelo.generate_content(prompt)
    
    return jsonify({
        "respuesta": respuesta_gemini.text,
        "pagina": 1
    }), 200

if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 8080)))