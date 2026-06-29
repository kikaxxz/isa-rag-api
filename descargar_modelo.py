from sentence_transformers import SentenceTransformer

print("Iniciando descarga del modelo en caché...")
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("Modelo descargado y listo para usarse.")