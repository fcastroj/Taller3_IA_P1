from django.shortcuts import render
from openai import OpenAI
import numpy as np
import os
from movie.models import Movie  # Asegúrate de importar el modelo de Movie si aún no lo has hecho

# Método para cargar la API Key desde el archivo openAI.env
def load_openai_key():
    # Cargar la clave API desde openAI.env en el directorio global
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'openAI.env'), 'r') as file:
        for line in file:
            if line.startswith('openai_apikey'):
                return line.split('=')[1].strip().replace('"', '')
    raise ValueError("API Key de OpenAI no encontrada en el archivo openAI.env")

# Cargar la clave de la API
api_key = load_openai_key()

# Inicializar el cliente de OpenAI con la clave obtenida
client = OpenAI(api_key=api_key)

# Función para calcular similitud de coseno
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recommend_movie(request):
    # Definir las variables para la película recomendada y la similitud
    best_movie = None
    max_similarity = -1

    # Si el formulario fue enviado
    if request.method == 'POST':
        prompt = request.POST.get('prompt', '')

        # Generar embedding del prompt
        response = client.embeddings.create(
            input=[prompt],
            model="text-embedding-3-small"
        )
        prompt_emb = np.array(response.data[0].embedding, dtype=np.float32)

        # Buscar la película más similar
        for movie in Movie.objects.all():
            movie_emb = np.frombuffer(movie.emb, dtype=np.float32)
            similarity = cosine_similarity(prompt_emb, movie_emb)

            if similarity > max_similarity:
                max_similarity = similarity
                best_movie = movie

    # Mostrar la película recomendada, si la hay
    context = {
        'movie': best_movie,
        'similarity': max_similarity
    }

    # Renderizar la página de recomendación (usando recommend.html)
    return render(request, 'recommendation/recommend.html', context)
