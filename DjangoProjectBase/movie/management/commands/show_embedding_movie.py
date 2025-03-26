import os
import numpy as np
import random
from django.core.management.base import BaseCommand
from movie.models import Movie

class Command(BaseCommand):
    help = "Display the embeddings of a randomly selected movie"

    def handle(self, *args, **kwargs):
        # ✅ Obtener todas las películas de la base de datos
        movies = Movie.objects.exclude(emb=None)  # Filtramos solo las que tienen embeddings
        if not movies.exists():
            self.stderr.write("❌ No movies with embeddings found in the database.")
            return

        # 🔀 Seleccionar una película al azar
        movie = random.choice(movies)
        self.stdout.write(f"🎬 Selected Movie: {movie.title}")
        
        try:
            # 📥 Convertir el embedding de binario a numpy array
            emb_array = np.frombuffer(movie.emb, dtype=np.float32)
            
            # 🔍 Mostrar los primeros valores del embedding
            self.stdout.write(f"🔢 Embedding (first 10 values): {emb_array[:10]}")
        except Exception as e:
            self.stderr.write(f"❌ Error processing embedding for {movie.title}: {e}")