from FastNoiseLite import FastNoiseLite
from PIL import Image
import numpy as np
import random

WIDTH, HEIGHT = 550, 500
FREQ = 0.008
OCTAVES = 5
LACUNARITY = 2.0
GAIN = 0.5

# Generar semilla aleatoria
seed = random.randint(0, 999999)
print(f"Usando semilla aleatoria: {seed}")

noise = FastNoiseLite(seed)
noise.noise_type = FastNoiseLite.TYPE_PERLIN
noise.set_frequency(FREQ)
noise.set_fractal_octaves(OCTAVES)
noise.set_fractal_lacunarity(LACUNARITY)
noise.set_fractal_gain(GAIN)

data = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
for y in range(HEIGHT):
    for x in range(WIDTH):
        v = noise.get_noise(x, y)
        # normalizar aprox a 0..1
        data[y, x] = (v + 1) * 0.5

img = Image.fromarray((data * 255).astype('uint8'), mode='L')
img.save("perlin_from_fastnoiselite_port.png")
print("Saved perlin_from_fastnoiselite_port.png")
