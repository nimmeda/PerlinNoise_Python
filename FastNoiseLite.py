# FastNoiseLite.py
# Pure-Python implementation of 2D Perlin-like FastNoise with fractal octaves.
# Target: provide a FastNoiseLite-compatible API for simple use:
#   noise = FastNoiseLite(seed)
#   noise.noise_type = FastNoiseLite.TYPE_PERLIN
#   noise.set_frequency(...)
#   noise.set_fractal_octaves(...)
#   noise.set_fractal_lacunarity(...)
#   noise.set_fractal_gain(...)
#   v = noise.get_noise(x, y)  # returns roughly in [-1, 1]
#
# This implementation uses a Ken-Perlin style permutation + gradient approach
# and sums octaves with lacunarity/gain to emulate FastNoiseLite's fractal behavior.

import math
import random

class FastNoiseLite:
    TYPE_OPEN_SIMPLEX_2 = 0
    TYPE_OPEN_SIMPLEX_2S = 1
    TYPE_CELLULAR = 2
    TYPE_PERLIN = 3
    TYPE_VALUE_CUBIC = 4
    TYPE_VALUE = 5

    TYPE_STR = {
        TYPE_OPEN_SIMPLEX_2: "OPEN_SIMPLEX_2",
        TYPE_OPEN_SIMPLEX_2S: "OPEN_SIMPLEX_2S",
        TYPE_CELLULAR: "CELLULAR",
        TYPE_PERLIN: "PERLIN",
        TYPE_VALUE_CUBIC: "VALUE_CUBIC",
        TYPE_VALUE: "VALUE",
    }

    def __init__(self, seed=1337):
        self.seed = int(seed) & 0xFFFFFFFF
        self.noise_type = self.TYPE_PERLIN
        self.frequency = 0.01
        # fractal parameters
        self.octaves = 3
        self.lacunarity = 2.0
        self.gain = 0.5

        # build permutation table (512 entries)
        self._build_perm_table(self.seed)

    def _build_perm_table(self, seed):
        rnd = random.Random(seed)
        perm = list(range(256))
        rnd.shuffle(perm)
        # duplicate for overflow
        self.perm = perm + perm
        # Precompute gradient vectors (8 directions)
        self.grad2 = [
            (1,0),(-1,0),(0,1),(0,-1),
            (1,1),(-1,1),(1,-1),(-1,-1)
        ]

    def set_frequency(self, freq):
        self.frequency = float(freq)

    def set_fractal_octaves(self, octaves):
        self.octaves = int(octaves) if octaves >= 1 else 1

    def set_fractal_lacunarity(self, lac):
        self.lacunarity = float(lac)

    def set_fractal_gain(self, gain):
        self.gain = float(gain)

    # ----------------------
    # Utility functions
    # ----------------------
    @staticmethod
    def _fade(t):
        # 6t^5 - 15t^4 + 10t^3
        return t * t * t * (t * (t * 6 - 15) + 10)

    @staticmethod
    def _lerp(a, b, t):
        return a + t * (b - a)

    def _grad(self, hash_val, x, y):
        # choose one of 8 gradient directions
        g = self.grad2[hash_val & 7]
        return g[0] * x + g[1] * y

    def _perm_hash(self, ix, iy):
        # classic permutation hash for 2D
        return self.perm[(self.perm[ix & 255] + (iy & 255)) & 255]

    # ----------------------
    # Raw Perlin noise (2D)
    # Produces values roughly in [-1,1], not normalized across octaves.
    # ----------------------
    def _perlin2(self, x, y):
        # Scale input by frequency outside when calling if desired.
        xi = math.floor(x)
        yi = math.floor(y)
        xf = x - xi
        yf = y - yi

        u = self._fade(xf)
        v = self._fade(yf)

        # corner hashes
        h00 = self._perm_hash(xi, yi)
        h10 = self._perm_hash(xi + 1, yi)
        h01 = self._perm_hash(xi, yi + 1)
        h11 = self._perm_hash(xi + 1, yi + 1)

        # gradient dot products
        n00 = self._grad(h00, xf, yf)
        n10 = self._grad(h10, xf - 1.0, yf)
        n01 = self._grad(h01, xf, yf - 1.0)
        n11 = self._grad(h11, xf - 1.0, yf - 1.0)

        # interpolate
        nx0 = self._lerp(n00, n10, u)
        nx1 = self._lerp(n01, n11, u)
        nxy = self._lerp(nx0, nx1, v)

        # nxy is in roughly [-sqrt(2), sqrt(2)] depending on gradients; keep as-is
        # We'll rely on fractal normalization to keep values manageable.
        return nxy

    # ----------------------
    # Fractal combination (like FastNoiseLite fractal)
    # ----------------------
    def _perlin_fractal(self, x, y):
        total = 0.0
        amplitude = 1.0
        frequency = 1.0
        max_ampl = 0.0

        for _ in range(self.octaves):
            sample_x = x * frequency
            sample_y = y * frequency
            val = self._perlin2(sample_x, sample_y)
            total += val * amplitude
            max_ampl += amplitude
            amplitude *= self.gain
            frequency *= self.lacunarity

        # Normalize by max_ampl (still centered near 0)
        return total / max_ampl if max_ampl != 0 else 0.0

    # ----------------------
    # Public getter: get_noise(x, y)
    # Accepts floats or ints; returns value approximately in [-1,1]
    # We apply the instance frequency here.
    # ----------------------
    def get_noise(self, x, y):
        # multiply coordinates by frequency first (matching FastNoiseLite behavior)
        fx = x * self.frequency
        fy = y * self.frequency

        if self.noise_type == self.TYPE_PERLIN:
            v = self._perlin_fractal(fx, fy)
            # clamp to [-1,1] (safe)
            if v > 1.0: v = 1.0
            if v < -1.0: v = -1.0
            return v
        else:
            # Fallback to perlin if other types unsupported in this port
            v = self._perlin_fractal(fx, fy)
            if v > 1.0: v = 1.0
            if v < -1.0: v = -1.0
            return v

    # Convenience: allow direct call like noise(x,y)
    def __call__(self, x, y):
        return self.get_noise(x, y)

# If run directly, show a tiny demo (text output)
if __name__ == "__main__":
    n = FastNoiseLite(42)
    n.noise_type = FastNoiseLite.TYPE_PERLIN
    n.set_frequency(0.02)
    n.set_fractal_octaves(4)
    n.set_fractal_lacunarity(2.0)
    n.set_fractal_gain(0.5)

    for y in range(8):
        row = []
        for x in range(8):
            v = n.get_noise(x, y)
            row.append(f"{v:+0.3f}")
        print(" ".join(row))
