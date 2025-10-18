"""
This script visualizes the distribution of BF16 (bfloat16) representable values 
for two specific biased exponents (125 and 126). 

Interactivity is added using mplcursors:
   - Hovering over a point displays its exact value along with its biased exponent.
   - The values are formatted with sufficient decimal places to accurately 
     reflect the BF16 precision.

This visualization helps users intuitively understand the density, spacing, 
and relative precision of BF16 numbers at different exponent levels.
"""

import numpy as np
import matplotlib.pyplot as plt
import mplcursors

# BF16 mantissa bits
mantissa_bits = 7
mantissa_steps = 2**mantissa_bits

# Biased exponents to plot
biased_exponents = [125, 126]
colors = ['tab:blue', 'tab:orange']

plt.figure(figsize=(10, 2))
y_center = 1.0

scatter_points = []

for i, E in enumerate(biased_exponents):
    actual_exponent = E - 127
    values = [(1 + m / mantissa_steps) * 2**actual_exponent for m in range(mantissa_steps)]

    s = plt.scatter(values, [y_center]*len(values), c=colors[i], s=5, label=f'biased exponent {E}')
    scatter_points.append(s)


plt.xscale('log')
plt.yticks([])
plt.xlabel("BF16 Representable Values")
plt.title("BF16 Values Distribution (biased exponent = 125 and 126)")
plt.grid(True, axis='x', linestyle='--', alpha=0.5)
plt.legend(title="Biased Exponent")

cursor = mplcursors.cursor(scatter_points, hover=True)

@cursor.connect("add")
def on_add(sel):
    for i, s in enumerate(scatter_points):
        if sel.artist == s:
            biased_exp = biased_exponents[i]
            break
    sel.annotation.set_text(f"x = {sel.target[0]:.8f}\nbiased exponent = {biased_exp}")

plt.show()
