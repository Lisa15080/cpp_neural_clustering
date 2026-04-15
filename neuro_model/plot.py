import json
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap

base_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(base_dir, 'true_clusters.json')) as f:
    t = json.load(f)
with open(os.path.join(base_dir, 'predictions.json')) as f:
    p = json.load(f)

# Создаём дискретную цветовую схему: 0=синий, 1=красный
colors = ListedColormap(['#4472C4', '#ED7D31'])  # синий, оранжевый

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.scatter(t['x'], t['y'], c=t['labels'], cmap=colors, s=20, 
            edgecolors='k', linewidth=0.3, vmin=0, vmax=1)
ax1.set_title("Истинные метки")
ax1.set_aspect('equal')

ax2.scatter(p['x'], p['y'], c=p['labels'], cmap=colors, s=20, 
            edgecolors='k', linewidth=0.3, vmin=0, vmax=1)
ax2.set_title("Предсказания сети")
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'result.png'), dpi=150)
print("Готово: result.png")
plt.show()
