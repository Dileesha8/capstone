import sys
sys.path.append('.')
from src.predict_brain import model as brain_model
print('=== BRAIN ALL CONV LAYERS ===')
for layer in brain_model.layers:
    if 'conv' in layer.name.lower() or 'dense' in layer.name.lower():
        print(layer.name, '-', layer.__class__.__name__)