from app.models.grounding_dino import GroundingDINO
import numpy as np
detector = GroundingDINO()
print('Backend:', detector._backend)
img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
results = detector.detect(img, 'general')
print('Detections:', len(results))
