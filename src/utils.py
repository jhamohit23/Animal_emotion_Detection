import logging
import os

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler()
        ]
    )

def get_image_paths(base_dir, emotions):
    image_paths = []
    for emotion in emotions:
        emotion_dir = os.path.join(base_dir, emotion)
        for image_name in os.listdir(emotion_dir):
            if image_name.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(emotion_dir, image_name)
                image_paths.append(image_path)
    return image_paths