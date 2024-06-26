import os
import csv
import logging

logger = logging.getLogger(__name__)

def create_directories(base_dir, emotions):
    try:
        for emotion in emotions:
            os.makedirs(os.path.join(base_dir, emotion), exist_ok=True)
        logger.info(f"Created directories: {', '.join(emotions)}")
    except Exception as e:
        logger.error(f"Error creating directories: {e}")
        raise

def create_annotation_csv(base_dir, emotions, csv_file):
    try:
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['animal_images', 'animal_emotions', 'animal_type'])
            
            for emotion in emotions:
                emotion_dir = os.path.join(base_dir, emotion)
                if not os.path.exists(emotion_dir):
                    logger.warning(f"Directory not found: {emotion_dir}")
                    continue
                for image_name in os.listdir(emotion_dir):
                    if image_name.endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(emotion_dir, image_name)
                        animal_type = image_name.split(' ')[0].lower()  # Assuming filenames are like "Dog (2).jpg"
                        writer.writerow([image_path, emotion, animal_type])
        logger.info(f"Created annotation CSV with animal types: {csv_file}")
    except Exception as e:
        logger.error(f"Error creating annotation CSV: {e}")
        raise

    