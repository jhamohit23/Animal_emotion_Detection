import streamlit as st
from src.data_preparation import create_directories, create_annotation_csv
from src.model import train_model, predict
from src.utils import setup_logging, get_image_paths
import logging
import os

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        emotions = ['happy', 'sad', 'hungry']
        base_dir = 'dataset'
        csv_file = 'annotation.csv'
        
        create_directories(base_dir, emotions)
        
        # Check if image files exist and are correctly named
        for emotion in emotions:
            emotion_dir = os.path.join(base_dir, emotion)
            if os.path.exists(emotion_dir):
                for image_name in os.listdir(emotion_dir):
                    if image_name.endswith(('.png', '.jpg', '.jpeg')):
                        try:
                            animal_type = image_name.split(' ')[0].lower()
                        except IndexError:
                            logger.warning(f"File {image_name} in {emotion_dir} is not correctly named. It should be 'AnimalType (number).jpg'")
        
        create_annotation_csv(base_dir, emotions, csv_file)
        
        if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
            # Unpack the tuple returned by train_model
            model, emotion_encoder, animal_encoder = train_model(csv_file)
            
            st.title('Animal Emotion and Type Predictor')
            uploaded_file = st.file_uploader("Choose an animal image...", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
                st.write("Predicting...")
                
                image_path = f"temp_image.{uploaded_file.name.split('.')[-1]}"
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Pass model, image path, and both encoders to predict
                emotion, animal_type = predict(model, image_path, emotion_encoder=emotion_encoder, animal_encoder=animal_encoder)
                st.write(f"The predicted emotion is: {emotion}")
                st.write(f"The predicted animal type is: {animal_type}")
        else:
            st.error("No data available. Please make sure your dataset is correctly set up.")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()