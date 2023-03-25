from fer import FER
import matplotlib.pyplot as plt


def get_emotion(image):
    image_process = plt.imread(image)
    emo_detector = FER(mtcnn=True)
    # Capture all the emotions on the image
    #captured_emotions = emo_detector.detect_emotions(image_process)
    # Print all captured emotions with the image
    #print(captured_emotions)
    #plt.imshow(test_image_one)

    # Use the top Emotion() function to call for the dominant emotion in the image
    dominant_emotion, emotion_score = emo_detector.top_emotion(image_process)
    return dominant_emotion
  
    



