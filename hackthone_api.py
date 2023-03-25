from flask import Flask, jsonify, request
from facial_detection import * 
from spotify import * 
from enum import Enum

app = Flask(__name__)

@app.route('/', methods=['POST'])
def get_song_rec():
    image = request.files['file']
    token = request.headers['token']
    emmotion=get_emotion(image)
    emmotion_value=Emotions[emmotion].value 
    tracks=recommendations(token,emmotion_value)
    return jsonify(tracks,emmotion,emmotion_value)

class Emotions(Enum):
    angry = 2 #Calm
    disgust = 4 #Energetic
    fear = 2
    happy = 1
    neutral = 2
    sad = 3
    surprise = 2


if __name__=="__main__":
    app.run(host="127.0.0.1",port=8080,debug=True)