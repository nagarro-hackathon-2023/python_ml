from flask import Flask, Response, jsonify, request
from facial_detection import * 
from spotify import * 
from enum import Enum

app = Flask(__name__)

@app.route('/get_tracks', methods=['POST'])
def get_song_rec():
    image = request.files['file']
    token = request.headers['token']
    emmotion=get_emotion(image)
    emmotion_value=Emotions[emmotion].value 
    tracks=recommendations(token,emmotion_value)
    if tracks=='Token expired!!!':
        return Response('Token expired!!!', 401, {'WWW-Authenticate':'Basic realm="Login Required"'})
    else: 
        return jsonify(tracks,emmotion,emmotion_value)

class Emotions(Enum):
    angry = 0 #Calm
    disgust = 0 #Energetic
    fear = 0
    happy = 2
    neutral = 1
    sad = 0
    surprise = 2

if __name__ == '__main__':
   app.run()
# if __name__=="__main__":
#     app.run(host="127.0.0.1",port=8080,debug=True)