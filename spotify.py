import spotipy
from model import *
from enum import Enum

class EmotionsMood(Enum):
    """Creating constants to assign to different moods"""
    Calm = 2 #Calm
    Energetic = 4 
    happy = 1
    sad = 3


def get_songs_features(sp,ids):
    """Get features of songs to identify tyoe of music"""
    meta = sp.track(ids)
    features = sp.audio_features(ids)

    # meta
    name = meta['name']
    album = meta['album']['name']
    artist = meta['album']['artists'][0]['name']
    release_date = meta['album']['release_date']
    length = meta['duration_ms']
    popularity = meta['popularity']
    ids =  meta['id']

    # features
    acousticness = features[0]['acousticness']
    danceability = features[0]['danceability']
    energy = features[0]['energy']
    instrumentalness = features[0]['instrumentalness']
    liveness = features[0]['liveness']
    valence = features[0]['valence']
    loudness = features[0]['loudness']
    speechiness = features[0]['speechiness']
    tempo = features[0]['tempo']
    key = features[0]['key']
    time_signature = features[0]['time_signature']

    track = [name, album, artist, ids, release_date, popularity, length, danceability, acousticness,
            energy, instrumentalness, liveness, valence, loudness, speechiness, tempo, key, time_signature]
    columns = ['name','album','artist','id','release_date','popularity','length','danceability','acousticness','energy','instrumentalness',
                'liveness','valence','loudness','speechiness','tempo','key','time_signature']
    return track,columns

def recommendations(token,emmotion_value):
    """Get actual recommendations based on songs's features"""
    playlistIdUrl=[]
    track_ids = []
    recommended_track=[]
    rec_tracks = []

    if token:
        sp = spotipy.Spotify(auth=token)
    else:
        print("Can't get token")
        return

    userId=sp.current_user()['id']

    if userId:
        all_playlists = sp.current_user_playlists(limit=1,offset=0)
        for item in all_playlists['items']:
            if item['external_urls']['spotify'] not in playlistIdUrl:
                playlistIdUrl.append( item['external_urls']['spotify'])

        for playlistId in playlistIdUrl:
            Playlist = sp.user_playlist(userId, playlistId)
            tracks = Playlist["tracks"]
            songs = tracks["items"]
            for i in range(0, len(songs)):
                if songs[i]['track']['id'] != None and songs[i]['track']['id'] not in track_ids: # Removes the local tracks in your playlist if there is any
                    track_ids.append(songs[i]['track']['id'])
                    
        for id in track_ids: 
            rec_tracks += sp.recommendations(seed_tracks=[id], seed_genres=['indian, happy, calm, chill'], limit=2, min_valence=0.3, min_popularity=60)['tracks']        

        for track in rec_tracks:
            features=get_songs_features(sp,track["id"])
            mood=predict_mood(features)
            if mood.upper()==EmotionsMood(emmotion_value).name.upper():
                trackUrl='http://open.spotify.com/track/'+str(track["id"])
                if trackUrl not in recommended_track:
                    recommended_track.append(trackUrl)
        return recommended_track

        # playlist_recs = sp.user_playlist_create(username, 
        #                                         name='Recommended Songs for Playlist by Amit - {}'.format(sourcePlaylist['name']))

        # #Add tracks to the new playlist
        # for i in rec_array:
        #     sp.user_playlist_add_tracks(username, playlist_recs['id'], i)

    else:
        return ("Can't get User")

