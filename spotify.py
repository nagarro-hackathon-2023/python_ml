import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util
import pandas as pd
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns;
from sklearn.ensemble._forest import RandomForestRegressor, RandomForestClassifier
import numpy as np
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
from model import *

def train_model(rec_playlist_df,rec_track_names):

    playlist_df=pd.read_csv('Spotify_playlist_Dataset.csv')
    
    track_names=[]
    for songName in playlist_df['Song']:
        if songName not in track_names and pd.isnull(songName)==False:
            track_names.append(songName)
    
    v=TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 6), max_features=10000)
    X_names_sparse = v.fit_transform(track_names)
    X_names_sparse.shape

    # Analyze feature importances
    # from sklearn.ensemble._forest import RandomForestRegressor, RandomForestClassifier
    # import numpy as np

    X_train = playlist_df.drop(['id', 'ratings','Song'], axis=1)
    y_train = playlist_df['ratings']
    forest = RandomForestClassifier(random_state=42, max_depth=5, max_features=12) # Set by GridSearchCV below
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Print the feature rankings
    # print("Feature ranking:")
    
    # for f in range(len(importances)):
    #     print("%d. %s %f " % (f + 1, 
    #             X_train.columns[f], 
    #             importances[indices[f]]))

    # Apply pca to the scaled train set first

    # from sklearn import decomposition
    # from sklearn.preprocessing import StandardScaler
    # import matplotlib.pyplot as plt
    # import seaborn as sns; 
    sns.set(style='white')

    X_scaled = StandardScaler().fit_transform(X_train)

    pca = decomposition.PCA().fit(X_scaled)   

    # plt.figure(figsize=(10,7))
    # plt.plot(np.cumsum(pca.explained_variance_ratio_), color='k', lw=2)
    # plt.xlabel('Number of components')
    # plt.ylabel('Total explained variance')
    # plt.xlim(0, 12)
    # plt.yticks(np.arange(0, 1.1, 0.1))
    # plt.axvline(8, c='b') # Tune this so that you obtain at least a 95% total variance explained
    # plt.axhline(0.95, c='r')
    # plt.show()

    pca1 = decomposition.PCA(n_components=8)
    X_pca = pca1.fit_transform(X_scaled)


    tsne = TSNE(random_state=17)
    X_tsne = tsne.fit_transform(X_scaled)

    
    #print(X_pca,X_names_sparse)
    X_train_last = csr_matrix(hstack([X_pca, X_names_sparse])) # Check with X_tsne + X_names_sparse also


    # from sklearn.model_selection import StratifiedKFold, GridSearchCV
    # import warnings
    warnings.filterwarnings('ignore')

    # Initialize a stratified split for the validation process
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    from sklearn.tree import DecisionTreeClassifier

    tree = DecisionTreeClassifier()

    tree_params = {'max_depth': range(1,11), 'max_features': range(4,19)}

    tree_grid = GridSearchCV(tree, tree_params, cv=skf, n_jobs=-1, verbose=True)

    tree_grid.fit(X_train_last, y_train)
    tree_grid.best_estimator_, tree_grid.best_score_

    parameters = {'max_features': [4, 7, 8, 10], 'min_samples_leaf': [1, 3, 5, 8], 'max_depth': [3, 5, 8]}
    rfc = RandomForestClassifier(n_estimators=100, random_state=42, 
                                n_jobs=-1, oob_score=True)
    gcv1 = GridSearchCV(rfc, parameters, n_jobs=-1, cv=skf, verbose=1)
    gcv1.fit(X_train_last, y_train)
    gcv1.best_estimator_, gcv1.best_score_

    # from sklearn.neighbors import KNeighborsClassifier

    knn_params = {'n_neighbors': range(1, 10)}
    knn = KNeighborsClassifier(n_jobs=-1)

    knn_grid = GridSearchCV(knn, knn_params, cv=skf, n_jobs=-1, verbose=True)
    knn_grid.fit(X_train_last, y_train)
    knn_grid.best_params_, knn_grid.best_score_
    
        
    X_test_names = v.transform(rec_track_names)

    rec_playlist_df=rec_playlist_df[["acousticness", "danceability", "duration_ms", 
                            "energy", "instrumentalness",  "key", "liveness",
                            "loudness", "mode", "speechiness", "tempo", "valence"]]

    tree_grid.best_estimator_.fit(X_train_last, y_train)
    rec_playlist_df_scaled = StandardScaler().fit_transform(rec_playlist_df)
    rec_playlist_df_pca = pca1.transform(rec_playlist_df_scaled)
    X_test_last = csr_matrix(hstack([rec_playlist_df_pca, X_test_names]))
    y_pred_class = tree_grid.best_estimator_.predict(X_test_last)

    print(y_pred_class)

    rec_playlist_df['ratings']=y_pred_class
    rec_playlist_df = rec_playlist_df.sort_values('ratings', ascending = False)
    rec_playlist_df = rec_playlist_df.reset_index()

    return rec_playlist_df

def get_songs_features(sp,ids):

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

    if token:
        sp = spotipy.Spotify(auth=token)
    else:
        print("Can't get token")

    userId=sp.current_user()['id']

    if userId:
        features = []
        playlistIdUrl=[]
        user_all_Playlists = sp.current_user_playlists(limit=50,offset=0)
        for item in user_all_Playlists['items']:
            if item['external_urls']['spotify'] not in playlistIdUrl:
                playlistIdUrl.append( item['external_urls']['spotify'])

        track_ids = []
        track_names = []

        for playlistId in playlistIdUrl:
            Playlist = sp.user_playlist(userId, playlistId)
            tracks = Playlist["tracks"]
            songs = tracks["items"]
            for i in range(0, len(songs)):
                if songs[i]['track']['id'] != None and songs[i]['track']['id'] not in track_ids: # Removes the local tracks in your playlist if there is any
                    track_ids.append(songs[i]['track']['id'])
                    track_names.append(songs[i]['track']['name'])

        for i in range(0,len(track_ids)):
            audio_features = sp.audio_features(track_ids[i])
            for track in audio_features:      
                if track is None:
                    features.append({'danceability': 0, 'energy': 0, 'key': 0, 'loudness': 0, 'mode': 0, 'speechiness': 0, 'acousticness': 0, 'instrumentalness': 0, 'liveness': 0, 'valence': 0, 'tempo': 0, 'type': 'audio_features', 'id': '00000', 'uri': 'spotify:track:0', 'track_href': 'https://api.spotify.com/', 'analysis_url': 'https://api.spotify.com/', 'duration_ms': 0, 'time_signature': 0})
                else:
                    features.append(track)
                    
        #print(features)
        rec_tracks = []
        for i in features: 
            rec_tracks += sp.recommendations(seed_tracks=[i['id']], limit=int(len(features)/2))['tracks'];
        
        rec_track_ids = []
        rec_track_names = []
        for i in rec_tracks:
            rec_track_ids.append(i['id'])
            rec_track_names.append(i['name'])

        rec_track_mood=[]

        for trackId in rec_track_ids:
            preds=get_songs_features(sp,trackId)
            mood=predict_mood(trackId,preds)
            obj={'track':trackId,'mood':mood}
            rec_track_mood.append(obj)

        print("print test")
        print(rec_track_mood)
        # rec_features = []
        # for i in range(0,len(rec_track_ids)):
        #     rec_audio_features = sp.audio_features(rec_track_ids[i])
        #     for track in rec_audio_features:
        #         rec_features.append(track)
            
        # rec_playlist_df = pd.DataFrame(rec_features, index = rec_track_ids)
        # rec_playlist_df.head()

        # rec_playlist_df=train_model(rec_playlist_df,rec_track_names)

            
        # # Pick the top ranking tracks to add your new playlist 9, 10 will work
        # recs_to_add = rec_playlist_df[rec_playlist_df['ratings']==emmotion_value]['index'].values.tolist()
        # # print(recs_to_add)
        # # recs_to_add = recs_to_add[:4]
        # # print("Records to add: ", len(recs_to_add))

        # # print(len(rec_tracks)) 
        # # print(rec_playlist_df.shape)

        #rec_track=[]
        ## rec_array = np.reshape(recs_to_add, (2, 2))
        
        # for trackId in recs_to_add:
        #     trackUrl='http://open.spotify.com/track/'+str(trackId)
        #     if trackUrl not in rec_track:
        #         rec_track.append(trackUrl)
        
        return rec_track_mood

        # playlist_recs = sp.user_playlist_create(username, 
        #                                         name='Recommended Songs for Playlist by Amit - {}'.format(sourcePlaylist['name']))

        # #Add tracks to the new playlist
        # for i in rec_array:
        #     sp.user_playlist_add_tracks(username, playlist_recs['id'], i)

    else:
        return ("Can't get User")
token='BQBPjiHLvnZNEAFoUN3y8DWHAYgmMg_MEnjTJJo77I2g8YLJPYSMCQWASAFjCcY3MZ53SJZ5ZB0C-CpqKd2NkN_w2xn1WXrPF-OlGlTL_htdqAOV-d0AMQaw1sNQ7sa_tCtG45fyjMzsKEdF5qJOfrtgIOyvb9G5Xh8h9yf-fqpRg7Mz-kTyf5le46AgMB6u_r1o'
recommendations(token,1)