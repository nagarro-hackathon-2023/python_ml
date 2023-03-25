import operator
import spotipy
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

def train_model(rec_playlist_df,rec_track_names):

    playlist_df=pd.read_csv('Spotify_playlist_Dataset.csv')
    
    track_names=[]
    for songName in playlist_df['Song']:
        if songName not in track_names and pd.isnull(songName)==False:
            track_names.append(songName)
    
    v=TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 6), max_features=10000)
    X_names_sparse = v.fit_transform(track_names)

    # Analyze feature importances
    # from sklearn.ensemble._forest import RandomForestRegressor, RandomForestClassifier
    # import numpy as np

    X_train = playlist_df.drop(['id', 'ratings','Song'], axis=1)
    y_train = playlist_df['ratings']
    forest = RandomForestClassifier(random_state=42, max_depth=5, max_features=12) # Set by GridSearchCV below
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    #Print the feature rankings
    print("Feature ranking:")
    
    for f in range(len(importances)):
        print("%d. %s %f " % (f + 1, 
                X_train.columns[f], 
                importances[indices[f]]))

    #Apply pca to the scaled train set first

    from sklearn import decomposition
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import seaborn as sns; 
    sns.set(style='white')

    X_scaled = StandardScaler().fit_transform(X_train)

    pca = decomposition.PCA().fit(X_scaled)   

    plt.figure(figsize=(10,7))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), color='k', lw=2)
    plt.xlabel('Number of components')
    plt.ylabel('Total explained variance')
    plt.xlim(0, 12)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.axvline(8, c='b') # Tune this so that you obtain at least a 95% total variance explained
    plt.axhline(0.95, c='r')
    plt.show()

    pca1 = decomposition.PCA(n_components=3)
    X_pca = pca1.fit_transform(X_scaled)


    tsne = TSNE(random_state=17)
    X_tsne = tsne.fit_transform(X_scaled)

    
    print(X_pca,X_names_sparse)
    X_train_last = csr_matrix(hstack([X_pca, X_names_sparse])) # Check with X_tsne + X_names_sparse also


    # from sklearn.model_selection import StratifiedKFold, GridSearchCV
    # import warnings
    warnings.filterwarnings('ignore')

    # Initialize a stratified split for the validation process
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    from sklearn.tree import DecisionTreeClassifier
    #Test with Decision Tree
    tree = DecisionTreeClassifier()

    tree_params = {'max_depth': range(1,11), 'max_features': range(4,19)}

    tree_grid = GridSearchCV(tree, tree_params, cv=skf, n_jobs=-1, verbose=True)

    tree_grid.fit(X_train_last, y_train)
    print(tree_grid.best_estimator_)
    print(tree_grid.best_score_)

    #Test with Random Forest Classifier
    parameters = {'max_features': [4, 7, 8, 10], 'min_samples_leaf': [1, 3, 5, 8], 'max_depth': [3, 5, 8]}
    rfc = RandomForestClassifier(n_estimators=100, random_state=42, 
                                n_jobs=-1, oob_score=True)
    gcv1 = GridSearchCV(rfc, parameters, n_jobs=-1, cv=skf, verbose=1)
    gcv1.fit(X_train_last, y_train)
    print(gcv1.best_estimator_)
    print(gcv1.best_score_)

    from sklearn.neighbors import KNeighborsClassifier
    #Test with Nearest neighbour
    knn_params = {'n_neighbors': range(1, 10)}
    knn = KNeighborsClassifier(n_jobs=-1)

    knn_grid = GridSearchCV(knn, knn_params, cv=skf, n_jobs=-1, verbose=True)
    knn_grid.fit(X_train_last, y_train)
    knn_grid.best_params_, knn_grid.best_score_
    
        
    X_test_names = v.transform(rec_track_names)

    rec_playlist_df=rec_playlist_df[["danceability", "energy", "valence"]]

    tree_grid.best_estimator_.fit(X_train_last, y_train)
    rec_playlist_df_scaled = StandardScaler().fit_transform(rec_playlist_df)
    rec_playlist_df_pca = pca1.transform(rec_playlist_df_scaled)
    X_test_last = csr_matrix(hstack([rec_playlist_df_pca, X_test_names]))
    y_pred_class = tree_grid.best_estimator_.predict(X_test_last)

    # print(y_pred_class)

    rec_playlist_df['ratings']=y_pred_class
    rec_playlist_df = rec_playlist_df.sort_values('ratings', ascending = False)
    rec_playlist_df = rec_playlist_df.reset_index()

    return rec_playlist_df

def recommendations(token,emotion_value):
    """Get recommendations based on emotion"""
    
    if token:
        sp = spotipy.Spotify(auth=token)
    else:
        print("Can't get token")
        return

    userId=sp.current_user()['id']

    if userId:
        playlistIdUrl=[]
        #Get user's playlist
        user_all_Playlists = sp.current_user_playlists(limit=50, offset=0)
        if user_all_Playlists:
            for item in user_all_Playlists['items']:
                if item['external_urls']['spotify'] not in playlistIdUrl:
                    playlistIdUrl.append( item['external_urls']['spotify'])
        else:
            print('No playlists found')
            return
        
        track_ids = []
        track_names = []
        recommended_tracks = []
        rec_track_ids = []
        rec_track_names = []
        rec_features = []
        rec_track=[]

        for playlistId in playlistIdUrl:
            Playlist = sp.user_playlist(userId, playlistId)
            songs = Playlist["tracks"]["items"]
            for index in range(0, len(songs)):
                if songs[index]['track']['id'] and songs[index]['track']['id'] not in track_ids:
                    track_ids.append(songs[index]['track']['id'])
                    track_names.append(songs[index]['track']['name'])

        for id in track_ids: 
            recommended_tracks += sp.recommendations(seed_tracks=[id], seed_genres=['indian, happy, calm, chill'], limit=2, min_valence=0.3, min_popularity=60)['tracks']

        for i in recommended_tracks:
            rec_track_ids.append(i['id'])
            rec_track_names.append(i['name'])

        for i in range(0,len(rec_track_ids)):
            rec_audio_features = sp.audio_features(rec_track_ids[i])
            for feature in rec_audio_features:
                rec_features.append(feature)
            
        rec_playlist_df = pd.DataFrame(rec_features, index = rec_track_ids)
        rec_playlist_df.head()
        rec_playlist_df=train_model(rec_playlist_df,rec_track_names)

        recs_to_add = rec_playlist_df[rec_playlist_df["ratings"] == emotion_value]['index'].values.tolist()
        
        for trackId in recs_to_add:
            track_feature = [t for t in rec_features if t['id'] == trackId]
            if track_feature:
                track_feature = track_feature[0]
            else:
                print('Not found for ', trackId)
                continue
            trackUrl='http://open.spotify.com/track/'+str(trackId)
            rec_track.append({'url': trackUrl, 'valence': track_feature["valence"] })
        rec_track.sort(key=operator.itemgetter('valence'), reverse=True)
        return rec_track

    else:
        return ("Can't get User")

token="BQBQ1UzhRptAhRwICRp5BLP5IUzYK8qNWnvlLVnKtkvTJKlxLyIsQfzWvTI-bRlvEDEXU4G6et7FqnJhSB-ShVokB0iAxXBAeGGCvNZGI7iv7U03_S7tp8Ix6jH3SIRyecncqSvlDTmAcWTyVrOJG_lz_Wf7GKMs1xqKkJVh_57h1pTA8mIs7Yp44CY2M9g-9nbyNY7QjEmjgPyH-dWScPROKBysYsZXCmg"
recommendations(token,0)