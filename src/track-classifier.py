from pandas import read_csv
import numpy as np
from ast import literal_eval
from time import sleep
import neptune.new as neptune

# Initializes a neptune project on the Neptune.AI website
run = neptune.init(project='common/quickstarts',
                   api_token='ANONYMOUS')

# from spotipy.oauth2 import SpotifyOAuth
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# cleans up the genre column
def genre_cleaner(genres):
    genre = genres[0].lower()
    genre = genre.split(" ")
    return genre[len(genre) - 1]


# cleans up track feature response to get only necessary features
def extract_features(track_features):
    features_list = ["acousticness", "danceability", "duration_ms", "energy",
                     "instrumentalness", "liveness", "loudness", "speechiness", "tempo", "valence"]
    tracks = []
    for track_feature in track_features:
        track = []
        for feature in features_list:
            track.append(track_feature[feature])

        tracks.append(track)

    return np.array(tracks)


# takes in list of tracks and returns list of track ids
def get_track_ids(track_list):
    track_ids = []
    for track in track_list:
        track_ids.append(track['track']['id'])

    return track_ids


# takes in list of playlist objects, returns dictionary of {playlist names: playlist ids}
def get_playlist_names(playlists):
    playlist_dict = {}
    playlist_arr = []
    for playlist in playlists:
        playlist_dict[playlist['name']] = playlist['id']

    return playlist_dict


# reads song data (data_w_genres.csv) obtained from https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks
x_data = read_csv('../data/tracks.csv', converters={'genres': eval})

# BEGIN WRANGLING THE DATA
x_data.drop(columns=['artists', 'key', 'mode',
                     'count', 'popularity'], inplace=True)
x_data = x_data[x_data['genres'].map(lambda d: len(d)) > 0]
x_data['genres'] = x_data['genres'].apply(genre_cleaner)

# remove genres that have less tracks than min_tracks
genre_counts = x_data['genres'].value_counts()
min_tracks = 100

for index, row in genre_counts.iteritems():
    if row < min_tracks:
        x_data.drop(x_data.index[x_data['genres'] == index], inplace=True)

x_data = x_data[(x_data['genres'] == 'rock') | (x_data['genres'] == 'metal') | (x_data['genres'] == 'rap') | (x_data['genres'] == 'pop') | (x_data['genres'] == 'jazz') | (x_data['genres'] == 'country') | (x_data['genres'] == 'r&b')]

# split the data to make a classification set
y_data = x_data['genres']
x_data.drop(columns=['genres'], inplace=True)

x_data.reset_index(drop=True, inplace=True)
y_data.reset_index(drop=True, inplace=True)

x_data = x_data.to_numpy()
y_data = y_data.to_numpy()

# cross validation with K-Fold
kf = KFold(n_splits=5)
kf.get_n_splits(x_data)

accuracy = 0
for train_index, test_index in kf.split(x_data):
    X_train, X_test = x_data[train_index], x_data[test_index]
    Y_train, Y_test = y_data[train_index], y_data[test_index]

    rf = RandomForestClassifier(max_depth=10, max_features='sqrt')
    rf.fit(X_train, Y_train)
    Y_pred = rf.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    accuracy += acc

print("K-Fold with Random Forest")
print("avg accuracy: " + str(accuracy / 5))

# Logs (pretty much prints) to Neptune.ai dashboard
run['avg accuracy:'] = accuracy / 5
# Logs as a point in a Neptune.ai graph on the dashboard under "avg accuracy2:" name
run['avg accuracy2:'].log(accuracy / 5)

