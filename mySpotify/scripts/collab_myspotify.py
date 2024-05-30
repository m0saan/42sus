import os
import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix
import implicit
from implicit.evaluation import precision_at_k
from implicit.nearest_neighbours import CosineRecommender
import logging
from colorama import Fore, Style, init
import warnings
import argparse

# filter warnings
warnings.filterwarnings('ignore')

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format=f'{Fore.GREEN}%(asctime)s - %(levelname)s - %(message)s{Style.RESET_ALL}')

# Initialize colorama
init(autoreset=True)


class MusicData:
    def __init__(self, data):
        self.data = data
        self.song_id_to_name = pd.Series(data.song.values, index=data.song_id).to_dict()

    def get_user_song_names(self, user_id):
        user_data = self.data[self.data['user_id'] == user_id]
        user_songs = [self.song_id_to_name[song_id] for song_id in user_data['song_id'].unique()]
        return user_songs

    def get_user_song_ids(self, user_id):
      user_data = self.data[self.data['user_id'] == user_id]
      return user_data['song_idx'].unique()

    def get_song_users(self, song_id):
        song_data = self.data[self.data['song_id'] == song_id]
        song_users = song_data['user_id'].unique()
        return song_users

    def get_songs(self, song_ids):
        return self.data.filter(pl.col('user_idx') == song_ids ).select(['track_id', 'artist', 'title'])

    def get_song_name(self, song_id):
        return self.song_id_to_name[song_id]

class MusicRecommender:
    def __init__(self, music_data, train_data, test_data):
        self.music_data = music_data
        self.train_data = train_data
        self.test_data = test_data
        self.model : implicit.als.AlternatingLeastSquares = None
        self.model_path = './models/model.npz'
        logging.info('Converting data to COO format...')
        self.train_coo = self._convert_to_coo(self.train_data)
        self.test_coo = self._convert_to_coo(self.test_data)
        logging.info('Data converted successfully.')

    def _convert_to_coo(self, data):
        return coo_matrix((data.play_count.astype(float), (data.index.get_level_values(0), data.index.get_level_values(1),)))

    def _load_model(self):
        logging.info(f'Loading model from {self.model_path}...')
        self.model = self.model.load(self.model_path)
        logging.info('Model loaded successfully.')
        return self.model

    def train(self, factors=728, iterations=50, regularization=0.01, show_progress=True):
      logging.info(f'Model: AlternatingLeastSquares, Factors: {factors}, Iterations: {iterations}, Regularization: {regularization}')
      self.model = implicit.als.AlternatingLeastSquares(factors=factors,
                                                        iterations=iterations,
                                                        regularization=regularization,
                                                        random_state=42,
                                                        )
      logging.info('Checking if model already exists...')
      if os.path.exists(self.model_path):
        return self._load_model()

      logging.info('Training model...')
      self.model.fit(self.train_coo, show_progress=show_progress)
      logging.info('Model trained successfully.')
      return self.model

    def calculate_precision_at_k(self, k=10, num_threads=4):
        p = precision_at_k(self.model, self.train_coo.tocsr(), self.test_coo.tocsr(), K=k, num_threads=num_threads)
        return p

    def recommend_songs_by_user_id(self, user_id, n=10):

        user_idx = music_data.data[music_data.data['user_id'] == user_id].user_idx.values[0]
        print(f">>> Songs recommended for user with ID {user_id}: ")

        songs_ids, scores = self.model.recommend(user_idx, self.test_coo.tocsr()[n], N=n)

        user_songs_ids = music_data.get_user_song_ids(user_id)

        recommended_songs = [song for song in songs_ids if song not in user_songs_ids][:10]
        df = pd.DataFrame(songs[songs['song_idx'].isin(recommended_songs)].drop_duplicates(subset=['song_idx']).song)
        df.index.name = 'index number'

        return df


    def recommend_songs_by_song_id(self, song_id, n=10):
        filtred_songs = songs[songs['song_id'] == song_id]
        song_name, song_idx = filtred_songs.song.iloc[0], filtred_songs.song_idx.iloc[0]
        print(f'Recommendation for {song_name} with ID {song_id}: ')
        itemids, scores = self.model.similar_items(itemid=song_idx)
        df = pd.DataFrame(songs[songs['song_idx'].isin(itemids)].drop_duplicates(subset=['song_idx']).song)
        df.index.name = 'index number'

        return df


    def save_model(self):
        logging.info(f'Saving model to {self.model_path}...')
        self.model.save(self.model_path)
        logging.info('Model saved successfully.')

    def explain_recommendation_by_user_id(self, user_id, recommended_songs=None):
            user_songs = self.music_data.get_user_song_names(user_id)
            print(f"User with ID {user_id} has listened to:\n" + "\n".join([f"- {song}" for song in user_songs]))
            if recommended_songs is None:
                recommended_songs = self.recommend_songs_by_user_id(user_id, n=25)
            print(f"Recommended songs for user with ID {user_id}:\n" + "\n".join([f"- {song}" for song in recommended_songs['song']]))
            return recommended_songs

    def explain_recommendation_by_song_id(self, song_id):
        song_name = self.music_data.get_song_name(song_id)
        song_users = self.music_data.get_song_users(song_id)
        print(f"Song '{song_name}' has been listened to by {len(song_users)} users.")
        recommended_songs = self.recommend_songs_by_song_id(song_id)
        print(f"Recommended songs similar to '{song_name}':\n" + "\n".join([f"- {song}" for song in recommended_songs['song']]))
        return recommended_songs

    def new_recommendations_count_by_user_id(self, user_id, recommended_songs):
        user_songs_ids = self.music_data.get_user_song_ids(user_id)
        new_recommendations = [song for song in recommended_songs.index if song not in user_songs_ids]
        print(f"Out of {len(recommended_songs)} recommended songs, {len(new_recommendations)} are new to the user.")
        return len(new_recommendations)


def load_data(triplet_path, unique_tracks_path):
    logging.info('Loading data...')


    triplet_columns = ['user_id', 'song_id', 'play_count']
    track_columns = ['track_id', 'song_id', 'artist', 'title']

    triplet_df = pl.read_csv(triplet_path, separator='\t', new_columns=triplet_columns, use_pyarrow=True)
    unique_tracks_df = pl.from_pandas(pd.read_csv(unique_tracks_path, names=track_columns, sep="<SEP>", engine='python'))


    logging.info('Data loaded successfully.')

    logging.info('Merging songs...')

    triplet_df = triplet_df.filter(pl.col('play_count') > 1)
    songs = pd.merge(triplet_df.to_pandas(), unique_tracks_df.to_pandas(), on='song_id', how='left')
    songs['song'] = songs['title']+' - ' + songs['artist']
    songs = songs[['user_id', 'song_id', 'track_id', 'song', 'play_count']]
    print(f'Songs DataFrame:\n {songs}')

    songs['user_idx'] = pd.factorize(songs['user_id'])[0]
    songs['song_idx'] = pd.factorize(songs['song_id'])[0]

    logging.info('Songs merged successfully.')

    del triplet_df, unique_tracks_df

    # save the data
    # songs.to_csv('data/songs.csv', index=False)

    return songs


if __name__ == '__main__':
    import argparse

    # Argument parsing
    parser = argparse.ArgumentParser(description="Music recommendation system.")
    parser.add_argument('--user_id', type=str, help="User ID for song recommendations.")
    parser.add_argument('--song_id', type=str, help="Song ID for song recommendations.")
    parser.add_argument('--global_path', type=str, default='./data', help="Global path for data files.")
    parser.add_argument('--triplet_path', type=str, help="Path to the triplets data file.")
    parser.add_argument('--unique_tracks_path', type=str, help="Path to the unique tracks data file.")
    parser.add_argument('--p_at_k', type=bool, default=False, help="Calculate precision at k.")
    args = parser.parse_args()

    # Set default paths if not provided
    if not args.triplet_path:
        args.triplet_path = f"{args.global_path}/train_triplets.txt"
    if not args.unique_tracks_path:
        args.unique_tracks_path = f"{args.global_path}/p02_unique_tracks.txt"

    # Check if at least one of user_id or song_id is provided
    if not args.user_id and not args.song_id:
        logging.info('Please provide a user ID or a song ID to get recommendations.')
        parser.print_help()
        exit(1)

    # Load data
    try:
        songs = load_data(args.triplet_path, args.unique_tracks_path)
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        exit(1)

    music_data = MusicData(songs)

    # Validate user_id
    if args.user_id:
        if args.user_id not in music_data.data['user_id'].values:
            logging.error(f"User ID {args.user_id} not found in the data.")
            exit(1)

    # Validate song_id
    if args.song_id:
        if args.song_id not in music_data.data['song_id'].values:
            logging.error(f"Song ID {args.song_id} not found in the data.")
            exit(1)

    # Splitting the data into training and testing sets
    X = songs[['user_idx', 'song_idx', 'play_count']]
    train_data, test_data = train_test_split(X, test_size=0.2, random_state=42)
    train_data.set_index(["user_idx", "song_idx"], inplace=True)
    test_data.set_index(["user_idx", "song_idx"], inplace=True)

    # Create MusicRecommender instance
    try:
        music_recommender = MusicRecommender(music_data, train_data, test_data)
        music_recommender.train()
    except Exception as e:
        logging.error(f"Error initializing or training the model: {e}")
        exit(1)

    # Calculate p@k
    if args.p_at_k:
        logging.info('Calculating precision at k...')
        p_at_k = music_recommender.calculate_precision_at_k()
        logging.info(f"Precision at k: {p_at_k}")

    if args.user_id:
        try:
            logging.info('People similar to you listen 🎧')
            recommended_songs = music_recommender.recommend_songs_by_user_id(args.user_id, n=25)
            print(recommended_songs)
            logging.info('Explaining the recommendation...\n')
            music_recommender.explain_recommendation_by_user_id(args.user_id, recommended_songs)
            music_recommender.new_recommendations_count_by_user_id(args.user_id, recommended_songs)
        except Exception as e:
            logging.error(f"Error recommending songs for user ID {args.user_id}: {e}")
    if args.song_id:
        try:
            logging.info('People who listen to this track usually listen👂 🎧')
            recommended_songs = music_recommender.recommend_songs_by_song_id(args.song_id, n=25)
            print(recommended_songs)
            music_recommender.explain_recommendation_by_song_id(args.song_id)
        except Exception as e:
            logging.error(f"Error recommending songs for song ID {args.song_id}: {e}")
