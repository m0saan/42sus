import os
import pandas as pd
import polars as pl
import logging
from colorama import Fore, Style, init
import warnings

# Filter warnings
warnings.filterwarnings('ignore')

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format=f'{Fore.GREEN}%(asctime)s - %(levelname)s - %(message)s{Style.RESET_ALL}')

# Initialize colorama
init(autoreset=True)

class DataLoader:
    def __init__(self, global_path):
        self.global_path = global_path
        self.triplet_path = f"{global_path}/train_triplets.txt"
        self.unique_tracks_path = f"{global_path}/p02_unique_tracks.txt"

    def load_data(self):
        logging.info('Loading data...')
        triplet_columns = ['user_id', 'song_id', 'play_count']
        track_columns = ['track_id', 'song_id', 'artist', 'title']

        triplet_df = pl.read_csv(self.triplet_path, separator='\t', new_columns=triplet_columns, use_pyarrow=True)
        unique_tracks_df = pl.from_pandas(pd.read_csv(self.unique_tracks_path, names=track_columns, sep="<SEP>", engine='python'))

        logging.info('Data loaded successfully.')
        return triplet_df, unique_tracks_df

class SongDataProcessor:
    def __init__(self, triplet_df, unique_tracks_df):
        self.triplet_df = triplet_df
        self.unique_tracks_df = unique_tracks_df

    def merge_data(self):
        logging.info('Merging songs...')
        triplet_df = self.triplet_df.filter(pl.col('play_count') > 1)
        songs = pd.merge(triplet_df.to_pandas(), self.unique_tracks_df.to_pandas(), on='song_id', how='left')
        songs['song'] = songs['title'] + ' - ' + songs['artist']
        songs = songs[['user_id', 'song_id', 'track_id', 'song', 'play_count']]
        songs['user_idx'] = pd.factorize(songs['user_id'])[0]
        songs['song_idx'] = pd.factorize(songs['song_id'])[0]

        logging.info('Songs merged successfully.')
        return songs

class SongStatistics:
    def __init__(self, songs):
        self.songs = songs
        self.unique_songs_df = self.songs[['song_idx', 'song']].drop_duplicates(subset='song_idx')

    def save_data(self, global_path):
        self.unique_songs_df.to_csv(f"{global_path}/songs.csv", index=False)
        self.songs[['user_idx', 'song_idx', 'play_count']].to_csv(f"{global_path}/ratings.csv", index=False)

    def calculate_statistics(self):
        num_ratings = self.songs.groupby('song_idx')['play_count'].count()
        mean_rating = self.songs.groupby('song_idx')['play_count'].mean()
        sum_ratings = self.songs.groupby('song_idx')['play_count'].sum()

        damping_factor = 10
        global_mean_rating = self.songs['play_count'].mean()
        damped_numerator = sum_ratings + damping_factor * global_mean_rating
        damped_denominator = num_ratings + damping_factor
        damped_mean_rating = damped_numerator / damped_denominator

        self.unique_songs_df['num_ratings'] = self.unique_songs_df['song_idx'].map(num_ratings)
        self.unique_songs_df['mean_rating'] = self.unique_songs_df['song_idx'].map(mean_rating)
        self.unique_songs_df['damped_mean_rating'] = self.unique_songs_df['song_idx'].map(damped_mean_rating)

        return self.unique_songs_df

def main():
    global_path = './data'

    data_loader = DataLoader(global_path)
    triplet_df, unique_tracks_df = data_loader.load_data()

    processor = SongDataProcessor(triplet_df, unique_tracks_df)
    songs = processor.merge_data()

    stats = SongStatistics(songs)
    stats.save_data(global_path)
    unique_songs_df = stats.calculate_statistics()

    logging.info('Top 10 songs by number of ratings:')
    print(unique_songs_df.sort_values(by='num_ratings', ascending=False).head(10))

    logging.info('Top 10 songs by mean rating:')
    print(unique_songs_df.sort_values(by='mean_rating', ascending=False).head(10))

    logging.info('Top 10 songs by damped mean rating:')
    print(unique_songs_df.sort_values(by='damped_mean_rating', ascending=False).head(10))

if __name__ == "__main__":
    main()
