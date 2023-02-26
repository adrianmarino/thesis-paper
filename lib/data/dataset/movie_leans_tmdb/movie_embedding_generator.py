import data as dt
from sentence_transformers import SentenceTransformer
import util as ut
import os
import pandas as pd
import numpy as np



class MovieEmbeddingGenerator:
    def __init__(
        self,
        cache_path             = './tmp',
        filename               = 'movie_embedding.parquet',
        emb_columns            = ['title', 'genres', 'adults', 'language', 'overview', 'tags', 'year'],
        pre_trained_model_name = 'all-MiniLM-L6-v2',
        weights_uniform_range  = 0.1
    ):
        self._pre_trained_model_name = pre_trained_model_name
        self._emb_columns            = emb_columns
        self._cache_file_path        = f'{cache_path}/{filename}'
        self._weights_uniform_range  = weights_uniform_range
        ut.mkdir(cache_path)


    def __call__(self, df):
        if os.path.exists(self._cache_file_path):
            return pd.read_parquet(self._cache_file_path)

        df = self._to_movie_features(df, self._emb_columns)
        
        sentences = df['sentence'].values

        model      = SentenceTransformer(self._pre_trained_model_name)
        embeddings = model.encode(sentences)
        
        df['embedding'] = [emb for emb in embeddings]

        emb_size = embeddings.shape[1]
        df = pd.concat([
            df, 
            pd.DataFrame([
                { 
                    'id': 0,
                    'sentence': 'PADDING', 
                    "embedding": np.random.uniform(-self._weights_uniform_range, self._weights_uniform_range, size=emb_size)
                }])
        ])

        df.to_parquet(self._cache_file_path)

        return df 


    def _to_sentence(self, row, columns):
        output = ''
        for col in columns:
            value = row[col]
            if type(row[col]) == str:
                value = value.strip()

            output += f'{col.capitalize()}: {value}. '

        return output


    def _to_movie_features(self, df, emb_columns):
        df = df.drop(columns=[
            'user_id',
            'user_seq',
            'user_movie_rating',
            'movie_imdb_id',
            'user_movie_rating_mean',
            'user_movie_rating_norm',
            'user_movie_rating_year',
            'user_movie_rating_timestamp'
        ]).rename(columns={
            'movie_id'                : 'id',
            'user_movie_tags'         : 'tags',
            'movie_title'             :  'title',
            'movie_genres'            : 'genres',
            'movie_for_adults'        : 'adults',
            'movie_original_language' : 'language',
            'movie_overview'          :  'overview',
            'movie_release_year'      :  'year'
        }).groupby('id').agg({
            'tags'     : dt.AggFn.flatmap_join(),
            'title'    : 'max',
            'genres'   : dt.AggFn.flatmap_join(), 
            'adults'   : 'max',
            'language' : 'max',
            'overview' : 'max',
            'year'     : 'max'
        }).reset_index()

        df['sentence'] = [self._to_sentence(row, emb_columns) for _,row in df.iterrows()]

        return df[['id', 'sentence']]