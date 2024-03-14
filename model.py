import pandas as pd

from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, dot, Dropout
from tensorflow.keras.models import Model

# 1. DEFINE MF Model    
class MF(object):
    def __init__(self, n_users: int, n_movies:int, n_latent_factors: int, **kwargs):
        self.n_users = n_users 
        self.n_movies = n_movies
        self.n_latent_factors = n_latent_factors 
        self.model = self._build()

    def _build(self):
        user_input = Input(shape=(1,), name="user_input", dtype="int64") # 임베딩 벡터의 차원 수
        user_embedding = Embedding(
            self.n_users, self.n_latent_factors, name="user_embedding",
        )(user_input)
        user_vec = Flatten(name="FlattenUsers")(user_embedding)

        movie_input = Input(shape=(1,), name="movie_input", dtype="int64")
        movie_embedding = Embedding(
            self.n_movies, self.n_latent_factors, name="movie_embedding",
        )(movie_input)
        movie_vec = Flatten(name="FlattenMovies")(movie_embedding)

        sim_dot_product = dot(
            [user_vec, movie_vec],
            name="Similarity-Dot-Product",
            axes=1,
        )

        model = Model(inputs = [user_input, movie_input], outputs = sim_dot_product)
        model.compile(optimizer="adam", loss = "mse")
        return model

    def summary(self):
        return self.model.summary()

    def train(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, **kwargs):
        X_train, y_train = [train_df.user_id, train_df.movie_id], train_df.rating
        X_valid, y_valid = [valid_df.user_id, valid_df.movie_id], valid_df.rating

        history = self.model.fit(
            X_train,
            y_train,
            validation_data = (X_valid, y_valid),
            **kwargs
            )
        return history
        
    def predict(self, test_df:pd.DataFrame, **kwargs):
        X_test = [test_df['user_id'], test_df["movie_id"]]
        return self.model.predict(X_test, **kwargs)

    def evalute(self, test_df:pd.DataFrame, **kwargs):
        X_test, y_test = [test_df["user_id"], test_df["movie_id"]], test_df["rating"]
        return self.model.evaluate(X_test, y_test, **kwargs)
    

class DeepMF(MF): # MF 상속을 의미
    def __init__(self,n_users:int, n_movies:int, n_latent_factors: int):
        super().__init__(n_users = n_users, n_movies=n_movies, n_latent_factors=n_latent_factors)
        self.model = self._build()

    def _build(self):
        user_input = Input(shape=(1,), name="user_input", dtype="int64")
        user_embedding = Embedding(
            self.n_users, self.n_latent_factors, name="user_embedding",
        )(user_input)
        user_vec = Flatten(name="FlattenUsers")(user_embedding)

        movie_input = Input(shape=(1,), name="movie_input", dtype="int64")
        movie_embedding = Embedding(
            self.n_movies, self.n_latent_factors, name="movie_embedding",
        )(movie_input)
        movie_vec = Flatten(name="FlattenMovies")(movie_embedding)

        sim_dot_product = dot(
            [user_vec, movie_vec],
            name="Similarity-Dot-Product",
            axes=1,
        )
        
        x = Dense(256, activation="relu")(sim_dot_product)
        x = Dropout(0.25)(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.25)(x)
        final_output = Dense(1, activation="relu")(x)

        model = Model(inputs = [user_input, movie_input], outputs = final_output)
        model.compile(optimizer="adam", loss = "mse")
        
        return model