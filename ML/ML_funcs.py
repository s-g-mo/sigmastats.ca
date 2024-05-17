import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report as metrics


class FixedBiasConstraint(tf.keras.constraints.Constraint):
    def __init__(self, value=None, vector=None):
        self.value = value
        self.vector = vector

    def __call__(self, w):
        if self.value is not None:
            return tf.fill(w.shape, self.value)
        elif self.vector is not None:
            bias = tf.cast(self.vector, dtype=w.dtype)
            return tf.reshape(bias, w.shape)
        else:
            return w

    def get_config(self):
        return {"value": self.value, "vector": self.vector}


def ELO_v1(N_teams: int, fixed_bias: float = None):
    team_vector = layers.Input(shape=(N_teams,))

    if fixed_bias is not None:
        win_prob = layers.Dense(
            1,
            bias_constraint=FixedBiasConstraint(value=fixed_bias),
            activation="sigmoid",
        )(team_vector)

    else:
        win_prob = layers.Dense(1, activation="sigmoid")(team_vector)

    return Model(inputs=team_vector, outputs=win_prob)


def ELO_v2(N_teams: int, fixed_bias: np.ndarray = None):
    team_vector = layers.Input(shape=(N_teams,))

    if fixed_bias is not None:
        win_prob = layers.Dense(
            3,
            bias_constraint=FixedBiasConstraint(vector=fixed_bias),
            activation="softmax",
        )(team_vector)

    else:
        win_prob = layers.Dense(3, activation="softmax")(team_vector)

    return Model(inputs=team_vector, outputs=win_prob)


def train_ELO_v1_model(ELO_model, X, Y):

    ELO_model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["binary_accuracy"],
    )
    ELO_model.summary()

    es = keras.callbacks.EarlyStopping(
        patience=3,
        verbose=1,
        monitor="val_loss",
        restore_best_weights=True,
    )

    history = ELO_model.fit(
        x=X,
        y=Y,
        epochs=200,
        batch_size=1,
        validation_split=0.1,
        shuffle=False,
        callbacks=[es],
    )


def train_ELO_v2_model(ELO_model, X, Y):

    ELO_model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["categorical_accuracy"],
    )
    ELO_model.summary()

    es = keras.callbacks.EarlyStopping(
        patience=3,
        verbose=1,
        monitor="val_loss",
        restore_best_weights=True,
    )
    plateau = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", patience=3, min_delta=0.001
    )

    history = ELO_model.fit(
        x=X,
        y=Y,
        epochs=200,
        batch_size=1,
        validation_split=0.1,
        shuffle=False,
        callbacks=[es, plateau],
    )


def evaluate_ELO_v1(ELO_model, X, Y):
    predictions = ELO_model.predict(X)
    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0
    print(metrics(Y, predictions, target_names=["away_win", "home_win"]))


def evaluate_ELO_v2(ELO_model, X, Y):
    predictions = ELO_model.predict(X)
    predictions = np.argmax(predictions, axis=1)
    print(
        metrics(
            np.argmax(Y, axis=1),
            predictions,
            target_names=["away_win", "tie", "home_win"],
        )
    )


def construct_ELO_v1_training_data(df_event, N_teams, team_to_idx, N_test_games=20):
    team_vectors = np.zeros(shape=(len(df_event), N_teams))
    results = np.zeros(shape=(len(df_event)))

    for i, event in df_event.iterrows():
        home = event.Home
        away = event.Away

        home_score = int(event.Score.split("–")[0])  # this dash is a funny
        away_score = int(event.Score.split("–")[1])  # uni-code char

        team_vectors[i, team_to_idx[home]] = 1
        team_vectors[i, team_to_idx[away]] = -1

        if home_score > away_score:
            results[i] = 1

    # Split - preserve temporal order when splitting in this case
    X_train = team_vectors[0:-N_test_games]
    X_test = team_vectors[-N_test_games:]
    Y_train = results[0:-N_test_games]
    Y_test = results[-N_test_games:]

    return (X_train, X_test, Y_train, Y_test)


def construct_ELO_v2_training_data(df_event, N_teams, team_to_idx, N_test_games=20):
    team_vectors = np.zeros(shape=(len(df_event), N_teams))
    results = np.zeros(shape=(len(df_event), 3))

    for i, event in df_event.iterrows():
        home = event.Home
        away = event.Away

        home_score = int(event.Score.split("–")[0])  # this dash is a funny
        away_score = int(event.Score.split("–")[1])  # uni-code char

        team_vectors[i, team_to_idx[home]] = 1
        team_vectors[i, team_to_idx[away]] = -1

        if away_score > home_score:
            results[i, 0] = 1
        if home_score == away_score:
            results[i, 1] = 1
        if home_score > away_score:
            results[i, 2] = 1

    # Split - preserve temporal order when splitting in this case
    X_train = team_vectors[0:-N_test_games]
    X_test = team_vectors[-N_test_games:]
    Y_train = results[0:-N_test_games]
    Y_test = results[-N_test_games:]

    return (X_train, X_test, Y_train, Y_test)
