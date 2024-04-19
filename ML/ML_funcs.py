import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report as metrics

# Optional way to supply a user-defined bias
class FixedValueConstraint(tf.keras.constraints.Constraint):
    def __init__(self, value):
        self.value = value

    def __call__(self, w):
        return tf.fill(w.shape, self.value)

    def get_config(self):
        return {'value': self.value}


def ELO_v1(N_teams: int, fixed_bias: None):
    team_vector = layers.Input(shape=(N_teams,))

    if fixed_bias is not None:
        win_prob = layers.Dense(
            1, 
            bias_constraint=FixedValueConstraint(fixed_bias), 
            activation='sigmoid'
        )(team_vector)

    else:
        win_prob = layers.Dense(1, activation='sigmoid')(team_vector)

    return Model(inputs=team_vector, outputs=win_prob)


def train_ELO_v1_model(ELO_model, X, Y):

    ELO_model.compile(
        optimizer=Adam(learning_rate=0.01), 
        loss=tf.keras.losses.BinaryCrossentropy(), 
        metrics=['binary_accuracy']
    )
    ELO_model.summary()

    es = keras.callbacks.EarlyStopping(
        patience=3,
        verbose=1,
        monitor='val_loss',
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


def evaluate_ELO_v1(ELO_model, X, Y):

    predictions = ELO_model.predict(X)
    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0
    print(metrics(Y, predictions, target_names=['away_win', 'home_win']))


def construct_ELO_training_data(df_event, N_teams, team_to_idx, N_test_games=20):
    team_vectors = np.zeros(shape=(len(df_event), N_teams))
    results = np.zeros(shape=(len(df_event)))

    for i, event in df_event.iterrows():
        home = event.Home
        away = event.Away

        home_score = int(event.Score.split('–')[0]) # this dash is a funny
        away_score = int(event.Score.split('–')[1]) # uni-code char

        team_vectors[i, team_to_idx[home]] = 1
        team_vectors[i, team_to_idx[away]] = -1

        if home_score > away_score:
            results[i] = 1

    # Split - preserve temporal order when splitting in this case
    X_train = team_vectors[0:-N_test_games]
    X_test =  team_vectors[-N_test_games:]
    Y_train = results[0:-N_test_games]
    Y_test = results[-N_test_games:]

    return (X_train, X_test, Y_train, Y_test)


