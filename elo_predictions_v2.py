import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from ML import ML_funcs
from elo_funcs import elo_funcs
from data_utils import data_utils

from IPython.core.debugger import set_trace

# Load team/idx mapping
team_to_idx = pickle.load(open("./data/teams/team_idx_dict.pkl", "rb"))

# Load rankings
df_ranks = pd.read_csv("./data/ranks/ranks_v2_matchday_1.csv", index_col=0)
bias = np.array([-0.76274047, -1.08718359, -0.28266953])

# Load df_event
url = "https://fbref.com/en/comps/211/schedule/Canadian-Premier-League-Scores-and-Fixtures"
df_event = data_utils.load_df_event_FBRef(
    url=url, save_to_disk=True, fpath="./data/event_CSVs/df_event.csv"
)

# Split df_event based on past and future
t0 = pd.Timestamp.utcnow().date()
t0 = pd.to_datetime(t0)
df_future = df_event[(df_event.date > t0) | (df_event.date.isna())].copy()
df_past = df_event[df_event.date < t0].copy()

# Figure out most recent matchday
last_matchday = df_past.matchday.max() if not np.isnan(df_past.matchday.max()) else 0
next_matchday = last_matchday + 1

df_event["correct_pred"] = 0

ELO_model = tf.keras.models.load_model(
    "./elo_model_v2.keras",
    custom_objects={"FixedBiasConstraint": ML_funcs.FixedBiasConstraint(bias)},
)
team_ranks = df_ranks.sort_values("team").iloc[:, 1:4].values
set_trace()

for matchday in range(1, next_matchday + 1):

    for _, event in df_event[df_event.matchday == matchday].iterrows():
        event_id = event.event_id
        home = event.home
        away = event.away

        if home == "York United":
            home = "York9 FC"
        if away == "York United":
            away = "York9 FC"

        home_score = event.home_score
        away_score = event.away_score
        goal_differential = np.abs(home_score - away_score)

        home_win = event.home_win
        away_win = event.away_win
        tie = event.tie

        # Probabilities
        team_vector = np.zeros(len(team_ranks))
        team_vector[team_to_idx[home]] = 1
        team_vector[team_to_idx[away]] = -1
        probs = ELO_model.predict(team_vector.reshape(1, len(team_ranks))).flatten()

        pred = np.zeros(len(probs))
        pred[np.argmax(probs)] = 1

        # Write predictions to df_event
        df_event.loc[df_event.event_id == event_id, "pred_away_win_prob"] = probs[0]
        df_event.loc[df_event.event_id == event_id, "pred_tie_prob"] = probs[1]
        df_event.loc[df_event.event_id == event_id, "pred_home_win_prob"] = probs[2]

        if away_win and pred[0] == 1:
            df_event.loc[df_event.event_id == event_id, "correct_pred"] = 1
        if tie and pred[1] == 1:
            df_event.loc[df_event.event_id == event_id, "correct_pred"] = 1
        if home_win and pred[2] == 1:
            df_event.loc[df_event.event_id == event_id, "correct_pred"] = 1

        # Evaluate predictions
        df_event.loc[df_event.event_id == event_id, "briar_score"] = (
            elo_funcs.briar_score(probs, pred)
        )

        # Current ranks
        current_home_away_rank = team_ranks[team_to_idx[home]][0]
        current_home_tie_rank = team_ranks[team_to_idx[home]][1]
        current_home_home_rank = team_ranks[team_to_idx[home]][2]

        current_away_away_rank = team_ranks[team_to_idx[away]][0]
        current_away_tie_rank = team_ranks[team_to_idx[away]][0]
        current_away_home_rank = team_ranks[team_to_idx[away]][0]

        # Update rankings
        if away_win:

            new_home_rank = elo_funcs.update_rank_exp(
                current_home_away_rank, goal_differential, home_win, probs[0]
            )
            team_ranks[team_to_idx[home]][0] = new_home_rank

            new_away_rank = elo_funcs.update_rank_exp(
                current_away_away_rank, goal_differential, away_win, probs[0]
            )
            team_ranks[team_to_idx[away]][0] = new_away_rank

            df_ranks.loc[
                df_ranks.team == home, f"matchday_{matchday}_away_rank_exp"
            ] = new_home_rank

            df_ranks.loc[
                df_ranks.team == away, f"matchday_{matchday}_away_rank_exp"
            ] = new_away_rank

            df_ranks.loc[df_ranks.team == home, f"matchday_{matchday}_tie_rank_exp"] = (
                current_home_tie_rank
            )

            df_ranks.loc[df_ranks.team == away, f"matchday_{matchday}_tie_rank_exp"] = (
                current_away_tie_rank
            )

            df_ranks.loc[
                df_ranks.team == home, f"matchday_{matchday}_home_rank_exp"
            ] = current_home_home_rank

            df_ranks.loc[
                df_ranks.team == away, f"matchday_{matchday}_home_rank_exp"
            ] = current_away_home_rank

        if tie:

            new_home_rank = elo_funcs.update_rank_exp(
                current_home_away_rank, goal_differential, home_win, probs[1]
            )
            team_ranks[team_to_idx[home]][1] = new_home_rank

            new_away_rank = elo_funcs.update_rank_exp(
                current_away_away_rank, goal_differential, away_win, probs[1]
            )
            team_ranks[team_to_idx[away]][1] = new_away_rank

            df_ranks.loc[
                df_ranks.team == home, f"matchday_{matchday}_away_rank_exp"
            ] = current_home_away_rank

            df_ranks.loc[
                df_ranks.team == away, f"matchday_{matchday}_away_rank_exp"
            ] = current_away_away_rank

            df_ranks.loc[df_ranks.team == home, f"matchday_{matchday}_tie_rank_exp"] = (
                new_home_rank
            )

            df_ranks.loc[df_ranks.team == away, f"matchday_{matchday}_tie_rank_exp"] = (
                new_away_rank
            )

            df_ranks.loc[
                df_ranks.team == home, f"matchday_{matchday}_home_rank_exp"
            ] = current_home_home_rank

            df_ranks.loc[
                df_ranks.team == away, f"matchday_{matchday}_home_rank_exp"
            ] = current_away_home_rank

        if home_win:

            new_home_rank = elo_funcs.update_rank_exp(
                current_home_away_rank, goal_differential, home_win, probs[2]
            )
            team_ranks[team_to_idx[home]][2] = new_home_rank

            new_away_rank = elo_funcs.update_rank_exp(
                current_away_away_rank, goal_differential, away_win, probs[2]
            )
            team_ranks[team_to_idx[away]][2] = new_away_rank

            df_ranks.loc[
                df_ranks.team == home, f"matchday_{matchday}_away_rank_exp"
            ] = current_home_away_rank

            df_ranks.loc[
                df_ranks.team == away, f"matchday_{matchday}_away_rank_exp"
            ] = current_away_away_rank

            df_ranks.loc[df_ranks.team == home, f"matchday_{matchday}_tie_rank_exp"] = (
                current_home_tie_rank
            )

            df_ranks.loc[df_ranks.team == away, f"matchday_{matchday}_tie_rank_exp"] = (
                current_away_tie_rank
            )

            df_ranks.loc[
                df_ranks.team == home, f"matchday_{matchday}_home_rank_exp"
            ] = new_home_rank

            df_ranks.loc[
                df_ranks.team == away, f"matchday_{matchday}_home_rank_exp"
            ] = new_away_rank

        # Update ELO model weights
        ELO_model.layers[-1].set_weights([team_ranks, bias])

        # df_ranks.loc[
        # 	df_ranks.team == home,
        # 	f'matchday_{matchday}_rank_standard'
        # ] = elo_funcs.convert_to_standard_elo(new_home_rank)
        # df_ranks.loc[
        # 	df_ranks.team == away,
        # 	f'matchday_{matchday}_rank_standard'
        # ] = elo_funcs.convert_to_standard_elo(new_away_rank)

set_trace()

df_ranks = df_ranks.sort_values(
    f"matchday_{last_matchday}_rank_exp", ascending=False
).reset_index(drop=True)
df_ranks.to_csv(f"./data/ranks/ranks_matchday_{matchday}.csv")

df_event["briar_score_running_mean"] = df_event.briar_score.expanding().mean()
df_event["running_briar_skill_score"] = elo_funcs.briar_skill_score(
    df_event["briar_score_running_mean"]
)

df_event_out = df_event[df_event.matchday <= next_matchday].copy()
df_event_out.to_csv(f"./data/event_CSVs/event_matchday_{matchday}.csv")

print(df_event_out)
print()
print(df_ranks)
