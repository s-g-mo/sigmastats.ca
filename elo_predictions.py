import pickle
import numpy as np
import pandas as pd

from elo_funcs import elo_funcs
from data_utils import data_utils


def _carry_ranks_forward_if_necessary(df_ranks: pd.DataFrame, matchday: int):
    current_rank_exp_col = f"matchday_{matchday}_rank_exp"
    previous_rank_exp_col = f"matchday_{matchday - 1}_rank_exp"

    current_rank_standard_col = f"matchday_{matchday}_rank_standard"
    previous_rank_standard_col = f"matchday_{matchday - 1}_rank_standard"

    check_for_null_ranks = df_ranks[current_rank_exp_col].isna()

    if check_for_null_ranks.any():
        df_ranks.loc[check_for_null_ranks, current_rank_exp_col] = df_ranks.loc[
            check_for_null_ranks, previous_rank_exp_col
        ]
        df_ranks.loc[check_for_null_ranks, current_rank_standard_col] = df_ranks.loc[
            check_for_null_ranks, previous_rank_standard_col
        ]


pd.set_option("display.max_rows", 500)

# Load team/idx mapping
team_to_idx = pickle.load(open("./data/teams/team_idx_dict.pkl", "rb"))

# Load rankings
df_ranks = pd.read_csv("./data/ranks/ranks_matchday_1.csv", index_col=0)
bias = df_ranks.data_driven_bias.values[0]

# Retcon Vancouver FC's rating - set equal to worst team
df_ranks.loc[df_ranks.team == "Vancouver FC", "matchday_0_rank_exp"] = -0.223884

# Load df_event
url = "https://fbref.com/en/comps/211/schedule/Canadian-Premier-League-Scores-and-Fixtures"
df_event = data_utils.load_df_event_FBRef(
    url=url, save_to_disk=True, fpath="./data/event_CSVs/df_event.csv"
)
df_event.sort_values("date", inplace=True)

# Split df_event based on past and future
t0 = pd.Timestamp.utcnow().date()
t0 = pd.to_datetime(t0)
df_future = df_event[(df_event.date > t0) | (df_event.date.isna())].copy()
df_past = df_event[df_event.date < t0].copy()

# Figure out most recent matchday
last_matchday = df_past.matchday.max() if not np.isnan(df_past.matchday.max()) else 0
next_matchday = last_matchday + 1

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

        # Predicted win probability - based on ranks going into current matchday
        home_rank_exp = df_ranks[df_ranks.team == home][
            f"matchday_{matchday - 1}_rank_exp"
        ].values
        away_rank_exp = df_ranks[df_ranks.team == away][
            f"matchday_{matchday - 1}_rank_exp"
        ].values

        home_win_prob = elo_funcs.σ_exp(home_rank_exp, away_rank_exp, bias)
        away_win_prob = 1 - home_win_prob

        # Write predictions to df_event
        df_event.loc[df_event.event_id == event_id, "pred_home_win_prob"] = (
            home_win_prob
        )
        df_event.loc[df_event.event_id == event_id, "pred_away_win_prob"] = (
            away_win_prob
        )

        if (home_win_prob >= 0.5) and home_win:
            df_event.loc[df_event.event_id == event_id, "correct_pred"] = 1
        if (away_win_prob >= 0.5) and away_win:
            df_event.loc[df_event.event_id == event_id, "correct_pred"] = 1
        if (home_win_prob >= 0.5) and away_win:
            df_event.loc[df_event.event_id == event_id, "correct_pred"] = 0
        if (away_win_prob >= 0.5) and home_win:
            df_event.loc[df_event.event_id == event_id, "correct_pred"] = 0
        if tie == 1:
            df_event.loc[df_event.event_id == event_id, "correct_pred"] = 2

        # Evaluate predictions
        if (home_win == 1) or (away_win == 1):
            df_event.loc[df_event.event_id == event_id, "briar_score"] = (
                elo_funcs.briar_score(home_win_prob, home_win)
            )
        if tie == 1:
            df_event.loc[df_event.event_id == event_id, "briar_score"] = (
                elo_funcs.modified_briar_score(home_win_prob)
            )

        # Update rankings
        new_home_rank = elo_funcs.update_rank_exp(
            home_rank_exp, goal_differential, home_win, home_win_prob
        )

        new_away_rank = elo_funcs.update_rank_exp(
            away_rank_exp, goal_differential, away_win, away_win_prob
        )

        df_ranks.loc[df_ranks.team == home, f"matchday_{matchday}_rank_exp"] = (
            new_home_rank
        )
        df_ranks.loc[df_ranks.team == away, f"matchday_{matchday}_rank_exp"] = (
            new_away_rank
        )

        df_ranks.loc[df_ranks.team == home, f"matchday_{matchday}_rank_standard"] = (
            elo_funcs.convert_to_standard_elo(new_home_rank)
        )

        df_ranks.loc[df_ranks.team == away, f"matchday_{matchday}_rank_standard"] = (
            elo_funcs.convert_to_standard_elo(new_away_rank)
        )

    if matchday != next_matchday:
        _carry_ranks_forward_if_necessary(df_ranks=df_ranks, matchday=matchday)

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

elo_funcs.plot_elo_evolution(df_ranks, next_matchday)

print(df_event_out)
print()
print(df_ranks)

print(f"Number of ties: {(df_event.tie).sum()}")
print(
    f"Total Number Correct: {(df_event.correct_pred == 1).sum()} / "
    f"{(df_event.home_win.notna()).sum()}"
)
