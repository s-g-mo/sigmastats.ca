import numpy as np
import pandas as pd
from data_utils import data_utils


def beautify_event_table(df_event: pd.DataFrame):
    event_display_cols = [
        "matchday",
        "date",
        "home",
        "score",
        "away",
        "home_score",
        "away_score",
        "pred_home_win_prob",
        "pred_away_win_prob",
        "correct_pred",
        "briar_score",
        "briar_score_running_mean",
        "running_briar_skill_score",
    ]
    df_event = df_event[event_display_cols]

    event_display_name_mapping = {
        "matchday": "Matchday",
        "date": "Date",
        "home": "Home Team",
        "score": "Score",
        "away": "Away Team",
        "home_score": "Home Score",
        "away_score": "Away Score",
        "pred_home_win_prob": "Predicted Home Win Prob.",
        "pred_away_win_prob": "Predicted Away Win Prob.",
        "correct_pred": "Prediction Correct?",
        "briar_score": "Prediction Briar Score",
        "briar_score_running_mean": "Running Mean Briar Score",
        "running_briar_skill_score": "Running Briar Skill Score",
    }
    df_event = df_event.rename(columns=event_display_name_mapping)

    df_event = df_event.replace("York9 FC", "York United")

    # Format data types to make more presentable
    df_event["Prediction Briar Score"] = df_event["Prediction Briar Score"].round(3)
    df_event["Running Mean Briar Score"] = df_event["Running Mean Briar Score"].round(3)
    df_event["Running Briar Skill Score"] = df_event["Running Briar Skill Score"].round(
        3
    )
    df_event.loc[
        df_event["Running Mean Briar Score"].diff() == 0, "Running Mean Briar Score"
    ] = np.nan
    df_event.loc[
        df_event["Running Briar Skill Score"].diff() == 0, "Running Briar Skill Score"
    ] = np.nan
    df_event = df_event.fillna("")
    df_event["Home Score"] = df_event["Home Score"].astype(str)
    df_event["Away Score"] = df_event["Away Score"].astype(str)
    df_event[["Home Score", "Away Score"]] = df_event[["Home Score", "Away Score"]].map(
        lambda x: x.replace(".0", "")
    )
    df_event["Predicted Home Win Prob."] = df_event["Predicted Home Win Prob."].map(
        lambda x: f"{x * 100:.2f}%" if isinstance(x, float) else ""
    )
    df_event["Predicted Away Win Prob."] = df_event["Predicted Away Win Prob."].map(
        lambda x: f"{x * 100:.2f}%" if isinstance(x, float) else ""
    )
    df_event["Prediction Correct?"] = df_event["Prediction Correct?"].astype(str)
    df_event.loc[
        (df_event["Prediction Correct?"] == "0.0") & (df_event["Home Score"] == ""),
        "Prediction Correct?",
    ] = ""
    df_event.loc[(df_event["Prediction Correct?"] == "1.0"), "Prediction Correct?"] = (
        "True"
    )
    df_event.loc[(df_event["Prediction Correct?"] == "0.0"), "Prediction Correct?"] = (
        "False"
    )
    df_event.loc[(df_event["Prediction Correct?"] == "2.0"), "Prediction Correct?"] = (
        "False"
    )

    return df_event


def beautify_rank_table(df_rank: pd.DataFrame, next_matchday: int):
    rank_display_cols = ["team"] + [
        f"matchday_{i}_rank_standard" for i in range(0, next_matchday)
    ]
    df_rank = df_rank[rank_display_cols]

    rank_col_mapping_1 = {"team": "Team"}
    rank_col_mapping_2 = {
        f"matchday_{i}_rank_standard": f"Rank Into Matchday {i+1}"
        for i in range(0, next_matchday)
    }
    rank_display_name_mapping = {**rank_col_mapping_1, **rank_col_mapping_2}
    df_rank = df_rank.rename(columns=rank_display_name_mapping)

    numeric_columns = [col for col in df_rank.columns if col != "Team"]
    df_rank[numeric_columns] = df_rank[numeric_columns].map(
        lambda x: round(x) if not pd.isna(x) else 0
    )
    df_rank[numeric_columns] = df_rank[numeric_columns].astype(int)
    df_rank = df_rank.replace(0, "")

    df_rank = df_rank.replace("York9 FC", "York United")
    df_rank = df_rank[df_rank.Team != "FC Edmonton"]

    return df_rank


def determine_next_matchday():
    url = "https://fbref.com/en/comps/211/schedule/Canadian-Premier-League-Scores-and-Fixtures"
    df_event = data_utils.scrape_season_schedule_FBRef(url)
    current_matchday = int(df_event[df_event.Score.notna()].Wk.max())
    return current_matchday + 1
