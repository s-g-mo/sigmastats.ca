import pickle
import numpy as np
import pandas as pd

from data_utils import data_utils
from elo_funcs import elo_funcs
from ML import ML_funcs

# Load data from past 3 seasons, concatenate, clean
df_CPL_2021 = pd.read_csv('./data/historic/CPL_FBRef_2021.csv',)
df_CPL_2022 = pd.read_csv('./data/historic/CPL_FBRef_2022.csv',)
df_CPL_2023 = pd.read_csv('./data/historic/CPL_FBRef_2023.csv',)
df_event = pd.concat([df_CPL_2021, df_CPL_2022, df_CPL_2023])
df_event = data_utils.preprocess_historic_FBRef(df_event)

# Estimate home win, away win, and tie rates from past seasons
away_win_rate = df_event.away_win.sum() / len(df_event)
tie_rate = df_event.tie.sum() / len(df_event)
home_win_rate = df_event.home_win.sum() / len(df_event)
bias = elo_funcs.convert_prob_to_elo_diff_exp(
    np.array([away_win_rate, tie_rate, home_win_rate])
)

# Load team to idx mapping
team_to_idx = pickle.load(open('./data/teams/team_idx_dict.pkl', 'rb'))
N_teams = len(team_to_idx)

# Create a dataframe that will eventually hold team rankings
df_team = pd.DataFrame.from_dict(
    {v:k for k,v in team_to_idx.items()}, 
    orient='index'
)
df_team.columns = ['team']

# Build training/testing
X_train, X_test, Y_train, Y_test = ML_funcs.construct_ELO_v2_training_data(
    df_event, 
    N_teams, 
    team_to_idx, 
    N_test_games=50
)

# Learn initial team ratings from an ensemble of 10 ELO models
N_models = 10
learned_ranks = np.zeros(shape=(N_models, N_teams, 3))
for i in range(N_models):
    ELO_model = ML_funcs.ELO_v2(N_teams, fixed_bias=bias)
    ML_funcs.train_ELO_v2_model(ELO_model, X_train, Y_train)
    ML_funcs.evaluate_ELO_v2(ELO_model, X_test, Y_test)
    learned_ranks[i] = ELO_model.layers[-1].get_weights()[0]

rank_weights = np.array([-away_win_rate, tie_rate, home_win_rate])

mean_ranks_exp = learned_ranks.mean(axis=0)
mean_ranks_standard = elo_funcs.convert_to_standard_elo(
    mean_ranks_exp
).reshape(mean_ranks_exp.shape)

df_team[['away_rank_exp', 'tie_rank_exp', 'home_rank_exp']] = mean_ranks_exp
df_team['composite_rank_exp'] = np.dot(mean_ranks_exp, rank_weights)

df_team[[
    'away_rank_standard', 
    'tie_rank_standard', 
    'home_rank_standard'
]] = mean_ranks_standard

df_team['composite_rank_standard'] = 3 * np.dot(mean_ranks_standard, rank_weights)

df_team.sort_values('composite_rank_exp', ascending=False, inplace=True)
df_team.to_csv('./data/ranks/learned_v2_model_params_2024.csv')

df_elo_2024 = df_team.copy()
df_elo_2024.rename(
    columns={c: 'matchday_0_' + c for c in df_elo_2024.columns if c != 'team'},
    inplace=True
)

df_elo_2024.to_csv('./data/ranks/ranks_v2_matchday_1.csv')
print(df_elo_2024)