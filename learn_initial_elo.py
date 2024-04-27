import pickle
import numpy as np
import pandas as pd

from data_utils import data_utils
from elo_funcs import elo_funcs
from ML import ML_funcs

# Load data from past 2 seasons, concatenate, clean
df_CPL_2022 = pd.read_csv('./data/historic/CPL_FBRef_2022.csv',)
df_CPL_2023 = pd.read_csv('./data/historic/CPL_FBRef_2023.csv',)
df_event = pd.concat([df_CPL_2022, df_CPL_2023])
df_event = data_utils.preprocess_historic_FBRef(df_event)

# Estimate the home win advantage using past two seasons of data
home_win_rate = df_event.home_win.sum() / len(df_event)
tie_rate = df_event.tie.sum() / len(df_event)
data_driven_bias = elo_funcs.convert_prob_to_elo_diff_exp(
    home_win_rate + tie_rate / 2
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
X_train, X_test, Y_train, Y_test = ML_funcs.construct_ELO_v1_training_data(
    df_event, 
    N_teams, 
    team_to_idx, 
    N_test_games=20
)

# Learn initial team ratings from an ensemble of 10 ELO models
N_models = 10
for i in range(N_models):
    ELO_model = ML_funcs.ELO_v1(N_teams, fixed_bias=data_driven_bias)
    ML_funcs.train_ELO_v1_model(ELO_model, X_train, Y_train)
    ML_funcs.evaluate_ELO_v1(ELO_model, X_test, Y_test)

    learned_ranks = ELO_model.layers[-1].get_weights()[0].flatten()
    df_team[f'rank_{i+1}'] = elo_funcs.convert_to_standard_elo(learned_ranks)

df_team['mean_learned_rank'] = df_team.iloc[:, 1:].mean(axis=1)
df_team.sort_values('mean_learned_rank', ascending=False, inplace=True)
df_team.to_csv('./data/ranks/learned_model_params_2024.csv')

df_elo_2024 = df_team.copy()
df_elo_2024 = df_elo_2024[['team', 'mean_learned_rank']]
df_elo_2024.rename(
    columns={'mean_learned_rank': 'matchday_0_rank_standard'},
    inplace=True
)
df_elo_2024['matchday_0_rank_exp'] = elo_funcs.convert_to_exp_elo(
    df_elo_2024['matchday_0_rank_standard'].to_numpy()
)
df_elo_2024['data_driven_bias'] = data_driven_bias
df_elo_2024 = df_elo_2024[[
    'team',
    'data_driven_bias',
    'matchday_0_rank_exp',
    'matchday_0_rank_standard',
]]

df_elo_2024.to_csv('./data/ranks/2024_matchday_1.csv')
print(df_elo_2024)