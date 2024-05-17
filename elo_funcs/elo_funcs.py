import numpy as np


def convert_to_standard_elo(rating):
    factor = 100 / (-np.log(10 ** (-0.25)))
    return rating.flatten() * factor + 1500


def convert_to_exp_elo(rating):
    factor = 100 / (-np.log(10 ** (-0.25)))
    return (rating.flatten() - 1500) / factor


def convert_prob_to_elo_diff_exp(prob, B=0):
    return -np.log((1 / prob) - 1) - B


def convert_prob_to_elo_diff_standard(prob, B=0):
    return -400 * np.log10((1 / prob) - 1) - B


def σ_exp(elo_H, elo_A, B):
    return 1 / (1 + np.exp(-(elo_H - elo_A + B)))


def σ_standard(elo_H, elo_A, B):
    return 1 / (1 + 10 ** ((-(elo_H - elo_A + B) / 400)))


def briar_score(pred, true):
    return np.mean((pred - true) ** 2)


def modified_briar_score(home_win_prob):
    return 2 * np.abs(0.5 - home_win_prob)


def briar_skill_score(briar_score, ref_score=0.25):
    return 1 - (briar_score / ref_score)


# For updating rankings set K = 50 ("standard" is K=20). If two teams play and
# neither has an advantage (so win prob = 50/50) then the winner would increase
# their rating by k x (1 - 0.5) -> 50 / 2 = 25 (loser would decrease by -25). So
# the next time they play, the prev winner's win prob would be 1525 vs 1475
# (57%). Hmmm, that feels high. Whatever, try K = 30 -> 1515 vs 1485 -> 54.3%
# win prob for higher rated team next time. Actually try 30 but adjust K by goal
# differential


def update_rank_exp(
    rank: float = 0, goal_differential: int = 1, outcome: int = 1, win_prob: float = 0.5
):
    K_exp = 0.175
    rank = rank + goal_differential * K_exp * (outcome - win_prob)
    return rank


def update_rank_standard(
    rank: float = 1500,
    goal_differential: int = 1,
    outcome: int = 1,
    win_prob: float = 0.5,
):
    K_standard = 30
    rank = rank + goal_differential * K_standard * (outcome - win_prob)
    return rank
