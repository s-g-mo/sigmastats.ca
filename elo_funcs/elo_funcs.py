import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


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
    if goal_differential == 0:
        if win_prob >= 0.5:
            rank = rank - K_exp * win_prob
        if win_prob <= 0.5:
            rank = rank + K_exp * win_prob
    else:
        rank = rank + K_exp * (outcome - win_prob)
    return rank


def plot_elo_evolution(df_ranks, next_matchday):

    team_colors = {
        "Forge FC": np.array([226, 78, 29]) / 255,
        "Cavalry FC": np.array([166, 44, 46]) / 255,
        "Atlético Ottawa": np.array([193, 52, 40]) / 255,
        "Valour FC": np.array([148, 119, 73]) / 255,
        "Vancouver FC": np.array([18, 18, 18]) / 255,
        "Pacific FC": np.array([68, 27, 110]) / 255,
        "HFX Wanderers": np.array([84, 168, 220]) / 255,
        "York9 FC": np.array([34, 93, 51]) / 255,
    }

    plot_columns = ["team"] + [col for col in df_ranks.columns if "standard" in col]

    fig, ax = plt.subplots(dpi=300, figsize=(10, 6))

    for team_idx, row in df_ranks[plot_columns].iterrows():

        if row.team == "FC Edmonton":
            continue

        ax.scatter(
            np.arange(0, next_matchday + 1),
            row.values[1:],
            s=15,
            color=team_colors[row.team],
            edgecolor="k",
        )
        ax.plot(
            np.arange(0, next_matchday + 1),
            row.values[1:],
            color=team_colors[row.team],
            label=row.team,
        )

    for spine in ax.spines.values():
        spine.set_linewidth(1.25)

    ax.tick_params(width=1.25)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("Team Rank [Standard ELO Units]", weight="bold")
    ax.set_xlabel("Matchday #", weight="bold")
    ax.set_xlim(-0.5, next_matchday)
    ax.set_ylim(1300, 1700)
    ax.set_title("ELO Evolution By Team", weight="bold")
    plt.legend(loc="upper left", ncols=4)
    plt.tight_layout()
    plt.savefig("./figures/elo_evolution.png")
