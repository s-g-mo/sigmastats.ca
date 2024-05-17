# Template for if need to set manual results
df_event.loc[df_event.event_id == 11, 'score'] = '2–1'
df_event.loc[df_event.event_id == 11, 'home_score'] = 2.0
df_event.loc[df_event.event_id == 11, 'away_score'] = 1.0
df_event.loc[df_event.event_id == 11, 'home_win'] = 1.0
df_event.loc[df_event.event_id == 11, 'away_win'] = 0.0
df_event.loc[df_event.event_id == 11, 'tie'] = 0.0

# If want to plot rank evolution
# import matplotlib.pyplot as plt
# from IPython.core.debugger import set_trace

# fig, ax = plt.subplots(figsize=(10,5))
# for _, team_ranks in df_rank.iterrows():
#     ranks = team_ranks.values[1:]
#     team = team_ranks.Team
#     ax.plot(np.arange(1, next_matchday + 1), ranks, label=team, lw=2.0)
# ax.set_xlabel('Matchday', weight='bold')
# ax.set_ylabel('Rank', weight='bold')
# ax.set_title('Team Rank Evolution [2024]', weight='bold')
# for tick in ax.get_xticklabels():
#     tick.set_fontweight('bold')
# for tick in ax.get_yticklabels():
#     tick.set_fontweight('bold')
# from matplotlib.ticker import MaxNLocator
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# #plt.grid(visible=True, which='major', axis='both', zorder=-1, alpha=0.5)
# #plt.show()
