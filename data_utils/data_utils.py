import numpy as np
import pandas as pd


def load_df_event_disk(fpath: str):
    df = pd.read_csv(fpath, index_col=0)
    df.date = pd.to_datetime(df.date)
    return df


def load_df_event_FBRef(url: str, save_to_disk: bool=True, fpath: str='./'):
    df_event = scrape_season_schedule_FBRef(url)
    df_event = preprocess_season_schedule_FBRef(df_event)
    if save_to_disk:
        df_event.to_csv(fpath)
    return df_event


def scrape_season_schedule_FBRef(url: str):
    return(pd.read_html(url)[0])


def preprocess_season_schedule_FBRef(df: pd.DataFrame):
    df = df.dropna(how='all', axis=0).reset_index(drop=True).copy()
    df.Wk = df.Wk.astype(int)
    df.Date = pd.to_datetime(df.Date)
    df.sort_values(['Wk', 'Date'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    cols_drop = [
        'Match Report',
        'Day',
        'Time', 
        'Notes',
        'Referee',
        'Attendance',
        'Venue'
    ]
    
    df.drop(columns=cols_drop, inplace=True)
    df.rename(columns={'Wk': 'matchday'}, inplace=True)
    df.rename(columns={col: col.lower() for col in df.columns}, inplace=True)
    df['event_id'] = np.arange(0, len(df))
    df = df[['event_id', 'matchday', 'date', 'home', 'score', 'away']]

    # Add scores to games that have concluded
    for i, event in df.iterrows():
        if type(event.score) == str:
            home_score = int(event.score.split('–')[0])
            away_score = int(event.score.split('–')[1])
            df.loc[i, 'home_score'] = home_score
            df.loc[i, 'away_score'] = away_score
            df.loc[i, 'home_win'] = 1 if home_score > away_score else 0
            df.loc[i, 'away_win'] = 1 if away_score > home_score else 0
            df.loc[i, 'tie'] = 1 if home_score == away_score else 0
    return df


def preprocess_historic_FBRef(df: pd.DataFrame):
    df = df[['Date', 'Time', 'Home', 'Score', 'Away']]
    df = df.dropna()
    df = df[df.Date != 'Date']
    df = df.sort_values(['Date', 'Time']).reset_index(drop=True)

    df['home_score'] = [int(S.split('–')[0]) for S in df.Score]
    df['away_score'] = [int(S.split('–')[1]) for S in df.Score]
    df['home_win'] = df.home_score > df.away_score
    df['away_win'] = df.away_score > df.home_score
    df['tie'] = df.away_score == df.home_score

    return df