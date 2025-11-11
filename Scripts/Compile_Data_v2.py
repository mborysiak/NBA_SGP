#%%

import pandas as pd 
import numpy as np
from ff.db_operations import DataManage
from ff import general as ffgeneral
import ff.data_clean as dc
from scipy.stats.morestats import shapiro
import datetime as dt

from scipy.stats import poisson, truncnorm
from typing import Dict

root_path = ffgeneral.get_main_path('NBA_SGP')
db_path = f'{root_path}/Data/'
dm = DataManage(db_path)

pd.set_option('display.max_columns', 999)

def rolling_stats(df, gcols, rcols, period, agg_type='mean'):
    '''
    Calculate rolling mean stats over a specified number of previous weeks
    '''
    
    rolls = df.groupby(gcols)[rcols].rolling(period, min_periods=1).agg(agg_type).reset_index(drop=True)
    rolls.columns = [f'r{agg_type}{period}_{c}' for c in rolls.columns]

    return rolls


def rolling_expand(df, gcols, rcols, agg_type='mean'):
    '''
    Calculate rolling mean stats over a specified number of previous weeks
    '''
    
    # if agg type is in form of percentile (e.g. p80) then use quantile
    if agg_type[0]=='p':

        # pull out perc amount and convert to decimal float to calculate quantile rolling
        perc_amt = float(agg_type[1:])/100
        rolls =  df.groupby(gcols)[rcols].apply(lambda x: x.expanding().quantile(perc_amt))

    # otherwise, use the string argument of aggregation
    else:
        rolls = df.groupby(gcols)[rcols].apply(lambda x: x.expanding().agg(agg_type))
    
    # clean up the rolled dataset indices and column name prior to returning 
    rolls = rolls.reset_index(drop=True)
    rolls.columns = [f'{agg_type}all_{c}' for c in rolls.columns]

    return rolls


def add_rolling_stats(df, gcols, rcols):

    df = df.sort_values(by=[gcols[0], 'game_date']).reset_index(drop=True)

    for lag in [3, 6, 10]:
        for agg_func in ['mean', 'median', 'min', 'max']:
            cur_roll = rolling_stats(df, gcols, rcols, lag, agg_type=agg_func)
            df = pd.concat([df, cur_roll], axis=1)

    # q6_25 = rolling_stats(df, gcols, rcols, 6, agg_type=lambda x: np.percentile(x, 25))
    # q6_75 = rolling_stats(df, gcols, rcols, 6, agg_type=lambda x: np.percentile(x, 75))
    # q10_25 = rolling_stats(df, gcols, rcols, 6, agg_type=lambda x: np.percentile(x, 25))
    # q10_75 = rolling_stats(df, gcols, rcols, 6, agg_type=lambda x: np.percentile(x, 75))

    std_6 = rolling_stats(df, gcols, rcols, 6, agg_type='std')
    std_10 = rolling_stats(df, gcols, rcols, 10, agg_type='std')
    df = pd.concat([df, std_6, std_10, 
                   # q6_25, q6_75, q10_25, q10_75
                    ], axis=1)

    return df


def forward_fill(df, cols=None):
    
    if cols is None: cols = df.columns
    df = df.sort_values(by=['player', 'game_date'])
    df = df.groupby('player', as_index=False)[cols].fillna(method='ffill')
    df = df.sort_values(by=['player', 'game_date'])

    return df


def fantasy_data():

    fd = dm.read(f'''SELECT *, 100*three_pointers/three_point_pct as three_pointers_att
                    FROM FantasyData 
                    ''', 'Player_Stats').fillna({'three_pointers_att': 0})
    fd = fd.drop(['rank'], axis=1)
    fd.columns = ['fd_' + c if c not in ('player', 'team', 'position', 'opponent', 'game_date') else c for c in fd.columns ]
    fd.fd_minutes = fd.fd_minutes.apply(lambda x: float(str(x).split(':')[0]))

    return fd


def sportsline_data(df):

    sl = dm.read(f'''SELECT player, game_date,
                            sl_points, sl_trb, sl_assists, sl_blocks,
                            sl_steals, sl_turnovers, sl_fg_pct, sl_ft_pct,
                            sl_minutes, sl_fantasy_points, sl_free_throws
                     FROM Sportsline_Projections 
                    ''', 'Player_Stats')
    
    for c in ['sl_points', 'sl_trb', 'sl_assists', 'sl_blocks',
              'sl_steals', 'sl_turnovers', 'sl_fg_pct', 'sl_ft_pct',
              'sl_minutes', 'sl_fantasy_points', 'sl_free_throws']:
        try: sl[c] = sl[c].apply(lambda x: float(str(x).replace('-', '0')))
        except: pass

    df = pd.merge(df, sl, on=['player', 'game_date'], how='left')

    return df


def fantasy_pros(df):

    fp = dm.read(f'''SELECT * 
                    FROM FantasyPros 
                    ''', 'Player_Stats')

    fp['is_home'] = np.where(fp.opponent.apply(lambda x: x.split(' ')[0])=='vs', 1, 0)     
    fp = fp.drop(['position', 'games_played', 'opponent'], axis=1)
    fp.columns = ['fp_' + c if c not in ('player', 'team', 'opponent', 'game_date', 'is_home') else c for c in fp.columns ]
    fp = fp.rename(columns={'team': 'team_fp'})

    df = pd.merge(df, fp, on=['player', 'game_date'], how='outer')
    df.loc[~df.team_fp.isnull(), 'team'] = df.loc[~df.team_fp.isnull(), 'team_fp']
    df = df.drop('team_fp', axis=1)
    df = df.fillna({'is_home': 0.5})
    df.fp_ft_pct = df.fp_ft_pct.astype('float')

    return df

def numberfire(df):
    
    nf = dm.read('''SELECT player, fantasy_points, salary, value, minutes, 
                           points, rebounds, assists, steals, blocks, turnovers,
                           three_pointers, game_date
                           FROM NumberFire_Projections''', 'Player_Stats')
    nf.salary = nf.salary.apply(lambda x: int(x.replace('$', '').replace(',', '')))
    nf.columns = ['nf_' + c if c not in ('player', 'game_date') else c for c in nf.columns ]

    df = pd.merge(df, nf, on=['player', 'game_date'], how='outer')

    return df


def fix_fp_returns(df):
    df.loc[(df.fd_points > 0) & (df.nf_points > 0) & (df.fp_points==0), [c for c in df.columns if 'fp' in c]] = np.nan
    return df


def consensus_fill(df):

    to_fill = {

        # stat fills
        'proj_points': ['fp_points', 'nf_points', 'fd_points', 'sl_points'],
        'proj_rebounds': ['fp_rebounds', 'nf_rebounds', 'fd_rebounds', 'sl_trb'],
        'proj_assists': ['fp_assists', 'nf_assists', 'fd_assists', 'sl_assists'],
        'proj_blocks': ['fp_blocks', 'nf_blocks', 'fd_blocks', 'sl_blocks'],
        'proj_steals': ['fp_steals', 'nf_steals', 'fd_steals', 'sl_steals'],
        'proj_turnovers': ['fp_turnovers', 'nf_turnovers', 'fd_turnovers', 'sl_turnovers'],
        'proj_three_pointers': ['fp_three_pointers', 'nf_three_pointers', 'fd_three_pointers'],
        'proj_fg_pct': ['fp_fg_pct', 'fd_fg_pct', 'sl_fg_pct'],
        'proj_ft_pct': ['fp_ft_pct', 'fd_ft_pct', 'sl_ft_pct'],
        'proj_minutes': ['fp_minutes', 'nf_minutes', 'fd_minutes', 'sl_minutes'],
        'proj_fantasy_points': ['fd_fantasy_points', 'nf_fantasy_points', 'sl_fantasy_points'],
        'proj_ft_made': ['fd_ft_made', 'sl_free_throws'],
        }

    for k, tf in to_fill.items():

        # find columns that exist in dataset
        tf = [c for c in tf if c in df.columns]
        
        # fill in nulls based on available data
        for c in tf:
            df.loc[df[c].isnull(), c] = df.loc[df[c].isnull(), tf].mean(axis=1)
        
        # fill in the average for all cols
        df['avg_' + k] = df[tf].mean(axis=1)

    return df



def add_proj_market_share(df):
    team_stats = df.sort_values(by=['team', 'game_date', 'fd_points'],
                                ascending=[True, True, False]).copy().reset_index(drop=True)
    team_stats = team_stats[team_stats.fd_points > 0].reset_index(drop=True)

    team_stats['team_rank'] = team_stats.groupby(['team', 'game_date']).cumcount()
    team_stats = team_stats[team_stats.team_rank <= 7].reset_index(drop=True)

    share_cols = {
                    'avg_proj_points': 'sum',
                    'avg_proj_rebounds': 'sum',
                    'avg_proj_assists': 'sum',
                    'avg_proj_blocks': 'sum',
                    'avg_proj_steals': 'sum',
                    'fd_ft_made': 'sum',
                    'fd_two_point_made':'sum',
                    'avg_proj_three_pointers': 'sum',
                    'fd_turnovers': 'sum',
                    'fd_minutes': 'sum',
                    'fd_three_pointers_att': 'sum'
                    }
    team_stats = team_stats.groupby(['team', 'game_date']).agg(share_cols).reset_index()
    team_stats.columns = [f'team_{c}' if c not in ('team', 'game_date') else c for c in team_stats.columns]

    df = pd.merge(df, team_stats, on=['team', 'game_date'], how='left')

    for k, _ in share_cols.items():
        df[f'share_{k}'] = df[k] / df[f'team_{k}']

    return df



# def rolling_proj_stats(df, proj_cols):
#     df = forward_fill(df)
#     df = add_rolling_stats(df, ['player'], proj_cols)

#     for c in proj_cols:
#         df[f'trend_diff_{c}3v10'] = df[f'rmean3_{c}'] - df[f'rmean10_{c}']
#         df[f'trend_chg_{c}3v10'] = df[f'trend_diff_{c}3v10'] / (df[f'rmean10_{c}']+5)

#         df[f'trend_diff_{c}6v10'] = df[f'rmean3_{c}'] - df[f'rmean10_{c}']
#         df[f'trend_chg_{c}6v10'] = df[f'trend_diff_{c}6v10'] / (df[f'rmean10_{c}']+5)

#         df[f'trend_diff_{c}1v6'] = df[c] - df[f'rmean6_{c}']
#         df[f'trend_chg_{c}1v6'] = df[f'trend_diff_{c}1v6'] / (df[f'rmean6_{c}']+5)

#     return df

def rolling_proj_stats(df, proj_cols):
    df = forward_fill(df)
    df = add_rolling_stats(df, ['player'], proj_cols)
    
    # Create all new columns at once using a dictionary, then join them
    new_columns = {}
    
    for c in proj_cols:
        # Calculate all values first
        rmean10_plus5 = df[f'rmean10_{c}'] + 5
        rmean6_plus5 = df[f'rmean6_{c}'] + 5
        
        trend_diff_3v10 = df[f'rmean3_{c}'] - df[f'rmean10_{c}']
        trend_diff_6v10 = df[f'rmean6_{c}'] - df[f'rmean10_{c}']
        trend_diff_1v6 = df[c] - df[f'rmean6_{c}']
        
        # Store in dictionary instead of adding to DataFrame individually
        new_columns[f'trend_diff_{c}3v10'] = trend_diff_3v10
        new_columns[f'trend_chg_{c}3v10'] = trend_diff_3v10 / rmean10_plus5
        
        new_columns[f'trend_diff_{c}6v10'] = trend_diff_6v10
        new_columns[f'trend_chg_{c}6v10'] = trend_diff_6v10 / rmean10_plus5
        
        new_columns[f'trend_diff_{c}1v6'] = trend_diff_1v6
        new_columns[f'trend_chg_{c}1v6'] = trend_diff_1v6 / rmean6_plus5
    
    # Add all new columns at once to prevent fragmentation
    new_df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    
    return new_df



def proj_market_share(df, proj_col_name):

    proj_cols = [c for c in df.columns if proj_col_name in c]

    for proj_col in proj_cols:
        orig_col = proj_col.replace(proj_col_name, '')
        if orig_col in df.columns:
            df[f'{proj_col_name}share_{orig_col}'] = df[orig_col] / (df[proj_col]+3)
            df[f'{proj_col_name}share_diff_{orig_col}'] = df[orig_col] - df[proj_col]
            df[[f'{proj_col_name}share_{orig_col}', f'{proj_col_name}share_diff_{orig_col}']] = \
                df[[f'{proj_col_name}share_{orig_col}', f'{proj_col_name}share_diff_{orig_col}']].fillna(0)
    return df


#-------------------
# Final Cleanup
#-------------------

def remove_non_uniques(df):
    cols = df.nunique()[df.nunique()==1].index
    cols = [c for c in cols if c != 'pos']
    df = df.drop(cols, axis=1)
    return df

def drop_duplicate_players(df):
    df = df.sort_values(by=['player', 'game_date', 'avg_proj_points'],
                        ascending=[True, True, False])
    df = df.drop_duplicates(subset=['player', 'game_date'], keep='first').reset_index(drop=True)
    return df


def cleanup_minutes(df):
    df = df.dropna(subset=['MIN']).reset_index(drop=True)
    df.MIN = df.MIN.apply(lambda x: float(x.split(':')[0]) + float(x.split(':')[1])/60)
    return df

def get_box_score():
    bs = dm.read("SELECT * FROM Box_Score", 'Player_Stats')
    bs = bs.rename(columns={'PLAYER_NAME': 'player',
                            'PTS': 'points',
                            'REB': 'rebounds',
                            'AST': 'assists',
                            'STL': 'steals',
                            'BLK': 'blocks',
                            'FG3M': 'three_pointers' })

    bs = bs.drop(['GAME_ID', 'TEAM_ABBREVIATION', 'TEAM_ID', 'TEAM_CITY', 'PLAYER_ID', 
                  'NICKNAME', 'START_POSITION', 'COMMENT'], axis=1)
    bs = cleanup_minutes(bs)
    bs = bs.sort_values(by=['player', 'game_date']).reset_index(drop=True)

    return bs


def add_y_act(df, bs):

    y_act = bs[['player', 'game_date', 'points', 'rebounds', 'assists', 'three_pointers',# 'steals', 'blocks'
                ]].copy()
    y_act.columns = ['y_act_' + c if c not in ('player', 'game_date') else c for c in y_act.columns]
    df = pd.merge(df, y_act, on=['player', 'game_date'], how='left')

    return df


def add_last_game_box_score(df, box_score):
    
    box_score = box_score[['player', 'game_date', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3_PCT', 'three_pointers', 'FG3A', 
                           'FTM', 'FTA', 'OREB', 'DREB', 'rebounds', 'assists', 'steals', 'blocks', 'FT_PCT',
                            'TO', 'points', 'PLUS_MINUS']]
    box_score.columns = [f'last_game_{c}' if c not in ('player', 'game_date') else c for c in box_score.columns]

    box_score = box_score.sort_values(by=['player', 'game_date']).reset_index(drop=True)
    box_score.game_date = box_score.groupby('player')['game_date'].shift(-1)
    box_score = box_score.fillna({'game_date': max_date})
    df = pd.merge(df, box_score, on=['player', 'game_date'], how='left')

    return df


def remove_no_minutes(df, box_score):
    df = pd.merge(df, box_score.loc[box_score.MIN>0, ['player', 'game_date', 'MIN']], on=['player', 'game_date'], how='left')
    df = df[(df.game_date==max_date) | ~(df.MIN.isnull())].reset_index(drop=True)
    df = df.drop('MIN', axis=1)
    print('no minutes:', df.shape)
    return df

def remove_low_minutes(df, box_score):
   
    df = pd.merge(df, box_score.loc[box_score.MIN>0, ['player', 'game_date', 'MIN']], on=['player', 'game_date'], how='left')
    df = df.loc[((df.MIN > df.rmean3_MIN*0.4) & (df.MIN > df.rmean6_MIN*0.4)) | (df.game_date==max_date)].reset_index(drop=True)
    df = df.drop('MIN', axis=1)
    print('drop low minutes:', df.shape)
    return df


def box_score_rolling(df, bs):

    r_cols = [c for c in bs.columns if c not in ('player', 'team', 'game_date')]
    bs = add_rolling_stats(bs, gcols=['player'], rcols=r_cols)
    bs = bs.drop(['MIN', 'FGM', 'FGA', 'FG_PCT', 'three_pointers',
                  'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'rebounds',
                  'assists', 'steals', 'blocks', 'TO', 'PF', 'points', 'PLUS_MINUS'], axis=1)
    bs.game_date = bs.groupby('player')['game_date'].shift(-1)
    bs = bs.fillna({'game_date': max_date})

    df = pd.merge(df, bs, on=['player', 'game_date'])

    for c in ['MIN', 'FGM', 'FGA', 'FG_PCT', 'three_pointers',
              'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'rebounds',
              'assists', 'steals', 'blocks', 'TO', 'points', 'PLUS_MINUS']:
        df[f'{c}_last_vs_rmean3'] = df[f'last_game_{c}'] - df[f'rmean3_{c}']



    return df

def add_advanced_stats(df):
    adv = dm.read("SELECT * FROM Advanced_Stats", 'Player_Stats')
    adv = adv.rename(columns={'PLAYER_NAME': 'player'})

    adv = adv.drop(['GAME_ID', 'TEAM_ABBREVIATION', 'MIN', 'TEAM_ID', 'TEAM_CITY', 
                    'PLAYER_ID', 'NICKNAME', 'START_POSITION', 'COMMENT'], axis=1)
    adv = adv.sort_values(by=['player', 'game_date']).reset_index(drop=True)

    r_cols = [c for c in adv.columns if c not in ('player', 'game_date')]
    adv_roll = add_rolling_stats(adv, gcols=['player'], rcols=r_cols)

    adv_roll = adv_roll.drop(['E_OFF_RATING', 'OFF_RATING', 'E_DEF_RATING', 'DEF_RATING',
                              'E_NET_RATING', 'NET_RATING', 'AST_PCT', 'AST_TOV', 'AST_RATIO',
                              'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'TM_TOV_PCT', 'EFG_PCT', 'TS_PCT',
                              'USG_PCT', 'E_USG_PCT', 'E_PACE', 'PACE', 'PACE_PER40', 'POSS', 'PIE',
                            ], axis=1)
                
    adv_roll.game_date = adv_roll.groupby('player')['game_date'].shift(-1)
    adv_roll = adv_roll.fillna({'game_date': max_date})
    df = pd.merge(df, adv_roll, on=['player', 'game_date'], how='left')

    return df, adv


# add the rolling advanced stats and tracking stats
def add_last_game_advanced_stats(df, adv_stats):
    adv_stats_cols = adv_stats.columns
    adv_stats = adv_stats.rename(columns={k: f'last_game_{k}' for k in adv_stats.columns if k not in ('player', 'game_date')})
    adv_stats = adv_stats.sort_values(by=['player', 'game_date']).reset_index(drop=True)
    adv_stats.game_date = adv_stats.groupby('player')['game_date'].shift(-1)
    adv_stats = adv_stats.fillna({'game_date': max_date})
    df = pd.merge(df, adv_stats, on=['player', 'game_date'], how='left')

    for c in adv_stats_cols:
        if c not in ('player', 'game_date'):
            df[f'{c}_last_vs_rmean3'] = df[f'last_game_{c}'] - df[f'rmean3_{c}']

    return df


def add_tracking_stats(df):
    track = dm.read("SELECT * FROM Tracking_Data", 'Player_Stats')
    track = track.rename(columns={'PLAYER_NAME': 'player'})

    track.MIN = track.MIN.apply(lambda x: float(str(x).replace(':','.')))
    for c in ['TCHS', 'PASS', 'DIST', 'ORBC', 'DRBC', 'RBC']:
        track[f'{c}_per_min'] = track[c] / (track.MIN + 5)

    track = track.drop(['GAME_ID', 'TEAM_ABBREVIATION', 'MIN', 'TEAM_ID', 'TEAM_CITY', 
                        'PLAYER_ID', 'START_POSITION', 'COMMENT'], axis=1)
    track = track.sort_values(by=['player', 'game_date']).reset_index(drop=True)

    r_cols = [c for c in track.columns if c not in ('player', 'game_date')]
    track = add_rolling_stats(track, gcols=['player'], rcols=r_cols)

    track = track.drop(['SPD', 'DIST', 'ORBC', 'DRBC', 'RBC', 'TCHS', 'SAST', 'FTAST',
                        'PASS', 'AST', 'CFGM', 'CFGA', 'CFG_PCT', 'UFGM', 'UFGA', 'UFG_PCT',
                        'FG_PCT', 'DFGM', 'DFGA', 'DFG_PCT',
                        'TCHS_per_min', 'PASS_per_min', 'DIST_per_min', 'ORBC_per_min', 
                        'DRBC_per_min', 'RBC_per_min'
                  ], axis=1)
    
    track.game_date = track.groupby('player')['game_date'].shift(-1)
    track = track.fillna({'game_date': max_date})
    df = pd.merge(df, track, on=['player', 'game_date'], how='left')

    return df


def add_team_box_score(df, team_or_opp):
    team_bs = dm.read("SELECT * FROM Box_Score", 'Team_Stats')
    team_bs  = team_bs.rename(columns={'TEAM_ABBREVIATION': team_or_opp})

    team_bs = team_bs.drop(['GAME_ID', 'TEAM_ID', 'TEAM_CITY', 'TEAM_NAME'], axis=1)
    team_bs = team_bs.sort_values(by=[team_or_opp, 'game_date']).reset_index(drop=True)
    team_bs = cleanup_minutes(team_bs)
    team_bs['is_overtime'] = np.where(team_bs.MIN > 240, 1, 0)

    r_cols = [c for c in team_bs.columns if c not in (team_or_opp, 'game_date')]
    team_bs = add_rolling_stats(team_bs, gcols=[team_or_opp], rcols=r_cols)
    team_bs = team_bs.drop(r_cols, axis=1)
        
    team_bs.game_date = team_bs.groupby(team_or_opp)['game_date'].shift(-1)
    team_bs = team_bs.fillna({'game_date': max_date})

    team_bs.columns = [f'{team_or_opp}_{c}' if c not in (team_or_opp, 'game_date') else c for c in team_bs.columns]
    df = pd.merge(df, team_bs, on=[team_or_opp, 'game_date'], how='left')

    return df

def add_hustle_stats(df):
    hustle = dm.read("SELECT * FROM Hustle_Stats", 'Player_Stats')
    hustle = hustle.rename(columns={'PLAYER_NAME': 'player'})

    hustle.MINUTES = hustle.MINUTES.apply(lambda x: float(str(x).replace(':','.')))
    for c in ['CONTESTED_SHOTS', 'DEFLECTIONS', 'OFF_LOOSE_BALLS_RECOVERED', 'DEF_LOOSE_BALLS_RECOVERED']:
        hustle[f'{c}_per_min'] = hustle[c] / (hustle.MINUTES + 5)

    hustle = hustle.drop(['GAME_ID', 'TEAM_ABBREVIATION', 'TEAM_ID', 'TEAM_CITY', 
                        'PLAYER_ID', 'START_POSITION', 'COMMENT', 'MINUTES', 'PTS'], axis=1)
    hustle = hustle.sort_values(by=['player', 'game_date']).reset_index(drop=True)

    r_cols = [c for c in hustle.columns if c not in ('player', 'game_date')]
    hustle_roll = add_rolling_stats(hustle, gcols=['player'], rcols=r_cols)

    hustle_roll = hustle_roll.drop([ 'CONTESTED_SHOTS', 'CONTESTED_SHOTS_2PT',
                                    'CONTESTED_SHOTS_3PT', 'DEFLECTIONS', 'CHARGES_DRAWN', 'SCREEN_ASSISTS',
                                    'SCREEN_AST_PTS', 'OFF_LOOSE_BALLS_RECOVERED', 'BOX_OUTS',
                                    'DEF_LOOSE_BALLS_RECOVERED', 'LOOSE_BALLS_RECOVERED', 'OFF_BOXOUTS',
                                    'DEF_BOXOUTS', 'BOX_OUT_PLAYER_TEAM_REBS', 'BOX_OUT_PLAYER_REBS',
                                    'CONTESTED_SHOTS_per_min', 'DEFLECTIONS_per_min', 
                                    'OFF_LOOSE_BALLS_RECOVERED_per_min', 'DEF_LOOSE_BALLS_RECOVERED_per_min'
                                    ], axis=1)
                
    hustle_roll.game_date = hustle_roll.groupby('player')['game_date'].shift(-1)
    hustle_roll = hustle_roll.fillna({'game_date': max_date})
    df = pd.merge(df, hustle_roll, on=['player', 'game_date'], how='left')

    return df

def add_team_advanced_stats(df, team_or_opp):
    team_adv = dm.read("SELECT * FROM Advanced_Stats", 'Team_Stats')
    team_adv  = team_adv.rename(columns={'TEAM_ABBREVIATION': team_or_opp})

    team_adv = team_adv.drop(['GAME_ID', 'MIN', 'TEAM_ID', 'TEAM_CITY', 'TEAM_NAME'], axis=1)
    team_adv = team_adv.sort_values(by=[team_or_opp, 'game_date']).reset_index(drop=True)

    r_cols = [c for c in team_adv.columns if c not in (team_or_opp, 'game_date')]
    team_adv = add_rolling_stats(team_adv, gcols=[team_or_opp], rcols=r_cols)
    team_adv = team_adv.drop(r_cols, axis=1)
        
    team_adv.game_date = team_adv.groupby(team_or_opp)['game_date'].shift(-1)
    team_adv = team_adv.fillna({'game_date': max_date})
    team_adv.columns = [f'{team_or_opp}_{c}' if c not in (team_or_opp, 'game_date') else c for c in team_adv.columns]
    df = pd.merge(df, team_adv, on=[team_or_opp, 'game_date'], how='left')

    return df

def add_team_tracking(df, team_or_opp):
    team_tr = dm.read("SELECT * FROM Tracking_Data", 'Team_Stats')
    team_tr  = team_tr.rename(columns={'TEAM_ABBREVIATION': team_or_opp})

    team_tr = team_tr.drop(['GAME_ID', 'MIN', 'TEAM_ID', 'TEAM_CITY', 'TEAM_NAME'], axis=1)
    team_tr = team_tr.sort_values(by=[team_or_opp, 'game_date']).reset_index(drop=True)

    r_cols = [c for c in team_tr.columns if c not in (team_or_opp, 'game_date')]
    team_tr = add_rolling_stats(team_tr, gcols=[team_or_opp], rcols=r_cols)
    team_tr = team_tr.drop(r_cols, axis=1)
        
    team_tr.game_date = team_tr.groupby(team_or_opp)['game_date'].shift(-1)
    team_tr = team_tr.fillna({'game_date': max_date})
    team_tr.columns = [f'{team_or_opp}_{c}' if c not in (team_or_opp, 'game_date') else c for c in team_tr.columns]
    df = pd.merge(df, team_tr, on=[team_or_opp, 'game_date'], how='left')

    return df

def add_team_hustle_stats(df, team_or_opp):
    team_hustle = dm.read("SELECT * FROM Hustle_Stats", 'Team_Stats')
    team_hustle.MINUTES = team_hustle.MINUTES.apply(lambda x: float(str(x).replace(':','')))
    team_hustle  = team_hustle.rename(columns={'TEAM_ABBREVIATION': team_or_opp})

    team_hustle = team_hustle.drop(['GAME_ID', 'TEAM_ID', 'TEAM_CITY', 'TEAM_NAME', 'PTS'], axis=1)
    team_hustle = team_hustle.sort_values(by=[team_or_opp, 'game_date']).reset_index(drop=True)

    r_cols = [c for c in team_hustle.columns if c not in (team_or_opp, 'game_date')]
    team_hustle = add_rolling_stats(team_hustle, gcols=[team_or_opp], rcols=r_cols)
    team_hustle = team_hustle.drop(r_cols, axis=1)
        
    team_hustle.game_date = team_hustle.groupby(team_or_opp)['game_date'].shift(-1)
    team_hustle = team_hustle.fillna({'game_date': max_date})
    team_hustle.columns = [f'{team_or_opp}_{c}' if c not in (team_or_opp, 'game_date') else c for c in team_hustle.columns]
    df = pd.merge(df, team_hustle, on=[team_or_opp, 'game_date'], how='left')

    return df


def available_stats(df, missing):

    avail_columns = [
                    'player', 'team', 'game_date',
                    'rmean6_MIN', 'rmean6_FGM', 'rmean6_FGA', 'rmean6_three_pointers', 
                    'rmean6_FG3A', 'rmean6_FTM', 'rmean6_FTA', 'rmean6_OREB', 'rmean6_DREB', 
                    'rmean6_rebounds', 'rmean6_assists', 'rmean6_steals', 'rmean6_blocks', 
                    'rmean6_TO', 'rmean6_points', 'rmean6_PLUS_MINUS'
                    ]

    missing_data = df[avail_columns].copy().rename(columns={'game_date': 'last_game_date'})
    missing = pd.merge(missing, missing_data, on='player')

    missing = missing[missing.game_date > missing.last_game_date]
    max_missing = missing.groupby(['player', 'game_date']).agg({'last_game_date': 'max'}).reset_index()
    missing = pd.merge(missing, max_missing, on=['player', 'game_date', 'last_game_date'])

    missing['days_missing'] = (pd.to_datetime(missing.game_date) - pd.to_datetime(missing.last_game_date)).dt.days
    missing = missing[missing.days_missing < 12]
    agg_cols = {k: 'sum' for k in avail_columns if k not in ('player', 'team', 'game_date')}
    missing = missing.groupby(['team', 'game_date']).agg(agg_cols).reset_index()

    missing.columns = [f'avail_{c}' if c not in ('team', 'game_date') else c for c in missing.columns ]
    df = pd.merge(df, missing, on=['team', 'game_date'], how='left')

    fill_cols = {f'avail_{k}': 0 for k in avail_columns if k not in ('player', 'team', 'game_date')}
    df = df.fillna(fill_cols)

    return df


def add_dk_team_lines(df):
    lines = dm.read("SELECT * FROM Draftkings_Odds", 'Team_Stats')
    lines['implied_points_for'] = (lines.over / 2) - (lines.spread/2)
    lines['implied_points_against'] = (lines.over / 2) + (lines.spread/2)
    lines = lines[['team', 'game_date', 'moneyline_odds', 'over', 'implied_points_for', 'implied_points_against', 'spread']]

    lines = add_rolling_stats(lines, ['team'], ['moneyline_odds', 'over', 'implied_points_for', 'implied_points_against'])
    df = pd.merge(df, lines, on=['team', 'game_date'])
    return df


def get_columns(train_date):
    try:
        cols = dm.read(f"SELECT * FROM Model_Data_{train_date}", 'Model_Features').columns.tolist()
        try: 
            cols.extend(dm.read(f"SELECT * FROM Model_Data_{train_date}v2", 'Model_Features').columns.tolist())
        except: 
            pass
        
        return cols

    except:
        return None

def trunc_normal(mean_val, sdev, min_sc, max_sc, num_samples=500):

    import scipy.stats as stats

    # create truncated distribution
    lower_bound = (min_sc - mean_val) / sdev, 
    upper_bound = (max_sc - mean_val) / sdev
    trunc_dist = stats.truncnorm(lower_bound, upper_bound, loc=mean_val, scale=sdev)
    
    estimates = trunc_dist.rvs(num_samples)

    return estimates


class PlayerPropsCalculator:
    """
    Vectorized calculator for player props expected values using 
    Poisson and Truncated Normal distributions.
    """
    
    def __init__(self, 
                 max_iterations: int = 20, 
                 max_lambda: float = 50.0):
        """
        Initialize calculator with configuration parameters.
        
        Args:
            max_iterations: Maximum iterations for binary search
            max_lambda: Maximum lambda value for Poisson

        """
        self.max_iterations = max_iterations
        self.max_lambda = max_lambda
        
   
    
    def calculate_discrete_probabilities_vectorized(self, lambda_params: np.ndarray, 
                                                  thresholds: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Vectorized calculation of discrete probabilities for Poisson.
        """
        floor_thresholds = np.floor(thresholds).astype(int)
        ceil_thresholds = np.ceil(thresholds).astype(int)
        
        floor_probs = poisson.cdf(floor_thresholds, lambda_params)
        ceil_probs = poisson.cdf(ceil_thresholds, lambda_params)
        weights = ceil_thresholds - thresholds
        
        return {
            'floor_probs': floor_probs,
            'ceil_probs': ceil_probs,
            'weights': weights
        }
    
    def find_poisson_parameters_vectorized(self, points: np.ndarray, 
                                         target_probs: np.ndarray) -> np.ndarray:
        """
        Vectorized binary search to find lambda parameters.
        """
        low = np.zeros_like(points)
        high = np.full_like(points, self.max_lambda)
        
        for _ in range(self.max_iterations):
            mid = (low + high) / 2
            probs = self.calculate_discrete_probabilities_vectorized(mid, points)
            interpolated_probs = (
                probs['floor_probs'] * probs['weights'] + 
                probs['ceil_probs'] * (1 - probs['weights'])
            )
            over_probs = 1 - interpolated_probs
            high = np.where(over_probs > target_probs, mid, high)
            low = np.where(over_probs <= target_probs, mid, low)
        
        return (low + high) / 2

    def calculate_truncnorm_ev_vectorized(self, means: np.ndarray, 
                                        stds: np.ndarray) -> np.ndarray:
        """
        Calculate expected values for truncated normal with configured bounds.
        """
        min_value = self.truncnorm_config['min_value']
        max_value = self.truncnorm_config['max_value']
        
        # Calculate normalized bounds
        a = (min_value - means) / stds
        b = (max_value - means) / stds
        
        # Calculate expected value using truncated normal formula
        expected_values = truncnorm.mean(a, b, loc=means, scale=stds)
        
        return expected_values
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process entire DataFrame of player props using vectorized operations.
        """
        required_columns = {'point', 'price', 'prop_type', 'game_date'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        result_df = df.copy()
        result_df['point'] = pd.to_numeric(result_df['point'], errors='coerce')
        result_df['price'] = pd.to_numeric(result_df['price'], errors='coerce')
        
        valid_mask = (
            ~result_df['point'].isna() & 
            ~result_df['price'].isna() & 
            (result_df['price'] > 1) & 
            (result_df['point'] > 0)
        )
        
        # Initialize columns
        result_df['ev_poisson'] = np.nan
        
        if valid_mask.any():
            points = result_df.loc[valid_mask, 'point'].values
            prices = result_df.loc[valid_mask, 'price'].values
            implied_probs = 1 / prices
            
            # Poisson calculations
            lambda_params = self.find_poisson_parameters_vectorized(points, implied_probs)
            poisson_evs = lambda_params
            
            # Assign results
            result_df.loc[valid_mask, 'ev_poisson'] = poisson_evs
        
        return result_df
    

def pull_vegas_stats(prop):
    
    df = dm.read(f'''SELECT *
                     FROM Game_Odds
                     WHERE name IN ('Over', 'Yes')
                           AND prop_type = '{prop}'
        ''', 'Player_Stats')
    
    df = df.rename(columns={'description': 'player'})

    return df


def get_stat_constraints(player_data, stat_cols):
    
    stat_mean = player_data[stat_cols].sum(axis=1).mean()
    stat_std = player_data[stat_cols].sum(axis=1).std()
    stat_min = player_data[stat_cols].sum(axis=1).min()
    stat_max = player_data[stat_cols].sum(axis=1).max()

    stat_cv = stat_std / stat_mean

    return stat_mean, stat_cv, stat_min, stat_max

def get_all_vegas_stats():

    
    player_data = dm.read(f'''SELECT PLAYER_NAME player, 
                                    game_date,
                                    MIN minutes,
                                    PTS points,
                                    REB rebounds,
                                    AST assists,
                                    STL steals,
                                    BLK blocks,
                                    FG3M three_pointers
                            FROM Box_Score 
                        ''', 'Player_Stats')
    player_data.minutes = player_data.minutes.fillna('0:').apply(lambda x: x.split(':')[0]).astype(float)
    player_data = player_data[player_data.minutes > 10].reset_index(drop=True)

    pos_stats = [
            ['player_points', ['points']],
            ['player_assists', ['assists']],
            ['player_rebounds', ['rebounds']],
            ['player_steals', ['steals']],
            ['player_blocks', ['blocks']],
            ['player_points_rebounds', ['points', 'rebounds']],
            ['player_points_assists', ['points', 'assists']],
            ['player_points_rebounds_assists', ['points', 'rebounds', 'assists']],
            ['player_threes', ['three_pointers']],
            ['player_blocks_steals', ['blocks', 'steals']],
        ]
    
    i = 0
    for prop_types, stat_cols in pos_stats:
        odds = pull_vegas_stats(prop_types)
        cur_stat = pd.merge(odds, player_data, on=['player', 'game_date'])
        _, stat_cv, stat_min, stat_max = get_stat_constraints(cur_stat, stat_cols)

        odds['ev_trunc_norm'] = np.nan
        for point in odds.point.unique():
            td_dist = trunc_normal(point, point*stat_cv, stat_min, stat_max, num_samples=10000)
            odds.loc[odds.point==point, 'ev_trunc_norm'] = np.percentile(td_dist, 100/odds.loc[odds.point==point, 'price'])

        calculator = PlayerPropsCalculator(max_lambda=odds.point.max())
        odds = calculator.process_dataframe(odds)
        odds = odds.pivot_table(index=['player', 'game_date'], columns='prop_type', values=['ev_trunc_norm', 'ev_poisson']).reset_index()
        odds.columns = [f"{c[1]}_{c[0]}" if c[1]!='' else c[0] for c in odds.columns]
        if i == 0: player_vegas_stats = odds.copy()
        else: player_vegas_stats = pd.merge(player_vegas_stats, odds, on=['player', 'game_date'], how='outer')
        i += 1

    return player_vegas_stats


def fill_vegas_stats(df, player_vegas_stats):

    df = pd.merge(df, player_vegas_stats, on=['player', 'game_date'], how='left')
    
    df['avg_proj_points_rebounds_assists'] = df[['avg_proj_points', 'avg_proj_rebounds', 'avg_proj_assists']].sum(axis=1)
    df['avg_proj_points_rebounds'] = df[['avg_proj_points', 'avg_proj_rebounds']].sum(axis=1)
    df['avg_proj_points_assists'] = df[['avg_proj_points', 'avg_proj_assists']].sum(axis=1)
    df['avg_proj_steals_blocks'] = df[['avg_proj_steals', 'avg_proj_blocks']].sum(axis=1)

    fill_cols = {
       'player_points_ev_poisson': ['avg_proj_points'],
       'player_points_ev_trunc_norm': ['avg_proj_points'], 
       'player_assists_ev_poisson': ['avg_proj_assists'],
       'player_assists_ev_trunc_norm': ['avg_proj_assists'], 
       'player_rebounds_ev_poisson': ['avg_proj_rebounds'],
       'player_rebounds_ev_trunc_norm': ['avg_proj_rebounds'], 
       'player_steals_ev_poisson': ['avg_proj_steals'],
       'player_steals_ev_trunc_norm': ['avg_proj_steals'], 
       'player_blocks_ev_poisson': ['avg_proj_blocks'],
       'player_blocks_ev_trunc_norm': ['avg_proj_blocks'], 
       'player_points_rebounds_ev_poisson': ['avg_proj_points_rebounds'],
       'player_points_rebounds_ev_trunc_norm': ['avg_proj_points_rebounds'],
       'player_points_assists_ev_poisson': ['avg_proj_points_assists'],
       'player_points_assists_ev_trunc_norm': ['avg_proj_points_assists'],
       'player_points_rebounds_assists_ev_poisson': ['avg_proj_points_rebounds_assists'],
       'player_points_rebounds_assists_ev_trunc_norm': ['avg_proj_points_rebounds_assists'],
       'player_threes_ev_poisson': ['avg_proj_three_pointers'], 
       'player_threes_ev_trunc_norm': ['avg_proj_three_pointers'],
       'player_blocks_steals_ev_poisson': ['avg_proj_steals_blocks'],
       'player_blocks_steals_ev_trunc_norm': ['avg_proj_steals_blocks']
    }

    cols = [c for c in player_vegas_stats.columns if 'ev' in c]
    for c in cols:
        ratio = (df.loc[~df[c].isnull(), c] / (df.loc[~df[c].isnull(), fill_cols[c]].sum(axis=1)+0.1)).median()
        df.loc[df[c].isnull(), c] = ratio * df.loc[df[c].isnull(), fill_cols[c]].sum(axis=1)
        df['avg_'+c] = df[[c, fill_cols[c][0]]].mean(axis=1)
    
    return df


def pull_odds(metric, game_dates):

    odds = dm.read(f'''SELECT player, game_date, value
                       FROM Draftkings_Odds 
                       WHERE stat_type='{metric}'
                             AND over_under='over'
                             AND decimal_odds < 2.5
                             AND decimal_odds > 1.5
                             AND game_date >= '{game_dates[-1]}'
                    ''', 'Player_Stats')

    return odds

def create_value_columns(df, metric):
    """Vectorized version of value column creation"""
    # Convert value column to float once
    try:
        df['value'] = df['value'].astype(float)
    except:
        pass
    
    # Get all relevant columns in one go
    metric_cols = [c for c in df.columns if metric in c and 'y_act' not in c]
    
    # Convert all relevant columns to float at once
    try:
        df[metric_cols] = df[metric_cols].astype(float)
        
        # Create all new columns at once using DataFrame operations
        vs_value_df = df[metric_cols].subtract(df['value'], axis=0)
        over_value_df = df[metric_cols].div(df['value'], axis=0)
        
        # Rename the new columns
        vs_value_df.columns = [f"{col}_vs_value" for col in vs_value_df.columns]
        over_value_df.columns = [f"{col}_over_value" for col in over_value_df.columns]
        
        # Concatenate all at once
        df = pd.concat([df, vs_value_df, over_value_df], axis=1)
    except:
        pass
    
    return df

def remove_low_counts(df):
    cnts = df.groupby('game_date').agg({'player': 'count'})
    cnts = cnts[cnts.player > 5].reset_index().drop('player', axis=1)
    df = pd.merge(df, cnts, on='game_date')
    return df

def get_over_under_class(df, metric, game_dates):
    
    odds = pull_odds(metric, game_dates)
    df = pd.merge(df, odds, on=['player', 'game_date'])
    df = df.sort_values(by='game_date').reset_index(drop=True)

    df[f'y_act_{metric}_over'] = np.where(df[f'y_act_{metric}'] >= df.value, 1, 0)
    df = remove_low_counts(df)

    return df


def filter_and_sort_features(feature_cols, importance_scores, n_features=500):
 
    # Sort features by importance and filter out value columns
    sorted_features = [feat for feat, _ in sorted(zip(feature_cols, importance_scores), 
                                                key=lambda x: x[1], 
                                                reverse=True)
                      if 'vs_value' not in feat and 'over_value' not in feat]
    return sorted_features[:n_features]

def select_features_for_category(df, target_col, n_features=500, is_classification=False):
    """
    Select features using multiple methods and return top N from each
    """
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, SelectKBest, f_classif, f_regression
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier, XGBRegressor
    
    # Get all numeric feature columns for training
    feature_cols = df.select_dtypes(include=['float64', 'int64']).columns
    feature_cols = [c for c in feature_cols if 'y_act' not in c]
    X = df[feature_cols]
    y = df[target_col]
    
    # Standardize features
    X_scaled = StandardScaler().fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    
    selected_features = []
    
    # MI Selection
    print('MI Selection')
    if is_classification:
        mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
    else:
        mi_scores = mutual_info_regression(X_scaled, y, random_state=42)
    selected_features.extend(filter_and_sort_features(feature_cols, mi_scores, n_features))
    
    # KBest Selection
    print('KBest Selection')
    if is_classification:
        kbest = SelectKBest(score_func=f_classif, k=len(feature_cols))
    else:
        kbest = SelectKBest(score_func=f_regression, k=len(feature_cols))
    kbest.fit(X_scaled, y)
    selected_features.extend(filter_and_sort_features(feature_cols, kbest.scores_, n_features))
    
    # LightGBM Selection
    print('LightGBM Selection')
    if is_classification:
        model = LGBMClassifier(n_jobs=-1, random_state=42)
    else:
        model = LGBMRegressor(n_jobs=-1, random_state=42)
    model.fit(X, y)
    selected_features.extend(filter_and_sort_features(feature_cols, model.feature_importances_, n_features))
    
    # XGBoost Selection
    print('XGBoost Selection')
    if is_classification:
        xgb_model = XGBClassifier(n_jobs=-1, random_state=42)
    else:
        xgb_model = XGBRegressor(n_jobs=-1, random_state=42)
    xgb_model.fit(X, y)
    selected_features.extend(filter_and_sort_features(feature_cols, xgb_model.feature_importances_, n_features))
    
    return selected_features


def get_reg_class_features(df, game_dates):
    all_reg_features = []
    all_class_features = []
    for metric in ['points', 'rebounds', 'assists', 'three_pointers']:
        df_class = get_over_under_class(df.copy(), metric, game_dates)
        df_class = create_value_columns(df_class, metric)
        df_class = df_class.dropna(axis=0).reset_index(drop=True)

        # Regression features
        print(f'\n{metric} regression features')
        reg_features = select_features_for_category(
                    df.dropna(axis=0).reset_index(drop=True), 
                    f'y_act_{metric}', 
                    n_features=1000,
                    is_classification=False
                )
        
        print(f'\n{metric} classification features')
        class_features = select_features_for_category(
                    df_class, 
                    f'y_act_{metric}_over', 
                    n_features=1000,
                    is_classification=True
                )
        all_reg_features.extend(reg_features)
        all_class_features.extend(class_features)

    return all_reg_features, all_class_features

def reconcile_features(all_features, min_count=5):
    """
    Reconcile features based on how many times they were selected
    """
    all_features = pd.Series(all_features).value_counts().sort_values(ascending=False)
    
    for i in range(1,11):
        print('Num features for top', i, 'features:', all_features[all_features >= i].shape[0])

    all_features = all_features[all_features >= min_count]
    print(all_features.head(50))
    final_features = all_features.index.tolist()
    
    return final_features

def get_final_features(all_reg_features, all_class_features, class_min=6, reg_min=7, remaining_min=8):

    final_class_features = reconcile_features(
        all_class_features,
        min_count=class_min
    )

    final_reg_features = reconcile_features(
        all_reg_features,
        min_count=7
    )

    print('class features:', len(final_class_features))
    print('reg features:', len(final_reg_features))
    final_features = final_class_features + final_reg_features
    remain_features = [c for c in all_class_features+all_reg_features if c not in final_features]

    final_remaining_features = reconcile_features(
        remain_features,
        min_count=8)

    final_features = final_features + final_remaining_features
    final_features = list(set(final_features))

    print('final features:', len(final_features))

    other_cols = ['player', 'game_date', 'team', 'opponent']
    other_cols.extend([c for c in df.columns if 'y_act' in c])
    other_cols.extend(final_features)

    return other_cols

#%%

train_date = '2025-10-23'
max_date = dm.read("SELECT max(game_date) FROM FantasyData", 'Player_Stats').values[0][0]

df = fantasy_data()

days_since_training = (dt.datetime.now() - dt.datetime.strptime(train_date, '%Y-%m-%d')).days
game_dates = df.game_date.sort_values(ascending=False).unique()
game_dates = game_dates[:90+days_since_training]
df = df[df.game_date.isin(game_dates)].reset_index(drop=True)

df = fantasy_pros(df)
df = numberfire(df)
df = sportsline_data(df)
df = fix_fp_returns(df)

missing = df.loc[(df.fd_points==0) | ((df.fp_points==0) & (df.nf_points==0)), 
                 ['player', 'game_date']].copy().reset_index(drop=True)

df = consensus_fill(df)
df = forward_fill(df)
df = df.dropna().reset_index(drop=True)

vegas_stats = get_all_vegas_stats()
df = fill_vegas_stats(df, vegas_stats)

df = df[(df.fd_points>0) & (df.fp_points>0) & (df.nf_points>0)].reset_index(drop=True)
df = add_proj_market_share(df)

proj_cols = [c for c in df.columns if 'fd' in c or 'fp' in c or 'nf' in c or 'proj' in c or '_ev_' in c]
df = rolling_proj_stats(df, proj_cols)
print(df.shape)

# get box score stats androll the box score stats and shift back a game
box_score = get_box_score()
df = remove_no_minutes(df, box_score)
df = add_last_game_box_score(df, box_score)
df = box_score_rolling(df, box_score)
df = remove_low_minutes(df, box_score)

df = available_stats(df, missing)
print('available stats:', df.shape)

df, adv_stats = add_advanced_stats(df)
df = add_last_game_advanced_stats(df, adv_stats)

df = add_tracking_stats(df)
df = add_hustle_stats(df)
print('after adv stats:', df.shape)

# add team stats
df = add_team_box_score(df, 'team')
df = add_team_box_score(df, 'opponent')
df = add_team_advanced_stats(df, 'team')
df = add_team_advanced_stats(df, 'opponent')
df = add_team_tracking(df, 'team')
df = add_team_tracking(df, 'opponent')

df = add_team_hustle_stats(df, 'team')
df = add_team_hustle_stats(df, 'opponent')

print('after team stats:', df.shape)

df = add_dk_team_lines(df)
df = forward_fill(df)
print('after team lines', df.shape)

df = add_y_act(df, box_score)
df = df.dropna(axis=0, subset=[c for c in df.columns if 'y_act' not in c]).reset_index(drop=True)
train_date = train_date.replace('-', '')

print(df.shape)
print(df.game_date.max())

final_cols = get_columns(train_date)
if final_cols is None:
    all_reg_features, all_class_features = get_reg_class_features(df, game_dates)
    final_cols = get_final_features(all_reg_features, all_class_features, class_min=6, reg_min=7, remaining_min=8)

print('Number of features:', len(final_cols))


#%%

df = df[final_cols]
print(df.shape)

dm.write_to_db(df.iloc[:,:2000], 'Model_Features', f'Model_Data_{train_date}', if_exist='replace')
if df.shape[1] > 2000:
    dm.write_to_db(df.iloc[:,2000:], 'Model_Features', f'Model_Data_{train_date}v2', if_exist='replace')


#%%




# %%

def team_proj(df):
    team_stats = df.sort_values(by=['team', 'game_date', 'fd_points'],
                            ascending=[True, True, False]).copy().reset_index(drop=True)
    team_stats = team_stats[team_stats.fd_points > 0].reset_index(drop=True)

    team_stats['team_rank'] = team_stats.groupby(['team', 'game_date']).cumcount()
    team_stats = team_stats[team_stats.team_rank <= 9].reset_index(drop=True)

    obj_cols = team_stats.dtypes[team_stats.dtypes == object].index.tolist()
    agg_cols = {c: 'mean' if 'pct' in c else 'sum' for c in team_stats if c not in obj_cols}
    agg_cols
    team_stats = team_stats.groupby(['team', 'opponent', 'game_date']).agg(agg_cols).reset_index()

    return team_stats

def team_rolling_proj_stats(team_df):
    opp_df = team_df.copy()
    opp_df.columns = ['opp_' + c if c not in ['team', 'opponent', 'game_date'] else c for c in opp_df.columns]

    team_df = rolling_proj_stats(team_df.rename(columns={'team': 'player'}))
    opp_df = rolling_proj_stats(opp_df.rename(columns={'team': 'player'}))

    team_df = team_df.rename(columns={'player': 'team'})
    opp_df = opp_df.drop('opponent', axis=1).rename(columns={'player': 'opponent'})
    team_df = pd.merge(team_df, opp_df, on=['opponent', 'game_date'])
    return team_df


def add_team_y_act(team_df):

    team_bs = dm.read("SELECT * FROM Box_Score", 'Team_Stats')
    team_bs = team_bs[['TEAM_ABBREVIATION', 'game_date', 'PTS']]
    team_bs.columns = ['team', 'game_date', 'y_act_team_pts']

    team_df = pd.merge(team_df, team_bs, on=['team', 'game_date'])

    team_bs.columns = ['opponent', 'game_date', 'y_act_opponent_pts']
    team_df = pd.merge(team_df, team_bs, on=['opponent', 'game_date'])
        
    team_df['y_act_total_points'] = team_df.y_act_team_pts + team_df.y_act_opponent_pts
    team_df['y_act_spread'] =  team_df.y_act_opponent_pts - team_df.y_act_team_pts

    team_df['y_act_over_total_points'] = np.where(team_df.y_act_total_points > team_df.over, 1, 0)
    team_df['y_act_over_spread'] = np.where( team_df.spread > team_df.y_act_spread, 1, 0)


    return team_df


df = fantasy_data()
df = fantasy_pros(df)
df = numberfire(df)
df = fix_fp_returns(df)

missing = df.loc[(df.fd_points==0) | ((df.fp_points==0) & (df.nf_points==0)), 
                 ['player', 'game_date']].copy().reset_index(drop=True)

df = consensus_fill(df)
df = forward_fill(df)
df = df.dropna().reset_index(drop=True)

df = df[(df.fd_points>0) & (df.fp_points>0) & (df.nf_points>0)].reset_index(drop=True)
df = add_proj_market_share(df)

# get box score stats androll the box score stats and shift back a game
box_score = get_box_score()
df = remove_no_minutes(df, box_score)
df = add_last_game_box_score(df, box_score)
df = box_score_rolling(df, box_score)
df = remove_low_minutes(df, box_score)

df = available_stats(df, missing)
print('available stats:', df.shape)

team_df = team_proj(df)
team_df = add_team_box_score(team_df, 'team')
team_df = add_team_box_score(team_df, 'opponent')
team_df = add_team_advanced_stats(team_df, 'team')
team_df = add_team_advanced_stats(team_df, 'opponent')
team_df = add_team_tracking(team_df, 'team')
team_df = add_team_tracking(team_df, 'opponent')
team_df = add_dk_team_lines(team_df)

team_df = add_team_y_act(team_df)
team_df = remove_low_corrs(team_df, threshold=0.05)


dm.write_to_db(team_df.iloc[:,:2000], 'Model_Features', f'Team_Model_Data_{train_date}', if_exist='replace')
if df.shape[1] > 2000:
    dm.write_to_db(team_df.iloc[:,2000:], 'Model_Features', f'Team_Model_Data_{train_date}v2', if_exist='replace')

df.groupby(['player', 'game_date']).size().sort_values(ascending=False)

# %%

team_df[['team', 'opponent', 'game_date', 'y_act_team_pts', 'y_act_opponent_pts', 'y_act_spread', 'spread', 'y_act_over_spread']]

#%%

df = dm.read("SELECT * FROM Model_Data_20240118v2", 'Model_Features')
df.corr()['y_act_steals'].sort_values()
# %%
df.corr()['y_act_assists'].sort_values()

# %%
