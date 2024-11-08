#%%

import pandas as pd 
import pyarrow.parquet as pq
import numpy as np
from ff.db_operations import DataManage
from ff import general as ffgeneral
import ff.data_clean as dc
from scipy.stats.morestats import shapiro
import datetime as dt

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


def fantasy_pros(df):

    fp = dm.read(f'''SELECT * 
                    FROM FantasyPros 
                    ''', 'Player_Stats')

    fp['is_home'] = np.where(fp.opponent.apply(lambda x: x.split(' ')[0])=='vs', 1, 0)     
    fp = fp.drop(['position', 'team', 'games_played', 'opponent'], axis=1)
    fp.columns = ['fp_' + c if c not in ('player', 'team', 'opponent', 'game_date', 'is_home') else c for c in fp.columns ]
    
    df = pd.merge(df, fp, on=['player', 'game_date'], how='outer')
    df = df.fillna({'is_home': 0.5})

    return df

def numberfire(df):
    
    nf = dm.read(f"SELECT * FROM NumberFire_Projections", 'Player_Stats')
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
        'proj_points': ['fp_points', 'nf_points', 'fd_points'],
        'proj_rebounds': ['fp_rebounds', 'nf_rebounds', 'fd_rebounds'],
        'proj_assists': ['fp_assists', 'nf_assists', 'fd_assists'],
        'proj_blocks': ['fp_blocks', 'nf_blocks', 'fd_blocks'],
        'proj_steals': ['fp_steals', 'nf_steals', 'fd_steals'],
        'proj_turnovers': ['fp_turnovers', 'nf_turnovers', 'fd_turnovers'],
        'proj_three_pointers': ['fp_three_pointers', 'nf_three_pointers', 'fd_three_pointers'],
        'proj_fg_pct': ['fp_fg_pct', 'fd_fg_pct'],
        'proj_ft_pct': ['fp_ft_pct', 'fd_ft_pct'],
        'proj_minutes': ['fp_minutes', 'nf_minutes', 'fd_minutes'],
        'proj_fantasy_points': ['fd_fantasy_points', 'nf_fantasy_points']
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



def rolling_proj_stats(df):
    df = forward_fill(df)
    proj_cols = [c for c in df.columns if 'fd' in c or 'fp' in c or 'nf' in c or 'proj' in c]
    df = add_rolling_stats(df, ['player'], proj_cols)

    for c in proj_cols:
        df[f'trend_diff_{c}3v10'] = df[f'rmean3_{c}'] - df[f'rmean10_{c}']
        df[f'trend_chg_{c}3v10'] = df[f'trend_diff_{c}3v10'] / (df[f'rmean10_{c}']+5)

        df[f'trend_diff_{c}6v10'] = df[f'rmean3_{c}'] - df[f'rmean10_{c}']
        df[f'trend_chg_{c}6v10'] = df[f'trend_diff_{c}6v10'] / (df[f'rmean10_{c}']+5)

    return df



# def calc_market_share(df):
    
#     player_cols = ['rush_rush_attempt_sum', 'rec_xyac_mean_yardage_sum', 'rush_yards_gained_sum',
#                    'rush_ep_sum', 'rush_ydstogo_sum', 'rush_touchdown_sum',
#                    'rush_yardline_100_sum', 'rush_first_down_sum', 'rec_yards_gained_sum', 
#                    'rec_first_down_sum', 'rec_epa_sum', 'rec_qb_epa_sum', 'rec_cp_sum','rec_xyac_epa_sum', 
#                    'rec_pass_attempt_sum', 'rec_xyac_success_sum', 'rec_comp_air_epa_sum',
#                    'rec_air_yards_sum', 'rec_touchdown_sum', 'rec_xyac_median_yardage_sum',
#                    'rec_complete_pass_sum', 'rec_qb_dropback_sum']

#     team_cols = ['team_' + c for c in player_cols]

#     for p, t in zip(player_cols, team_cols):
#         df[p+'_share'] = df[p] / (df[t]+0.5)

#     share_cols = [c+'_share' for c in player_cols]
#     df[share_cols] = df[share_cols].fillna(0)
#     df = add_rolling_stats(df, gcols=['player'], rcols=share_cols)

#     df = forward_fill(df)
#     share_cols = [c for c in df.columns if 'share' in c]
#     df[share_cols] = df[share_cols].fillna(0)

#     return df


# def add_player_comparison(df, cols):
    
#     to_agg = {c: [np.mean, np.max, np.min] for c in cols}
#     team_stats = df.groupby(['team', 'week', 'year']).agg(to_agg)

#     diff_df = df[['player', 'team', 'week', 'year']].drop_duplicates()
#     for c in cols:
#         tmp_df = team_stats[c].reset_index()
#         tmp_df = pd.merge(tmp_df, df[['player', 'team', 'week', 'year', c]], on=['team', 'week', 'year'])

#         for a in ['mean', 'amin', 'amax']:
#             tmp_df[f'{c}_{a}_diff'] = tmp_df[c] - tmp_df[a]
    
#         tmp_df = tmp_df[['player', 'team', 'week', 'year', f'{c}_mean_diff', f'{c}_amax_diff', f'{c}_amin_diff']]
#         diff_df = pd.merge(diff_df, tmp_df, on=['player', 'team', 'week', 'year'])
        
#     diff_df = diff_df.drop_duplicates()
#     team_stats.columns = [f'{c[0]}_{c[1]}' for c in team_stats.columns]
#     team_stats = team_stats.reset_index().drop_duplicates()

#     df = pd.merge(df, team_stats, on=['team', 'week', 'year'])
#     df = pd.merge(df, diff_df, on=['player', 'team', 'week', 'year'])

#     return df


# def pos_rank_stats(df, team_pos_rank, pos):
    
#     pos_stats = dm.read(f'''SELECT * 
#                             FROM {pos}_Stats 
#                             WHERE (season = 2020 AND week != 17)
#                                     OR (season >=2021 AND week != 18)
#                             ''', 'FastR')
#     pos_stats = pos_stats.rename(columns={'season': 'year'})

#     pos_rank_cols = ['rush_rush_attempt_sum', 'rec_xyac_mean_yardage_sum', 'rush_yards_gained_sum',
#                     'rush_ep_sum', 'rush_ydstogo_sum', 'rush_touchdown_sum',
#                     'rush_yardline_100_sum', 'rush_first_down_sum', 'rec_yards_gained_sum', 
#                     'rec_first_down_sum', 'rec_epa_sum', 'rec_qb_epa_sum', 'rec_cp_sum','rec_xyac_epa_sum', 
#                     'rec_pass_attempt_sum', 'rec_xyac_success_sum', 'rec_comp_air_epa_sum',
#                     'rec_air_yards_sum', 'rec_touchdown_sum', 'rec_xyac_median_yardage_sum',
#                     'rec_complete_pass_sum', 'rec_qb_dropback_sum'
#     ]
#     agg_cols = {c: 'sum' for c in pos_rank_cols}

#     pos_stats = pd.merge(team_pos_rank, pos_stats, on=['player', 'team', 'week', 'year'], how='left')

#     pos_stats = pos_stats.groupby(['pos_rank', 'team', 'week','year']).agg(agg_cols)
#     pos_stats.columns = ['pos_rank_' + c for c in pos_stats.columns]
#     pos_stats = pos_stats.reset_index()

#     gcols = ['team', 'pos_rank']
#     rcols=['pos_rank_' + c for c in pos_rank_cols]
#     pos_stats = pos_stats.sort_values(by=['team', 'pos_rank', 'year', 'week']).reset_index(drop=True)

#     rolls3_mean = rolling_stats(pos_stats, gcols, rcols, 3, agg_type='mean')
#     rolls3_max = rolling_stats(pos_stats, gcols, rcols, 3, agg_type='max')

#     rolls8_mean = rolling_stats(pos_stats, gcols, rcols, 8, agg_type='mean')
#     rolls8_max = rolling_stats(pos_stats, gcols, rcols, 8, agg_type='max')

#     pos_stats = pd.concat([pos_stats, rolls8_mean, rolls8_max, rolls3_mean, rolls3_max], axis=1)

#     pos_stats = pd.merge(team_pos_rank, pos_stats, on=['pos_rank', 'team', 'week', 'year'])
#     pos_stats = pos_stats.drop(['pos_rank_' + c for c in pos_rank_cols], axis=1)

#     pos_stats['week'] = pos_stats['week'] + 1
#     pos_stats = switch_seasons(pos_stats)
#     pos_stats = fix_bye_week(pos_stats)

#     df = pd.merge(df, pos_stats, on=['player', 'team', 'pos', 'week', 'year'], how='left')

#     return df

# def create_pos_rank(df, extra_pos=False):
#     df = df.sort_values(by=['team', 'pos', 'year', 'week', 'avg_proj_points'],
#                         ascending=[True, True, True, True, False]).reset_index(drop=True)

#     df['pos_rank'] = df.pos + df.groupby(['team', 'pos', 'year', 'week']).cumcount().apply(lambda x: str(x))
#     if extra_pos:
#         df = df[df['pos_rank'].isin(['RB0', 'RB1', 'RB2', 'WR0', 'WR1', 'WR2', 'WR3', 'WR4', 'TE0', 'TE1'])].reset_index(drop=True)
#     else:
#         df = df[df['pos_rank'].isin(['QB0', 'RB0', 'RB1', 'WR0', 'WR1', 'WR2', 'TE0'])].reset_index(drop=True)
#     return df


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


# def results_vs_predicted(df, col):

#     df = df.sort_values(by=['player','year', 'week']).reset_index(drop=True)
#     df[f'{col}_miss'] = df.y_act - df[col]
#     df[f'{col}_miss'] = (df.groupby('player')[f'{col}_miss'].shift(1)).fillna(0)
#     df = add_rolling_stats(df, ['player'], [f'{col}_miss'], perform_check=False)
#     df[f'{col}_miss_recent_vs8'] = df[f'rmean3_{col}_miss'] - df[f'rmean8_{col}_miss']

#     good_cols = [c for c in df.columns if 'miss' in c or c in ('player', 'team', 'week', 'year')]
#     df = df[good_cols]

#     return df

# def projected_pts_vs_predicted(df):

#     proj_pts_miss = df[['player', 'team', 'week', 'year']].copy()
#     for c in ['ffa_points', 'projected_points', 'fantasyPoints', 'ProjPts', 
#               'fc_proj_fantasy_pts_fc', 'fft_proj_pts', 'avg_proj_points']:
#         if c in df.columns:
#             cur_miss = results_vs_predicted(df[['player', 'team', 'week', 'year', 'y_act', c]].copy(), c)
#             proj_pts_miss = pd.merge(proj_pts_miss, cur_miss, on=['player', 'team', 'week', 'year']) 

#     df = pd.merge(df, proj_pts_miss, on=['player', 'team', 'week', 'year'], how='left')
#     df = forward_fill(df)

#     miss_cols = [c for c in df.columns if 'miss' in c]
#     df[miss_cols] = df[miss_cols].fillna(0)

#     return df

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

    y_act = bs[['player', 'game_date', 'points', 'rebounds', 'assists', 'three_pointers', 'steals', 'blocks']].copy()
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

    track = track.drop(['GAME_ID', 'TEAM_ABBREVIATION', 'MIN', 'TEAM_ID', 'TEAM_CITY', 
                    'PLAYER_ID', 'START_POSITION', 'COMMENT'], axis=1)
    track = track.sort_values(by=['player', 'game_date']).reset_index(drop=True)

    r_cols = [c for c in track.columns if c not in ('player', 'game_date')]
    track = add_rolling_stats(track, gcols=['player'], rcols=r_cols)

    track = track.drop(['SPD', 'DIST', 'ORBC', 'DRBC', 'RBC', 'TCHS', 'SAST', 'FTAST',
                        'PASS', 'AST', 'CFGM', 'CFGA', 'CFG_PCT', 'UFGM', 'UFGA', 'UFG_PCT',
                        'FG_PCT', 'DFGM', 'DFGA', 'DFG_PCT'
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

    hustle = hustle.drop(['GAME_ID', 'TEAM_ABBREVIATION', 'TEAM_ID', 'TEAM_CITY', 
                        'PLAYER_ID', 'START_POSITION', 'COMMENT', 'MINUTES', 'PTS'], axis=1)
    hustle = hustle.sort_values(by=['player', 'game_date']).reset_index(drop=True)

    r_cols = [c for c in hustle.columns if c not in ('player', 'game_date')]
    hustle_roll = add_rolling_stats(hustle, gcols=['player'], rcols=r_cols)

    hustle_roll = hustle_roll.drop([ 'CONTESTED_SHOTS', 'CONTESTED_SHOTS_2PT',
                                    'CONTESTED_SHOTS_3PT', 'DEFLECTIONS', 'CHARGES_DRAWN', 'SCREEN_ASSISTS',
                                    'SCREEN_AST_PTS', 'OFF_LOOSE_BALLS_RECOVERED', 'BOX_OUTS',
                                    'DEF_LOOSE_BALLS_RECOVERED', 'LOOSE_BALLS_RECOVERED', 'OFF_BOXOUTS',
                                    'DEF_BOXOUTS', 'BOX_OUT_PLAYER_TEAM_REBS', 'BOX_OUT_PLAYER_REBS'
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



def remove_low_corrs(df, threshold):

    orig_cols = df.shape[1]
    obj_cols = list(df.dtypes[df.dtypes=='object'].index)
    corrs = pd.DataFrame(np.corrcoef(df.dropna().drop(obj_cols, axis=1).values, rowvar=False), 
                            columns=[c for c in df.columns if c not in obj_cols],
                            index=[c for c in df.columns if c not in obj_cols])
    
    corrs = corrs[[y for y in corrs.columns if 'y_act' in y]]
    corrs = corrs[~corrs.index.str.contains('y_act')]
    good_corrs = list(corrs[abs(corrs) >= threshold].dropna(how='all').index)

    obj_cols.extend(good_corrs)
    obj_cols.extend(corrs.columns)
    print(f'Kept {len(obj_cols)}/{orig_cols} columns')

    return df[obj_cols]


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


def get_columns(df, train_date, threshold=0.05):
    try:
        cols = dm.read(f"SELECT * FROM Model_Data_{train_date}", 'Model_Features').columns.tolist()
        cols.extend(dm.read(f"SELECT * FROM Model_Data_{train_date}v2", 'Model_Features').columns.tolist())
        df = df[cols]
    except:
        df = remove_low_corrs(df, threshold=threshold)

    return df

#%%

train_date = '2024-10-22'
max_date = dm.read("SELECT max(game_date) FROM FantasyData", 'Player_Stats').values[0][0]

df = fantasy_data()

days_since_training = (dt.datetime.now() - dt.datetime.strptime(train_date, '%Y-%m-%d')).days
game_dates = df.game_date.sort_values(ascending=False).unique()
df = df[df.game_date.isin(game_dates[:100+days_since_training])].reset_index(drop=True)

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
df = rolling_proj_stats(df)
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
print(df.shape)

train_date = train_date.replace('-', '')
df = get_columns(df, train_date, threshold=0.05)
print(df.game_date.max())
print(df.shape)

#%%

dm.write_to_db(df.iloc[:,:2000], 'Model_Features', f'Model_Data_{train_date}', if_exist='replace')
if df.shape[1] > 2000:
    dm.write_to_db(df.iloc[:,2000:], 'Model_Features', f'Model_Data_{train_date}v2', if_exist='replace')

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
# %%

team_df[['team', 'opponent', 'game_date', 'y_act_team_pts', 'y_act_opponent_pts', 'y_act_spread', 'spread', 'y_act_over_spread']]

#%%

df = dm.read("SELECT * FROM Model_Data_20240118v2", 'Model_Features')
df.corr()['y_act_steals'].sort_values()
# %%
df.corr()['y_act_assists'].sort_values()

# %%
