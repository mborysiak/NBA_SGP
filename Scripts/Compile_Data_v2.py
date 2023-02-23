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

    std_6 = rolling_stats(df, gcols, rcols, 6, agg_type='std')
    std_10 = rolling_stats(df, gcols, rcols, 10, agg_type='std')
    df = pd.concat([df, std_6, std_10], axis=1)

    return df


def forward_fill(df, cols=None):
    
    if cols is None: cols = df.columns
    df = df.sort_values(by=['player', 'game_date'])
    df = df.groupby('player', as_index=False)[cols].fillna(method='ffill')
    df = df.sort_values(by=['player', 'game_date'])

    return df


def fantasy_data():

    fd = dm.read(f'''SELECT * 
                    FROM FantasyData 
                    ''', 'Player_Stats')
    fd = fd.drop(['rank'], axis=1)
    fd.columns = ['fd_' + c if c not in ('player', 'team', 'position', 'opponent', 'game_date') else c for c in fd.columns ]

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



def rolling_proj_stats(df):
    df = forward_fill(df)
    proj_cols = [c for c in df.columns if 'fd' in c or 'fp' in c or 'nf' in c or 'proj' in c]
    df = add_rolling_stats(df, ['player'], proj_cols)

    for c in proj_cols:
        df[f'trend_diff_{c}'] = df[f'rmean3_{c}'] - df[f'rmean10_{c}']
        df[f'trend_chg_{c}'] = df[f'trend_diff_{c}'] / (df[f'rmean10_{c}']+5)

    return df



def calc_market_share(df):
    
    player_cols = ['rush_rush_attempt_sum', 'rec_xyac_mean_yardage_sum', 'rush_yards_gained_sum',
                   'rush_ep_sum', 'rush_ydstogo_sum', 'rush_touchdown_sum',
                   'rush_yardline_100_sum', 'rush_first_down_sum', 'rec_yards_gained_sum', 
                   'rec_first_down_sum', 'rec_epa_sum', 'rec_qb_epa_sum', 'rec_cp_sum','rec_xyac_epa_sum', 
                   'rec_pass_attempt_sum', 'rec_xyac_success_sum', 'rec_comp_air_epa_sum',
                   'rec_air_yards_sum', 'rec_touchdown_sum', 'rec_xyac_median_yardage_sum',
                   'rec_complete_pass_sum', 'rec_qb_dropback_sum']

    team_cols = ['team_' + c for c in player_cols]

    for p, t in zip(player_cols, team_cols):
        df[p+'_share'] = df[p] / (df[t]+0.5)

    share_cols = [c+'_share' for c in player_cols]
    df[share_cols] = df[share_cols].fillna(0)
    df = add_rolling_stats(df, gcols=['player'], rcols=share_cols)

    df = forward_fill(df)
    share_cols = [c for c in df.columns if 'share' in c]
    df[share_cols] = df[share_cols].fillna(0)

    return df


def add_player_comparison(df, cols):
    
    to_agg = {c: [np.mean, np.max, np.min] for c in cols}
    team_stats = df.groupby(['team', 'week', 'year']).agg(to_agg)

    diff_df = df[['player', 'team', 'week', 'year']].drop_duplicates()
    for c in cols:
        tmp_df = team_stats[c].reset_index()
        tmp_df = pd.merge(tmp_df, df[['player', 'team', 'week', 'year', c]], on=['team', 'week', 'year'])

        for a in ['mean', 'amin', 'amax']:
            tmp_df[f'{c}_{a}_diff'] = tmp_df[c] - tmp_df[a]
    
        tmp_df = tmp_df[['player', 'team', 'week', 'year', f'{c}_mean_diff', f'{c}_amax_diff', f'{c}_amin_diff']]
        diff_df = pd.merge(diff_df, tmp_df, on=['player', 'team', 'week', 'year'])
        
    diff_df = diff_df.drop_duplicates()
    team_stats.columns = [f'{c[0]}_{c[1]}' for c in team_stats.columns]
    team_stats = team_stats.reset_index().drop_duplicates()

    df = pd.merge(df, team_stats, on=['team', 'week', 'year'])
    df = pd.merge(df, diff_df, on=['player', 'team', 'week', 'year'])

    return df


def pos_rank_stats(df, team_pos_rank, pos):
    
    pos_stats = dm.read(f'''SELECT * 
                            FROM {pos}_Stats 
                            WHERE (season = 2020 AND week != 17)
                                    OR (season >=2021 AND week != 18)
                            ''', 'FastR')
    pos_stats = pos_stats.rename(columns={'season': 'year'})

    pos_rank_cols = ['rush_rush_attempt_sum', 'rec_xyac_mean_yardage_sum', 'rush_yards_gained_sum',
                    'rush_ep_sum', 'rush_ydstogo_sum', 'rush_touchdown_sum',
                    'rush_yardline_100_sum', 'rush_first_down_sum', 'rec_yards_gained_sum', 
                    'rec_first_down_sum', 'rec_epa_sum', 'rec_qb_epa_sum', 'rec_cp_sum','rec_xyac_epa_sum', 
                    'rec_pass_attempt_sum', 'rec_xyac_success_sum', 'rec_comp_air_epa_sum',
                    'rec_air_yards_sum', 'rec_touchdown_sum', 'rec_xyac_median_yardage_sum',
                    'rec_complete_pass_sum', 'rec_qb_dropback_sum'
    ]
    agg_cols = {c: 'sum' for c in pos_rank_cols}

    pos_stats = pd.merge(team_pos_rank, pos_stats, on=['player', 'team', 'week', 'year'], how='left')

    pos_stats = pos_stats.groupby(['pos_rank', 'team', 'week','year']).agg(agg_cols)
    pos_stats.columns = ['pos_rank_' + c for c in pos_stats.columns]
    pos_stats = pos_stats.reset_index()

    gcols = ['team', 'pos_rank']
    rcols=['pos_rank_' + c for c in pos_rank_cols]
    pos_stats = pos_stats.sort_values(by=['team', 'pos_rank', 'year', 'week']).reset_index(drop=True)

    rolls3_mean = rolling_stats(pos_stats, gcols, rcols, 3, agg_type='mean')
    rolls3_max = rolling_stats(pos_stats, gcols, rcols, 3, agg_type='max')

    rolls8_mean = rolling_stats(pos_stats, gcols, rcols, 8, agg_type='mean')
    rolls8_max = rolling_stats(pos_stats, gcols, rcols, 8, agg_type='max')

    pos_stats = pd.concat([pos_stats, rolls8_mean, rolls8_max, rolls3_mean, rolls3_max], axis=1)

    pos_stats = pd.merge(team_pos_rank, pos_stats, on=['pos_rank', 'team', 'week', 'year'])
    pos_stats = pos_stats.drop(['pos_rank_' + c for c in pos_rank_cols], axis=1)

    pos_stats['week'] = pos_stats['week'] + 1
    pos_stats = switch_seasons(pos_stats)
    pos_stats = fix_bye_week(pos_stats)

    df = pd.merge(df, pos_stats, on=['player', 'team', 'pos', 'week', 'year'], how='left')

    return df
    



def get_team_stats():
    
    team_stats = dm.read(f'''SELECT * 
                             FROM Team_Stats 
                             WHERE (season = 2020 AND week != 17)
                                    OR (season >=2021 AND week != 18)''', 'FastR')

    rcols_team = [c for c in team_stats.columns if 'rush_' in c or 'rec_' in c]

    team_stats = team_stats.rename(columns={'season': 'year'})
    team_stats = add_rolling_stats(team_stats, ['team'], rcols_team)

    team_stats = team_stats[team_stats.year >= 2020].reset_index(drop=True)
    team_stats['week'] = team_stats.week + 1
    team_stats = switch_seasons(team_stats)
    team_stats = fix_bye_week(team_stats)

    return team_stats


def format_lines(lines, is_home):
    if is_home==1: label = 'home'
    else: label = 'away'

    lines = lines[[f'{label}_team', f'{label}_line', 'over_under', 'week', 'year']]
    lines = lines.assign(is_home=is_home)
    lines.columns = ['team', 'line', 'over_under', 'week', 'year', 'is_home']
    
    lines = dc.convert_to_float(lines)
    lines['implied_points_for'] = (lines.over_under / 2) - (lines.line / 2) 
    lines['implied_points_against'] = (lines.over_under / 2) + (lines.line / 2) 
    
    return lines

def format_scores(scores, is_home):
    if is_home==1: label = 'home'
    else: label = 'away'

    scores = scores[[f'{label}_team', f'{label}_score', 'week', 'year']]
    scores = scores.assign(is_home=is_home)
    scores.columns = ['team', 'final_score', 'week', 'year', 'is_home']
    
    return scores

def join_lines_scores(lines, final_scores):

    home_lines = format_lines(lines, is_home=1)
    away_lines = format_lines(lines, is_home=0)
    all_lines = pd.concat([home_lines, away_lines], axis=0)

    home_scores = format_scores(final_scores, is_home=1)
    away_scores = format_scores(final_scores, is_home=0)
    scores = pd.concat([home_scores, away_scores], axis=0)
    
    scores = pd.merge(all_lines, scores, on=['team', 'week', 'year', 'is_home'])

    return scores

def create_scores_lines_table(WEEK, YEAR):

    lines = dm.read("SELECT * FROM Gambling_Lines WHERE year>=2020", 'Pre_TeamData')
    final_scores = dm.read("SELECT * FROM Final_Scores WHERE year>=2020", 'FastR')
    scores_lines = join_lines_scores(lines, final_scores)
    try:
        cur_lines = dm.read(f"SELECT * FROM Gambling_Lines WHERE year={YEAR} AND week={WEEK}", 'Pre_TeamData')
        cur_home = format_lines(cur_lines, is_home=1)
        cur_away = format_lines(cur_lines, is_home=0)
        cur_lines = pd.concat([cur_home, cur_away], axis=0)
        scores_lines = pd.concat([scores_lines, cur_lines], axis=0)
    except:
        print('Current week not available')

    dm.write_to_db(scores_lines, 'Model_Features', 'Scores_Lines', 'replace')


def add_gambling_lines(df):

    lines = dm.read("SELECT * FROM Gambling_Lines", 'Pre_TeamData')
    home_lines = format_lines(lines, is_home=1)
    away_lines = format_lines(lines, is_home=0)
    lines = pd.concat([home_lines, away_lines], axis=0)

    df = pd.merge(df, lines, on=['team', 'week', 'year'], how='left')

    return df

#-----------------------
# Attach Points
#-----------------------

def results_vs_predicted(df, col):

    df = df.sort_values(by=['player','year', 'week']).reset_index(drop=True)
    df[f'{col}_miss'] = df.y_act - df[col]
    df[f'{col}_miss'] = (df.groupby('player')[f'{col}_miss'].shift(1)).fillna(0)
    df = add_rolling_stats(df, ['player'], [f'{col}_miss'], perform_check=False)
    df[f'{col}_miss_recent_vs8'] = df[f'rmean3_{col}_miss'] - df[f'rmean8_{col}_miss']

    good_cols = [c for c in df.columns if 'miss' in c or c in ('player', 'team', 'week', 'year')]
    df = df[good_cols]

    return df

def projected_pts_vs_predicted(df):

    proj_pts_miss = df[['player', 'team', 'week', 'year']].copy()
    for c in ['ffa_points', 'projected_points', 'fantasyPoints', 'ProjPts', 
              'fc_proj_fantasy_pts_fc', 'fft_proj_pts', 'avg_proj_points']:
        if c in df.columns:
            cur_miss = results_vs_predicted(df[['player', 'team', 'week', 'year', 'y_act', c]].copy(), c)
            proj_pts_miss = pd.merge(proj_pts_miss, cur_miss, on=['player', 'team', 'week', 'year']) 

    df = pd.merge(df, proj_pts_miss, on=['player', 'team', 'week', 'year'], how='left')
    df = forward_fill(df)

    miss_cols = [c for c in df.columns if 'miss' in c]
    df[miss_cols] = df[miss_cols].fillna(0)

    return df


def attach_y_act(df, pos, defense=False, rush_or_pass=''):

    if defense:
        y_act = dm.read(f'''SELECT defTeam player, week, season year, fantasy_pts y_act
                            FROM {pos}_Stats
                            WHERE season >= 2020''', 'FastR')
    
        df = pd.merge(df, y_act, on=['player',  'week', 'year'], how='left')
    
    elif pos=='QB':
        y_act = dm.read(f'''SELECT player, team, week, season year,
                                   fantasy_pts{rush_or_pass} y_act
                            FROM {pos}_Stats
                            WHERE season >= 2020
                                  AND pass_pass_attempt_sum > 15''', 'FastR')
    
        df = pd.merge(df, y_act, on=['player', 'team', 'week', 'year'], how='left')
    
    else:
        y_act = dm.read(f'''SELECT player, team, week, season year, fantasy_pts y_act
                            FROM {pos}_Stats
                            WHERE season >= 2020''', 'FastR')
                            
        snaps = get_snap_data()
        proj = dm.read('''SELECT player, week, year, fantasyPoints
                          FROM PFF_Proj_Ranks''', 'Pre_PlayerData')

        y_act = pd.merge(y_act, snaps, on=['player', 'week', 'year'], how='left')
        y_act = pd.merge(y_act, proj, on=['player', 'week', 'year'], how='left')

        y_act = y_act[~((y_act.fantasyPoints > 12) & \
                        (y_act.snap_pct < y_act.avg_snap_pct*0.5) & \
                        (y_act.snap_pct <= 0.4) & \
                        (y_act.snap_pct > 0))].drop(['snap_pct', 'avg_snap_pct', 'fantasyPoints'], axis=1)
        
        df = pd.merge(df, y_act, on=['player', 'team', 'week', 'year'], how='left')

    return df

#-------------------
# Final Cleanup
#-------------------

def drop_y_act_except_current(df, week, year):
    
    
    df = df[~(df.y_act.isnull()) | ((df.week==week) & (df.year==year))].reset_index(drop=True)
    df.loc[((df.week==week) & (df.year==year)), 'y_act'] = 0

    return df

def remove_non_uniques(df):
    cols = df.nunique()[df.nunique()==1].index
    cols = [c for c in cols if c != 'pos']
    df = df.drop(cols, axis=1)
    return df

def drop_duplicate_players(df):
    df = df.sort_values(by=['player', 'year', 'week', 'projected_points', 'ffa_points'],
                    ascending=[True, True, True, False, False])
    df = df.drop_duplicates(subset=['player', 'year', 'week'], keep='first').reset_index(drop=True)
    return df


def remove_low_corrs(df, corr_cut=0.015):
    obj_cols = df.dtypes[df.dtypes=='object'].index
    corrs = pd.DataFrame(np.corrcoef(df.drop(obj_cols, axis=1).values, rowvar=False), 
                         columns=[c for c in df.columns if c not in obj_cols],
                         index=[c for c in df.columns if c not in obj_cols])
    corrs = corrs['y_act']
    low_corrs = list(corrs[abs(corrs) < corr_cut].index)
    low_corrs = [c for c in low_corrs if c not in ('week', 'year')]
    df = df.drop(low_corrs, axis=1)
    print(f'Removed {len(low_corrs)}/{df.shape[1]} columns')
    
    corrs = corrs.dropna().sort_values()
    display(corrs.iloc[:20])
    display(corrs.iloc[-20:])
    display(corrs[~corrs.index.str.contains('ffa|proj|expert|rank|Proj|fc|salary|Points|Rank') | \
                   corrs.index.str.contains('team') | \
                   corrs.index.str.contains('qb')].iloc[:20])
    display(corrs[~corrs.index.str.contains('ffa|proj|expert|rank|Proj|fc|salary|Points|Rank') | \
                   corrs.index.str.contains('team') | \
                   corrs.index.str.contains('qb')].iloc[-20:])
    return df


#--------------------
# Data to apply to all datasets
#--------------------


def create_pos_rank(df, extra_pos=False):
    df = df.sort_values(by=['team', 'pos', 'year', 'week', 'avg_proj_points'],
                        ascending=[True, True, True, True, False]).reset_index(drop=True)

    df['pos_rank'] = df.pos + df.groupby(['team', 'pos', 'year', 'week']).cumcount().apply(lambda x: str(x))
    if extra_pos:
        df = df[df['pos_rank'].isin(['RB0', 'RB1', 'RB2', 'WR0', 'WR1', 'WR2', 'WR3', 'WR4', 'TE0', 'TE1'])].reset_index(drop=True)
    else:
        df = df[df['pos_rank'].isin(['QB0', 'RB0', 'RB1', 'WR0', 'WR1', 'WR2', 'TE0'])].reset_index(drop=True)
    return df


def get_team_projections():

    team_proj = pd.DataFrame()
    for pos in ['QB', 'RB', 'WR', 'TE']:

        tp = fantasy_pros_new(pos)
        tp = pff_experts_new(tp, pos)
        tp = ffa_compile(tp, 'FFA_Projections', pos)
        tp = ffa_compile(tp, 'FFA_RawStats', pos)
        tp = fftoday_proj(tp, pos)
        tp = fantasy_cruncher(tp, pos)
        tp = get_salaries(tp, pos)

        tp = consensus_fill(tp)
        tp = fill_ratio_nulls(tp)
        tp = log_rank_cols(tp)
        team_proj = pd.concat([team_proj, tp], axis=0)

    team_proj = create_pos_rank(team_proj)
    team_proj = forward_fill(team_proj)
    team_proj = team_proj.fillna(0)

    cnts = team_proj.groupby(['team', 'week', 'year']).agg({'avg_proj_points': 'count'})
    print('Team counts that do not equal 7:', cnts[cnts.avg_proj_points!=7])

    cols = [
            'projected_points', 'ProjPts', 'ffa_points', 'fc_proj_fantasy_pts_fc', 'avg_proj_points', 
            'fantasyPoints', 'dk_salary', 'fd_salary',
            'log_ffa_rank', 'log_avg_proj_rank', 'log_expertConsensus', 'log_rankadj_fp_rank', 'log_playeradj_fp_rank', 'log_fp_rank',
            'rushAtt', 'rushYds', 'rushTd', 'recvTargets', 'recvReceptions', 'recvYds', 'recvTd',
            'ffa_rush_yds', 'ffa_rush_tds','ffa_rec', 'ffa_rec_yds','ffa_rec_tds',
            'fft_rush_att', 'fft_rush_yds', 'fft_rush_td', 'fft_rec', 'fft_rec_yds', 'fft_rec_td', 'fft_proj_pts',
            'fc_proj_rushing_stats_att', 'fc_proj_rushing_stats_yrds', 'fc_proj_rushing_stats_tds',
            'fc_proj_receiving_stats_tar', 'fc_proj_receiving_stats_rec',
            'fc_proj_receiving_stats_yrds', 'fc_proj_receiving_stats_tds', 
            'avg_proj_rush_att', 'avg_proj_rush_td', 'avg_proj_rec', 'avg_proj_rec_tgts','avg_proj_rec_yds','avg_proj_rec_td', 
            ]

    to_agg = {c: 'sum' for c in cols}

    # get the projections broken out by RB and WR/TE
    team_proj_pos = team_proj[team_proj.pos.isin(['RB', 'WR', 'TE'])].copy()
    team_proj_pos.loc[team_proj_pos.pos=='TE', 'pos'] = 'WR'
    team_proj_pos = team_proj_pos.groupby(['pos', 'team', 'week', 'year']).agg(to_agg)
    team_proj_pos.columns = [f'pos_proj_{c}' for c in team_proj_pos.columns]
    team_proj_pos = team_proj_pos.reset_index()
    team_proj_pos_te = team_proj_pos[team_proj_pos.pos=='WR'].copy()
    team_proj_pos_te['pos'] = 'TE'
    team_proj_pos = pd.concat([team_proj_pos, team_proj_pos_te], axis=0).reset_index(drop=True)
    
    # get the projections broken out by team
    team_proj = team_proj.groupby(['team', 'week', 'year']).agg(to_agg)
    team_proj.columns = [f'team_proj_{c}' for c in team_proj.columns]
    team_proj = team_proj.reset_index()

    return team_proj, team_proj_pos

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


#%%

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

def box_score_rolling(bs):

    r_cols = [c for c in bs.columns if c not in ('player', 'team', 'game_date')]
    bs = add_rolling_stats(bs, gcols=['player'], rcols=r_cols)
    bs = bs.drop(['MIN', 'FGM', 'FGA', 'FG_PCT', 'three_pointers',
                  'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'rebounds',
                  'assists', 'steals', 'blocks', 'TO', 'PF', 'points', 'PLUS_MINUS'], axis=1)
    bs.game_date = bs.groupby('player')['game_date'].shift(-1)
    bs = bs.fillna({'game_date': max_date})

    return bs

def add_advanced_stats(df):
    adv = dm.read("SELECT * FROM Advanced_Stats", 'Player_Stats')
    adv = adv.rename(columns={'PLAYER_NAME': 'player'})

    adv = adv.drop(['GAME_ID', 'TEAM_ABBREVIATION', 'MIN', 'TEAM_ID', 'TEAM_CITY', 
                    'PLAYER_ID', 'NICKNAME', 'START_POSITION', 'COMMENT'], axis=1)
    adv = adv.sort_values(by=['player', 'game_date']).reset_index(drop=True)

    r_cols = [c for c in adv.columns if c not in ('player', 'game_date')]
    adv = add_rolling_stats(adv, gcols=['player'], rcols=r_cols)

    adv = adv.drop(['E_OFF_RATING', 'OFF_RATING', 'E_DEF_RATING', 'DEF_RATING',
                  'E_NET_RATING', 'NET_RATING', 'AST_PCT', 'AST_TOV', 'AST_RATIO',
                  'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'TM_TOV_PCT', 'EFG_PCT', 'TS_PCT',
                  'USG_PCT', 'E_USG_PCT', 'E_PACE', 'PACE', 'PACE_PER40', 'POSS', 'PIE',
                  ], axis=1)
    
    adv.game_date = adv.groupby('player')['game_date'].shift(-1)
    adv = adv.fillna({'game_date': max_date})
    df = pd.merge(df, adv, on=['player', 'game_date'], how='left')

    return df

#%%

max_date = dm.read("SELECT max(game_date) FROM FantasyData", 'Player_Stats').values[0][0]

df = fantasy_data()
df = fantasy_pros(df)
df = numberfire(df)
df= consensus_fill(df)
df = forward_fill(df)
df = df.dropna().reset_index(drop=True)
df = df[(df.fd_points>0) & (df.fp_points>0) & (df.nf_points>0)].reset_index(drop=True)
df = rolling_proj_stats(df)

# get box score stats and join to filter to players with minutes played
box_score = get_box_score()
df = pd.merge(df, box_score.loc[box_score.MIN>0, ['player', 'game_date', 'MIN']], on=['player', 'game_date'], how='left')
df = df[(df.game_date==max_date) | ~(df.MIN.isnull())]

# roll the box score stats and shift back a game
box_score_roll = box_score_rolling(box_score)
df = pd.merge(df, box_score_roll, on=['player', 'game_date'])

df = add_advanced_stats(df)
df = forward_fill(df)
df = add_y_act(df, box_score)
df = df.dropna(axis=0, subset=[c for c in df.columns if 'y_act' not in c]).reset_index(drop=True)

#%%

dm.write_to_db(df.iloc[:,:2000], 'Model_Features', 'Model_Data', if_exist='replace')
if df.shape[1] > 2000:
    dm.write_to_db(df.iloc[:,2000:], 'Model_Features', 'Model_Data2', if_exist='replace')

# %%

# %%
