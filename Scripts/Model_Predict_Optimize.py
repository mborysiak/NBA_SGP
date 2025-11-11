#%%
# from ff import general as ffgeneral
# from ff.db_operations import DataManage

# root_path = ffgeneral.get_main_path('NBA_SGP')
# db_path = f'{root_path}/Data/'
# dm = DataManage(db_path, timeout=200)

# # If starting a new year, use this code to initiate the database
# old_year = 2025
# new_year = 2026

# df = dm.read("SELECT * FROM Stack_Model_Predict LIMIT 0", f'Stack_Predict_{old_year}')
# dm.write_to_db(df, f'Stack_Predict_{new_year}', 'Stack_Model_Predict', 'replace', create_backup=False)

# df = dm.read("SELECT * FROM Stack_Model_Predict_Staging LIMIT 0", f'Stack_Predict_{old_year}')
# dm.write_to_db(df, f'Stack_Predict_{new_year}', 'Stack_Model_Predict_Staging', 'replace', create_backup=False)

#%%
# core packages

import pandas as pd
import numpy as np
import os
import gzip
import pickle
import datetime as dt
import itertools
import time
import gc
import matplotlib.pyplot as plt
from ff.db_operations import DataManage
from ff import general as ffgeneral
import ff.data_clean as dc
import optuna
from collections import defaultdict

from sklearn.preprocessing import StandardScaler
from Fix_Standard_Dev import *
from joblib import Parallel, delayed
from skmodel import SciKitModel
from sklearn.pipeline import Pipeline

from hyperopt.pyll import scope
from joblib import Parallel, delayed
from hyperopt import Trials, hp
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


root_path = ffgeneral.get_main_path('NBA_SGP')
db_path = f'{root_path}/Data/'
dm = DataManage(db_path, timeout=200)

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)

        
#====================
# Stacking Functions
#====================

def get_skm(skm_df, model_obj, to_drop):
    
    skm_options = {
        'reg': SciKitModel(skm_df, model_obj='reg', r2_wt=r2_wt, sera_wt=sera_wt, mse_wt=mse_wt, mae_wt=mae_wt),
        'class': SciKitModel(skm_df, model_obj='class', brier_wt=brier_wt, matt_wt=matt_wt),
        'quantile': SciKitModel(skm_df, model_obj='quantile')
    }
    
    skm = skm_options[model_obj]
    X, y = skm.Xy_split(y_metric='y_act', to_drop=to_drop)

    return skm, X, y


def get_full_pipe(skm, m, alpha=None, stack_model=False, min_samples=10, bayes_rand='rand'):

    if skm.model_obj=='class': kb = 'k_best_c'
    else: kb = 'k_best'

    stack_models = {

        'full_stack': skm.model_pipe([
                                      skm.piece('std_scale'), 
                                      skm.feature_union([
                                                    skm.piece('agglomeration'), 
                                                    skm.piece(kb),
                                                    skm.piece('pca')
                                                    ]),
                                      skm.piece(kb),
                                      skm.piece(m)
                                      ]),

        'random_full_stack': skm.model_pipe([
                                      skm.piece('random_sample'),
                                      skm.piece('std_scale'), 
                                      skm.feature_union([
                                                    skm.piece('agglomeration'), 
                                                    skm.piece(kb),
                                                    skm.piece('pca')
                                                    ]),
                                      skm.piece(kb),
                                      skm.piece(m)
                                      ]),

        'random_full_stack_ind_cats': skm.model_pipe([
                                        skm.piece('random_sample'),
                                        skm.piece('std_scale'), 
                                        skm.feature_union([
                                                        skm.piece('agglomeration'), 
                                                        skm.piece(kb),
                                                        skm.piece('pca')
                                                        ]),
                                        skm.piece(kb),
                                        skm.piece(m)
                                      ]),

        'kbest': skm.model_pipe([
                                 skm.piece('std_scale'),
                                 skm.piece(kb),
                                 skm.piece(m)
                                 ]),

        'random' : skm.model_pipe([
                                    skm.piece('random_sample'),
                                    skm.piece('std_scale'),
                                    skm.piece(m)
                                    ]),

        'random_kbest': skm.model_pipe([
                                        skm.piece('random_sample'),
                                        skm.piece('std_scale'),
                                        skm.piece(kb),
                                        skm.piece(m)
                                        ])
    }

    pipe = stack_models[stack_model]
    params = skm.default_params(pipe, bayes_rand=bayes_rand, min_samples=min_samples)
    
    if skm.model_obj == 'quantile':
        if m in ('qr_q', 'gbmh_q'): pipe.set_params(**{f'{m}__quantile': alpha})
        elif m in ('rf_q', 'knn_q'): pipe.set_params(**{f'{m}__q': alpha})
        else: pipe.set_params(**{f'{m}__alpha': alpha})

    if stack_model=='random_full_stack' and run_params['opt_type']=='bayes': 
        params['random_sample__frac'] = hp.uniform('random_sample__frac', 0.5, 1)

    elif stack_model=='random_full_stack' and run_params['opt_type']=='rand':
        params['random_sample__frac'] = np.arange(0.5, 1, 0.05)

        # params['select_perc__percentile'] = hp.uniform('percentile', 0.5, 1)
        # params['feature_union__agglomeration__n_clusters'] = scope.int(hp.quniform('n_clusters', 2, 10, 1))
        # params['feature_union__pca__n_components'] = scope.int(hp.quniform('n_components', 2, 10, 1))

    if m in ('gbm', 'gbm_c', 'gbm_q'):
        params[f'{m}__n_estimators'] = scope.int(hp.quniform('n_estimators', 20, 80, 2))
        params[f'{m}__max_depth'] = scope.int(hp.quniform('max_depth', 2, 18, 2))
        params[f'{m}__max_features'] = hp.uniform('max_features', 0.6, 0.95)
        params[f'{m}__subsample'] = hp.uniform('subsample', 0.6, 0.95)

    return pipe, params

def show_calibration_curve(y_true, y_pred, n_bins=10, strategy='quantile'):

    from sklearn.calibration import calibration_curve

    # Plot perfectly calibrated
    plt.plot([0, 1], [0, 1], linestyle = '--', label = 'Ideally Calibrated')
    
    # Plot model's calibration curve
    x, y = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy='quantile')
    plt.plot(y, x, marker = '.', label = 'Quantile')

    # Plot model's calibration curve
    x, y = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy='uniform')
    plt.plot(y, x, marker = '+', label = 'Uniform')
    
    leg = plt.legend(loc = 'upper left')
    plt.xlabel('Average Predicted Probability in each bin')
    plt.ylabel('Ratio of positives')
    plt.show()

def create_value_compare_col(X):
    for c in X.columns:
        if 'class' not in c and 'diff' not in c:
            X[c + '_vs_value'] = X[c] - X.value
        if 'class' not in c:
            X[c + '_over_value'] = X[c] / X.value
    return X


#-----------------------
# Saving validations
#-----------------------

def save_pickle(obj, path, fname, protocol=-1):
    with gzip.open(f"{path}/{fname}.p", 'wb') as f:
        pickle.dump(obj, f, protocol)

    print(f'Saved {fname} to path {path}')

def load_pickle(path, fname):
    with gzip.open(f"{path}/{fname}.p", 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object

#==========
# General Setting
#==========

#---------------
# Settings
#---------------

run_params = {

    # opt params
    'opt_type': 'bayes',
    'n_iters': 50,
    
    'n_splits': 5,
    'num_k_folds': 3,
    'show_plot': True,
    'print_coef': True,
    'min_include': 2,

}

r2_wt = 0
sera_wt = 0
mse_wt = 1
mae_wt = 0
brier_wt = 1
matt_wt = 0



def predict_odds_reduce(cur_df, num_choices, lr_odds_reduce):
    num_games = list(cur_df.team.unique())
    num_games.extend(list(cur_df.opponent.unique()))
    num_games = int(len(set(num_games))/2)
    odds_reduce = lr_odds_reduce.predict([[num_games, num_choices]])[0][0]
    return odds_reduce

def train_odds_reduce():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv("c:/Users/borys/OneDrive/Desktop/sgp_decay.csv")
    X = df[['games', 'legs']]
    y = df[['pct_chg']]

    lr = LinearRegression()
    lr.fit(X, y)
    print(lr.coef_)
    print(lr.score(X,y))

    return lr

lr_odds_reduce = train_odds_reduce()

import sqlite3

def create_indexes(db_path, tablename):
    """Create indexes for the betting prediction database"""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    try:
        # Create single column index
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_game_date 
            ON {tablename} (game_date)
        """)
        
        # Create composite index for common filters
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_prediction_filters 
            ON {tablename} (
                game_date,
                value_cut_greater,
                value_cut_less,
                decimal_cut_greater,
                decimal_cut_less,
                ens_vers,
                wt_col
            )
        """)
        
        conn.commit()
        
        # Check created indexes
        cur.execute(f"""
            SELECT name, sql 
            FROM sqlite_master 
            WHERE type='index' 
            AND tbl_name='{tablename}'
        """)
        indexes = cur.fetchall()
        print("Created indexes:")
        for idx_name, idx_sql in indexes:
            print(f"- {idx_name}")
            
    except Exception as e:
        print(f"Error creating indexes: {e}")
        
    finally:
        conn.close()

#============
# Choice Dictionary Management
#============

def get_choices_dict():
    """Initialize dictionary structure but use numpy arrays internally"""
    all_choices = {}
    
    # Using smaller dtypes for memory efficiency
    for win_type in ['num_correct', 'num_wins', 'winnings', 'num_trials']:
        all_choices[win_type] = {}
        for start_spot in range(2):
            all_choices[win_type][start_spot] = {}
            for num_choices in range(1,7):
                # Choose appropriate dtype for each metric
                if win_type == 'winnings':
                    all_choices[win_type][start_spot][num_choices] = np.array([], dtype=np.float16)
                elif win_type in ['num_correct', 'num_trials']:
                    all_choices[win_type][start_spot][num_choices] = np.array([], dtype=np.uint16)
                else:  # num_wins
                    all_choices[win_type][start_spot][num_choices] = np.array([], dtype=np.uint8)

    return all_choices

def fill_choices_dict(all_choices, preds):
    """Fill choices dictionary using numpy operations but maintain dictionary structure"""
    # Convert pandas series to numpy arrays for faster operations
    y_act = preds.y_act.values.astype(np.int8)
    decimal_odds = preds.decimal_odds.values.astype(np.float16)
    total_rows = len(preds)
    
    for start_spot in range(2):
        for num_choices in range(1,7):
            if start_spot + num_choices > total_rows:
                continue
                
            # Calculate wins and odds using numpy operations
            slice_y = y_act[start_spot:start_spot + num_choices]
            wins = np.sum(slice_y)
            odds = np.prod(decimal_odds[start_spot:start_spot + num_choices])
            
            # Still need to use pandas for odds_reduce calculation
            odds_reduce = predict_odds_reduce(preds.iloc[start_spot:start_spot+num_choices], 
                                              num_choices, lr_odds_reduce)
            odds = np.float16(np.round(odds * odds_reduce,3))
            
            # Append to numpy arrays
            all_choices['num_correct'][start_spot][num_choices] = np.append(
                all_choices['num_correct'][start_spot][num_choices], 
                wins
            )
            
            all_choices['num_trials'][start_spot][num_choices] = np.append(
                all_choices['num_trials'][start_spot][num_choices], 
                1
            )
            
            if wins == num_choices:
                all_choices['winnings'][start_spot][num_choices] = np.append(
                    all_choices['winnings'][start_spot][num_choices], 
                    odds
                )
                all_choices['num_wins'][start_spot][num_choices] = np.append(
                    all_choices['num_wins'][start_spot][num_choices], 
                    1
                )
            else:
                all_choices['winnings'][start_spot][num_choices] = np.append(
                    all_choices['winnings'][start_spot][num_choices], 
                    -1
                )
                all_choices['num_wins'][start_spot][num_choices] = np.append(
                    all_choices['num_wins'][start_spot][num_choices], 
                    0
                )
    
    return all_choices



def aggregate_choices(all_choices):
    for k,v in all_choices.items():
        for k2, v2 in v.items():
            all_choices[k][k2] = np.sum(v2)
    return pd.DataFrame(all_choices)

#====================
# Stack Model Functions
#====================

def create_save_path(decimal_cut_greater, decimal_cut_less, val_greater, val_less, wt_col, include_under, foldername='pick_choices'):
    decimal_cut_greater_lbl = decimal_cut_greater.replace('>=', 'greaterequal').replace('>', 'greater')
    decimal_cut_less_lbl = decimal_cut_less.replace('<=', 'lessequal')
    val_greater_lbl = val_greater.replace('>', 'greater')
    val_less_lbl = val_less.replace('<', 'less')
    save_name = f'{wt_col}_{decimal_cut_greater_lbl}_{decimal_cut_less_lbl}_{include_under}_{val_greater_lbl}_{val_less_lbl}'
    save_path = f'{root_path}/Model_Outputs/{save_name}'
    return save_name

def get_date_info(df):
    df['real_date'] = pd.to_datetime(df['game_date'].astype('str'), format='%Y%m%d')
    df['day_of_week'] = df['real_date'].dt.dayofweek
    df['month'] = df['real_date'].dt.month
    return df

def train_split(train_pred, test_date, num_back_days=45, cv_time_input=None, i=20):

    train_dates = train_pred.loc[train_pred.game_date < test_date, 'game_date'].sort_values().unique()[-num_back_days:]
    test_pred = train_pred[train_pred.game_date == test_date]
    train_pred = train_pred[(train_pred.game_date < test_date) & (train_pred.game_date.isin(train_dates))]

    if cv_time_input is None: 
        train_pred = train_pred.sample(frac=1, random_state=i).reset_index(drop=True)
        cv_time_input = None
    else:  
        train_pred = train_pred.sort_values(by='game_date').reset_index(drop=True)
        cv_time_input = train_pred.game_date.unique()[-cv_time_input]

    return train_pred, test_pred, cv_time_input

def preprocess_X(df, wt_col, cv_time_input=None):
    X = df[['decimal_odds', 'value', 'prob_over', 'pred_mean', 'pred_q25', 'pred_q50', 'pred_q75']].copy()
    X = create_value_compare_col(X)

    if wt_col=='decimal_odds_twomax':
        X['decimal_odds_twomax'] = X.decimal_odds
        X.loc[X.decimal_odds_twomax > 2, 'decimal_odds_twomax'] = 2-(1/X.loc[X.decimal_odds_twomax > 2, 'decimal_odds_twomax'])

    X = pd.concat([
                   X, 
                   pd.get_dummies(df.metric), 
                   pd.get_dummies(df.day_of_week, prefix='day'),
                   pd.get_dummies(df.month, prefix='month'),
                   pd.get_dummies(df.train_date, prefix='train_date'),
                   ], axis=1)
    
    if cv_time_input is not None: 
        X = pd.concat([df[['player','team', 'game_date']], X], axis=1)
        X['week'] = X.game_date.apply(lambda x: int(str(x)[-4:]))
        X['year'] = X.game_date.apply(lambda x: int(str(x)[:4]))

    return X

def flip_probs(df, pred_col='final_pred'):
    df.y_act = df.y_act.astype(int)
    df.loc[df[pred_col] < 0.5, 'y_act'] = 1 - df.loc[df[pred_col] < 0.5, 'y_act']
    
    df.loc[df[pred_col] < 0.5, 'decimal_odds'] = (1 / (1 - (1/df.loc[df[pred_col] < 0.5, 'decimal_odds']))) - 0.2
    df.loc[df[pred_col] < 0.5, pred_col] = 1 - df.loc[df[pred_col] < 0.5, pred_col]
    return df

#=================
# Past Run Information
#=================


def get_past_runs(ens_vers, tablename, foldername, dbname='Simulation'):

    past_runs = dm.read(f'''SELECT DISTINCT wt_col, decimal_cut_greater, decimal_cut_less, value_cut_greater, value_cut_less
                            FROM {tablename}
                            WHERE ens_vers='{ens_vers}'
                        ''', dbname)
    
    for c in past_runs.columns:
        past_runs[c] = past_runs[c].astype(str)

    past_runs_list = []
    for i, row in past_runs.iterrows():
        decimal_cut_greater = row.decimal_cut_greater
        decimal_cut_less = row.decimal_cut_less
        val_greater = row.value_cut_greater
        val_less = row.value_cut_less
        wt_col = row.wt_col
        include_under = ''

        save_path = create_save_path(decimal_cut_greater, decimal_cut_less, val_greater, val_less, wt_col, include_under, foldername)
        if os.path.exists(save_path) and f'{ens_vers}.p' in os.listdir(save_path):
            past_runs_list.append(save_path.split('/')[-1])

    return past_runs_list

def find_last_run(ens_vers, tablename, dbname='Simulation'):

    last_run = dm.read(f'''SELECT max(game_date) as game_date
                           FROM {tablename}
                           WHERE ens_vers = '{ens_vers}'
                        ''', dbname).values[0][0]
    return last_run

def get_past_trials(ens_vers, decimal_cut_greater, decimal_cut_less, val_greater, val_less, wt_col, include_under, foldername='pick_choices'):
    check_path = create_save_path(decimal_cut_greater, decimal_cut_less, val_greater, val_less, wt_col, include_under, foldername)
    try: trial_obj = load_pickle(check_path, ens_vers)                            
    except: trial_obj = Trials()
    return trial_obj

def pull_game_dates(q):
    game_dates = dm.read(q, 'Simulation').game_date.unique()
    return game_dates


def rename_existing(new_study_db, study_name):

    import datetime as dt
    new_study_name = str(study_name) + '_' + dt.datetime.now().strftime('%Y%m%d%H%M%S')
    optuna.copy_study(from_study_name=study_name, from_storage=new_study_db, to_storage=new_study_db, to_study_name=new_study_name)
    optuna.delete_study(study_name=study_name, storage=new_study_db)


def get_new_study(old_db, new_db, old_name, new_name, num_trials):
    
    old_storage = optuna.storages.RDBStorage(
                                url=old_db,
                                engine_kwargs={"pool_size": 64, 
                                            "connect_args": {"timeout": 60},
                                            },
                                )
    
    new_storage = optuna.storages.RDBStorage(
                                url=new_db,
                                engine_kwargs={"pool_size": 64, 
                                            "connect_args": {"timeout": 60},
                                            },
                                )
    
    if old_name is not None:
        old_study = optuna.create_study(
            study_name=old_name,
            storage=old_storage,
            load_if_exists=True
        )
    
    try:
        next_study = optuna.create_study(
            study_name=new_name, 
            storage=new_storage, 
            load_if_exists=False
        )

    except:
        rename_existing(new_storage, new_name)
        next_study = optuna.create_study(
            study_name=new_name, 
            storage=new_storage, 
            load_if_exists=False
        )
    
    if old_name is not None and len(old_study.trials) > 0:
        print(f"Loaded {new_name} study with {old_name} {len(old_study.trials)} trials")
        next_study.add_trials(old_study.trials[-num_trials:])

    return next_study
    

def get_optuna_study(save_name, game_date, last_game_date):
    time.sleep(1*np.random.random())
    old_name = last_game_date
    new_name = game_date
    old_db = f"sqlite:///optuna/predict_optimize/{save_name}.sqlite3"
    new_db = f"sqlite:///optuna/predict_optimize/{save_name}.sqlite3"
    next_study = get_new_study(old_db, new_db, old_name, new_name, num_trials=50)
    return next_study


#=============
# SGP Functions
#=============

def remove_combo_stats(pred_df, prob_col):
    pred_df = pred_df[(~pred_df.metric.str.contains('_') | (pred_df.metric=='three_pointers'))].reset_index(drop=True)
    pred_df = pred_df.sort_values(by=prob_col, ascending=False).reset_index(drop=True)
    return pred_df

def remove_threes_df(pred_df, prob_col):
    pred_df = pred_df[(pred_df.metric!='three_pointers')].reset_index(drop=True)
    pred_df = pred_df.sort_values(by=prob_col, ascending=False).reset_index(drop=True)
    return pred_df


def get_max_pred(pred_df, grp_col, prob_col):
    
    max_prob = pred_df.groupby(grp_col).agg({prob_col: 'max'}).reset_index()
    pred_df = pd.merge(pred_df, max_prob, on=[grp_col, prob_col])
    pred_df = pred_df.sort_values(by=prob_col, ascending=False).reset_index(drop=True)
    
    return pred_df

def get_game_mapping(preds):
    game_mapping = {}
    games = preds[['team', 'opponent']].drop_duplicates().values
    for g in games:
        sorted_g = sorted(g)
        game_mapping[g[0]] = '_'.join(sorted_g)

    game_mapping = pd.DataFrame(game_mapping, index=[0]).T.reset_index()
    game_mapping.columns = ['team', 'matchup']
    preds = pd.merge(preds, game_mapping, on='team')
    return preds

def get_top_matchups(preds, prob_col, num_players=5):
    preds = preds.sort_values(by=prob_col, ascending=False).reset_index(drop=True)
    preds['top_matchup_players'] = preds.groupby('matchup').cumcount().values
    preds = preds[preds.top_matchup_players < num_players]
    top_matchups = preds.groupby('matchup').agg({prob_col: 'mean', 'player':'count'}).sort_values(by=prob_col, ascending=False).reset_index()
    return top_matchups


#=================
# Choice and Winnings Functions
#=================

def top_sgp_choices(cur_lbl, df, prob_col, choices, matchup_rank, num_matchups, remove_combos, remove_threes):
  
    if remove_combos: df = remove_combo_stats(df, prob_col)
    else: df = get_max_pred(df, 'player', prob_col)

    if remove_threes: df = remove_threes_df(df, prob_col)
    
    df = get_game_mapping(df)
    top_matchups = get_top_matchups(df, prob_col, num_players=5)
    top_matchups = top_matchups[top_matchups.player >= 5]

    df = df[df.matchup.isin(top_matchups.iloc[matchup_rank:matchup_rank+num_matchups, 0])]
    df = df.sort_values(by=prob_col, ascending=False).reset_index(drop=True)
    choices[cur_lbl] = fill_choices_dict(choices[cur_lbl], df)

    return df, choices

def top_all_choices(df, prob_col, all_label, choices, remove_combos, remove_threes):
    if remove_combos: df = remove_combo_stats(df, prob_col)
    else: df = get_max_pred(df, 'player', prob_col)

    if remove_threes: df = remove_threes_df(df, prob_col)

    choices[all_label] = fill_choices_dict(choices[all_label], df)

    return df, choices


#==============
# Output Functions
#==============
    
def reset_table(read_tablename, write_tablename, db_name='Simulation'):
    try:
        df = dm.read(f"SELECT * FROM {read_tablename} LIMIT 1",db_name)
        df = df.drop(0, axis=0)
        dm.write_to_db(df, db_name, write_tablename, 'replace', create_backup=False)
    except Exception as e:
        print(e)
        pass


def calc_all_sgp_winnings(prob_types, prob_dfs, prob_cols, choices):

    # calculate winnings from various sgp choices
    for prob_type, prob_df, prob_col in zip(prob_types, prob_dfs, prob_cols):
        for remove_combos in [True, False]:
            for remove_threes in [True, False]:
                all_label = f'all_{prob_type}_{remove_combos}_{remove_threes}'
                if all_label not in choices.keys(): choices[all_label] = get_choices_dict()
                _, choices = top_all_choices(prob_df, prob_col, all_label, choices, remove_combos, remove_threes)
                
                for matchup_rank in [0, 1, 2]:
                    for num_matchups in [1, 2, 3]:
                        cur_lbl = f'{prob_type}_{remove_combos}_{remove_threes}_{matchup_rank}_{num_matchups}'
                        if cur_lbl not in choices.keys(): choices[cur_lbl] = get_choices_dict()
                        _, choices = top_sgp_choices(cur_lbl, prob_df, prob_col, choices, matchup_rank, num_matchups, remove_combos, remove_threes)
    return choices


def format_choices_output(choice_df, win_type):
    choice_df = pd.melt(choice_df.T.reset_index(), id_vars=['index'])
    choice_df.columns = ['start_spot', 'num_choices', win_type]

    return choice_df

def calc_pct_stats(df):
    for c in ['num_correct', 'num_wins', 'winnings', 'num_trials', 'num_choices']:
        df[c] = df[c].astype(float)
    df['num_correct_pct'] = df.num_correct / ((df.num_trials*df.num_choices)+0.000001)
    df['num_wins_pct'] = df.num_wins / (df.num_trials+0.000001)
    df.num_correct_pct = df.num_correct_pct.round(3)
    df.num_wins_pct = df.num_wins_pct.round(3)
    df.winnings = df.winnings.round(1)
    return df

# save out all the various combinations by extracting from dictionary
def save_sgp_results(dbname, tablename, choices, game_dates, prob_cols, val_greater, val_less, wt_col, decimal_cut_greater, decimal_cut_less, include_under, ens_vers):
    for prob_type in prob_cols:
        for remove_combos in [True, False]:
            for remove_threes in [True, False]:
                for matchup_rank in [0, 1, 2]:
                    for num_matchups in [1, 2, 3]:
                        cur_lbl = f'{prob_type}_{remove_combos}_{remove_threes}_{matchup_rank}_{num_matchups}'
                        for i, win_type in enumerate(['num_correct', 'num_wins', 'winnings', 'num_trials']):
                            choice_df = aggregate_choices(choices[cur_lbl][win_type])
                            choice_df = format_choices_output(choice_df, win_type)
                            if i==0: choice_df_all = choice_df.copy()
                            else: choice_df_all = pd.merge(choice_df_all, choice_df, on=['start_spot', 'num_choices'])

                        choice_df_all = calc_pct_stats(choice_df_all)
                        choice_df_all = choice_df_all.assign(value_cut_greater=val_greater, value_cut_less=val_less, wt_col=wt_col, 
                                                            decimal_cut_greater=decimal_cut_greater, decimal_cut_less=decimal_cut_less,rank_order=prob_type, 
                                                            include_under=include_under, last_date=game_dates[-1], ens_vers=ens_vers)
                        choice_df_all = choice_df_all.assign(bet_type='sgp', matchup_rank=matchup_rank, num_matchups=num_matchups, 
                                                             no_combos=remove_combos, remove_threes=remove_threes)

                        dm.write_to_db(choice_df_all, dbname, tablename, 'append', create_backup=False)

def save_all_results(dbname, tablename, choices, game_dates, prob_cols, val_greater, val_less, wt_col, decimal_cut_greater, decimal_cut_less, include_under, ens_vers):
    for prob_type in prob_cols:
        for remove_combos in [True, False]:
            for remove_threes in [True, False]:
                cur_lbl = f'all_{prob_type}_{remove_combos}_{remove_threes}'
                for i, win_type in enumerate(['num_correct', 'num_wins', 'winnings', 'num_trials']):
                    choice_df = aggregate_choices(choices[cur_lbl][win_type])
                    choice_df = format_choices_output(choice_df, win_type)
                    if i==0: choice_df_all = choice_df.copy()
                    else: choice_df_all = pd.merge(choice_df_all, choice_df, on=['start_spot', 'num_choices'])

                
                choice_df_all = calc_pct_stats(choice_df_all)
                choice_df_all = choice_df_all.assign(value_cut_greater=val_greater, value_cut_less=val_less, wt_col=wt_col, 
                                                    decimal_cut_greater=decimal_cut_greater, decimal_cut_less=decimal_cut_less,rank_order=prob_type, 
                                                    include_under=include_under, last_date=game_dates[-1], ens_vers=ens_vers)
                choice_df_all = choice_df_all.assign(bet_type='all', matchup_rank=np.nan, num_matchups=np.nan, 
                                                     no_combos=remove_combos, remove_threes=remove_threes)

                dm.write_to_db(choice_df_all, dbname, tablename, 'append', create_backup=False)
    
    
#%%


def calc_stack_model(dbname, tablename, last_run_date, ens_vers, wt_col, decimal_cut_greater, decimal_cut_less, val_greater, val_less):

    print('\n=======\n', val_greater, val_less, wt_col, decimal_cut_greater, decimal_cut_less, '\n=======\n')

    q = f'''SELECT * 
            FROM Over_Probability_New
            WHERE value {val_greater}
                AND value {val_less}
                AND decimal_odds {decimal_cut_greater}
                AND decimal_odds {decimal_cut_less}
                AND ens_vers = '{ens_vers}'
                AND y_act IS NOT NULL
                AND metric IN ('points', 'assists', 'rebounds', 'three_pointers',
                               'assists_rebounds', 'points_assists', 'points_rebounds')
                AND game_date > 20240601
            ORDER BY game_date ASC
            '''

    save_name = create_save_path(decimal_cut_greater, decimal_cut_less, val_greater, val_less, wt_col, include_under='', foldername='pick_choices')
    game_dates = pull_game_dates(q)

    print(game_dates)
    game_dates = [d for d in game_dates if d > last_run_date]
    num_back_days = 60
    num_trials = 20; i=20
    last_test_date = last_run_date
    if len(game_dates)>0:
        train_pred_all = dm.read(q, 'Simulation')

        for test_date in game_dates:

            try:
                print('Date:', test_date)

                train_pred = train_pred_all.copy()
                train_pred = train_pred.drop('y_act', axis=1).rename(columns={'y_act_prob': 'y_act'})
                train_pred = get_date_info(train_pred)
                train_pred = train_pred.dropna().reset_index(drop=True)
                train_pred, test_pred, cv_time_input = train_split(train_pred, test_date=test_date, num_back_days=num_back_days, i=i)
                
                X_train = preprocess_X(train_pred, wt_col, cv_time_input)
                X_test = preprocess_X(test_pred, wt_col, cv_time_input)
                y_train = train_pred.y_act

                trials = get_optuna_study(save_name, test_date, last_test_date)

                skm, _, _ = get_skm(pd.concat([X_train, y_train], axis=1), model_obj='class', to_drop=[])
                pipe, params = get_full_pipe(skm,  'lr_c', stack_model='random_kbest', alpha=None, 
                                            min_samples=10, bayes_rand='optuna')
                
                params['random_sample__frac'] = ['real', 0.2, 1]
                params['k_best_c__k'] = ['int', 3, X_train.shape[1]]

                best_model, _, _, _ = skm.best_stack(pipe, params, X_train, y_train, 
                                                    n_iter=num_trials, alpha=None, wt_col=wt_col,
                                                    trials=trials, bayes_rand='optuna',
                                                    run_adp=False, print_coef=False,
                                                    proba=True, num_k_folds=run_params['num_k_folds'],
                                                    random_state=(i*2)+(i*7), optuna_timeout=120)

                for c in X_train.columns:
                    if c not in X_test.columns:
                        X_test[c] = 0

                X_test = X_test[X_train.columns]
                best_model.fit(X_train, y_train)

                preds_orig = test_pred.reset_index(drop=True).copy()
                preds_stack = pd.Series(np.round(best_model.predict_proba(X_test)[:,1], 3), name='final_pred')
                preds_stack = pd.concat([preds_stack, preds_orig], axis=1)
                preds_stack = preds_stack[['player', 'game_date', 'team', 'opponent', 'metric', 'value', 'decimal_odds', 
                                            'prob_over', 'final_pred', 'y_act']]
                
                preds_stack = preds_stack.assign(value_cut_greater=val_greater, value_cut_less=val_less, wt_col=wt_col, 
                                                decimal_cut_greater=decimal_cut_greater, decimal_cut_less=decimal_cut_less, ens_vers=ens_vers)
                last_test_date = test_date
                dm.write_to_db(preds_stack, dbname, tablename, 'append', create_backup=False)
            
            except:
                pass

#-------------
# Set Tables
#-------------
save_tablename = 'Stack_Model_Predict_Staging'
last_run_tablename = 'Stack_Model_Predict'
db_stack_predict = 'Stack_Predict_2025'
db_stack_predict_last = 'Stack_Predict_2025'


#%%
#--------------
# Set Params
#--------------

for ens_vers in [
    'random_kbest_matt0_brier1_include3_kfold3', 
     'random_full_stack_matt0_brier1_include3_kfold3',
     'random_full_stack_ind_cats_matt0_brier1_include3_kfold3'
    ]:

    dm.delete_from_db(db_stack_predict, 'Stack_Model_Predict_Staging', f"ens_vers='{ens_vers}'", create_backup=False)
    
    last_run_date = find_last_run(ens_vers, tablename=last_run_tablename, dbname=db_stack_predict_last)
    last_run_date = 20250313#int(last_run_date)
    
    print('Last Run Date', last_run_date)

    wt_col_list=[None, 'decimal_odds', 'decimal_odds_twomax']
    decimal_cut_greater_list = ['>=1.5', '>=1.75']
    decimal_cut_less_list = ['<=2.0', '<=2.5']
    val_greater_list = ['>1.5', '>2.5', '>3.5', '>4.5', '>5.5']
    val_less_list = ['<100', '<30', '<20', '<10']

    iter_cats = list(set(itertools.product(wt_col_list, decimal_cut_greater_list, decimal_cut_less_list, val_greater_list, val_less_list)))

    #------------
    # Run
    #------------

    # for wt_col, decimal_cut_greater, decimal_cut_less, val_greater, val_less in iter_cats[100:102]:
    #     calc_stack_model(db_stack_predict, save_tablename, last_run_date, ens_vers, wt_col, decimal_cut_greater, decimal_cut_less, val_greater, val_less)

    out = Parallel(n_jobs=16, verbose=50)(
        delayed(calc_stack_model)
        (db_stack_predict, save_tablename, last_run_date, ens_vers, wt_col, decimal_cut_greater, decimal_cut_less, val_greater, val_less) 
        for wt_col, decimal_cut_greater, decimal_cut_less, val_greater, val_less in iter_cats
    )

#------------
# Transfer from Staging to Prod
#------------

for ens_vers in [
                'random_kbest_matt0_brier1_include3_kfold3', 
                'random_full_stack_matt0_brier1_include3_kfold3',
                'random_full_stack_ind_cats_matt0_brier1_include3_kfold3'
                 ]:
    print(ens_vers)
    df = dm.read(f"SELECT * FROM Stack_Model_Predict_Staging WHERE ens_vers='{ens_vers}'", db_stack_predict)
    dm.write_to_db(df, db_stack_predict, 'Stack_Model_Predict', 'append', create_backup=False)
    del df
    gc.collect()
    
create_indexes(f'{db_path}/{db_stack_predict}.sqlite3', 'Stack_Model_Predict_Staging')


#%%

def run_past_choices(db_stack_predict, db_results, pull_tablename, save_tablename, 
                     ens_vers, wt_col, decimal_cut_greater, decimal_cut_less, include_under, val_greater, val_less):

    if wt_col is None: wt_col_q = 'IS NULL'
    else: wt_col_q = f"= '{wt_col}'"

    # Get unique dates first
    game_dates = dm.read(f'''SELECT DISTINCT game_date 
                            FROM {pull_tablename}
                            WHERE value_cut_greater = '{val_greater}'
                            AND value_cut_less = '{val_less}'
                            AND decimal_cut_greater = '{decimal_cut_greater}'
                            AND decimal_cut_less = '{decimal_cut_less}'
                            AND ens_vers = '{ens_vers}'
                            AND wt_col {wt_col_q}
                            AND y_act IS NOT NULL
                            AND game_date >= 20250207
                            ORDER BY game_date''', db_stack_predict).game_date.values
    
    all_sgp_choices = {}
    if len(game_dates)>0:
        for test_date in game_dates:

            preds = dm.read(f'''SELECT * 
                                FROM {pull_tablename}
                                WHERE game_date = {test_date}
                                AND value_cut_greater = '{val_greater}'
                                AND value_cut_less = '{val_less}'
                                AND decimal_cut_greater = '{decimal_cut_greater}'
                                AND decimal_cut_less = '{decimal_cut_less}'
                                AND ens_vers = '{ens_vers}'
                                AND wt_col {wt_col_q}
                                AND metric IN ('points', 'assists', 'rebounds', 'three_pointers',
                                               'assists_rebounds', 'points_assists', 'points_rebounds')
                                AND y_act IS NOT NULL''', db_stack_predict)
            
            # Convert numeric columns once
            numeric_cols = ['value', 'decimal_odds', 'prob_over', 'final_pred']
            preds[numeric_cols] = preds[numeric_cols].astype(float)
            
            # Create views instead of copies where possible
            preds_orig = preds.copy()
            preds_stack = preds.copy()
            preds_avg = preds.copy()
            preds_avg['avg_prob'] = preds_avg[['final_pred', 'prob_over']].mean(axis=1)

            del preds

            if include_under:
                preds_orig = flip_probs(preds_orig, pred_col='prob_over')
                preds_stack = flip_probs(preds_stack, pred_col='final_pred')
                preds_avg = flip_probs(preds_avg, pred_col='avg_prob')

            if wt_col is None:
                prob_types = ['stack_model', 'original', 'avg']
                prob_dfs = [preds_stack, preds_orig, preds_avg]
                prob_cols = ['final_pred', 'prob_over', 'avg_prob']
            else:
                prob_types = ['stack_model', 'avg']
                prob_dfs = [preds_stack, preds_avg]
                prob_cols = ['final_pred', 'avg_prob']

            preds_stack.y_act = preds_stack.y_act.astype(int)
            preds_orig.y_act = preds_orig.y_act.astype(int)
            preds_avg.y_act = preds_avg.y_act.astype(int)
            
            all_sgp_choices = calc_all_sgp_winnings(prob_types, prob_dfs, prob_cols, all_sgp_choices)

            
            del preds_orig
            del preds_stack
            del preds_avg
            gc.collect()
            
        save_sgp_results(db_results, save_tablename, all_sgp_choices, game_dates, prob_types, val_greater, val_less, wt_col, decimal_cut_greater, decimal_cut_less, include_under, ens_vers)
        save_all_results(db_results, save_tablename, all_sgp_choices, game_dates, prob_types, val_greater, val_less, wt_col, decimal_cut_greater, decimal_cut_less, include_under, ens_vers)

        del game_dates
        all_sgp_choices.clear()
        gc.collect()
    

#-------------
# Set Tables
#-------------
pull_tablename = 'Stack_Model_Predict_Staging'
save_tablename = 'Probability_Choices_Staging'
db_results = 'Results'
db_stack_predict = 'Stack_Predict_2025'

#--------------
# Set Params
#--------------

ens_vers_list = [
                 'random_kbest_matt0_brier1_include3_kfold3', 
                 'random_full_stack_matt0_brier1_include3_kfold3', 
                 'random_full_stack_ind_cats_matt0_brier1_include3_kfold3'
                 ]

for ens_vers in ens_vers_list:
    dm.delete_from_db('Results', 'Probability_Choices_Staging', f"ens_vers='{ens_vers}'", create_backup=False)

    print(ens_vers)
    wt_col_list=[None, 'decimal_odds', 'decimal_odds_twomax']
    decimal_cut_greater_list = ['>=1.5', '>=1.75']
    decimal_cut_less_list = ['<=2.0', '<=2.5']
    val_greater_list = ['>1.5', '>2.5', '>3.5', '>4.5', '>5.5']
    val_less_list = ['<100', '<30', '<20', '<10']
    include_under_list = [True]

    iter_cats = list(set(itertools.product(wt_col_list, decimal_cut_greater_list, decimal_cut_less_list, 
                                           include_under_list, val_greater_list, val_less_list)))

    #------------
    # Run
    #------------
    
    # for wt_col, decimal_cut_greater, decimal_cut_less, include_under, val_greater, val_less in iter_cats[:1]:
    #     past_choice_result = run_past_choices(db_stack_predict, db_results, pull_tablename, save_tablename, ens_vers, wt_col, decimal_cut_greater, decimal_cut_less, include_under, val_greater, val_less)

    
    out = Parallel(n_jobs=8, verbose=1)(
        delayed(run_past_choices)
        (db_stack_predict, db_results, pull_tablename, save_tablename, ens_vers, wt_col, decimal_cut_greater, decimal_cut_less, include_under, val_greater, val_less) 
        for wt_col, decimal_cut_greater, decimal_cut_less, include_under, val_greater, val_less in iter_cats
    )
#%%

prob_choices_staging = dm.read(f"SELECT * FROM Probability_Choices_Staging", 'Results')
prob_choices_staging[['matchup_rank', 'num_matchups']] = prob_choices_staging[['matchup_rank', 'num_matchups']].fillna(-1).astype(int)
prob_choices_staging.wt_col = prob_choices_staging.wt_col.fillna('None')

query_cuts_df = dm.read(f"SELECT * FROM Best_Choices", 'Simulation')
query_cuts_df.date_run = pd.to_datetime(query_cuts_df.date_run)
query_cuts_df = query_cuts_df[query_cuts_df.date_run==query_cuts_df.date_run.max()].reset_index(drop=True)
query_cuts_df.wt_col = query_cuts_df.wt_col.fillna('None')

query_cuts = {}
for i, row in query_cuts_df.iterrows():
    row.label = row.label + row.ens_vers
    query_cuts[row.label] = {}
    query_cuts[row.label]= row.to_dict()
    # if query_cuts[row.label]['wt_col'] == 'None': query_cuts[row.label]['wt_col'] = None

all_staging = pd.DataFrame()
for cut_name, cut_dict in query_cuts.items():

    ens_vers = cut_dict['ens_vers']
    start_spot = cut_dict['start_spot']
    num_choices = cut_dict['num_choices']
    decimal_cut_greater = cut_dict['decimal_cut_greater']
    decimal_cut_less = cut_dict['decimal_cut_less']
    val_greater = cut_dict['value_cut_greater']
    val_less = cut_dict['value_cut_less']
    wt_col = cut_dict['wt_col']
    include_under = int(cut_dict['include_under'])
    rank_order = cut_dict['rank_order']
    bet_type = cut_dict['bet_type']
    remove_combos = int(cut_dict['no_combos'])
    remove_threes = int(cut_dict['remove_threes'])
    if bet_type=='sgp':
        matchup_rank = int(cut_dict['matchup_rank'])
        num_matchups = int(cut_dict['num_matchups'])
    else:
        matchup_rank = -1
        num_matchups = -1

    cur_staging = (
        prob_choices_staging[
            (prob_choices_staging.ens_vers==ens_vers) &
            (prob_choices_staging.decimal_cut_greater==decimal_cut_greater) &
            (prob_choices_staging.decimal_cut_less==decimal_cut_less) &
            (prob_choices_staging.value_cut_greater==val_greater) &
            (prob_choices_staging.value_cut_less==val_less) &
            (prob_choices_staging.wt_col==wt_col) &
            (prob_choices_staging.include_under==include_under) &
            (prob_choices_staging.ens_vers==ens_vers) &
            (prob_choices_staging.bet_type==bet_type) &
            (prob_choices_staging.rank_order==rank_order) &
            (prob_choices_staging.no_combos==remove_combos) &
            (prob_choices_staging.remove_threes==remove_threes) &
            (prob_choices_staging.matchup_rank==matchup_rank) &
            (prob_choices_staging.num_matchups==num_matchups) &
            (prob_choices_staging.start_spot==start_spot) &
            (prob_choices_staging.num_choices==num_choices)
        ])

    all_staging = pd.concat([all_staging, cur_staging], axis=0)

all_staging.sort_values(by='winnings', ascending=False)

#%%

min_date, max_date = dm.read(f'''SELECT min(game_date), max(game_date) 
                                 FROM Stack_Model_Predict_Staging
                                ''', db_stack_predict).values[0]


staging = dm.read(f'''SELECT * 
                    FROM Probability_Choices_Staging 
                ''', db_results)
staging.ens_vers = staging.ens_vers.apply(lambda x: x.split('_matt0')[0])


if i==0: dm.write_to_db(staging, db_results, f'Probability_Choices_{min_date}_{max_date}', 'replace')
else: dm.write_to_db(staging, db_results, f'Probability_Choices_{min_date}_{max_date}', 'append')

final = dm.read(f'''SELECT * 
                    FROM Probability_Choices
                ''', db_results)
final.ens_vers = final.ens_vers.apply(lambda x: x.split('_matt0')[0])
orig_cols = final.columns

update_cols = {'winnings': 'winnings_add', 
                'last_date': 'last_date_new',
                'num_correct': 'num_correct_add',
                'num_wins': 'num_wins_add',
                'num_trials': 'num_trials_add'}
staging = staging.rename(columns=update_cols).drop(['num_correct_pct', 'num_wins_pct'], axis=1)
for k, v in update_cols.items(): 
    staging[v] = staging[v].astype(float)
    final[k] = final[k].astype(float)

merge_cols = ['start_spot', 'num_choices', 'value_cut_greater', 'value_cut_less', 'wt_col', 'decimal_cut_greater', 
            'decimal_cut_less', 'rank_order', 'include_under', 'ens_vers', 'bet_type', 'matchup_rank', 
            'num_matchups', 'no_combos', 'remove_threes']

for c in merge_cols:
    staging[c] = staging[c].astype(final[c].dtypes)

final = pd.merge(final, staging, on=merge_cols, how='left')

for k,v in update_cols.items():
    if k!='last_date': final[k] = final[k] + final[v]
    else: final[k] = final[v]

final['num_correct_pct'] = (final.num_correct / ((final.num_trials*final.num_choices)+0.000001)).round(3)
final['num_wins_pct'] = (final.num_wins / (final.num_trials+0.000001)).round(3)
final = final[orig_cols]

dm.write_to_db(final, db_results, 'Probability_Choices', 'replace', create_backup=True)

del final
del staging
gc.collect()

#%%

from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold

def choice_param_training_data(use_cats=True, frac=1):

    choice_params = dm.read(f'''
                                SELECT * 
                                FROM Probability_Choices
                                WHERE NOT (rank_order='original' 
                                           AND wt_col IS NOT NULL)
                                      AND NOT (bet_type='all' 
                                               AND matchup_rank IS NOT NULL)    
                                      AND num_trials > (SELECT 0.8*max(num_trials) FROM Probability_Choices)
                            ''', 'Results').sample(frac=frac).reset_index(drop=True)

    choice_params['winnings_rnd'] = choice_params.winnings.astype('int')
    choice_params = choice_params.drop_duplicates(subset=['num_correct', 'num_wins', 
                                                          'num_trials', 'winnings_rnd']).reset_index(drop=True)

    X = choice_params.drop(['winnings', 'num_correct', 'num_wins', 'num_correct_pct', 'num_wins_pct', 'winnings_rnd'], axis=1)
    y_winnings = choice_params.winnings
    y_win_pct = choice_params.num_wins_pct
    y_num_correct_pct = choice_params.num_correct_pct
    if not use_cats:
        X.value_cut_greater = X.value_cut_greater.apply(lambda x: x.replace('>', '')).astype(float)
        X.value_cut_less = X.value_cut_less.apply(lambda x: x.replace('<', '')).astype(float)

    for c in X.columns:
        if X[c].dtypes=='object': X[c] = X[c].astype('category')

    return X, y_winnings, y_win_pct, y_num_correct_pct


class FoldPredict:

    def __init__(self, save_path, retrain=True):
        self.save_path = save_path
        self.retrain = retrain

    def cross_fold_train(self, model_type, model, params, X, y, n_iter=10):

        for i, (train_idx, test_idx) in enumerate(KFold(n_splits=4, shuffle=True).split(X)):
            print(f'Fold {i+1}')
            X_train, _ = X.iloc[train_idx], X.iloc[test_idx]
            y_train, _ = y.iloc[train_idx], y.iloc[test_idx]

            grid = RandomizedSearchCV(model, params, n_iter=n_iter, scoring='neg_mean_absolute_error', n_jobs=2, cv=4)
            grid.fit(X_train,y_train)
            
            scores = pd.concat([pd.DataFrame(grid.cv_results_['params']), 
                                pd.DataFrame(grid.cv_results_['mean_test_score'])], axis=1).sort_values(by=0)
            print(scores)

            best_model = grid.best_estimator_
    
            if not os.path.exists(self.save_path): os.makedirs(self.save_path)
            save_pickle(best_model, self.save_path, f'{model_type}_fold{i}')

    def cross_fold_predict(self, model_type, X, y):

        predictions = pd.DataFrame()
        for _, (train_idx, test_idx) in enumerate(KFold(n_splits=4, shuffle=True).split(X)):

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            cur_predict = pd.DataFrame()
            for i in range(4):
                model = load_pickle(self.save_path, f'{model_type}_fold{i}')
                if self.retrain: model.fit(X_train, y_train)
                model_i_predict = pd.DataFrame(model.predict(X_test), index=test_idx, columns=[f'score_{i}'])
                cur_predict = pd.concat([cur_predict, model_i_predict], axis=1)
            
            cur_predict = pd.DataFrame(cur_predict.mean(axis=1), columns=[f'{model_type}_pred'])
            print('MAE:', np.round(np.mean(np.abs(cur_predict[f'{model_type}_pred'] - y_test)), 5))
            cur_predict = pd.concat([cur_predict, y_test], axis=1)
            predictions = pd.concat([predictions, cur_predict], axis=0)
                
        predictions = pd.merge(predictions, X,left_index=True, right_index=True)

        return predictions

def calc_kelly_criterion(pred):

    pred['num_wins'] = pred.num_wins_pct * pred.num_trials
    pred['avg_win_odds'] = (pred.score + pred.num_trials - pred.num_wins) / pred.num_wins
    pred['num_wins_pct_choices'] = pred.num_correct_pct ** pred.num_choices

    pred['kc_wins'] = pred.num_wins_pct - ((1-pred.num_wins_pct)/(pred.avg_win_odds-1))
    pred['kc_num_choices'] = pred.num_wins_pct_choices - ((1-pred.num_wins_pct_choices)/(pred.avg_win_odds-1))
    pred['kc_avg'] = pred[['kc_wins', 'kc_num_choices']].mean(axis=1)
    pred['kc_avg'] * pred.num_trials

    return pred

params = {
    'n_estimators': range(100, 300, 25),
    'num_leaves': range(100, 300, 25),
    'min_child_samples': range(50, 400, 25),
    'learning_rate': [0.25, 0.3, 0.35, 0.4, 0.45],
    'subsample': [0.75, 0.8, 0.85, 0.9, 0.95, 1]
}

lgbm = LGBMRegressor(n_jobs=16) 
X, y_winnings, y_win_pct, y_num_correct_pct = choice_param_training_data(use_cats=True, frac=1)
fp = FoldPredict(f'{root_path}/Model_Outputs/Final_LGBM/', retrain=True)

#%%
fp.cross_fold_train('winnings', lgbm, params, X, y_winnings, n_iter=10)
fp.cross_fold_train('win_pct', lgbm, params, X, y_win_pct, n_iter=10)
fp.cross_fold_train('num_correct_pct', lgbm, params, X, y_num_correct_pct, n_iter=10)

#%%

X, y_winnings, y_win_pct, y_num_correct_pct = choice_param_training_data(use_cats=True, frac=1)
winnings_pr = fp.cross_fold_predict('winnings', X, y_winnings)
win_pct_pr = fp.cross_fold_predict('win_pct', X, y_win_pct)
num_correct_pct_pr = fp.cross_fold_predict('num_correct_pct', X, y_num_correct_pct)

pred = pd.merge(win_pct_pr[['num_wins_pct', 'win_pct_pred']], winnings_pr, left_index=True, right_index=True)
pred = pd.merge(num_correct_pct_pr[['num_correct_pct', 'num_correct_pct_pred']], pred, left_index=True, right_index=True)
# pred = calc_kelly_criterion(pred)

print('Winnings MAE:', np.round(np.mean(np.abs(pred.winnings - pred.winnings_pred)), 5))
print('R2 Score', np.round(r2_score(pred.winnings, pred.winnings_pred), 4))
pred.plot.scatter(x='winnings', y='winnings_pred')

pred.num_matchups = pred.num_matchups.astype(float)
pred['winnings_pred_comb'] = (pred.winnings_pred + pred.winnings)/2
pred = pred.sort_values(by='winnings_pred_comb', ascending=False).reset_index(drop=True)

pred['win_pct_comb'] = (pred.win_pct_pred + pred.num_wins_pct)/2
pred['num_correct_pct_comb'] = (pred.num_correct_pct_pred + pred.num_correct_pct)/2
# pred = pred.sort_values(by='win_pct_comb', ascending=False).reset_index(drop=True)

#%%

import pprint
best_results = {}

for i, ens_v in enumerate(['random_kbest',
                            'random_full_stack', 
                            'random_full_stack_ind_cats']):

    print('=============\n', ens_v, '\n=============')
    
    best_sgp_10 = (
        pred[
             (pred.ens_vers==ens_v) & 
             (pred.win_pct_comb > 0.12) & 
             (pred.num_choices >= 4) &
             (pred.num_matchups==1)
             ]).sort_values(by='winnings_pred_comb', ascending=False).head(5)
    display(best_sgp_10) 

    best_sgp_15 = (
        pred[
             (pred.ens_vers==ens_v) & 
             (pred.win_pct_comb > 0.18) &
             (pred.num_choices >= 3) &
             (pred.num_matchups==1)
             ]).sort_values(by='winnings_pred_comb', ascending=False).head(5)
    display(best_sgp_15) 

    best_overall_10 = (
        pred[
             (pred.ens_vers==ens_v) & 
             (pred.win_pct_comb > 0.12) & 
             (pred.num_choices >= 4)
             ]).sort_values(by='winnings_pred_comb', ascending=False).head(5)
    display(best_overall_10) 

    best_overall_15 = (
        pred[
             (pred.ens_vers==ens_v) & 
             (pred.win_pct_comb > 0.18) &
             (pred.num_choices >= 3)
             ]).sort_values(by='winnings_pred_comb', ascending=False).head(5)
    display(best_overall_15) 

    
    save_out = pd.DataFrame(best_sgp_10.iloc[0]).T.assign(label = 'min_12%_sgp', date_run = pd.to_datetime('today').strftime('%Y-%m-%d'))
    
    best_overall_10 = best_overall_10[best_overall_10.index.isin(save_out.index)==False]
    save_out = pd.concat([save_out, pd.DataFrame(best_overall_10.iloc[0]).T.assign(label = 'min_12%', date_run = pd.to_datetime('today').strftime('%Y-%m-%d'))])
    
    best_sgp_15 = best_sgp_15[best_sgp_15.index.isin(save_out.index)==False]
    save_out = pd.concat([save_out, pd.DataFrame(best_sgp_15.iloc[0]).T.assign(label = 'min_18%_sgp', date_run = pd.to_datetime('today').strftime('%Y-%m-%d'))])
    
    best_overall_15 = best_overall_15[best_overall_15.index.isin(save_out.index)==False]
    save_out = pd.concat([save_out, pd.DataFrame(best_overall_15.iloc[0]).T.assign(label = 'min_18%', date_run = pd.to_datetime('today').strftime('%Y-%m-%d'))])
  #  save_out = pd.concat([save_out, pd.DataFrame(best_high_win_actual.iloc[0]).T.assign(label = 'min_25%_actual', date_run = pd.to_datetime('today').strftime('%Y-%m-%d'))])

    save_out = save_out.drop_duplicates()

    for c in ['start_spot','include_under', 'last_date', 'no_combos', 'remove_threes']:
        save_out[c] = save_out[c].astype('int')

    for c in ['num_correct_pct', 'num_correct_pct_pred', 'num_wins_pct', 'win_pct_pred', 'win_pct_comb', 'num_correct_pct_comb', 'num_correct_pct_pred', 'winnings', 'winnings_pred', 'winnings_pred_comb']:
        save_out[c] = save_out[c].astype('float').round(3)

    del_str = f"ens_vers = '{ens_v}' AND date_run = '{pd.to_datetime('today').strftime('%Y-%m-%d')}'"
    dm.delete_from_db('Simulation', 'Best_Choices', del_str, create_backup=False)
    dm.write_to_db(save_out, 'Simulation', 'Best_Choices', 'append', create_backup=False)

#%%