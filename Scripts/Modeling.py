#%%
# core packages
import pandas as pd
import numpy as np
import os
import gzip
import pickle
import datetime as dt
import matplotlib.pyplot as plt

from ff.db_operations import DataManage
from ff import general as ffgeneral
from skmodel import SciKitModel

import pandas_bokeh
pandas_bokeh.output_notebook()

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

pd.set_option('display.max_columns', 999)
# from sklearn import set_config
# set_config(display='diagram')

#==========
# General Setting
#==========

# set the root path and database management object
root_path = ffgeneral.get_main_path('NBA_SGP')
db_path = f'{root_path}/Data/'
dm = DataManage(db_path)

#---------------
# Settings
#---------------

run_params = {
    
    # set year and week to analyze
    'cv_time_input': '2023-02-03',
    'train_time_split': '2023-02-25',
    'metrics': ['points', 'rebounds', 'assists', 'three_pointers'],

    'n_iters': 25,
    'n_splits': 5
}

run_params['cv_time_input'] = int(run_params['cv_time_input'].replace('-', ''))
run_params['train_time_split'] = int(run_params['train_time_split'].replace('-', ''))

# set weights for running model
r2_wt = 1
sera_wt = 0
mse_wt = 5
matt_wt = 0
brier_wt = 1

# set version and iterations
vers = 'mse5_rsq1_lowsample_perc'

#----------------
# Data Loading
#----------------

def create_pkey_output_path(metric, run_params, vers):

    pkey = f"{metric}_{run_params['train_time_split']}_{vers}"
    model_output_path = f"{root_path}/Model_Outputs/{pkey}/"
    if not os.path.exists(model_output_path): os.makedirs(model_output_path)
    
    return pkey, model_output_path

def load_data(run_params):

    # load data and filter down
    df = dm.read(f'''SELECT * FROM Model_Data''', 'Model_Features')
   
    if df.shape[1]==2000:
        df2 = dm.read(f'''SELECT * FROM Model_Data2''', 'Model_Features')
        df = pd.concat([df, df2], axis=1)

    df.game_date = df.game_date.apply(lambda x: int(x.replace('-', '')))
    df = df.sort_values(by=['game_date']).reset_index(drop=True)
    
    drop_cols = list(df.dtypes[df.dtypes=='object'].index)
    run_params['drop_cols'] = drop_cols
    print(drop_cols)

    return df, run_params


def train_predict_split(df, run_params):

    # # get the train / predict dataframes and output dataframe
    df_train = df[df.game_date < run_params['train_time_split']].reset_index(drop=True)
    df_train = df_train.dropna(subset=['y_act']).reset_index(drop=True)

    df_predict = df[df.game_date == run_params['train_time_split']].reset_index(drop=True)
    output_start = df_predict[['player', 'game_date']].copy().drop_duplicates()

    # get the minimum number of training samples for the initial datasets
    min_samples = int(df_train[df_train.game_date < run_params['cv_time_input']].shape[0])  
    print('Shape of Train Set', df_train.shape)

    return df_train, df_predict, output_start, min_samples

def create_value_columns(df, metric):

    for c in df.columns:
        if metric in c:
            df[c + '_vs_value'] = df[c] - df.value
            df[c + '_over_value'] = df[c] / df.value

    return df

def pull_odds(metric):

    odds = dm.read(f'''SELECT player, stat_type, game_date year, value, over_under
                       FROM Draftkings_Odds 
                    ''', 'Player_Stats')
    odds.stat_type = odds.stat_type.apply(lambda x: x.lower())
    odds.year = odds.year.apply(lambda x: int(x.replace('-', '')))

    odds = odds[odds.stat_type==metric].reset_index(drop=True)
    odds = odds.loc[odds.over_under=='over', ['player', 'year', 'value']]

    return odds

def get_over_under_class(df, metric):
    
    odds = pull_odds(metric)
    df = pd.merge(df, odds, on=['player', 'year'])
    df['y_act'] = np.where(df.y_act > df.value, 1, 0)
    df = create_value_columns(df, metric)

    return df

#----------------
# Modeling Functions
#----------------

def output_dict():
    return {'pred': {}, 'actual': {}, 'scores': {}, 'models': {}, 'full_hold':{}}


def update_output_dict(label, m, suffix, out_dict, oof_data, best_models):

    # append all of the metric outputs
    lbl = f'{label}_{m}{suffix}'
    out_dict['pred'][lbl] = oof_data['hold']
    out_dict['actual'][lbl] = oof_data['actual']
    out_dict['scores'][lbl] = oof_data['scores']
    out_dict['models'][lbl] = best_models
    out_dict['full_hold'][lbl] = oof_data['full_hold']

    return out_dict


def get_skm(skm_df, model_obj, to_drop):
    
    skm_options = {
        'reg': SciKitModel(skm_df, model_obj='reg', r2_wt=r2_wt, sera_wt=sera_wt, mse_wt=mse_wt),
        'class': SciKitModel(skm_df, model_obj='class', brier_wt=brier_wt, matt_wt=matt_wt),
        'quantile': SciKitModel(skm_df, model_obj='quantile')
    }
    
    skm = skm_options[model_obj]
    X, y = skm.Xy_split(y_metric='y_act', to_drop=to_drop)

    return skm, X, y


def get_full_pipe(skm, m, alpha=None, stack_model=False, min_samples=10):

    if m == 'adp':
        
        
        # set up the ADP model pipe
        pipe = skm.model_pipe([skm.piece('feature_select'), 
                               skm.piece('std_scale'), 
                               skm.piece('k_best'),
                               skm.piece('lr')])

    elif stack_model:
        pipe = skm.model_pipe([
                            skm.piece('std_scale'),
                            skm.piece('k_best'), 
                            skm.piece(m)
                        ])

    elif skm.model_obj == 'reg':
        pipe = skm.model_pipe([skm.piece('random_sample'),
                                skm.piece('std_scale'), 
                                skm.piece('select_perc'),
                                skm.feature_union([
                                                skm.piece('agglomeration'), 
                                                skm.piece('k_best'),
                                                skm.piece('pca')
                                                ]),
                                skm.piece('k_best'),
                                skm.piece(m)])
        
    elif skm.model_obj == 'class':
        pipe = skm.model_pipe([skm.piece('random_sample'),
                               skm.piece('std_scale'), 
                               skm.piece('select_perc_c'),
                               skm.feature_union([
                                                skm.piece('agglomeration'), 
                                                skm.piece('k_best_c'),
                                                ]),
                               skm.piece('k_best_c'),
                               skm.piece(m)])
    
    elif skm.model_obj == 'quantile':
        pipe = skm.model_pipe([
                                skm.piece('random_sample'),
                                skm.piece('std_scale'), 
                                skm.piece('k_best'), 
                                skm.piece(m)
                                ])

        if m == 'qr_q': pipe.steps[-1][-1].quantile = alpha
        elif m in ('rf_q', 'knn_q'): pipe.steps[-1][-1].q = alpha
        else: pipe.steps[-1][-1].alpha = alpha
    


    # get the params for the current pipe and adjust if needed
    params = skm.default_params(pipe, 'rand')
    if m=='knn': params['knn__n_neighbors'] = range(1, min_samples-1)
    if m=='knn_c': params['knn_c__n_neighbors'] = range(1, min_samples-1)
    if m=='knn_q': params['knn_q__n_neighbors'] = range(1, min_samples-1)
    if stack_model: params['k_best__k'] = range(2, 40)

    return pipe, params


def get_model_output(model_name, cur_df, model_obj, out_dict, run_params, i, min_samples=10, alpha=''):

    print(f'\n{model_name}\n============\n')

    skm, X, y = get_skm(cur_df, model_obj, to_drop=run_params['drop_cols'])
    pipe, params = get_full_pipe(skm, model_name, alpha, min_samples=min_samples)
    
    if model_obj == 'class': proba = True
    else: proba = False

    # fit and append the ADP model
    import time
    start = time.time()
    best_models, oof_data, _ = skm.time_series_cv(pipe, X, y, params, n_iter=run_params['n_iters'], 
                                                  n_splits=run_params['n_splits'], col_split='game_date', 
                                                  bayes_rand='custom_rand', time_split=run_params['cv_time_input'],
                                                  proba=proba, random_seed=(i+7)*19+(i*12)+6, alpha=alpha)

    print('Time Elapsed:', np.round((time.time()-start)/60,1), 'Minutes')
    
    col_label = str(alpha)
    out_dict = update_output_dict(model_obj, model_name, col_label, out_dict, oof_data, best_models)

    return out_dict, best_models, oof_data

#-----------------
# Saving Data / Handling
#-----------------

def save_pickle(obj, path, fname, protocol=-1):
    with gzip.open(f"{path}/{fname}.p", 'wb') as f:
        pickle.dump(obj, f, protocol)

    print(f'Saved {fname} to path {path}')

def load_pickle(path, fname):
    with gzip.open(f"{path}/{fname}.p", 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object

def save_output_dict(out_dict, model_output_path, label):

    save_pickle(out_dict['pred'], model_output_path, f'{label}_pred')
    save_pickle(out_dict['actual'], model_output_path, f'{label}_actual')
    save_pickle(out_dict['models'], model_output_path, f'{label}_models')
    save_pickle(out_dict['scores'], model_output_path, f'{label}_scores')
    save_pickle(out_dict['full_hold'], model_output_path, f'{label}_full_hold')



#%%

for metric in run_params['metrics']:

    print(f"\n==================\n{metric} {run_params['train_time_split']} {vers}\n====================")

    #==========
    # Pull and clean compiled data
    #==========

    # load data and filter down
    pkey, model_output_path = create_pkey_output_path(metric, run_params, vers)
    df, run_params = load_data(run_params)

    df = df.drop([c for c in df.columns if 'y_act' in c and metric not in c], axis=1)
    df = df.rename(columns={f'y_act_{metric}': 'y_act'})
    df['week'] = 1
    df['year'] = df.game_date
    df['team'] = 0

    df_train, df_predict, output_start, min_samples = train_predict_split(df, run_params)
    df_train['y_act'] = df_train.y_act + (np.random.random(size=len(df_train)) / 1000)
    df_train_class = get_over_under_class(df, metric)

    # set up blank dictionaries for all metrics
    out_reg, out_quant, out_class = output_dict(),  output_dict(), output_dict()

    #=========
    # Run Models
    #=========

    # run all models
    model_list = [ 'bridge', 'huber', 'lgbm', 'ridge', 'svr', 'lasso', 'enet', 'xgb', 'knn', 'gbm', 'gbmh', 'rf']
    for i, m in enumerate(model_list):
        out_reg, _, _ = get_model_output(m, df_train, 'reg', out_reg, run_params, i, min_samples)
    save_output_dict(out_reg, model_output_path, 'reg')

    model_list = ['lr_c', 'xgb_c',  'lgbm_c', 'gbm_c', 'rf_c', 'knn_c', 'gbmh_c'] 
    for i, m in enumerate(model_list):
        out_class, _, _= get_model_output(m, df_train_class, 'class', out_class, run_params, i, min_samples)
    save_output_dict(out_class, model_output_path, 'class')

    # run all other models
    model_list = ['gbm_q', 'lgbm_q', 'qr_q', 'knn_q', 'rf_q']
    for i, m in enumerate(model_list):
        for alph in [0.1, 0.25, 0.75, 0.9]:
            out_quant, _, _ = get_model_output(m, df_train, 'quantile', out_quant, run_params, i, alpha=alph)
    save_output_dict(out_quant, model_output_path, 'quant')
# %%
