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

from hyperopt.pyll import scope
from hyperopt import Trials, hp
from wakepy import keep
from joblib import Parallel, delayed
import time

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

verbosity=50
run_params = {
    
    # set year and week to analyze
    'cv_time_input_back_days': 40,
    'train_time_split': '2024-03-08',
    'metrics': [
                 #'points', 'assists', 'rebounds',
                #   'three_pointers', 'points_assists',
                #   'points_rebounds','points_rebounds_assists', 
                  'assists_rebounds', 'steals_blocks','blocks', 'steals', 
                # 'total_points', 'spread'  
                ],
    'n_iters': 15,
    'n_splits': 4,
    'parlay': False,
    'opt_type': 'bayes'
}

# run_params['cv_time_input'] = int(run_params['cv_time_input'].replace('-', ''))
run_params['train_time_split'] = int(run_params['train_time_split'].replace('-', ''))

# set weights for running model
r2_wt = 0
mae_wt = 0
sera_wt = 0
mse_wt = 1
matt_wt = 0
brier_wt = 1

# set version and iterations
vers = 'mse1_brier1'

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
    df = dm.read(f'''SELECT * FROM Model_Data_{run_params['train_time_split']}''', 'Model_Features')
   
    if df.shape[1]==2000:
        df2 = dm.read(f'''SELECT * FROM Model_Data_{run_params['train_time_split']}v2''', 'Model_Features')
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

#------------------
# Create Odds Modeling Columns
#------------------


def create_metric_split_columns(df, metric_split):
    if len(metric_split)==2:
        ms1 = metric_split[0]
        ms2 = metric_split[1]

        for c in df.columns:
            if ms1 in c:
                try: df[f'{c}_{ms2}'] = df[c] + df[c.replace(ms1, ms2)]
                except: pass

    elif len(metric_split)==3:
        ms1 = metric_split[0]
        ms2 = metric_split[1]
        ms3 = metric_split[2]

        for c in df.columns:
            if ms1 in c:
                try: df[f'{c}_{ms2}_{ms3}'] = df[c] + df[c.replace(ms1, ms2)] + df[c.replace(ms1, ms3)]
                except: pass

    return df


def create_y_act(df, metric):

    if metric in ('points_assists', 'points_rebounds', 'points_rebounds_assists', 'steals_blocks', 'assists_rebounds'):
        metric_split = metric.split('_')
        df[f'y_act_{metric}'] = df[['y_act_' + c for c in metric_split]].sum(axis=1)
        df = create_metric_split_columns(df, metric_split)

    df = df.drop([c for c in df.columns if 'y_act' in c and metric not in c], axis=1)
    df = df.rename(columns={f'y_act_{metric}': 'y_act'})
    return df

def parlay_values(df):
    df['value'] = np.select([
                                df.value <= 3, 
                                (df.value > 3) & (df.value < 10),
                                (df.value >= 10) & (df.value < 24),
                                (df.value >= 24) & (df.value < 30),
                                (df.value >= 30) & (df.value < 35),
                                (df.value >= 35) & (df.value < 40),
                                (df.value >= 40) & (df.value < 45),
                                (df.value >= 45)
                            ], 
                            
                            [
                                df.value,
                                df.value - 1,
                                df.value - 2,
                                df.value - 3,
                                30,
                                35,
                                40,
                                df.value - 5
                            ]
                            )
    
    return df

def pull_odds(metric, run_params):

    odds = dm.read(f'''SELECT player, game_date year, value
                       FROM Draftkings_Odds 
                       WHERE stat_type='{metric}'
                             AND over_under='over'
                             AND decimal_odds < 2.5
                             AND decimal_odds > 1.5
                    ''', 'Player_Stats')
    odds.year = odds.year.apply(lambda x: int(x.replace('-', '')))
    if run_params['parlay']: odds = parlay_values(odds)

    return odds

def create_value_columns(df, metric):

    for c in df.columns:
        if metric in c:
            df = pd.concat([df, pd.Series(df[c]-df.value, name=f'{c}_vs_value')], axis=1)
            df = pd.concat([df, pd.Series(df[c]/df.value, name=f'{c}_over_value')], axis=1)

    return df

def remove_low_counts(df):
    cnts = df.groupby('game_date').agg({'player': 'count'})
    cnts = cnts[cnts.player > 5].reset_index().drop('player', axis=1)
    df = pd.merge(df, cnts, on='game_date')
    return df

def get_over_under_class(df, metric, run_params, model_obj='class'):
    
    odds = pull_odds(metric, run_params)
    df = pd.merge(df, odds, on=['player', 'year'])
    df = df.sort_values(by='game_date').reset_index(drop=True)

    if model_obj == 'class': df['y_act'] = np.where(df.y_act >= df.value, 1, 0)
    elif model_obj == 'reg': df['y_act'] = df.y_act - df.value

    df = create_value_columns(df, metric)
    df = remove_low_counts(df)
    df_train_class, df_predict_class, _, _ = train_predict_split(df, run_params)

    return df_train_class, df_predict_class

#----------------
# Modeling Functions
#----------------

def output_dict():
    return {'pred': {}, 'actual': {}, 'scores': {}, 'models': {}, 'full_hold':{}, 'param_scores': {}, 'trials': {}}


def get_trial_times(root_path, run_params):

    recent_save = get_newest_folder_with_keywords(f"{root_path}/Model_Outputs/", [run_params['cur_metric']], [str(run_params['train_time_split'])])
    all_trials = load_pickle(recent_save, 'all_trials')

    times = []
    for k,v in all_trials.items():
        if k!='reg_adp':
            max_trial = len(v.trials) - 1
            trial_times = []
            for i in range(max_trial-25, max_trial):
                trial_times.append(v.trials[i]['refresh_time'] - v.trials[i]['book_time'])
            trial_time = np.mean(trial_times).seconds
            times.append([k, np.round(trial_time / 60, 2)])

    time_per_trial = pd.DataFrame(times, columns=['model', 'time_per_trial']).sort_values(by='time_per_trial', ascending=False)
    time_per_trial['total_time'] = time_per_trial.time_per_trial * 25
    return time_per_trial

def calc_num_trials(time_per_trial, run_params):

    n_iters = run_params['n_iters']
    time_per_trial['percentile_90_time'] = time_per_trial.time_per_trial.quantile(0.4)
    time_per_trial['num_trials'] = n_iters * time_per_trial.percentile_90_time / time_per_trial.time_per_trial
    time_per_trial['num_trials'] = time_per_trial.num_trials.apply(lambda x: np.min([n_iters, np.max([x, 4])])).astype('int')
    
    return {k:v for k,v in zip(time_per_trial.model, time_per_trial.num_trials)}

def reg_params(df_train, min_samples, num_trials, run_params):
    model_list = ['gbm', 'gbmh', 'rf', 'xgb', 'lgbm', 'ridge', 'svr', 'lasso', 'bridge', 'enet', 'knn']
    label = 'reg'
    func_params_reg = []
    for i, m  in enumerate(model_list):
        try: num_trial = num_trials[f'{label}_{m}']
        except: num_trial = run_params['n_iters']
        func_params_reg.append([m, label, df_train, 'reg', i, min_samples, '', num_trial])

    return func_params_reg

def class_params(df_train_class, min_samples, num_trials, run_params, is_parlay=False):
    model_list = ['gbm_c', 'rf_c', 'gbmh_c', 'lgbm_c', 'lr_c', 'knn_c','xgb_c']
    func_params_c = []
    if is_parlay: label = 'parlay_class'
    else: label = 'class'
 
    for i, m  in enumerate(model_list):
        try: num_trial = num_trials[f'{label}_{m}']
        except: num_trial = run_params['n_iters']
        func_params_c.append([m, label, df_train_class, 'class', i, min_samples, '', num_trial])

    return func_params_c

def quant_params(df_train, alphas, min_samples, num_trials, run_params):
    model_list =  ['qr_q', 'lgbm_q', 'gbm_q', 'gbmh_q']
    func_params_q = []
    for alph in alphas:
        label = f'quant_{alph}'
        for i, m  in enumerate(model_list):
            try: num_trial = num_trials[f'{label}_{m}']
            except: num_trial = run_params['n_iters']
            func_params_q.append([m, label, df_train, 'quantile', i, min_samples, alph, num_trial])

    return func_params_q


def order_func_params(func_params, trial_times):
    
    if trial_times is not None:
        missing_trials = [f'{x[1]}_{x[0]}' for x in func_params if f'{x[1]}_{x[0]}' not in trial_times.values]
        trial_order = missing_trials + list(trial_times.model.values)
        func_params = sorted(func_params, key=lambda x: trial_order.index(f'{x[1]}_{x[0]}'))

    return func_params

def get_skm(skm_df, model_obj, to_drop):
    
    skm_options = {
        'reg': SciKitModel(skm_df, model_obj='reg', r2_wt=r2_wt, sera_wt=sera_wt, mse_wt=mse_wt),
        'class': SciKitModel(skm_df, model_obj='class', brier_wt=brier_wt, matt_wt=matt_wt),
        'quantile': SciKitModel(skm_df, model_obj='quantile')
    }
    
    skm = skm_options[model_obj]
    X, y = skm.Xy_split(y_metric='y_act', to_drop=to_drop)

    return skm, X, y


def get_full_pipe(skm, m, alpha=None, stack_model=False, min_samples=10, bayes_rand='rand'):

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

        if m in ('qr_q', 'gbmh_q'): pipe.set_params(**{f'{m}__quantile': alpha})
        elif m in ('rf_q', 'knn_q'): pipe.set_params(**{f'{m}__q': alpha})
        else: pipe.set_params(**{f'{m}__alpha': alpha})

    # get the params for the current pipe and adjust if needed
    params = skm.default_params(pipe, bayes_rand, min_samples=min_samples)
    if m=='adp': 
        params['feature_select__cols'] = [
                                            ['game_date', 'year', 'week', 'ProjPts', 'dk_salary', 'projected_points', 'fantasyPoints', 'ffa_points', 'avg_proj_points', 'fc_proj_fantasy_pts_fc', 'log_fp_rank', 'log_avg_proj_rank'],
                                            ['year', 'week',  'ProjPts', 'dk_salary', 'projected_points', 'fantasyPoints', 'ffa_points', 'avg_proj_points', 'fc_proj_fantasy_pts_fc', 'log_fp_rank', 'log_avg_proj_rank'],
                                            [ 'ProjPts', 'dk_salary','projected_points', 'fantasyPoints', 'ffa_points', 'avg_proj_points', 'fc_proj_fantasy_pts_fc', 'log_fp_rank', 'log_avg_proj_rank']
                                        ]
        params['k_best__k'] = range(1, 14)

    if m in ('gbm', 'gbm_c', 'gbm_q'):
        params[f'{m}__n_estimators'] = scope.int(hp.quniform('n_estimators', 20, 50, 2))
        params[f'{m}__max_depth'] = scope.int(hp.quniform('max_depth', 2, 12, 2))
        params[f'{m}__max_features'] = hp.uniform('max_features', 0.4, 0.8)
        params[f'{m}__subsample'] = hp.uniform('subsample', 0.4, 0.8)
    
    return pipe, params

def update_trials_params(trials, m, params, pipe):

    m_params = [p.split('__')[1] for p in params.keys() if m in p]

    hyper_params = {
        p: pipe.steps[-1][1].get_params()[p] for p in m_params
    }

    for trial in trials:
        # Update each trial with the new hyperparameters
        for k,v in hyper_params.items():
            if k not in trial['misc']['vals']:
                trial['misc']['vals'][k] = [v]
                trial['misc']['idxs'][k] = [trial['tid']]

    return trials

def get_proba(model_obj):
    if model_obj == 'class': proba = True
    else: proba = False
    return proba

def get_newest_folder_with_keywords(path, keywords, ignore_keywords=None):
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    
    # Apply ignore_keywords if provided
    if ignore_keywords:
        folders = [f for f in folders if not any(ignore_keyword in f for ignore_keyword in ignore_keywords)]
    
    matching_folders = [f for f in folders if all(keyword in f for keyword in keywords)]
    
    if not matching_folders:
        return None
    
    newest_folder = max(matching_folders, key=lambda f: os.path.getctime(os.path.join(path, f)))
    return os.path.join(path, newest_folder)


def get_trials(label, m, bayes_rand):

    recent_save = get_newest_folder_with_keywords(f"{root_path}/Model_Outputs/", [run_params['cur_metric']], [str(run_params['train_time_split'])])

    if recent_save is not None and bayes_rand=='bayes': 
        try:
            trials = load_pickle(recent_save, 'all_trials')
            trials = trials[f'{label}_{m}']
            print('Loading previous trials')
        except:
            print('No Previous Trials Exist')
            trials = Trials()

    elif bayes_rand=='bayes':
        print('Creating new Trials object')
        trials = Trials()

    else:
        trials = None

    return trials
    

def get_model_output(model_name, label, cur_df, model_obj, run_params, i, min_samples=10, alpha='', n_iter=20):

    print(f'\n{model_name}\n============\n')
    
    bayes_rand = run_params['opt_type']
    proba = get_proba(model_obj)
    trials = get_trials(label, model_name, bayes_rand)

    skm, X, y = get_skm(cur_df, model_obj, to_drop=run_params['drop_cols'])
    pipe, params = get_full_pipe(skm, model_name, alpha, min_samples=min_samples, bayes_rand=bayes_rand)
    if trials is not None: trials = update_trials_params(trials, model_name, params, pipe)

    # fit and append the ADP model
    start = time.time()
    best_models, oof_data, param_scores, trials = skm.time_series_cv(pipe, X, y, params, n_iter=n_iter, 
                                                                     n_splits=run_params['n_splits'], col_split='game_date', 
                                                                     time_split=run_params['cv_time_input'],
                                                                     bayes_rand=bayes_rand, proba=proba, trials=trials,
                                                                     random_seed=(i+7)*19+(i*12)+6, alpha=alpha)
    best_models = [bm.fit(X,y) for bm in best_models]
    print('Time Elapsed:', np.round((time.time()-start)/60,1), 'Minutes')
    
    return best_models, oof_data, param_scores, trials

#-----------------
# Saving Data / Handling
#-----------------


def update_output_dict(out_dict, label, m, result):

    best_models, oof_data, param_scores, trials = result

    # append all of the metric outputs
    lbl = f'{label}_{m}'
    out_dict['pred'][lbl] = oof_data['hold']
    out_dict['actual'][lbl] = oof_data['actual']
    out_dict['scores'][lbl] = oof_data['scores']
    out_dict['models'][lbl] = best_models
    out_dict['full_hold'][lbl] = oof_data['full_hold']
    out_dict['param_scores'][lbl] = param_scores
    out_dict['trials'][lbl] = trials

    return out_dict


def unpack_results(out_dict, func_params, results):
    for fp, result in zip(func_params, results):
        model_name, label, _, _, _, _, _, _ = fp
        out_dict = update_output_dict(out_dict, label, model_name, result)

    return out_dict


def save_pickle(obj, path, fname, protocol=-1):
    with gzip.open(f"{path}/{fname}.p", 'wb') as f:
        pickle.dump(obj, f, protocol)

    print(f'Saved {fname} to path {path}')


def load_pickle(path, fname):
    with gzip.open(f"{path}/{fname}.p", 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object


def save_output_dict(out_dict, label, model_output_path):

    label = label.split('_')[0]
    save_pickle(out_dict['pred'], model_output_path, f'{label}_pred')
    save_pickle(out_dict['actual'], model_output_path, f'{label}_actual')
    save_pickle(out_dict['models'], model_output_path, f'{label}_models')
    save_pickle(out_dict['scores'], model_output_path, f'{label}_scores')
    save_pickle(out_dict['full_hold'], model_output_path, f'{label}_full_hold')
    save_pickle(out_dict['param_scores'], model_output_path, f'{label}_param_scores')
    save_pickle(out_dict['trials'], model_output_path, f'{label}_trials')

def show_calibration_curve(y_true, y_pred, n_bins=10):

    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

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

    print('Brier Score:', brier_score_loss(y_true, y_pred))

#%%

for metric in run_params['metrics']:

    print(f"\n==================\n{metric} {run_params['train_time_split']} {vers}\n====================")

    #==========
    # Pull and clean compiled data
    #==========

    # load data and filter down
    pkey, model_output_path = create_pkey_output_path(metric, run_params, vers)
    df, run_params = load_data(run_params)
    
    game_dates = df.game_date.sort_values(ascending=False).unique()
    run_params['cv_time_input'] = game_dates[run_params['cv_time_input_back_days']]

    run_params['cur_metric'] = metric
    df = remove_low_counts(df)
    df = create_y_act(df, metric)
    
    df['week'] = 1
    df['year'] = df.game_date
    df['team'] = 0

    df_train, df_predict, output_start, min_samples = train_predict_split(df, run_params)
    df_train['y_act'] = df_train.y_act + (np.random.random(size=len(df_train)) / 1000)

    run_params['parlay'] = False
    df_train_class, df_predict_class = get_over_under_class(df, metric, run_params, model_obj='class')
    df_train_diff, df_predict_diff = get_over_under_class(df, metric, run_params, model_obj='reg')

    run_params['parlay'] = True
    df_train_parlay, df_predict_parlay = get_over_under_class(df, metric, run_params, model_obj='class')
    df_train_diff, df_predict_diff = get_over_under_class(df, metric, run_params, model_obj='reg')

    try:
        trial_times = get_trial_times(root_path, run_params)
        num_trials = calc_num_trials(trial_times, run_params)
        print(trial_times)
        print('Lower trials:', {k:v for k,v in num_trials.items() if v < run_params['n_iters']})
    except:
        num_trials = None
        trial_times = None
        print('No Trials Exist')

    func_params = []
    func_params.extend(quant_params(df_train, [0.25, 0.75], min_samples,  num_trials, run_params))
    func_params.extend(reg_params(df_train, min_samples, num_trials, run_params))
    func_params.extend(class_params(df_train_class, int(min_samples/10), num_trials, run_params, is_parlay=False))
    func_params.extend(class_params(df_train_parlay, int(min_samples/10), num_trials, run_params, is_parlay=True))
    func_params = order_func_params(func_params, trial_times)
    
    # run all models in parallel
    results = Parallel(n_jobs=-1, verbose=verbosity)(
                    delayed(get_model_output)
                    (m, label, df, model_obj, run_params, i, min_samples, alpha, n_iter) \
                        for m, label, df, model_obj, i, min_samples, alpha, n_iter in func_params
                    )
    
    # save output for all models
    out_dict = output_dict()
    out_dict = unpack_results(out_dict, func_params, results)
    save_output_dict(out_dict, 'all', model_output_path)

#%%

for m, label, df, model_obj, i, min_samples, alpha, n_iter in func_params[-1:]:

    model_name = m
    print(model_name)
    cur_df = df.copy()

    bayes_rand = run_params['opt_type']
    proba = get_proba(model_obj)
    trials = get_trials(label, model_name, bayes_rand)
    
    skm, X, y = get_skm(cur_df, model_obj, to_drop=run_params['drop_cols'])
    pipe, params = get_full_pipe(skm, model_name, alpha, min_samples=min_samples, bayes_rand=bayes_rand)
    if trials is not None: trials = update_trials_params(trials, model_name, params, pipe)

    # fit and append the ADP model
    start = time.time()
    best_models, oof_data, param_scores, trials = skm.time_series_cv(pipe, X, y, params, n_iter=20, 
                                                                     n_splits=run_params['n_splits'], col_split='game_date', 
                                                                     time_split=run_params['cv_time_input'],
                                                                     bayes_rand=bayes_rand, proba=proba, trials=trials,
                                                                     random_seed=(i+7)*19+(i*12)+6, alpha=alpha)

#%%

scores = load_pickle(model_output_path, 'all_param_scores')
scores['reg_gbm'].sort_values(by='score').head(20)

#%%

metric = 'steals'

# load data and filter down
pkey, model_output_path = create_pkey_output_path(metric, run_params, vers)
df, run_params = load_data(run_params)

game_dates = df.game_date.sort_values(ascending=False).unique()
run_params['cv_time_input'] = game_dates[run_params['cv_time_input_back_days']]

df = remove_low_counts(df)
df = create_y_act(df, metric)

df['week'] = 1
df['year'] = df.game_date
df['team'] = 0

df_train, df_predict, output_start, min_samples = train_predict_split(df, run_params)
df_train['y_act'] = df_train.y_act + (np.random.random(size=len(df_train)) / 1000)
df_train_class, df_predict_class = get_over_under_class(df, metric, run_params, model_obj='class')
df_train_diff, df_predict_diff = get_over_under_class(df, metric, run_params, model_obj='reg')

# set up blank dictionaries for all metrics
out_reg, out_quant, out_class, out_diff = output_dict(),  output_dict(), output_dict(), output_dict()

#%%

# 'qr_q', 'gbmh_q'): pipe.set_params(**{f'{m}__quantile': alpha})
#         elif m in ('rf_q', 'knn_q')

cur_df = df_train.copy()
model_obj = 'reg'
model_name = 'lgbm'
out_dict={}
alpha=None
i=50

skm, X, y = get_skm(cur_df, model_obj, to_drop=run_params['drop_cols'])
# pipe, params = get_full_pipe(skm, model_name, alpha, min_samples=min_samples, bayes_rand='bayes')
# pipe.set_params(**{'lgbm__objective': 'tweedie'})

pipe = skm.model_pipe([skm.piece('random_sample'),
                       skm.piece('std_scale'),
                       skm.piece('pca'),
                       skm.piece(model_name)])
params = skm.default_params(pipe, 'bayes', min_samples=min_samples)

#%%
if model_obj == 'class': proba = True
else: proba = False

print(f'\n{model_name}\n============\n')

# fit and append the ADP model
import time
start = time.time()
best_models, oof_data, _, trials = skm.time_series_cv(pipe, X, y, params, n_iter=3, trials=Trials(),
                                                n_splits=run_params['n_splits'], col_split='game_date', 
                                                bayes_rand='bayes', time_split=run_params['cv_time_input'],
                                                proba=proba, random_seed=(i+7)*19+(i*12)+6, alpha=alpha)

print('Time Elapsed:', np.round((time.time()-start)/60,1), 'Minutes')

oof_data['full_hold'].plot.scatter(x='pred', y='y_act')
# col_label = str(alpha)
# out_dict = update_output_dict(model_obj, model_name, col_label, out_dict, oof_data, best_models)

#%%

all_df = pd.DataFrame()
for i in range(50):
    cur_param = pd.DataFrame(trials.trials[i]['misc']['vals'])
    cur_loss = pd.DataFrame({'loss': trials.trials[i]['result']['loss']}, index=[0])
    cur = pd.concat([cur_param, cur_loss], axis=1)
    all_df = pd.concat([all_df, cur], axis=0)

all_df = all_df.sort_values(by='loss').reset_index(drop=True)
all_df

#%%
pipeline = best_models[1]
pipeline.fit(X,y)

# get the feature names from the original dataset
feature_names = X.columns

# get the indices of the selected features
selected_features = pipeline.named_steps['k_best_c'].get_support()

# filter the feature names to get only the selected features
selected_feature_names = feature_names[selected_features]

# get the sorted indices of the coefficients
coef_indices = np.argsort(np.abs(pipeline.named_steps['lr_c'].coef_[0]))[::-1]

# print the feature names and corresponding coefficients in order
for feature, coef in zip(selected_feature_names[coef_indices], pipeline.named_steps['lr_c'].coef_[0][coef_indices]):
    print(f'{feature}: {coef:.4f}')

# %%