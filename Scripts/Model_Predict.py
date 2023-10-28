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
import ff.data_clean as dc

from sklearn.preprocessing import StandardScaler
from Fix_Standard_Dev import *
from joblib import Parallel, delayed
from skmodel import SciKitModel
from sklearn.pipeline import Pipeline

from joblib import Parallel, delayed
from hyperopt import Trials, hp
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


root_path = ffgeneral.get_main_path('NBA_SGP')
db_path = f'{root_path}/Data/'
dm = DataManage(db_path)

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)


#====================
# Data Loading Functions
#====================

def create_pkey_output_path(metric, run_params):

    pkey = f"{metric}_{run_params['train_date']}_{run_params['pred_vers']}"
    model_output_path = f"{root_path}/Model_Outputs/{pkey}/"
    run_params['model_output_path'] = model_output_path
    if not os.path.exists(model_output_path): os.makedirs(model_output_path)
    
    return pkey, model_output_path

def load_data(run_params): 

    train_date = run_params['train_date']

    # load data and filter down
    df = dm.read(f'''SELECT * FROM Model_Data_{train_date}''', 'Model_Features')
   
    if df.shape[1]==2000:
        df2 = dm.read(f'''SELECT * FROM Model_Data_{train_date}v2''', 'Model_Features')
        df = pd.concat([df, df2], axis=1)

    df.game_date = df.game_date.apply(lambda x: int(x.replace('-', '')))
    df = df.sort_values(by=['game_date']).reset_index(drop=True)
    
    drop_cols = list(df.dtypes[df.dtypes=='object'].index)
    run_params['drop_cols'] = drop_cols
    print(drop_cols)

    return df, run_params


def train_predict_split(df, run_params):

    # # get the train / predict dataframes and output dataframe
    df_train = df[df.game_date < run_params['test_time_split']].reset_index(drop=True)
    df_train = df_train.dropna(subset=['y_act']).reset_index(drop=True)

    df_predict = df[df.game_date == run_params['test_time_split']].reset_index(drop=True)
    output_start = df_predict[['player', 'game_date']].copy().drop_duplicates()

    # get the minimum number of training samples for the initial datasets
    min_samples = int(df_train[df_train.game_date < run_params['cv_time_input']].shape[0])  
    print('Shape of Train Set', df_train.shape)

    return df_train, df_predict, output_start, min_samples



        
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
        if m in ('qr_q', 'gbmh_q'): pipe.steps[-1][-1].quantile = alpha
        elif m in ('rf_q', 'knn_q'): pipe.steps[-1][-1].q = alpha
        else: pipe.steps[-1][-1].alpha = alpha

    if stack_model=='random_full_stack' and run_params['opt_type']=='bayes': 
        params['random_sample__frac'] = hp.uniform('random_sample__frac', 0.5, 1)
    elif stack_model=='random_full_stack' and run_params['opt_type']=='rand':
        params['random_sample__frac'] = np.arange(0.5, 1, 0.05)
        # params['select_perc__percentile'] = hp.uniform('percentile', 0.5, 1)
        # params['feature_union__agglomeration__n_clusters'] = scope.int(hp.quniform('n_clusters', 2, 10, 1))
        # params['feature_union__pca__n_components'] = scope.int(hp.quniform('n_components', 2, 10, 1))

    return pipe, params



def load_all_pickles(model_output_path, label):
    models = load_pickle(model_output_path, f'{label}_models')
    full_hold = load_pickle(model_output_path, f'{label}_full_hold')
    return models, full_hold

def X_y_stack(full_hold):
    df = full_hold['reg_ridge']
    df_class = full_hold['class_lr_c']

    y = df[['player', 'year', 'y_act']]
    y_class = df_class[['player', 'year', 'y_act']]

    df = df[['player', 'year']]
    df_class = df_class[['player', 'year']]

    for k, v in full_hold.items():
        df_cur = v[['player', 'year', 'pred']].rename(columns={'pred': k})
        if 'class' not in k: df = pd.merge(df, df_cur, on=['player', 'year'])
        df_class = pd.merge(df_class, df_cur, on=['player', 'year'])

    X = df.reset_index(drop=True)
    y = pd.merge(X[['player', 'year']], y, on=['player','year'])

    X_class = df_class.reset_index(drop=True)  
    y_class = pd.merge(X_class[['player', 'year']], y_class, on=['player','year'])

    return X, X_class, y, y_class


def show_scatter_plot(y_pred, y, label='Total', r2=True):
    plt.scatter(y_pred, y)
    plt.xlabel('predictions');plt.ylabel('actual')
    plt.show()

    from sklearn.metrics import r2_score
    if r2: print(f'{label} R2:', r2_score(y, y_pred))
    else: print(f'{label} Corr:', np.corrcoef(y, y_pred)[0][1])



def load_all_stack_pred(model_output_path):

    # load the regregression predictions
    models, full_hold = load_all_pickles(model_output_path, 'all')
    X_stack, X_stack_class, y_stack, y_stack_class = X_y_stack(full_hold)

    models_reg = {k: v for k, v in models.items() if 'reg' in k}
    models_class = {k: v for k, v in models.items() if 'class' in k}
    models_quant = {k: v for k, v in models.items() if 'quant' in k}

    return X_stack, X_stack_class, y_stack, y_stack_class, models_reg, models_class, models_quant


def fit_and_predict(m, df_predict, X, y, proba):

    try:
        cols = m.steps[0][-1].columns
        cols = [c for c in cols if c in X.columns]
        X = X[cols]
        X_predict = df_predict[cols]
        m = Pipeline(m.steps[1:])
    except:
        X_predict = df_predict[X.columns]
        
    try:
        m.fit(X,y)

        if proba: cur_predict = m.predict_proba(X_predict)[:,1]
        else: cur_predict = m.predict(X_predict)
    
    except:
        cur_predict = []

    return cur_predict

def create_stack_predict(df_predict, models, X, y, proba=False):

    # create the full stack pipe with meta estimators followed by stacked model
    X_predict = pd.DataFrame()
    for k, ind_models in models.items():
   
        predictions = Parallel(n_jobs=-1, verbose=0)(delayed(fit_and_predict)(m, df_predict, X, y, proba) for m in ind_models)
        predictions = pd.Series(pd.DataFrame(predictions).T.mean(axis=1), name=k)
        X_predict = pd.concat([X_predict, predictions], axis=1)

    return X_predict

def create_output_class(df_predict, best_predictions_prob, output_teams):
    output_class = pd.concat([df_predict[['player', 'game_date', 'metric', 'value']], 
                            pd.Series(best_predictions_prob.mean(axis=1), name='prob_over')], axis=1)
    output_class = pd.merge(output_class, output_teams, on=['player'])
    
    return output_class


def get_group_col(df, teams):
    df = pd.merge(df.copy(), teams, on=['player', 'year'], how='left')
    df = df.fillna({'team': 'unknown', 'opponent': 'unknown'})
    df.loc[:, 'grp'] = df.team + df.opponent + df.year.astype('str')
    grp = df.grp.values
    df = df.drop(['team', 'opponent', 'grp'], axis=1)
    return grp


def get_stack_predict_data(df_train, df_predict, df_train_prob, df_predict_prob, run_params, 
                           models_reg, models_quant, models_class):

    _, X, y = get_skm(df_train, 'reg', to_drop=run_params['drop_cols'])
    print('Predicting Regression Models')
    X_predict = create_stack_predict(df_predict, models_reg, X, y)
    X_predict = pd.concat([df_predict[['player', 'week', 'year']], X_predict], axis=1)

    print('Predicting Quant Models')
    X_predict_quant = create_stack_predict(df_predict, models_quant, X, y)
    X_predict = pd.concat([X_predict, X_predict_quant], axis=1)

    _, X, y = get_skm(df_train_prob, 'class', to_drop=run_params['drop_cols'])
    print('Predicting Class Models')
    X_predict_class = create_stack_predict(df_predict_prob, models_class, X, y, True)
    X_predict_class = pd.concat([df_predict_prob[['player', 'week', 'year']], X_predict_class], axis=1)
    X_predict_class = pd.merge(X_predict, X_predict_class, on=['player', 'week', 'year'])

    return X_predict, X_predict_class


def stack_predictions(X_predict, best_models, final_models, model_obj='reg'):
    
    predictions = pd.DataFrame()
    for bm, fm in zip(best_models, final_models):
        
        if model_obj in ('reg', 'quantile'): cur_prediction = np.round(bm.predict(X_predict), 2)
        elif model_obj=='class': cur_prediction = np.round(bm.predict_proba(X_predict)[:,1], 3)
        
        cur_prediction = pd.Series(cur_prediction, name=fm)
        predictions = pd.concat([predictions, cur_prediction], axis=1)

    return predictions


def best_average_models(scores, final_models, y_stack, stack_val_pred, predictions, model_obj, min_include = 3, wts=None):
    
    skm, _, _ = get_skm(df_train, model_obj=model_obj, to_drop=[])
    
    n_scores = []
    models_included = []
    for i in range(len(scores)-min_include+1):
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=False)[:i+min_include]
        models_included.append(top_n)
        model_idx = np.array(final_models)[top_n]
        
        n_score = skm.custom_score(y_stack, stack_val_pred[model_idx].mean(axis=1), sample_weight=wts)
        n_scores.append(n_score)
        
    print('All Average Scores:', np.round(n_scores, 3))
    best_n = np.argmin(n_scores)
    best_score = n_scores[best_n]
    top_models = models_included[best_n]

    model_idx = np.array(final_models)[top_models]

    print('Top Models:', model_idx)
    best_val = stack_val_pred[model_idx]
    best_predictions = predictions[model_idx]

    return best_val, best_predictions, best_score


def average_stack_models(scores, final_models, y_stack, stack_val_pred, predictions, model_obj, show_plot=True, min_include=3, wts=None):
    
    best_val, best_predictions, best_score = best_average_models(scores, final_models, y_stack, stack_val_pred, predictions, 
                                                                 model_obj=model_obj, min_include=min_include, wts=wts)
    
    if show_plot:
        show_scatter_plot(best_val.mean(axis=1), y_stack, r2=True)
    if show_plot and model_obj=='class':
        show_calibration_curve(y_stack, best_val.mean(axis=1), n_bins=6)

    
    return best_val, best_predictions, best_score


def create_output(output_start, predictions, labels):

    output = output_start.copy()
    for lab, pred in zip(labels, predictions):
        output[lab] = pred.mean(axis=1)

    return output


def add_actual(df):

    metric_lookup = {
        'points': 'PTS',
        'rebounds': 'REB',
        'assists': 'AST',
        'three_pointers': 'FG3M'
    }

    stat_pull = metric_lookup[metric]

    d = str(run_params['test_time_split'])
    actual_pts = dm.read(f'''SELECT PLAYER_NAME player, {d} as game_date, {stat_pull} actual_{metric} 
                             FROM Box_Score 
                             WHERE game_date='{dt.date(int(d[:4]), int(d[4:6]), int(d[6:]))}' 
                            ''', 'Player_Stats')
                
    if len(actual_pts) > 0:
        df = pd.merge(df, actual_pts, on=['player', 'game_date'], how='left')
    return df


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

#-----------------------
# Over Under Classification
#-----------------------

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


def pull_odds(metric, parlay=False):

    odds = dm.read(f'''SELECT player, game_date year, value, decimal_odds
                       FROM Draftkings_Odds 
                       WHERE stat_type='{metric}'
                             AND over_under='over'
                             AND decimal_odds < 2.5
                             AND decimal_odds > 1.5
                    ''', 'Player_Stats')
    
    if parlay: odds = parlay_values(odds)
        
    odds.year = odds.year.apply(lambda x: int(x.replace('-', '')))

    return odds

def create_value_columns(df, metric):

    for c in df.columns:
        if metric in c:
            df[c + '_vs_value'] = df[c] - df.value
            df[c + '_over_value'] = df[c] / df.value

    return df

def get_over_under_class(df, metric, run_params, model_obj='class'):
    
    odds = pull_odds(metric, run_params['parlay'])
    df = pd.merge(df, odds, on=['player', 'year'])

    if model_obj == 'class': df['y_act'] = np.where(df.y_act >= df.value, 1, 0)
    elif model_obj == 'reg': df['y_act'] = df.y_act - df.value

    df = create_value_columns(df, metric)
    df_train_class, df_predict_class, _, _ = train_predict_split(df, run_params)

    return df_train_class, df_predict_class

def add_dk_lines(df):
    lines = dm.read('''SELECT player, game_date, stat_type metric, decimal_odds
                       FROM Draftkings_Odds
                       WHERE over_under='over'
                    ''', 'Player_Stats')
    lines.game_date = lines.game_date.apply(lambda x: int(x.replace('-', '')))
    df = pd.merge(df, lines, on=['player', 'game_date', 'metric'])
    return df

def X_y_stack_class(df, metric, run_params, pickle_name='class'):

    df_train_prob, df_predict_prob = get_over_under_class(df, metric, run_params)
    _, _, models_prob, _, full_hold_prob = load_all_pickles(model_output_path, pickle_name)
    X_stack_prob, y_stack_prob, df_labels = X_y_stack('class', full_hold_prob)
    _, X_prob, y_prob = get_skm(df_train_prob, 'class', to_drop=run_params['drop_cols'])
    X_predict_prob = create_stack_predict(df_predict_prob, models_prob, X_prob, y_prob, proba=True)
    
    df_train_diff, df_predict_diff = get_over_under_class(df, metric, run_params, model_obj='reg')
    _, _, models_diff, _, full_hold_diff = load_all_pickles(model_output_path, 'diff')
    X_stack_diff, _, _ = X_y_stack('reg', full_hold_diff)
    _, X_diff, y_diff = get_skm(df_train_diff, 'reg', to_drop=run_params['drop_cols'])
    X_predict_diff = create_stack_predict(df_predict_diff, models_diff, X_diff, y_diff)
    X_stack_diff.columns = [c.replace('reg', 'diff') for c in X_stack_diff.columns]
    X_predict_diff.columns = [c.replace('reg', 'diff') for c in X_predict_diff.columns]
    
    X_stack_prob = pd.concat([df_labels[['player', 'week', 'year']], X_stack_prob, 
                              X_stack_diff
                              ], axis=1)

    X_predict_prob = pd.concat([df_predict_prob[['player', 'year']], X_predict_prob, 
                                X_predict_diff
                                ], axis=1)
    
    X_stack_prob = pd.merge(X_stack_prob, df_train_prob[['player', 'year', 'value', 'decimal_odds']], on=['player', 'year'])
    X_predict_prob = pd.merge(X_predict_prob, df_predict_prob[['player', 'year', 'value', 'decimal_odds']], on=['player', 'year'])

    return df_predict_prob, X_stack_prob, y_stack_prob, X_predict_prob

def create_value_columns(df, metric):

    for c in df.columns:
        if metric in c:
            df[c + '_vs_value'] = df[c] - df.value
            df[c + '_over_value'] = df[c] / df.value

    return df


def join_train_features(X_stack_player, X_stack_class, y_stack_class):

    X_stack_class = pd.merge(X_stack_class, X_stack_player, on=['player', 'week', 'year'], how='left')
    X_stack_class = X_stack_class.drop(['player', 'week', 'year'], axis=1).dropna(axis=0)
    y_stack_class = y_stack_class[y_stack_class.index.isin(X_stack_class.index)]
    X_stack_class, y_stack_class = X_stack_class.reset_index(drop=True), y_stack_class.reset_index(drop=True)

    return X_stack_class, y_stack_class

def join_predict_features(df_predict, X_predict, X_predict_class):
    X_predict_player = pd.concat([df_predict[['player', 'year']], X_predict.copy()], axis=1)
    X_predict_player = X_predict_player.loc[:,~X_predict_player.columns.duplicated()].copy()
    X_predict_class = pd.merge(X_predict_class, X_predict_player, on=['player', 'year'])
    X_predict_class = X_predict_class.drop(['player', 'year'], axis=1)
    return X_predict_class

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

def col_ordering(X):
    col_order = [c for c in X.columns if 'reg' in c]
    col_order.extend([c for c in X.columns if 'class' in c])
    col_order.extend([c for c in X.columns if 'quant' in c])
    return X[col_order]

def save_pickle(obj, path, fname, protocol=-1):
    with gzip.open(f"{path}/{fname}.p", 'wb') as f:
        pickle.dump(obj, f, protocol)

    print(f'Saved {fname} to path {path}')

def load_pickle(path, fname):
    with gzip.open(f"{path}/{fname}.p", 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object

def get_newest_folder_with_keywords(path, keywords, ignore_keywords=None, req_fname=None):
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    
    # Apply ignore_keywords if provided
    if ignore_keywords:
        folders = [f for f in folders if not any(ignore_keyword in f for ignore_keyword in ignore_keywords)]
    
    matching_folders = [f for f in folders if all(keyword in f for keyword in keywords)]
    
    if req_fname is not None:
        matching_folders = [f for f in matching_folders if os.path.isfile(os.path.join(path, f, req_fname))]
    
    if not matching_folders:
        return None
    
    newest_folder = max(matching_folders, key=lambda f: os.path.getctime(os.path.join(path, f)))
    return os.path.join(path, newest_folder)

def get_trial_times(root_path, fname, run_params):

    recent_save = get_newest_folder_with_keywords(f"{root_path}/Model_Outputs/", [run_params['cur_metric']], [str(run_params['train_date'])])
    all_trials = load_pickle(recent_save, fname)['trials']

    times = []
    for k,v in all_trials.items():
        if k!='reg_adp':
            max_trial = len(v.trials) - 1
            trial_times = []
            for i in range(max_trial-50, max_trial):
                trial_times.append(v.trials[i]['refresh_time'] - v.trials[i]['book_time'])
            trial_time = np.mean(trial_times).seconds
            times.append([k, np.round(trial_time / 60, 4)])

    time_per_trial = pd.DataFrame(times, columns=['model', 'time_per_trial']).sort_values(by='time_per_trial', ascending=False)
    time_per_trial['total_time'] = time_per_trial.time_per_trial * 50
    return time_per_trial


def calc_num_trials(time_per_trial, run_params):

    n_iters = run_params['n_iters']
    time_per_trial['percentile_90_time'] = time_per_trial.time_per_trial.quantile(0.6)
    time_per_trial['num_trials'] = n_iters * (time_per_trial.percentile_90_time + 0.001) / (time_per_trial.time_per_trial +  0.001)
    time_per_trial['num_trials'] = time_per_trial.num_trials.apply(lambda x: np.min([n_iters, np.max([x, n_iters/10])])).astype('int')
    
    return {k:v for k,v in zip(time_per_trial.model, time_per_trial.num_trials)}


def get_proba_adp_coef(model_obj, final_m, run_params):
    if model_obj == 'class': proba = True
    else: proba = False

    run_adp = 'False'

    if ('gbmh' in final_m 
        or 'knn' in final_m 
        or 'full_stack' in run_params['stack_model'] 
        or run_params['opt_type']=='bayes'): print_coef = False
    else: print_coef = run_params['print_coef']

    return proba, run_adp, print_coef


def get_trials(fname, final_m, bayes_rand):

    recent_save = get_newest_folder_with_keywords(f"{root_path}/Model_Outputs/", [run_params['cur_metric']], [str(run_params['train_date'])])
    if recent_save is not None and bayes_rand=='bayes': 
        try:
            trials = load_pickle(recent_save, fname)
            trials = trials['trials'][final_m]
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

def run_stack_models(fname, final_m, i, model_obj, alpha, X_stack, y_stack, run_params, num_trials, is_million=None, wt_col=None):

    print(f'\n{final_m}')

    min_samples = int(len(y_stack)/10)
    proba, _, print_coef = get_proba_adp_coef(model_obj, final_m, run_params)
    skm, _, _ = get_skm(pd.concat([X_stack, y_stack], axis=1), model_obj, to_drop=[])

    pipe, params = get_full_pipe(skm, final_m, stack_model=run_params['stack_model'], alpha=alpha, 
                                 min_samples=min_samples, bayes_rand=run_params['opt_type'])
    
    trials = get_trials(fname, final_m, run_params['opt_type'])
    try: n_iter = num_trials[final_m]
    except: n_iter = run_params['n_iters']

    best_model, stack_scores, stack_pred, trial = skm.best_stack(pipe, params, X_stack, y_stack, 
                                                                n_iter=n_iter, alpha=alpha, wt_col=wt_col,
                                                                trials=trials, bayes_rand=run_params['opt_type'],
                                                                run_adp=False, print_coef=print_coef,
                                                                proba=proba, num_k_folds=run_params['num_k_folds'],
                                                                random_state=(i*2)+(i*7))
    
    return best_model, stack_scores, stack_pred, trial

def get_func_params(model_obj, alpha):

    model_list = {
        'reg': ['rf', 'gbm', 'gbmh', 'huber', 'xgb', 'lgbm', 'knn', 'ridge', 'lasso', 'bridge'],
        'class': ['rf_c', 'gbm_c', 'gbmh_c', 'xgb_c','lgbm_c', 'knn_c', 'lr_c'],
        'quantile': ['qr_q', 'gbm_q', 'lgbm_q', 'gbmh_q', 'rf_q']#, 'knn_q']
    }

    func_params = [[m, i, model_obj, alpha] for i, m  in enumerate(model_list[model_obj])]

    return model_list[model_obj], func_params

def unpack_results(model_list, results):
    best_models = [r[0] for r in results]
    scores = [r[1]['stack_score'] for r in results]
    stack_val_pred = pd.concat([pd.Series(r[2]['stack_pred'], name=m) for r,m in zip(results, model_list)], axis=1)
    trials = {m: r[3] for m, r in zip(model_list, results)}
    return best_models, scores, stack_val_pred, trials
    
def cleanup_X_y(X, y):
    X_player = X[['player', 'team', 'week', 'year']].copy()
    X = X.drop(['player', 'team', 'week', 'year'], axis=1).dropna(axis=0)
    y = y[y.index.isin(X.index)].y_act
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    X = col_ordering(X)
    X_player = pd.concat([X_player, X], axis=1)
    return X_player, X, y


def save_stack_runs(model_output_path, fname, best_models, scores, stack_val_pred, trials):
    stack_out = {}
    stack_out['best_models'] = best_models
    stack_out['scores'] = scores
    stack_out['stack_val_pred'] = stack_val_pred
    stack_out['trials'] = trials
    save_pickle(stack_out, model_output_path, fname, protocol=-1)

def load_stack_runs(model_output_path, fname):

    stack_in = load_pickle(model_output_path, fname)
    return stack_in['best_models'], stack_in['scores'], stack_in['stack_val_pred']

def remove_knn_rf_q(X):
    return X[[c for c in X.columns if 'knn_q' not in c and 'rf_q' not in c]]

def load_run_models(run_params, X_stack, y_stack, X_predict, model_obj, alpha=None, is_parlay=False):
    
    if is_parlay: model_obj_label = 'is_parlay'
    else: model_obj_label = model_obj

    if model_obj=='reg': ens_vers = run_params['reg_ens_vers']
    elif model_obj=='class': ens_vers = run_params['class_ens_vers']
    elif model_obj=='quantile': 
        ens_vers = run_params['quant_ens_vers']
        model_obj_label = f"{model_obj_label}_{alpha}"
    

    path = run_params['model_output_path']
    fname = f"{model_obj_label}_{run_params['cur_metric']}_{ens_vers}"    
    model_list, func_params = get_func_params(model_obj, alpha)

    try:
        time_per_trial = get_trial_times(root_path, fname, run_params)
        print(time_per_trial)
        num_trials = calc_num_trials(time_per_trial, run_params)
    except: 
        num_trials = {m: run_params['n_iters'] for m in model_list}
    
    print(num_trials)
    print(path, fname)

    if os.path.exists(f"{path}/{fname}.p"):
        best_models, scores, stack_val_pred = load_stack_runs(path, fname)
    
    else:
        
        if run_params['opt_type']=='bayes':
            results = Parallel(n_jobs=-1, verbose=50)(
                            delayed(run_stack_models)
                            (fname, final_m, i, model_obj, alpha, X_stack, y_stack, run_params, num_trials, is_parlay) 
                            for final_m, i, model_obj, alpha in func_params
                            )
            best_models, scores, stack_val_pred, trials = unpack_results(model_list, results)
            save_stack_runs(path, fname, best_models, scores, stack_val_pred, trials)

        elif run_params['opt_type']=='rand':
            best_models = []; scores = []; stack_val_pred = pd.DataFrame()
            for final_m, i, model_obj, alpha in func_params:
                best_model, stack_scores, stack_pred, trials = run_stack_models(fname, final_m, i, model_obj, alpha, X_stack, y_stack, run_params, num_trials, is_million)
                best_models.append(best_model)
                scores.append(stack_scores['stack_score'])
                stack_val_pred = pd.concat([stack_val_pred, pd.Series(stack_pred['stack_pred'], name=final_m)], axis=1)
            
            save_stack_runs(path, fname, best_models, scores, stack_val_pred, trials)

    X_predict = X_predict[X_stack.columns]
    predictions = stack_predictions(X_predict, best_models, model_list, model_obj=model_obj)
    best_val, best_predictions, _ = average_stack_models(scores, model_list, y_stack, stack_val_pred, 
                                                         predictions, model_obj=model_obj, 
                                                         show_plot=run_params['show_plot'], 
                                                         min_include=run_params['min_include'])

    return best_val, best_predictions


# for backfilling events
def create_metric_split_columns_backfill(df, metric_split):
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


def create_y_act_backfill(df, metric):

    if metric in ('points_assists', 'points_rebounds', 'points_rebounds_assists', 'steals_blocks', 'assists_rebounds'):
        metric_split = metric.split('_')
        df[f'y_act_{metric}'] = df[['y_act_' + c for c in metric_split]].sum(axis=1)
        df = create_metric_split_columns_backfill(df, metric_split)

    return df

def get_all_past_results(run_params):

    attach_pts, run_params = load_data(run_params)
    for metric in ['points_assists', 'points_rebounds', 'points_rebounds_assists', 'steals_blocks', 'assists_rebounds']:
        attach_pts = create_y_act_backfill(attach_pts, metric)

    attach_cols = ['player', 'game_date']
    attach_cols.extend([c for c in attach_pts.columns if 'y_act' in c])
    attach_pts = attach_pts[attach_cols]
    attach_pts.columns = [c.replace('y_act_', '') for c in attach_pts.columns]
    attach_pts = pd.melt(attach_pts, id_vars=['player', 'game_date'], var_name=['metric'], value_name='y_act_fill')
    attach_pts = attach_pts[attach_pts.metric!='points_assists_rebounds']
    return attach_pts

#%%
#==========
# General Setting
#==========

#---------------
# Settings
#---------------

run_params = {
    
    # set year and week to analyze
    'cv_time_input_orig': '2023-02-20',
    'train_date_orig': '2023-03-14',
    'test_time_split_orig': '2023-03-15',

    'metrics':  [
                 'points','assists', 'rebounds', 'three_pointers',   
                 'points_assists', 'points_rebounds',
                 'points_rebounds_assists', 'assists_rebounds',
                 'steals_blocks', 'steals', #'blocks', 
               
                ],

    # opt params
    'opt_type': 'bayes',
    'n_iters': 50,
    
    'n_splits': 5,
    'num_k_folds': 3,
    'show_plot': True,
    'print_coef': True,
    'min_include': 2,

    # set version and iterations
    'pred_vers': 'mse1_brier1',
    'stack_model': 'random_kbest',
    'stack_model_class': 'random_kbest',

    'parlay': False,
    'std_dev_type': 'pred_quant_spline'

}

s_mod = run_params['stack_model']
prob_mod = run_params['stack_model_class']
min_inc = run_params['min_include']
kfold = run_params['num_k_folds']

r2_wt = 0
sera_wt = 0
mse_wt = 0
mae_wt = 1
brier_wt = 1
matt_wt = 0

alpha = 80
class_cut = 80

set_weeks = [6]

pred_vers = 'mse1_brier1'
reg_ens_vers = f"{s_mod}_mae{mae_wt}_rsq{r2_wt}_mse{mse_wt}_include{min_inc}_kfold{kfold}"
quant_ens_vers = f"{s_mod}_include{min_inc}_kfold{kfold}"
class_ens_vers = f"{prob_mod}_matt{matt_wt}_brier{brier_wt}_include{min_inc}_kfold{kfold}"

run_params['pred_vers'] = pred_vers
run_params['reg_ens_vers'] = reg_ens_vers
run_params['quant_ens_vers'] = quant_ens_vers
run_params['class_ens_vers'] = class_ens_vers

run_params['cv_time_input'] = int(run_params['cv_time_input_orig'].replace('-', ''))
run_params['train_date'] = int(run_params['train_date_orig'].replace('-', ''))
run_params['test_time_split'] = int(run_params['test_time_split_orig'].replace('-', ''))

teams = dm.read("SELECT player, game_date year, team, opponent FROM FantasyData", 'Player_Stats')
teams.year = teams.year.apply(lambda x: int(x.replace('-', '')))

# %%

for te_date, tr_date in [
                            # ['2023-03-21', '2023-03-14'],
                            # ['2023-03-22', '2023-03-14'],
                            # ['2023-03-23', '2023-03-14'],
                            # ['2023-03-24', '2023-03-14'],
                            # ['2023-03-25', '2023-03-14'],
                            # ['2023-03-26', '2023-03-14'],
                            # ['2023-03-27', '2023-03-14'],
                            # ['2023-10-24', '2023-03-14'],
                            # ['2023-10-25', '2023-03-14'],
                            # ['2023-10-26', '2023-03-14'],
                            ['2023-10-27', '2023-03-14'],
]:
    
    run_params['train_date'] = int(tr_date.replace('-', ''))
    run_params['test_time_split'] = int(te_date.replace('-', ''))

    for metric in run_params['metrics']:

        # load data and filter down
        pkey, model_output_path = create_pkey_output_path(metric, run_params)
        df, run_params = load_data(run_params)
        run_params['cur_metric'] = metric
        output_teams = df.loc[df.game_date==run_params['test_time_split'], ['player', 'team', 'opponent']]

        df = create_y_act(df, metric)
        df['week'] = 1
        df['year'] = df.game_date
        df['team'] = 0

        df_train, df_predict, output_start, min_samples = train_predict_split(df, run_params)
        df_train_prob, df_predict_prob = get_over_under_class(df, metric, run_params)

        # set up blank dictionaries for all metrics
        out_reg, out_class, out_quant, out_million = {}, {}, {}, {}

        #------------
        # Run the Stacking Models and Generate Output
        # #------------

        # get the training data for stacking and prediction data after stacking
        X_stack, X_stack_class, y_stack, y_stack_class, models_reg, models_class, models_quant = load_all_stack_pred(model_output_path)

        X_predict, X_predict_class = get_stack_predict_data(df_train, df_predict, df_train_prob, df_predict_prob, run_params, 
                                                            models_reg, models_quant, models_class)

        X_stack_player = X_stack.copy()
        X_predict_class_player = X_predict_class.copy()
        X_stack = X_stack.drop(['player', 'year'], axis=1)
        X_predict = X_predict.drop(['player', 'week', 'year'], axis=1)
        y_stack = y_stack.drop(['player', 'year'], axis=1).y_act

        odds = pull_odds(metric, run_params['parlay'])
        X_stack_class = pd.merge(X_stack_class, odds, on=['player', 'year'])
        X_predict_class = pd.merge(X_predict_class, odds, on=['player', 'year'])

        X_stack_class = create_value_columns(X_stack_class, metric)
        X_predict_class = create_value_columns(X_predict_class, metric)

        y_stack_class = pd.merge(y_stack_class, X_stack_class[['player', 'year']], on=['player', 'year'])
        y_stack_class = y_stack_class.drop(['player', 'year'], axis=1).y_act

        X_stack_class = X_stack_class.drop(['player', 'year'], axis=1)
        X_predict_class = X_predict_class.drop(['player', 'week', 'year'], axis=1)

        X_stack_class = create_value_compare_col(X_stack_class)
        X_predict_class = create_value_compare_col(X_predict_class)

        if X_predict_class.shape[0] > 0:
            best_val_prob, best_pred_prob = load_run_models(run_params, X_stack_class, y_stack_class, X_predict_class, 'class', alpha=None)         
            best_val_mean, best_pred_mean = load_run_models(run_params, X_stack, y_stack, X_predict, 'reg', alpha=None)
            best_val_q25, best_pred_q25 = load_run_models(run_params, X_stack, y_stack, X_predict, 'quantile', alpha=0.25)
            best_val_q50, best_pred_q50 = load_run_models(run_params, X_stack, y_stack, X_predict, 'quantile', alpha=0.5)
            best_val_q75, best_pred_q75 = load_run_models(run_params, X_stack, y_stack, X_predict, 'quantile', alpha=0.75)

            preds = [best_pred_mean, best_pred_q25, best_pred_q50, best_pred_q75]
            labels = ['pred_mean', 'pred_q25', 'pred_q50', 'pred_q75']
            output = create_output(output_start, preds, labels)
            output_prob = create_output_class(df_predict_prob.assign(metric=metric), best_pred_prob, output_teams)
            output_prob = pd.merge(output_prob, output, on=['player', 'game_date'])
            output_prob = add_dk_lines(output_prob)

            output_prob = pd.merge(output_prob, df_predict[['player', 'game_date', 'y_act']], on=['player', 'game_date'], how='left')
            output_prob['y_act_prob'] = np.where(output_prob.y_act >= output_prob.value, 1, 0)
            output_prob = output_prob[['player', 'game_date', 'team', 'opponent', 'metric', 'decimal_odds', 'value', 
                                        'prob_over', 'y_act_prob', 'y_act', 'pred_mean', 'pred_q25', 'pred_q50', 'pred_q75']]
            output_prob = output_prob.assign(pred_vers=run_params['pred_vers'], ens_vers=run_params['class_ens_vers'], 
                                            train_date=run_params['train_date'], parlay=run_params['parlay'])
            output_prob = np.round(output_prob,3).sort_values(by='prob_over', ascending=False)
            display(output_prob)

            del_str = f'''metric='{metric}'
                        AND game_date={run_params['test_time_split']} 
                        AND pred_vers='{run_params['pred_vers']}'
                        AND ens_vers='{run_params['class_ens_vers']}'
                        AND train_date={run_params['train_date']}
                        AND parlay={run_params['parlay']}
                        '''
            dm.delete_from_db('Simulation', 'Over_Probability_New', del_str, create_backup=False)
            dm.write_to_db(output_prob,'Simulation', 'Over_Probability_New', 'append')

#%%

past_pred = dm.read('''SELECT * 
                        FROM Over_Probability_New 
                        ''', 'Simulation')
attach_pts = get_all_past_results(run_params)

print(past_pred.shape[0])
past_pred = pd.merge(past_pred, attach_pts, on=['player', 'game_date', 'metric'])
print(past_pred.shape[0])
past_pred['y_act_prob_fill'] = np.where(past_pred.y_act_fill > past_pred.value, 1, 0)

missing_idx = past_pred.loc[past_pred.y_act.isnull()].index
for met in ['y_act', 'y_act_prob']:
    past_pred.loc[past_pred.index.isin(missing_idx), met] =  past_pred.loc[past_pred.index.isin(missing_idx), f'{met}_fill']
past_pred = past_pred.drop(['y_act_fill', 'y_act_prob_fill'], axis=1)
past_pred = past_pred.sort_values(by='game_date').reset_index(drop=True)

past_pred[past_pred.game_date==20231026]
dm.write_to_db(past_pred,'Simulation', 'Over_Probability_New', 'replace', create_backup=True)

#%%

import copy

def get_choices_dict():

    all_choices = {}
    for start_spot in range(10):
        all_choices[start_spot] = {}
        for num_choices in [1,2,3,4,5,6]:
            all_choices[start_spot][num_choices] = []
    return all_choices

def fill_choices_dict(all_choices, preds):
    for start_spot in range(10):
        for num_choices in [1,2,3,4,5,6]:
            if preds.iloc[start_spot:start_spot+num_choices].shape[0] >= num_choices:
                wins = preds.iloc[start_spot:start_spot+num_choices].y_act.sum()
                odds = np.prod(preds.iloc[start_spot:start_spot+num_choices].decimal_odds)
                if wins == num_choices:
                    all_choices[start_spot][num_choices].append(odds)
                else:
                    all_choices[start_spot][num_choices].append(-1)
    return all_choices

def aggregate_choices(all_choices):
    for k,v in all_choices.items():
        for k2, v2 in v.items():
            all_choices[k][k2] = np.sum(v2)
    return pd.DataFrame(all_choices)


i=20
model_obj = 'class'
alpha=None
run_params['cur_metric'] = 'points'

final_ens_choices = get_choices_dict()
orig_choices = get_choices_dict()

game_dates = dm.read('''SELECT DISTINCT game_date 
                        FROM Over_Probability_New 
                        WHERE value > 2.5 
                              AND decimal_odds > 1.5
                              AND game_date > 20230315
                              AND metric NOT IN ('steals', 'blocks', 'steal_blocks')
                        ORDER BY game_date ASC''', 'Simulation').game_date.values

for test_date in game_dates:

    train_pred = dm.read('''SELECT * 
                            FROM Over_Probability_New 
                            WHERE value > 2.5 
                                  AND decimal_odds > 1.7
                            ''', 'Simulation')
    train_pred = train_pred.drop('y_act', axis=1).rename(columns={'y_act_prob': 'y_act'})

    test_pred = train_pred[train_pred.game_date == test_date]
    train_pred = train_pred[train_pred.game_date < test_date].sample(frac=1, random_state=i).reset_index(drop=True)

    X_train = train_pred[['decimal_odds', 'value', 'prob_over', 'pred_mean', 'pred_q25', 'pred_q50', 'pred_q75']].copy()
    # X_train.loc[X_train.decimal_odds > 2, 'decimal_odds'] = 2-(1/X_train.loc[X_train.decimal_odds > 2, 'decimal_odds'])
    X_train = create_value_compare_col(X_train)
    X_train = pd.concat([X_train, pd.get_dummies(train_pred.metric)], axis=1)

    X_test = test_pred[['decimal_odds', 'value', 'prob_over', 'pred_mean', 'pred_q25', 'pred_q50', 'pred_q75']].copy()
    # X_test.loc[X_test.decimal_odds > 2, 'decimal_odds'] = 2-(1/X_test.loc[X_test.decimal_odds > 2, 'decimal_odds'])
    X_test = create_value_compare_col(X_test)
    X_test = pd.concat([X_test, pd.get_dummies(test_pred.metric)], axis=1)

    y_train = train_pred.y_act

    best_model, stack_scores, stack_pred, trial = run_stack_models('test', 'lr_c', i, model_obj, alpha, 
                                                                   X_train, y_train, run_params, 50, False, wt_col='decimal_odds')
    show_calibration_curve( stack_pred['y'],stack_pred['stack_pred'])

    for c in X_train.columns:
        if c not in X_test.columns:
            X_test[c] = 0

    X_test = X_test[X_train.columns]

    best_model.fit(X_train, y_train)
    preds = pd.Series(best_model.predict_proba(X_test)[:,1], name='final_pred')
    preds = pd.concat([preds, test_pred.reset_index(drop=True)], axis=1)

    player_max_pred = preds.groupby('player').agg({'final_pred': 'max'}).reset_index()
    preds = pd.merge(preds, player_max_pred, on=['player', 'final_pred'])
    preds = preds.sort_values(by='final_pred', ascending=False).reset_index(drop=True)

    player_max_pred_orig = test_pred.groupby('player').agg({'prob_over': 'max'}).reset_index()
    preds_orig = pd.merge(test_pred, player_max_pred_orig, on=['player', 'prob_over'])
    preds_orig = preds_orig.sort_values(by='prob_over', ascending=False).reset_index(drop=True)

    final_ens_choices = fill_choices_dict(final_ens_choices, preds)
    orig_choices = fill_choices_dict(orig_choices, preds_orig)

    
display(aggregate_choices(copy.deepcopy(final_ens_choices)))
display(aggregate_choices(copy.deepcopy(orig_choices)))

#%%

rs_cols = best_model.steps[0][1].columns
cols = rs_cols[best_model.steps[2][-1].get_support()]
pd.Series(best_model.steps[-1][-1].coef_[0], cols).sort_values()

#%%

i=20
model_obj = 'class'
alpha=None
run_params['cur_metric'] = 'points'

test_date = 20231027
train_pred = dm.read('''SELECT * 
                            FROM Over_Probability_New 
                            WHERE value > 2.5 
                                  AND decimal_odds > 1.7
                            ''', 'Simulation')
                        
train_pred = train_pred.drop('y_act', axis=1).rename(columns={'y_act_prob': 'y_act'})
test_pred = train_pred[train_pred.game_date == test_date]
train_pred = train_pred[train_pred.game_date < test_date].sample(frac=1, random_state=i).reset_index(drop=True)

X_train = train_pred[['decimal_odds', 'value', 'prob_over', 'pred_mean', 'pred_q25', 'pred_q50', 'pred_q75']].copy()
# X_train.loc[X_train.decimal_odds > 2, 'decimal_odds'] = 2-(1/X_train.loc[X_train.decimal_odds > 2, 'decimal_odds'])
X_train = create_value_compare_col(X_train)
X_train = pd.concat([X_train, pd.get_dummies(train_pred.metric)], axis=1)

X_test = test_pred[['decimal_odds', 'value', 'prob_over', 'pred_mean', 'pred_q25', 'pred_q50', 'pred_q75']].copy()
# X_test.loc[X_test.decimal_odds > 2, 'decimal_odds'] = 2-(1/X_test.loc[X_test.decimal_odds > 2, 'decimal_odds'])
X_test = create_value_compare_col(X_test)
X_test = pd.concat([X_test, pd.get_dummies(test_pred.metric)], axis=1)

y_train = train_pred.y_act

best_model, stack_scores, stack_pred, trial = run_stack_models('test', 'lr_c', i, model_obj, alpha, X_train, y_train, run_params, 100, False, wt_col='decimal_odds')
show_calibration_curve(stack_pred['y'],stack_pred['stack_pred'])

for c in X_train.columns:
    if c not in X_test.columns:
        X_test[c] = 0

X_test = X_test[X_train.columns]

best_model.fit(X_train, y_train)
preds = pd.Series(best_model.predict_proba(X_test)[:,1], name='final_pred')
preds = pd.concat([preds, test_pred.reset_index(drop=True)], axis=1)
preds.sort_values(by='final_pred', ascending=False).iloc[:50]


# %%

start_spot = 5
num_choices = 3

player_max_pred = preds.groupby('player').agg({'final_pred': 'max'}).reset_index()
preds_top = pd.merge(preds, player_max_pred, on=['player', 'final_pred'])
preds_top.sort_values(by='final_pred', ascending=False).iloc[start_spot:start_spot+num_choices]
# %%
