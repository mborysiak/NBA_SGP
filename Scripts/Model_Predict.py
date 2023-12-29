#%%

# core packages
import pandas as pd
import numpy as np
import os
import gzip
import pickle
import datetime as dt
import itertools

import matplotlib.pyplot as plt
from ff.db_operations import DataManage
from ff import general as ffgeneral
import ff.data_clean as dc

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

    print('Shape of Train Set', df_train.shape)

    return df_train, df_predict, output_start



        
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

    if m in ('gbm', 'gbm_c', 'gbm_q'):
        params[f'{m}__n_estimators'] = scope.int(hp.quniform('n_estimators', 20, 80, 2))
        params[f'{m}__max_depth'] = scope.int(hp.quniform('max_depth', 2, 18, 2))
        params[f'{m}__max_features'] = hp.uniform('max_features', 0.6, 0.95)
        params[f'{m}__subsample'] = hp.uniform('subsample', 0.6, 0.95)

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
    df_train_class, df_predict_class, _ = train_predict_split(df, run_params)

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

def get_trial_times(root_path, fname, run_params):

    recent_save = f"{root_path}/Model_Outputs/{run_params['cur_metric']}_{run_params['last_train_date']}_{run_params['pred_vers']}"
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

    recent_save = f"{root_path}/Model_Outputs/{run_params['cur_metric']}_{run_params['last_train_date']}_{run_params['pred_vers']}"
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

    try:
        best_model, stack_scores, stack_pred, trial = skm.best_stack(pipe, params, X_stack, y_stack, 
                                                                    n_iter=n_iter, alpha=alpha, wt_col=wt_col,
                                                                    trials=trials, bayes_rand=run_params['opt_type'],
                                                                    run_adp=False, print_coef=print_coef,
                                                                    proba=proba, num_k_folds=run_params['num_k_folds'],
                                                                    random_state=(i*2)+(i*7))
    except:
        print(f"Error with {final_m}. Running backup model")
        
        backup_models = {
            'reg': 'ridge',
            'class': 'lr_c',
            'quantile': 'qr_q'
        }
        
        trial = trials
        backup_model = backup_models[model_obj]
        pipe, params = get_full_pipe(skm, backup_model, stack_model='kbest', alpha=alpha, 
                                    min_samples=min_samples, bayes_rand=run_params['opt_type'])

        best_model, stack_scores, stack_pred, _ = skm.best_stack(pipe, params, X_stack, y_stack, 
                                                                    n_iter=n_iter, alpha=alpha, wt_col=None,
                                                                    trials=Trials(), bayes_rand=run_params['opt_type'],
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


def remove_bad_models(model_obj, run_params, alpha, func_params, model_list):

    if (run_params['cur_metric']=='assists' and run_params['train_date']==20230328 
        and model_obj=='quantile' and run_params['stack_model']=='random_full_stack_ind_cats'):
        func_params = [f for f in func_params if f[0]!='gbm_q']
        model_list = [m for m in model_list if m!='gbm_q']
    
    if (run_params['cur_metric']=='assists' and run_params['train_date']==20230328
        and model_obj=='reg' and run_params['stack_model']=='random_full_stack_ind_cats'):
        func_params = [f for f in func_params if f[0]!='gbm']
        model_list = [m for m in model_list if m!='gbm']

    if (run_params['cur_metric']=='points_assists' and run_params['train_date']==20231111 
        and model_obj=='quantile' and alpha==0.75 and run_params['stack_model']=='random_full_stack'):
        func_params = [f for f in func_params if f[0]!='rf_q']
        model_list = [m for m in model_list if m!='rf_q']

    if (run_params['cur_metric']=='blocks' and run_params['train_date']==20231201 
        and model_obj=='quantile' and alpha==0.75 and run_params['stack_model']=='random_full_stack'):
        func_params = [f for f in func_params if f[0]!='rf_q']
        model_list = [m for m in model_list if m!='rf_q']
    
    return model_list, func_params


def remove_low_preds(predictions, stack_val_pred, model_list, scores):
    
    preds_mean_check = pd.DataFrame(predictions.median(), columns=['preds'])
    val_mean_check = pd.DataFrame(stack_val_pred.median(), columns=['vals'])
    mean_checks = pd.merge(preds_mean_check, val_mean_check, left_index=True, right_index=True)
    mean_checks['pct_diff'] = (mean_checks.preds - mean_checks.vals) / (mean_checks.vals + 0.01)
    
    print(mean_checks)
    models_pre = mean_checks.index
    for cut in np.arange(0.15, 2, 0.1):
        mean_checks_idx = mean_checks[abs(mean_checks.pct_diff) <= cut].index
        if len(mean_checks_idx) >= run_params['min_include']: break

    print("models removed:", [m for m in models_pre if m not in mean_checks_idx])

    # auto remove any predictions that are negative or 0
    good_col = []
    good_idx = []

    for i, col in enumerate(predictions.columns):
        if col in mean_checks_idx:
            good_col.append(col)
            good_idx.append(i)

    predictions = predictions[good_col]
    stack_val_pred = stack_val_pred[good_col]
    model_list = list(np.array(model_list)[good_idx])
    scores = list(np.array(scores)[good_idx])

    return predictions, stack_val_pred, model_list, scores


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

    model_list, func_params = remove_bad_models(model_obj, run_params, alpha, func_params, model_list)
    print(num_trials)
    print(path, fname)

    if os.path.exists(f"{path}/{fname}.p"):
        best_models, scores, stack_val_pred = load_stack_runs(path, fname)
    
    else:
        
        results = Parallel(n_jobs=-1, verbose=50)(
                        delayed(run_stack_models)
                        (fname, final_m, i, model_obj, alpha, X_stack, y_stack, run_params, num_trials, is_parlay) 
                        for final_m, i, model_obj, alpha in func_params
                        )
        best_models, scores, stack_val_pred, trials = unpack_results(model_list, results)
        save_stack_runs(path, fname, best_models, scores, stack_val_pred, trials)


    X_predict = X_predict[X_stack.columns]
    predictions = stack_predictions(X_predict, best_models, model_list, model_obj=model_obj)

    predictions, stack_val_pred, model_list, scores = remove_low_preds(predictions, stack_val_pred, model_list, scores)

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


def save_individual_stats(individual_cats, metric, X_stack, X_stack_class, X_predict, X_predict_class):
    individual_cats[metric]['X_stack'] = X_stack.copy()
    individual_cats[metric]['X_stack_class'] = X_stack_class.copy()
    individual_cats[metric]['X_predict'] = X_predict.copy()
    individual_cats[metric]['X_predict_class'] = X_predict_class.copy()

    for k, v in individual_cats[metric].items():
        v.columns = [f'{metric}_{c}' if c not in ('player', 'week', 'year') else c for c in v.columns ]
        individual_cats[metric][k] = v

    return individual_cats


def create_metric_split_columns_stack(df, metric, individual_cats):

    met_cols = [k for k in individual_cats.keys() if k in metric]
    model_cols = ['_'.join(m.split('_')[1:]) for m in df.columns if met_cols[0] in m]
    metric_split = metric.split('_')

    if len(metric_split)==2:
        ms1 = metric_split[0]
        ms2 = metric_split[1]

        for c in model_cols:
            try: df[f'{ms1}_{ms2}_{c}'] = df[f'{ms1}_{c}'] + df[f'{ms2}_{c}']
            except: pass

    elif len(metric_split)==3:
        ms1 = metric_split[0]
        ms2 = metric_split[1]
        ms3 = metric_split[2]

        for c in model_cols:
            try: df[f'{ms1}_{ms2}_{ms3}_{c}'] = df[f'{ms1}_{c}'] + df[f'{ms2}_{c}'] + df[f'{ms3}_{c}']
            except: pass

    return df

#%%
#==========
# General Setting
#==========

#---------------
# Settings
#---------------

run_params = {
    
    # set year and week to analyze
    'last_train_date_orig': '2023-11-11',
    'train_date_orig': '2023-12-01',
    'test_time_split_orig': dt.date.today().strftime('%Y-%m-%d'),

    'metrics':  [
                'points', 'assists', 'rebounds',
                'points_assists', 'points_rebounds', 'points_rebounds_assists', 'assists_rebounds', 
                'three_pointers',  'blocks',  'steals', 'steals_blocks', 
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
    # 'stack_model': 'random_full_stack',
    # 'stack_model_class': 'random_full_stack',
    # 'stack_model': 'random_full_stack_ind_cats',
    # 'stack_model_class': 'random_full_stack_ind_cats',
    'parlay': False,

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

run_params['train_date'] = int(run_params['train_date_orig'].replace('-', ''))
run_params['test_time_split'] = int(run_params['test_time_split_orig'].replace('-', ''))
run_params['last_train_date'] = int(run_params['last_train_date_orig'].replace('-', ''))

teams = dm.read("SELECT player, game_date year, team, opponent FROM FantasyData", 'Player_Stats')
teams.year = teams.year.apply(lambda x: int(x.replace('-', '')))

# %%

# for te_date, tr_date in [
#                             # ['2023-03-15', '2023-03-14'],
#                             # ['2023-03-16', '2023-03-14'],
#                             # ['2023-03-17', '2023-03-14'],
#                             # ['2023-03-18', '2023-03-14'],
#                             # ['2023-03-19', '2023-03-14'],
#                             # ['2023-03-20', '2023-03-14'],
#                             # ['2023-03-21', '2023-03-14'],
#                             # ['2023-03-22', '2023-03-14'],
#                             # ['2023-03-23', '2023-03-14'],
#                             # ['2023-03-24', '2023-03-14'],
#                             # ['2023-03-25', '2023-03-14'],
#                             # ['2023-03-26', '2023-03-14'],
#                             # ['2023-03-27', '2023-03-14'],
#                             # ['2023-03-28', '2023-03-28'],
#                             # ['2023-03-29', '2023-03-28'],
#                             # ['2023-03-30', '2023-03-28'],
#                             # ['2023-03-31', '2023-03-28'],
#                             # ['2023-04-01', '2023-03-28'],
#                             # ['2023-04-02', '2023-03-28'],
#                             # ['2023-04-04', '2023-03-28'],
#                             # ['2023-04-05', '2023-03-28'],
#                             # ['2023-04-06', '2023-03-28'],
#                             # ['2023-04-07', '2023-03-28'],
#                             # ['2023-10-24', '2023-03-28'],
#                             # ['2023-10-25', '2023-03-28'],
#                             # ['2023-10-26', '2023-03-28'],
#                             # ['2023-10-27', '2023-03-28'],
#                             # ['2023-10-28', '2023-03-28'],
#                             # ['2023-10-29', '2023-03-28'],
#                             # ['2023-10-30', '2023-03-28'],
#                             # ['2023-10-31', '2023-03-28'],
#                             # ['2023-11-01', '2023-10-30'],
#                             # ['2023-11-02', '2023-10-30'],
#                             # ['2023-11-03', '2023-10-30'],
#                             # ['2023-11-04', '2023-10-30'],
#                             # ['2023-11-05', '2023-10-30'],
#                             # ['2023-11-06', '2023-10-30'],
#                             # ['2023-11-08', '2023-10-30'],
#                             # ['2023-11-09', '2023-10-30'],
#                             # ['2023-11-10', '2023-10-30'],
#                             # ['2023-11-11', '2023-10-30'],
#                             # ['2023-11-12', '2023-10-30'],
#                             # ['2023-11-13', '2023-10-30'],
#                             # ['2023-11-14', '2023-10-30'],
#                             # ['2023-11-15', '2023-10-30'],
#                             # ['2023-11-16', '2023-10-30'],
#                             # ['2023-11-17', '2023-11-11'],
#                             # ['2023-11-18', '2023-11-11'],
#                             # ['2023-11-19', '2023-11-11'],
#                             # ['2023-11-20', '2023-11-11'],
#                             # ['2023-11-21', '2023-11-11'],
#                             # ['2023-11-22', '2023-11-11'],
#                             # ['2023-11-24', '2023-11-11'],
#                             # ['2023-11-25', '2023-11-11'],
#                             # ['2023-11-26', '2023-11-11'],
#                             # ['2023-11-27', '2023-11-11'],
#                             # ['2023-11-28', '2023-11-11'],
#                             # ['2023-11-29', '2023-11-11'],
#                             # ['2023-11-30', '2023-11-11'],
#                             ['2023-12-01', '2023-12-01'],
#                             ['2023-12-02', '2023-12-01'],
#                             ['2023-12-04', '2023-12-01'],
#                             ['2023-12-05', '2023-12-01'],
#                             ['2023-12-06', '2023-12-01'],
#                             ['2023-12-07', '2023-12-01'],
#                             ['2023-12-08', '2023-12-01'],
#                             ['2023-12-09', '2023-12-01'],
#                             ['2023-12-11', '2023-12-01'],
#                             ['2023-12-12', '2023-12-01'],
#                             ['2023-12-13', '2023-12-01'],
#                             ['2023-12-14', '2023-12-01'],
#                             ['2023-12-15', '2023-12-01'],
#                             ['2023-12-16', '2023-12-01'],
#                             ['2023-12-17', '2023-12-01'],
#                             ['2023-12-18', '2023-12-01'],
#                             ['2023-12-19', '2023-12-01'],

#                             ]:
    
#     run_params['train_date'] = int(tr_date.replace('-', ''))
#     run_params['test_time_split'] = int(te_date.replace('-', ''))

individual_cats = {}
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

    df_train, df_predict, output_start = train_predict_split(df, run_params)
    df_train_prob, df_predict_prob = get_over_under_class(df, metric, run_params)
    output_start_prob = df_predict_prob[['player', 'game_date', 'value']].assign(metric=metric)

    # set up blank dictionaries for all metrics
    out_reg, out_class, out_quant, out_million = {}, {}, {}, {}

    #------------
    # Run the Stacking Models and Generate Output
    #------------

    # get the training data for stacking and prediction data after stacking
    X_stack, X_stack_class, y_stack, y_stack_class, models_reg, models_class, models_quant = load_all_stack_pred(model_output_path)

    X_predict, X_predict_class = get_stack_predict_data(df_train, df_predict, df_train_prob, df_predict_prob, run_params, 
                                                        models_reg, models_quant, models_class)
    

    
    if 'ind_cats' in run_params['stack_model'] and metric in ['points', 'rebounds', 'assists', 'steals', 'blocks']:
        individual_cats[metric] = {} 
        individual_cats = save_individual_stats(individual_cats, metric, X_stack, X_stack_class, X_predict, X_predict_class)

    combined_metrics = ['points_assists', 'points_rebounds', 'points_rebounds_assists', 'assists_rebounds', 'steals_blocks']
    if 'ind_cats' in run_params['stack_model'] and metric in combined_metrics:
        for k,v in individual_cats.items():
            if k in metric:
                X_stack = pd.merge(X_stack, individual_cats[k]['X_stack'], on=['player', 'year']).reset_index(drop=True)
                X_stack_class = pd.merge(X_stack_class, individual_cats[k]['X_stack_class'][['player', 'year']], on=['player', 'year']).reset_index(drop=True)
                X_predict = pd.merge(X_predict, individual_cats[k]['X_predict'], on=['player', 'week', 'year']).reset_index(drop=True)
                X_predict_class = pd.merge(X_predict_class, individual_cats[k]['X_predict_class'], on=['player', 'week', 'year']).reset_index(drop=True)

        X_predict_player_join = X_predict[['player', 'year']].rename(columns={'year':'game_date'})
        X_predict_class_player_join = X_predict_class[['player', 'year']].rename(columns={'year':'game_date'})
        output_start = pd.merge(output_start, X_predict_player_join, on=['player', 'game_date']).reset_index(drop=True)
        output_start_prob = pd.merge(output_start_prob, X_predict_class_player_join, on=['player', 'game_date']).reset_index(drop=True)

        X_stack = create_metric_split_columns_stack(X_stack, metric, individual_cats)
        X_stack_class = create_metric_split_columns_stack(X_stack_class, metric, individual_cats)
        X_predict = create_metric_split_columns_stack(X_predict, metric, individual_cats)
        X_predict_class = create_metric_split_columns_stack(X_predict_class, metric, individual_cats)

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
        output_prob = create_output_class(output_start_prob, best_pred_prob, output_teams)
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

past_pred[past_pred.game_date==20231109].sort_values(by='prob_over', ascending=False)
dm.write_to_db(past_pred,'Simulation', 'Over_Probability_New', 'replace', create_backup=True)

#%%


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


def get_date_info(df):
    df['real_date'] = pd.to_datetime(df['game_date'].astype('str'), format='%Y%m%d')
    df['day_of_week'] = df['real_date'].dt.dayofweek
    df['month'] = df['real_date'].dt.month
    return df


def train_split(train_pred, test_date, num_back_days=45, cv_time_input=None):

    train_dates = train_pred.game_date.sort_values().unique()[-num_back_days:]
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


def format_choices_output(choice_df, ens_vers, rank_order, val_greater, val_less, wt_col, decimal_cut_greater, decimal_cut_less, include_under, game_dates):
    choice_df = pd.melt(choice_df.T.reset_index(), id_vars=['index'])
    choice_df.columns = ['start_spot', 'num_choices', 'winnings']
    choice_df = choice_df.assign(value_cut_greater=val_greater, value_cut_less=val_less, wt_col=wt_col, decimal_cut_greater=decimal_cut_greater, 
                                decimal_cut_less=decimal_cut_less,rank_order=rank_order, 
                                include_under=include_under, last_date=game_dates[-1], ens_vers=ens_vers)

    return choice_df

def flip_probs(df, pred_col='final_pred'):

    df.loc[df[pred_col] < 0.5, 'y_act'] = np.where(df.loc[df[pred_col] < 0.5, 'y_act']==1, 0, 1)
    df.loc[df[pred_col] < 0.5, 'decimal_odds'] = (1 / (1 - (1/df.loc[df[pred_col] < 0.5, 'decimal_odds']))) - 0.1
    df.loc[df[pred_col] < 0.5, pred_col] = 1-df.loc[df[pred_col] < 0.5, pred_col]
    return df

def create_save_path(decimal_cut_greater, decimal_cut_less, val_greater, val_less, wt_col, include_under, foldername='pick_choices'):
    decimal_cut_greater_lbl = decimal_cut_greater.replace('>=', 'greaterequal').replace('>', 'greater')
    decimal_cut_less_lbl = decimal_cut_less.replace('<=', 'lessequal')
    val_greater_lbl = val_greater.replace('>', 'greater')
    val_less_lbl = val_less.replace('<', 'less')
    save_path = f'{root_path}/Model_Outputs/{foldername}/{wt_col}_{decimal_cut_greater_lbl}_{decimal_cut_less_lbl}_{include_under}_{val_greater_lbl}_{val_less_lbl}'
    return save_path

def get_past_runs(ens_vers):
    
    past_runs = dm.read(f'''SELECT DISTINCT wt_col, decimal_cut_greater, decimal_cut_less, include_under, value_cut_greater, value_cut_less
                           FROM Over_Probability_Choices
                           WHERE ens_vers='{ens_vers}'
                        ''', 'Simulation')
    for c in past_runs.columns:
        past_runs[c] = past_runs[c].astype(str)

    past_runs_list = []
    for i, row in past_runs.iterrows():
        decimal_cut_greater = row.decimal_cut_greater
        decimal_cut_less = row.decimal_cut_less
        val_greater = row.value_cut_greater
        val_less = row.value_cut_less
        wt_col = row.wt_col
        include_under = np.where(row.include_under=='1', True, False)

        save_path = create_save_path(decimal_cut_greater, decimal_cut_less, val_greater, val_less, wt_col, include_under)
        if os.path.exists(save_path) and f'{ens_vers}.p' in os.listdir(save_path):
            past_runs_list.append(save_path.split('/')[-1])

    return past_runs_list


##############################


def find_last_run(ens_vers, decimal_cut_greater, decimal_cut_less, val_greater, val_less, wt_col, include_under):
    include_under_q = np.where(include_under, 1, 0)
    wt_col_q = np.where(wt_col is None, 'IS NULL', f"='{wt_col}'")
    last_run = dm.read(f'''SELECT start_spot, num_choices, rank_order, last_date, winnings
                           FROM Over_Probability_Choices
                           WHERE value_cut_greater='{val_greater}'
                                 AND value_cut_less='{val_less}'
                                 AND wt_col {wt_col_q}
                                 AND decimal_cut_greater='{decimal_cut_greater}'
                                 AND decimal_cut_less='{decimal_cut_less}'
                                 AND include_under={include_under_q}
                                 AND ens_vers = '{ens_vers}'
                        ''', 'Simulation')
    return last_run

def fill_winnings_last_run(last_run):
    final_ens_choices = get_choices_dict()
    orig_choices = get_choices_dict()
    avg_choices = get_choices_dict()

    for ro in ['stack_model', 'original', 'avg']:
        for ro, ss, nc, w in last_run[['rank_order', 'start_spot', 'num_choices', 'winnings']].values:
            if ro=='stack_model': final_ens_choices[int(ss)][int(nc)] = [w]
            elif ro=='original': orig_choices[int(ss)][int(nc)] = [w]
            elif ro=='avg': avg_choices[int(ss)][int(nc)] = [w]

    return final_ens_choices, orig_choices, avg_choices

def get_past_trials(ens_vers, decimal_cut_greater, decimal_cut_less, val_greater, val_less, wt_col, include_under, foldername='pick_choices'):
    check_path = create_save_path(decimal_cut_greater, decimal_cut_less, val_greater, val_less, wt_col, include_under, foldername)
    try: trial_obj = load_pickle(check_path, ens_vers)                            
    except: trial_obj = Trials()
    return trial_obj

def pull_game_dates(q):
    game_dates = dm.read(q, 'Simulation').game_date.unique()
    return game_dates

#%%

def run_final_stack(pipe, params, X_train, y_train, cv_time_input, n_iter, wt_col, bayes_rand, trials_obj=Trials()):

    if cv_time_input is None:
        best_model, stack_scores, stack_pred, trial_obj = skm.best_stack(pipe, params, X_train, y_train, 
                                                                        n_iter=n_iter, alpha=None, wt_col=wt_col,
                                                                        trials=trials_obj, bayes_rand=bayes_rand,
                                                                        run_adp=False, print_coef=False,
                                                                        proba=True, num_k_folds=run_params['num_k_folds'],
                                                                        random_state=100)
    else: 
        best_models, oof_data, param_scores, trials = skm.time_series_cv(pipe, X_train, y_train, params, n_iter=n_iter, 
                                                                        n_splits=5, col_split='game_date', 
                                                                        time_split=cv_time_input,
                                                                        bayes_rand=bayes_rand, proba=True, trials=trials_obj,
                                                                        random_seed=(i+7)*19+(i*12)+6, alpha=None)
    return best_models, oof_data, param_scores, trials

#%%

i=20
test_date = run_params['test_time_split']
ens_vers = run_params['class_ens_vers']

query_cuts = {

    'no_under_high_cut': {

        #  determined using iterative modeling starting from stack_model, include_under=0
        'decimal_cut_greater': '>0',
        'decimal_cut_less': '<=3',
        'val_greater': '>4.5',
        'val_less': '<50',
        'wt_col': None,
        'include_under': False,
        'stack_model': True
    },

     'no_under_high_cut_decimal_odds': {

        #  determined using iterative modeling starting from stack_model, include_under=0
        'decimal_cut_greater': '>=1.7',
        'decimal_cut_less': '<=2.3',
        'val_greater': '>4.5',
        'val_less': '<20',
        'wt_col': 'decimal_odds',
        'include_under': False,
        'stack_model': True
    },

    'no_under_low_cut': {

        #  determined using iterative modeling starting from stack_model, include_under=0
        'decimal_cut_greater': '>0',
        'decimal_cut_less': '<=2.3',
        'val_greater': '>1.5',
        'val_less': '<50',
        'wt_col': 'decimal_odds_twomax',
        'include_under': False,
        'stack_model': True
    },

    'yes_under_high_cut': {

        # determined using iterative modeling starting from stack_model, include_under=1, start_spot<=1
        'decimal_cut_greater': '>0',
        'decimal_cut_less': '<=3',
        'val_greater': '>4.5',
        'val_less': '<40',
        'wt_col': 'decimal_odds_twomax',
        'include_under': True,
        'stack_model': True
    },

    'yes_under_low_cut': {

        #  determined using iterative modeling starting from stack_model, include_under=0, num_choices <= 4
        'decimal_cut_greater': '>0',
        'decimal_cut_less': '<=3',
        'val_greater': '>1.5',
        'val_less': '<40',
        'wt_col': 'decimal_odds_twomax',
        'include_under': True,
        'stack_model': True
    },
}

for cut_name, cut_dict in query_cuts.items():

    decimal_cut_greater = cut_dict['decimal_cut_greater']
    decimal_cut_less = cut_dict['decimal_cut_less']
    val_greater = cut_dict['val_greater']
    val_less = cut_dict['val_less']
    wt_col = cut_dict['wt_col']
    include_under = cut_dict['include_under']
    stack_model = cut_dict['stack_model']

    print('\n=======\n', val_greater, val_less, wt_col, decimal_cut_greater, decimal_cut_less, include_under, '\n=======\n')

    q = f'''SELECT * 
            FROM Over_Probability_New 
            WHERE value {val_greater}
                AND value {val_less}
                AND decimal_odds {decimal_cut_greater}
                AND decimal_odds {decimal_cut_less}
                AND ens_vers = '{ens_vers}'
            ORDER BY game_date ASC
            '''

    trials_obj = get_past_trials(ens_vers, decimal_cut_greater, decimal_cut_less, val_greater, val_less, wt_col, include_under)

    orig_predict = dm.read(q, 'Simulation')
    orig_predict = orig_predict[orig_predict.game_date==test_date]
    if include_under:
        orig_predict = flip_probs(orig_predict, pred_col='prob_over')

    if not stack_model:
        display(orig_predict.sort_values(by='prob_over', ascending=False).iloc[:50])

    else:

        run_params['cur_metric'] = 'points'
        train_pred = dm.read(q, 'Simulation')
        train_pred = get_date_info(train_pred)
        train_pred = train_pred.drop('y_act', axis=1).rename(columns={'y_act_prob': 'y_act'})

        train_pred, test_pred, cv_time_input = train_split(train_pred, test_date, num_back_days=75, cv_time_input=15)
        train_pred = train_pred.dropna().reset_index(drop=True)

        X_train = preprocess_X(train_pred, wt_col, cv_time_input)
        X_test = preprocess_X(test_pred, wt_col, cv_time_input)
        y_train = train_pred.y_act

        model_obj = 'class'
        final_m = 'lr_c'
        skm, X_train, y_train = get_skm(pd.concat([X_train, y_train], axis=1), model_obj, to_drop=['player', 'team', 'week', 'year'])
        pipe, params = get_full_pipe(skm, final_m, stack_model='random_kbest', alpha=None, 
                                    min_samples=10, bayes_rand='rand')
        pipe.steps[-1][-1].solver = 'saga'


        bayes_rand = 'rand'
        if bayes_rand == 'bayes':
            params['random_sample__frac'] = hp.uniform('frac', 0.5, 1)
            n_iter = 20
        else:
            params['random_sample__frac'] = np.arange(0.5, 1, 0.05)
            n_iter = 10
        
        best_models, oof_data, param_scores, trials = run_final_stack(pipe, params, X_train, y_train, cv_time_input, 
                                                                      n_iter, wt_col, bayes_rand=bayes_rand, trials_obj=Trials())
        

        show_calibration_curve(oof_data['actual'], oof_data['hold'], n_bins=8)

        for c in X_train.columns:
            if c not in X_test.columns:
                X_test[c] = 0

        X_test = X_test[X_train.columns]

        preds = pd.DataFrame()
        for best_model in best_models:
            best_model.fit(X_train, y_train)
            preds_cur = pd.Series(best_model.predict_proba(X_test)[:,1])
            preds = pd.concat([preds, preds_cur], axis=1)
            preds['final_pred'] = preds.mean(axis=1)

        preds = pd.concat([preds.final_pred, test_pred.reset_index(drop=True)], axis=1)

        if include_under:
            preds = flip_probs(preds)

        preds = preds.sort_values(by='final_pred', ascending=False)

        display(preds.iloc[:50])


#%%

import itertools

def run_past_choices(ens_vers, past_runs, wt_col, decimal_cut_greater, decimal_cut_less, include_under, val_greater, val_less):         
        
    print('\n=======\n', val_greater, val_less, wt_col, decimal_cut_greater, decimal_cut_less, include_under, '\n=======\n')

    q = f'''SELECT * 
            FROM Over_Probability_New
            WHERE value {val_greater}
                AND value {val_less}
                AND decimal_odds {decimal_cut_greater}
                AND decimal_odds {decimal_cut_less}
                AND ens_vers = '{ens_vers}'
                AND y_act IS NOT NULL
            ORDER BY game_date ASC
            '''

    save_path = create_save_path(decimal_cut_greater, decimal_cut_less, val_greater, val_less, wt_col, include_under)
    game_dates = pull_game_dates(q)

    if save_path.split('/')[-1] in past_runs:
        last_run = find_last_run(ens_vers, decimal_cut_greater, decimal_cut_less, val_greater, val_less, wt_col, include_under)
        final_ens_choices, orig_choices, avg_choices = fill_winnings_last_run(last_run)
        trial_obj = get_past_trials(ens_vers, decimal_cut_greater, decimal_cut_less, val_greater, val_less, wt_col, include_under)
        game_dates = [d for d in game_dates if d > last_run.last_date.unique()[0]]
        num_trials = 10
        num_back_days = 45

    else:
        final_ens_choices, orig_choices, avg_choices = get_choices_dict(), get_choices_dict(), get_choices_dict()
        trial_obj = Trials()
        game_dates = game_dates[3:]
        num_trials = 52
        num_back_days = 100

    if len(game_dates)>0:
        for test_date in game_dates:

            num_trials = np.max([num_trials-2, 10])
            num_back_days = np.max([num_back_days-1, 45])
            print('Date:', test_date)

            train_pred = dm.read(q, 'Simulation')
            train_pred = train_pred.drop('y_act', axis=1).rename(columns={'y_act_prob': 'y_act'})
            train_pred = get_date_info(train_pred)
            train_pred = train_pred.dropna().reset_index(drop=True)
            train_pred, test_pred = train_split(train_pred, test_date=test_date, num_back_days=num_back_days)
            
            X_train = preprocess_X(train_pred, wt_col)
            X_test = preprocess_X(test_pred, wt_col)
            y_train = train_pred.y_act

            model_obj = 'class'
            final_m = 'lr_c'
            skm, _, _ = get_skm(pd.concat([X_train, y_train], axis=1), model_obj, to_drop=[])
            pipe, params = get_full_pipe(skm, final_m, stack_model='random_kbest', alpha=None, 
                                        min_samples=10, bayes_rand='bayes')
            params['random_sample__frac'] = hp.uniform('frac', 0.5, 1)
            
            try:
                best_model, _, _, trial_obj = skm.best_stack(pipe, params, X_train, y_train, 
                                                            n_iter=num_trials, alpha=None, wt_col=wt_col,
                                                            trials=trial_obj, bayes_rand=run_params['opt_type'],
                                                            run_adp=False, print_coef=False,
                                                            proba=True, num_k_folds=run_params['num_k_folds'],
                                                            random_state=(i*2)+(i*7))
            except:
                best_model, _, _, trial_obj = skm.best_stack(pipe, params, X_train, y_train, 
                                                            n_iter=num_trials, alpha=None, wt_col=wt_col,
                                                            trials=Trials(), bayes_rand=run_params['opt_type'],
                                                            run_adp=False, print_coef=False,
                                                            proba=True, num_k_folds=run_params['num_k_folds'],
                                                            random_state=(i*2)+(i*7))

            for c in X_train.columns:
                if c not in X_test.columns:
                    X_test[c] = 0

            X_test = X_test[X_train.columns]

            best_model.fit(X_train, y_train)
            preds = pd.Series(best_model.predict_proba(X_test)[:,1], name='final_pred')
            preds = pd.concat([preds, test_pred.reset_index(drop=True)], axis=1)
            preds_avg = preds.copy()
            preds_avg['avg_prob'] = preds_avg[['final_pred', 'prob_over']].mean(axis=1)

            if include_under:
                preds = flip_probs(preds)
                preds_avg = flip_probs(preds_avg, pred_col='avg_prob')
                test_pred = flip_probs(test_pred, pred_col='prob_over')

            player_max_avg = preds_avg.groupby('player').agg({'avg_prob': 'max'}).reset_index()
            preds_avg = pd.merge(preds_avg, player_max_avg, on=['player', 'avg_prob'])
            preds_avg = preds_avg.sort_values(by='avg_prob', ascending=False).reset_index(drop=True)

            player_max_pred = preds.groupby('player').agg({'final_pred': 'max'}).reset_index()
            preds = pd.merge(preds, player_max_pred, on=['player', 'final_pred'])
            preds = preds.sort_values(by='final_pred', ascending=False).reset_index(drop=True)

            player_max_pred_orig = test_pred.groupby('player').agg({'prob_over': 'max'}).reset_index()
            preds_orig = pd.merge(test_pred, player_max_pred_orig, on=['player', 'prob_over'])
            preds_orig = preds_orig.sort_values(by='prob_over', ascending=False).reset_index(drop=True)

            avg_choices = fill_choices_dict(avg_choices, preds_avg)
            final_ens_choices = fill_choices_dict(final_ens_choices, preds)
            orig_choices = fill_choices_dict(orig_choices, preds_orig)

        final_ens = aggregate_choices(final_ens_choices)
        orig_ch = aggregate_choices(orig_choices)
        avg_choices = aggregate_choices(avg_choices)

        final_ens = format_choices_output(final_ens, ens_vers, rank_order='stack_model', val_greater=val_greater, val_less=val_less, wt_col=wt_col,
                                            decimal_cut_greater=decimal_cut_greater, decimal_cut_less=decimal_cut_less, include_under=include_under,
                                            game_dates=game_dates)
        orig_ch = format_choices_output(orig_ch, ens_vers, rank_order='original', val_greater=val_greater, val_less=val_less, wt_col=wt_col,
                                            decimal_cut_greater=decimal_cut_greater, decimal_cut_less=decimal_cut_less, include_under=include_under,
                                            game_dates=game_dates)
        avg_choices = format_choices_output(avg_choices, ens_vers, rank_order='avg', val_greater=val_greater, val_less=val_less, wt_col=wt_col,
                                            decimal_cut_greater=decimal_cut_greater, decimal_cut_less=decimal_cut_less, include_under=include_under,
                                            game_dates=game_dates)

        include_under_q = np.where(include_under, 1, 0)
        wt_col_q = np.where(wt_col is None, 'IS NULL', f"='{wt_col}'")

        del_str = f'''value_cut_greater='{val_greater}'
                    AND value_cut_less='{val_less}'
                    AND wt_col {wt_col_q}
                    AND decimal_cut_greater='{decimal_cut_greater}'
                    AND decimal_cut_less='{decimal_cut_less}'
                    AND include_under={include_under_q}
                    AND ens_vers = '{ens_vers}'
                '''

        dm.delete_from_db('Simulation', 'Over_Probability_Choices', del_str, create_backup=False)
        dm.write_to_db(final_ens,'Simulation', 'Over_Probability_Choices', 'append', create_backup=False)
        dm.write_to_db(orig_ch,'Simulation', 'Over_Probability_Choices', 'append', create_backup=False)
        dm.write_to_db(avg_choices,'Simulation', 'Over_Probability_Choices', 'append', create_backup=False)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_pickle(trial_obj, save_path, ens_vers)

    else:
        final_ens = aggregate_choices(final_ens_choices)
        orig_ch = aggregate_choices(orig_choices)
        avg_choices = aggregate_choices(avg_choices)

    return save_path, final_ens, orig_ch

i=20
model_obj = 'class'
alpha=None
run_params['cur_metric'] = 'points'

ens_vers = 'random_full_stack_matt0_brier1_include2_kfold3'#run_params['class_ens_vers']
print(ens_vers)
past_runs = get_past_runs(ens_vers)

wt_col_list=[None, 'decimal_odds', 'decimal_odds_twomax']
decimal_cut_greater_list = ['>0', '>=1.7']
decimal_cut_less_list = ['<=2.3', '<=3', '<=2.3']
include_under_list = [False, True]
val_greater_list = ['>0', '>0.5' ,'>1.5', '>2.5', '>3.5', '>4.5']
val_less_list = ['<100', '<50', '<40', '<30', '<20']

iter_cats = list(set(itertools.product(wt_col_list, decimal_cut_greater_list, decimal_cut_less_list, include_under_list,
                                       val_greater_list, val_less_list)))

out = Parallel(n_jobs=-1, verbose=50)(
    delayed(run_past_choices)
    (ens_vers, past_runs, wt_col, decimal_cut_greater, decimal_cut_less, include_under, val_greater, val_less) 
    for wt_col, decimal_cut_greater, decimal_cut_less, include_under, val_greater, val_less in iter_cats
)

# for wt_col, decimal_cut_greater, decimal_cut_less, include_under, val_greater, val_less in iter_cats[:1]:
#     run_past_choices(ens_vers, past_runs, wt_col, decimal_cut_greater, decimal_cut_less, include_under, val_greater, val_less)

#%%
    
def get_max_pred(pred_df, grp_col, prob_col):
    
    max_prob = pred_df.groupby(grp_col).agg({prob_col: 'max'}).reset_index()
    pred_df = pd.merge(pred_df, max_prob, on=[grp_col, prob_col])
    pred_df = pred_df.sort_values(by=prob_col, ascending=False).reset_index(drop=True)
    
    return pred_df

def remove_combo_stats(pred_df, prob_col):
    pred_df = pred_df[(~pred_df.metric.str.contains('_')) | (pred_df.metric=='three_pointers')].reset_index(drop=True)
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

def top_sgp_choices(cur_lbl, df, prob_col, choices, matchup_rank, num_matchups, remove_combos):
  
    if remove_combos: df = remove_combo_stats(df, prob_col)
    else: df = get_max_pred(df, 'player', prob_col)
    
    df = get_game_mapping(df)
    top_matchups = get_top_matchups(df, prob_col, num_players=5)
    top_matchups = top_matchups[top_matchups.player >= 5]

    df = df[df.matchup.isin(top_matchups.iloc[matchup_rank:matchup_rank+num_matchups, 0])]
    df = df.sort_values(by=prob_col, ascending=False).reset_index(drop=True)
    choices[cur_lbl] = fill_choices_dict(choices[cur_lbl], df)

    return df, choices

def top_all_choices(df, prob_col, choices):
    df = get_max_pred(df, 'player', prob_col)
    choices = fill_choices_dict(choices, df)
    return df, choices


def run_past_choices_sgp(ens_vers, past_runs, wt_col, decimal_cut_greater, decimal_cut_less, include_under, val_greater, val_less):

    print('\n=======\n', val_greater, val_less, wt_col, decimal_cut_greater, decimal_cut_less, include_under, '\n=======\n')

    q = f'''SELECT * 
            FROM Over_Probability_New
            WHERE value {val_greater}
                AND value {val_less}
                AND decimal_odds {decimal_cut_greater}
                AND decimal_odds {decimal_cut_less}
                AND ens_vers = '{ens_vers}'
                AND y_act IS NOT NULL
            ORDER BY game_date ASC
            '''

    save_path = create_save_path(decimal_cut_greater, decimal_cut_less, val_greater, val_less, wt_col, include_under)
    game_dates = pull_game_dates(q)

    if save_path.split('/')[-1] in past_runs:
        last_run = find_last_run(ens_vers, decimal_cut_greater, decimal_cut_less, val_greater, val_less, wt_col, include_under)
        final_ens_choices, orig_choices, avg_choices = fill_winnings_last_run(last_run)
        trial_obj = get_past_trials(ens_vers, decimal_cut_greater, decimal_cut_less, val_greater, val_less, wt_col, include_under)
        game_dates = [d for d in game_dates if d > last_run.last_date.unique()[0]]
        num_trials = 10
        num_back_days = 45

    else:
        sgp_choices = get_choices_dict()
        trial_obj = Trials()
        game_dates = game_dates[3:]
        num_trials = 52
        num_back_days = 100

    if len(game_dates)>0:
        for test_date in game_dates[:5]:

            num_trials = np.max([num_trials-2, 10])
            num_back_days = np.max([num_back_days-1, 45])
            print('Date:', test_date)

            train_pred = dm.read(q, 'Simulation')
            train_pred = train_pred.drop('y_act', axis=1).rename(columns={'y_act_prob': 'y_act'})
            train_pred = get_date_info(train_pred)
            train_pred = train_pred.dropna().reset_index(drop=True)
            train_pred, test_pred, cv_time_input = train_split(train_pred, test_date=test_date, num_back_days=num_back_days)
            
            X_train = preprocess_X(train_pred, wt_col, cv_time_input)
            X_test = preprocess_X(test_pred, wt_col, cv_time_input)
            y_train = train_pred.y_act

            model_obj = 'class'
            final_m = 'lr_c'
            skm, _, _ = get_skm(pd.concat([X_train, y_train], axis=1), model_obj, to_drop=[])
            pipe, params = get_full_pipe(skm, final_m, stack_model='random_kbest', alpha=None, 
                                    min_samples=10, bayes_rand=run_params['opt_type'])
            params['random_sample__frac'] = hp.uniform('frac', 0.5, 1)
            
            try:
                best_model, _, _, trial_obj = skm.best_stack(pipe, params, X_train, y_train, 
                                                            n_iter=num_trials, alpha=None, wt_col=wt_col,
                                                            trials=trial_obj, bayes_rand=run_params['opt_type'],
                                                            run_adp=False, print_coef=False,
                                                            proba=True, num_k_folds=run_params['num_k_folds'],
                                                            random_state=(i*2)+(i*7))
            except:
                best_model, _, _, trial_obj = skm.best_stack(pipe, params, X_train, y_train, 
                                                            n_iter=num_trials, alpha=None, wt_col=wt_col,
                                                            trials=Trials(), bayes_rand=run_params['opt_type'],
                                                            run_adp=False, print_coef=False,
                                                            proba=True, num_k_folds=run_params['num_k_folds'],
                                                            random_state=(i*2)+(i*7))

            for c in X_train.columns:
                if c not in X_test.columns:
                    X_test[c] = 0

            X_test = X_test[X_train.columns]
            best_model.fit(X_train, y_train)

            preds_orig = test_pred.reset_index(drop=True).copy()
            preds_stack = pd.Series(best_model.predict_proba(X_test)[:,1], name='final_pred')
            preds_stack = pd.concat([preds_stack, preds_orig], axis=1)

            preds_avg = preds_stack.copy()
            preds_avg['avg_prob'] = preds_avg[['final_pred', 'prob_over']].mean(axis=1)

            if include_under:
                preds_orig = flip_probs(preds_orig, pred_col='prob_over')
                preds_stack = flip_probs(preds_stack, pred_col='final_pred')
                preds_avg = flip_probs(preds_avg, pred_col='avg_prob')

            prob_types = ['stack_model', 'original', 'avg']
            prob_dfs = [preds_stack, preds_orig, preds_avg]
            prob_cols = ['final_pred', 'prob_over', 'avg_prob']
            
            # calculate winnings from various sgp choices
            for prob_type, prob_df, prob_col in zip(prob_types, prob_dfs, prob_cols):
                for remove_combos in [True, False]:
                    for matchup_rank in [0, 1, 2]:
                        for num_matchups in [1, 2, 3]:
                            cur_lbl = f'{prob_type}_{remove_combos}_{matchup_rank}_{num_matchups}'
                            if cur_lbl not in sgp_choices.keys(): sgp_choices[cur_lbl] = get_choices_dict()
                            _, sgp_choices = top_sgp_choices(cur_lbl, prob_df, prob_col, sgp_choices, matchup_rank, num_matchups, remove_combos)

        # save out all the various combinations by extracting from dictionary
        for prob_type in ['stack_model', 'original', 'avg']:
            for remove_combos in [True, False]:
                for matchup_rank in [0, 1, 2]:
                    for num_matchups in [1, 2, 3]:
                        cur_lbl = f'{prob_type}_{remove_combos}_{matchup_rank}_{num_matchups}'
                        cur_result = aggregate_choices(sgp_choices[cur_lbl])
                        cur_result = format_choices_output(cur_result, ens_vers, rank_order=prob_type, 
                                                        val_greater=val_greater, val_less=val_less, wt_col=wt_col,
                                                        decimal_cut_greater=decimal_cut_greater, decimal_cut_less=decimal_cut_less,
                                                        include_under=include_under, game_dates=game_dates)
                        cur_result = cur_result.assign(bet_type='sgp', matchup_rank=matchup_rank, num_matchups=num_matchups, no_combos=remove_combos)

                        include_under_q = np.where(include_under, 1, 0)
                        wt_col_q = np.where(wt_col is None, 'IS NULL', f"='{wt_col}'")
                        no_combos_q = np.where(remove_combos, 1, 0)

                        del_str = f'''value_cut_greater='{val_greater}'
                                    AND value_cut_less='{val_less}'
                                    AND wt_col {wt_col_q}
                                    AND decimal_cut_greater='{decimal_cut_greater}'
                                    AND decimal_cut_less='{decimal_cut_less}'
                                    AND include_under={include_under_q}
                                    AND rank_order='{prob_type}'
                                    AND ens_vers = '{ens_vers}'
                                    AND bet_type='sgp'
                                    AND matchup_rank={matchup_rank}
                                    AND num_matchups={num_matchups}
                                    AND no_combos={no_combos_q}
                                '''

                        dm.delete_from_db('Simulation', 'Probability_Choices_SGP', del_str, create_backup=False)
                        dm.write_to_db(cur_result, 'Simulation', 'Probability_Choices_SGP', 'append', create_backup=False)

        if not os.path.exists(save_path): os.makedirs(save_path)
        save_pickle(trial_obj, save_path, ens_vers)

    else:
        print('\n=======\n', val_greater, val_less, wt_col, decimal_cut_greater, decimal_cut_less, include_under, '\n=======\nDid Not Run')



i=20
model_obj = 'class'
alpha=None
run_params['cur_metric'] = 'points'

ens_vers = 'random_kbest_matt0_brier1_include2_kfold3'
print(ens_vers)
past_runs = []#get_past_runs(ens_vers)

wt_col_list=[None, 'decimal_odds', 'decimal_odds_twomax']
decimal_cut_greater_list = ['>0', '>=1.7']
decimal_cut_less_list = ['<=2.3', '<=3', '<=2.3']
include_under_list = [False, True]
val_greater_list = ['>0', '>0.5' ,'>1.5', '>2.5', '>3.5', '>4.5']
val_less_list = ['<100', '<50', '<40', '<30', '<20']

iter_cats = list(set(itertools.product(wt_col_list, decimal_cut_greater_list, decimal_cut_less_list, include_under_list,
                                       val_greater_list, val_less_list)))

# out = Parallel(n_jobs=-1, verbose=50)(
#     delayed(run_past_choices_sgp)
#     (ens_vers, past_runs, wt_col, decimal_cut_greater, decimal_cut_less, include_under, val_greater, val_less) 
#     for wt_col, decimal_cut_greater, decimal_cut_less, include_under, val_greater, val_less in iter_cats
# )

for wt_col, decimal_cut_greater, decimal_cut_less, include_under, val_greater, val_less in iter_cats[:1]:
    run_past_choices_sgp(ens_vers, past_runs, wt_col, decimal_cut_greater, decimal_cut_less, include_under, val_greater, val_less)

#%%


# results = dm.read("SELECT * FROM Over_Probability_Choices", 'Simulation')
# results['bet_type'] = 'all'
# results['matchup_rank'] = 0
# results['num_matchups'] = 7
# results['no_combos'] = 0

# dm.write_to_db(results, 'Simulation', 'Over_Probability_Choices', 'replace', create_backup=True)

#%%

num_matchups = dm.read("SELECT * FROM Over_Probability_New", 'Simulation')
num_matchups[['game_date', 'team']].drop_duplicates().groupby('game_date').count().reset_index().team.mean()

#%%

from sklearn.linear_model import ElasticNet
from lightgbm import LGBMRegressor
import shap

choice_params = dm.read('''SELECT * 
                           FROM Over_Probability_Choices
                           WHERE num_choices >= 3
                                 AND start_spot <= 3
                                 AND rank_order='stack_model'
                                 AND include_under=0
                                 AND ens_vers = 'random_kbest_matt0_brier1_include2_kfold3'
                                 AND value_cut_greater = '>1.5'
                                 AND decimal_cut_greater = '>0'
                                 AND value_cut_less = '<50'
                        
                                  
                           ''', 'Simulation')
X = choice_params.drop('winnings', axis=1)
for c in X.columns:
    X = pd.concat([X, pd.get_dummies(X[c], prefix=c)], axis=1).drop(c, axis=1)

y = choice_params.winnings

lr = ElasticNet(l1_ratio=0.01, alpha=0.0001)
lr.fit(X,y)
pd.Series(lr.coef_, index=X.columns).sort_values(ascending=False)

#%%
from sklearn.model_selection import cross_val_score

lgbm = LGBMRegressor(n_jobs=-1)
lgbm.fit(X,y)

scores = cross_val_score(lgbm, X, y, cv=5, scoring='neg_mean_squared_error')
scores = np.sqrt(-np.mean(scores))
print(scores)

shap_values = shap.TreeExplainer(lgbm).shap_values(X)
shap.summary_plot(shap_values, X, feature_names=X.columns, plot_size=(8,10), max_display=30, show=False)

#%%




# %%

# df = dm.read("SELECT * FROM Over_Probability_New", 'Simulation')
# df = df[~((df.game_date==20231119) & (df.team.isin(['PHI', 'BKN'])))]
# dm.write_to_db(df,'Simulation', 'Over_Probability_New', 'replace', create_backup=True)

#%%

# df = dm.read("SELECT * FROM Over_Probability_New", 'Simulation')
# df = df[~((df.game_date==20231206) & (df.train_date==20231111))]
# dm.write_to_db(df,'Simulation', 'Over_Probability_New', 'replace', create_backup=True)

#%%

df = dm.read("SELECT * FROM DraftKings_Odds", 'Player_Stats')
df = df[~((df.game_date==20231119) & (df.team.isin(['PHI', 'BKN'])))]

dm.write_to_db(df,'Player_Stats', 'DraftKings_Odds', 'replace', create_backup=True)

# %%

is_parlay = False
alpha=None
model_obj = 'class'

X_stack = X_stack_class
y_stack = y_stack_class
X_predict = X_predict_class

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
func_params = [f for f in func_params if f[0]!='rf_q']
model_list = [m for m in model_list if m!='rf_q']
try:
    time_per_trial = get_trial_times(root_path, fname, run_params)
    print(time_per_trial)
    num_trials = calc_num_trials(time_per_trial, run_params)
except: 
    num_trials = {m: run_params['n_iters'] for m in model_list}

if run_params['cur_metric']=='three_pointers' and run_params['train_date']==20230328 and alpha==0.5:
    num_trials['gbm_q'] = 20

if run_params['cur_metric']=='blocks' and run_params['train_date']==20231030 and model_obj=='reg':
    num_trials['gbm'] = 20

if run_params['cur_metric']=='points_assists' and run_params['train_date']==20231111 and model_obj=='quantile' and alpha==0.75:
    func_params = [f for f in func_params if f[0]!='rf_q']
    model_list = [m for m in model_list if m!='rf_q']
    print('filter worked')

print(model_list)

print(num_trials)
print(path, fname)

if os.path.exists(f"{path}/{fname}.p"):
    best_models, scores, stack_val_pred = load_stack_runs(path, fname)

else:
    
    # results = Parallel(n_jobs=-1, verbose=50)(
    #                 delayed(run_stack_models)
    #                 (fname, final_m, i, model_obj, alpha, X_stack, y_stack, run_params, num_trials, is_parlay) 
    #                 for final_m, i, model_obj, alpha in func_params
    #                 )
    # best_models, scores, stack_val_pred, trials = unpack_results(model_list, results)
    # save_stack_runs(path, fname, best_models, scores, stack_val_pred, trials)
    
    for final_m, i, model_obj, alpha in func_params:
        print(f'\n{final_m}')

        min_samples = int(len(y_stack)/10)
        proba, _, print_coef = get_proba_adp_coef(model_obj, final_m, run_params)
        skm, _, _ = get_skm(pd.concat([X_stack, y_stack], axis=1), model_obj, to_drop=[])

        pipe, params = get_full_pipe(skm, final_m, stack_model=run_params['stack_model'], alpha=alpha, 
                                    min_samples=min_samples, bayes_rand=run_params['opt_type'])
        
        trials = get_trials(fname, final_m, run_params['opt_type'])
        try: n_iter = int(num_trials[final_m]/2)
        except: n_iter = int(run_params['n_iters']/2)

        try:
            best_model, stack_scores, stack_pred, trial = skm.best_stack(pipe, params, X_stack, y_stack, 
                                                                        n_iter=n_iter, alpha=alpha, wt_col=None,
                                                                        trials=trials, bayes_rand=run_params['opt_type'],
                                                                        run_adp=False, print_coef=print_coef,
                                                                        proba=proba, num_k_folds=run_params['num_k_folds'],
                                                                        random_state=(i*2)+(i*7))
            
        except:
            backup_models = {
                'reg': 'ridge',
                'class': 'lr_c',
                'quantile': 'qr_q'
            }
            
            trial = trials
            backup_model = backup_models[model_obj]
            pipe, params = get_full_pipe(skm, backup_model, stack_model='kbest', alpha=alpha, 
                                        min_samples=min_samples, bayes_rand=run_params['opt_type'])

            best_model, stack_scores, stack_pred, _ = skm.best_stack(pipe, params, X_stack, y_stack, 
                                                                        n_iter=n_iter, alpha=alpha, wt_col=None,
                                                                        trials=Trials(), bayes_rand=run_params['opt_type'],
                                                                        run_adp=False, print_coef=print_coef,
                                                                        proba=proba, num_k_folds=run_params['num_k_folds'],
                                                                        random_state=(i*2)+(i*7))

X_predict = X_predict[X_stack.columns]
predictions = stack_predictions(X_predict, best_models, model_list, model_obj=model_obj)

# auto remove any predictions that are negative or 0
good_col = []
good_idx = []

for i, col in enumerate(predictions.columns):
    if predictions[col].mean() > 0:
        good_col.append(col)
        good_idx.append(i)

predictions = predictions[good_col]
stack_val_pred = stack_val_pred[good_col]
model_list = list(np.array(model_list)[good_idx])
scores = list(np.array(scores)[good_idx])

best_val, best_predictions, _ = average_stack_models(scores, model_list, y_stack, stack_val_pred, 
                                                        predictions, model_obj=model_obj, 
                                                        show_plot=run_params['show_plot'], 
                                                        min_include=run_params['min_include'])

# good_models = [c for c in best_predictions.columns if best_predictions[c].mean() > 0.1]
# best_predictions = best_predictions[good_models]
# best_val = best_val[good_models]

# %%

preds_mean_check = pd.DataFrame(predictions.median(), columns=['preds'])
val_mean_check = pd.DataFrame(stack_val_pred.median(), columns=['vals'])
mean_checks = pd.merge(preds_mean_check, val_mean_check, left_index=True, right_index=True)
mean_checks['pct_diff'] = (mean_checks.preds - mean_checks.vals) / mean_checks.vals
mean_checks = mean_checks[abs(mean_checks.pct_diff) < 0.25].index

# auto remove any predictions that are negative or 0
good_col = []
good_idx = []

for i, col in enumerate(predictions.columns):
    if col in mean_checks:
        good_col.append(col)
        good_idx.append(i)

predictions = predictions[good_col]
stack_val_pred = stack_val_pred[good_col]
model_list = list(np.array(model_list)[good_idx])
scores = list(np.array(scores)[good_idx])

# %%
