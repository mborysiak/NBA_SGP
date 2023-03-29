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

# set the root path and database management object
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


def get_full_pipe(skm, m, alpha=None, stack_model=False, std_model=False, min_samples=10):


    if stack_model and full_stack_features:
        if skm.model_obj=='class': kb = 'k_best_c'
        else: kb = 'k_best'

        pipe = skm.model_pipe([skm.piece('random_sample'),
                                skm.piece('std_scale'), 
                                skm.piece('select_perc'),
                                skm.feature_union([
                                                skm.piece('agglomeration'), 
                                                skm.piece(kb),
                                                skm.piece('pca')
                                                ]),
                                skm.piece(kb),
                                skm.piece(m)])

    elif stack_model:
        if skm.model_obj=='class': kb = 'k_best_c'
        else: kb = 'k_best'

        pipe = skm.model_pipe([
                            skm.piece('random_sample'),
                            skm.piece('std_scale'), 
                            skm.piece(kb),
                            skm.piece(m)
                        ])

    elif std_model:
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



    elif skm.model_obj == 'quantile':
        pipe = skm.model_pipe([
                                skm.piece('random_sample'),
                                skm.piece('std_scale'), 
                                skm.piece('k_best'), 
                                skm.piece(m)
                                ])

    # get the params for the current pipe and adjust if needed
    params = skm.default_params(pipe, 'rand')
    
    if skm.model_obj == 'quantile':
        if m == 'qr_q': pipe.steps[-1][-1].quantile = alpha
        elif m in ('rf_q', 'knn_q'): pipe.steps[-1][-1].q = alpha
        else: pipe.steps[-1][-1].alpha = alpha

    if m=='knn_c': params['knn_c__n_neighbors'] = range(1, min_samples-1)
    if m=='knn': params['knn__n_neighbors'] = range(1, min_samples-1)
    if m=='knn_q': params['knn_q__n_neighbors'] = range(1, min_samples-1)
    
    if stack_model and full_stack_features: 
        params['random_sample__frac'] = np.arange(0.6, 1.05, 0.05)
        params['select_perc__percentile'] = range(60, 105, 5)
        params['feature_union__agglomeration__n_clusters'] = range(2, 20, 1)
        params[f'feature_union__{kb}__k'] = range(5, 25, 2)
        params['feature_union__pca__n_components'] = range(2, 20, 1)
        params[f'{kb}__k'] = range(1, 40)
    
    elif stack_model:
        params['random_sample__frac'] = np.arange(0.3, 1, 0.05)
        params[f'{kb}__k'] = range(1, 30)

    return pipe, params


def load_all_pickles(model_output_path, label):
    pred = load_pickle(model_output_path, f'{label}_pred')
    actual = load_pickle(model_output_path, f'{label}_actual')
    models = load_pickle(model_output_path, f'{label}_models')
    scores = load_pickle(model_output_path, f'{label}_scores')
    try: full_hold = load_pickle(model_output_path, f'{label}_full_hold')
    except: full_hold = None
    return pred, actual, models, scores, full_hold


def X_y_stack(met, full_hold):
    i = 0
    for k, v in full_hold.items():
        if i == 0:
            df = v.copy()
            df = df.rename(columns={'pred': k})
        else:
            df_cur = v.rename(columns={'pred': k}).drop('y_act', axis=1)
            df = pd.merge(df, df_cur, on=['player', 'team', 'week','year'])
        i+=1

    X = df[[c for c in df.columns if met in c or 'y_act_' in c]].reset_index(drop=True)
    y = df['y_act'].reset_index(drop=True)
    return X, y, df


def show_scatter_plot(y_pred, y, label='Total', r2=True):
    plt.scatter(y_pred, y)
    plt.xlabel('predictions');plt.ylabel('actual')
    plt.show()

    from sklearn.metrics import r2_score
    if r2: print(f'{label} R2:', r2_score(y, y_pred))
    else: print(f'{label} Corr:', np.corrcoef(y, y_pred)[0][1])


def load_all_stack_pred(model_output_path, prefix=''):

    # load the regression predictions
    _, _, models_reg, _, full_hold_reg = load_all_pickles(model_output_path, prefix+'reg')
    X_stack, y_stack, _ = X_y_stack('reg', full_hold_reg)

    # load the quantile predictions
    _, _, models_quant, _, full_hold_quant = load_all_pickles(model_output_path, prefix+'quant')
    X_stack_quant, _, df_labels = X_y_stack('quant', full_hold_quant)

    # concat all the predictions together
    X_stack = pd.concat([df_labels[['player', 'week', 'year']], X_stack, X_stack_quant], axis=1)
    y_stack = pd.concat([df_labels[['player', 'week', 'year']], y_stack], axis=1)

    return X_stack, y_stack, models_reg, models_quant


def run_stack_models(final_m, i, X_stack, y_stack, best_models, scores, 
                     stack_val_pred, model_obj='reg', grp = None, alpha = None, show_plots=True,
                     num_k_folds=3, print_coef=True, proba=False):

    print(f'\n{final_m}')

    skm, _, _ = get_skm(pd.concat([X_stack, y_stack], axis=1), model_obj, to_drop=[])
    pipe, params = get_full_pipe(skm, final_m, stack_model=True, alpha=alpha)

    if model_obj=='class': proba = True
    else: proba = False
    
    if 'gbmh' in final_m or 'knn' in final_m: print_coef = False
    else: print_coef = print_coef

    best_model, stack_scores, stack_pred = skm.best_stack(pipe, params,
                                                          X_stack, y_stack, n_iter=run_params['n_iters'], 
                                                          run_adp=False, print_coef=print_coef,
                                                          sample_weight=False, proba=proba,
                                                          num_k_folds=num_k_folds, alpha=alpha,
                                                          random_state=(i*12)+(i*17), grp=grp)

    best_models.append(best_model)
    scores.append(stack_scores['stack_score'])
    stack_val_pred = pd.concat([stack_val_pred, pd.Series(stack_pred['stack_pred'], name=final_m)], axis=1)

    if show_plots:
        show_scatter_plot(stack_pred['stack_pred'], stack_pred['y'], r2=True)


    return best_models, scores, stack_val_pred


def fit_and_predict(m, df_predict, X, y, proba):

    
    try:
        m.fit(X,y)
        if proba: cur_predict = m.predict_proba(df_predict[X.columns])[:,1]
        else: cur_predict = m.predict(df_predict[X.columns])
    except:
        cur_predict = []

    return cur_predict

def create_stack_predict(df_predict, models, X, y, proba=False):

    # create the full stack pipe with meta estimators followed by stacked model
    X_predict = pd.DataFrame()
    for k, ind_models in models.items():
        # predictions = []
        # for m in ind_models:
        #     predictions.append(fit_and_predict(m, df_predict, X, y, proba))
        predictions = Parallel(n_jobs=-1, verbose=0)(delayed(fit_and_predict)(m, df_predict, X, y, proba) for m in ind_models)
        predictions = pd.Series(pd.DataFrame(predictions).T.mean(axis=1), name=k)
        X_predict = pd.concat([X_predict, predictions], axis=1)

    return X_predict

def get_stack_predict_data(df_train, df_predict, run_params, 
                           models_reg, models_quant):

    _, X, y = get_skm(df_train, 'reg', to_drop=run_params['drop_cols'])
    print('Predicting Regression Models')
    X_predict = create_stack_predict(df_predict, models_reg, X, y)
    X_predict = pd.concat([df_predict[['player', 'week', 'year']], X_predict], axis=1)

    print('Predicting Quant Models')
    X_predict_quant = create_stack_predict(df_predict, models_quant, X, y)
    X_predict = pd.concat([X_predict, X_predict_quant], axis=1)

    return X_predict


def stack_predictions(X_predict, best_models, final_models, model_obj='reg'):
    
    predictions = pd.DataFrame()
    for bm, fm in zip(best_models, final_models):
        
        if model_obj in ('reg', 'quantile'): cur_prediction = np.round(bm.predict(X_predict), 2)
        elif model_obj=='class': cur_prediction = np.round(bm.predict_proba(X_predict)[:,1], 3)
        
        cur_prediction = pd.Series(cur_prediction, name=fm)
        predictions = pd.concat([predictions, cur_prediction], axis=1)

    return predictions


def best_average_models(scores, final_models, y_stack, stack_val_pred, predictions, model_obj, min_include = 3):
    
    skm, _, _ = get_skm(df_train, model_obj=model_obj, to_drop=[])
    
    n_scores = []
    models_included = []
    for i in range(len(scores)-min_include+1):
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=False)[:i+min_include]
        models_included.append(top_n)
        model_idx = np.array(final_models)[top_n]
        
        n_score = skm.custom_score(y_stack, stack_val_pred[model_idx].mean(axis=1))
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


def average_stack_models(scores, final_models, y_stack, stack_val_pred, predictions, model_obj, show_plot=True, min_include=3):
    
    best_val, best_predictions, best_score = best_average_models(scores, final_models, y_stack, stack_val_pred, predictions, 
                                                                 model_obj=model_obj, min_include=min_include)
    
    if show_plot:
        show_scatter_plot(best_val.mean(axis=1), y_stack, r2=True)
    
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

def save_pickle(obj, path, fname, protocol=-1):
    with gzip.open(f"{path}/{fname}.p", 'wb') as f:
        pickle.dump(obj, f, protocol)

    print(f'Saved {fname} to path {path}')

def load_pickle(path, fname):
    with gzip.open(f"{path}/{fname}.p", 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object

def save_stack_runs(model_output_path, fname, best_models, scores, stack_val_pred):
    stack_out = {}
    stack_out['best_models'] = best_models
    stack_out['scores'] = scores
    stack_out['stack_val_pred'] = stack_val_pred
    save_pickle(stack_out, model_output_path, fname, protocol=-1)

def load_stack_runs(model_output_path, fname):
    stack_in = load_pickle(model_output_path, fname)
    return stack_in['best_models'], stack_in['scores'], stack_in['stack_val_pred']


def load_run_models(run_params, final_models, X_stack, y_stack, X_predict, model_obj, alpha=None, grp=None):
     
    model_output_path = run_params['model_output_path']
    ens_vers = run_params['ensemble_vers']

    if alpha is not None: alpha_label = alpha*100
    else: alpha_label = ''

    if model_obj == 'parlay_class': model_obj_actual = 'class'
    else: model_obj_actual = model_obj

    if os.path.exists(f"{model_output_path}{model_obj}{alpha_label}_{ens_vers}.p"):
            best_models, scores, stack_val_pred = load_stack_runs(model_output_path, f'{model_obj}{alpha_label}_' + run_params['ensemble_vers'])
    
    else:
        

        stack_val_pred = pd.DataFrame(); scores = []; best_models = []
        for i, fm in enumerate(final_models):
            try:
                best_models, scores, stack_val_pred = run_stack_models(fm, i+1, X_stack, y_stack, best_models, 
                                                                        scores, stack_val_pred, model_obj=model_obj_actual,
                                                                        grp=grp, alpha=alpha, show_plots=show_plot, 
                                                                        num_k_folds=num_k_folds, print_coef=print_coef)
            except:
                print('Error in model run', fm)
                final_models.remove(fm)

        save_stack_runs(model_output_path, f'{model_obj}{alpha_label}_' + ens_vers, best_models, scores, stack_val_pred)
        
    final_models = [f for f in final_models if f in stack_val_pred.columns]
    predictions = stack_predictions(X_predict, best_models, final_models, model_obj=model_obj_actual)
    best_val, best_predictions, _ = average_stack_models(scores, final_models, y_stack, stack_val_pred, 
                                                            predictions, model_obj=model_obj_actual, show_plot=show_plot, 
                                                            min_include=min_include)

    return best_val, best_predictions

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

#%%
#==========
# General Setting
#==========

#---------------
# Settings
#---------------

run_params = {
    
    # set year and week to analyze
    'cv_time_input_orig': '2023-02-06',
    'train_date_orig': '2023-03-21',
    'test_time_split_orig': '2023-03-27',
    'metrics':  [
                'points', 'assists', 'rebounds', 'three_pointers',   
                'steals', 'steals_blocks', 'blocks',# 'assists_rebounds',
                #'points_assists', 'points_rebounds', 'points_rebounds_assists', 
                ],

    'n_iters': 25,
    'n_splits': 5,

    # set version and iterations
    # 'pred_vers': 'mse5_rsq1_lowsample_perc',
    'pred_vers': 'mae1_rsq1_lowsample_perc',
    'ensemble_vers': 'mae1_rsq1_fullstack_allstats_diff_grp_no_odds',
    'parlay': True,
    'std_dev_type': 'pred_quant_spline'

}

run_params['cv_time_input'] = int(run_params['cv_time_input_orig'].replace('-', ''))
run_params['train_date'] = int(run_params['train_date_orig'].replace('-', ''))
run_params['test_time_split'] = int(run_params['test_time_split_orig'].replace('-', ''))


full_stack_features = True

min_include = 2
show_plot= True
print_coef = False
num_k_folds = 3

# set weights for running model
r2_wt = 1
sera_wt = 0
mse_wt = 0
mae_wt = 1

brier_wt = 1
matt_wt = 0

teams = dm.read("SELECT player, game_date year, team, opponent FROM FantasyData", 'Player_Stats')
teams.year = teams.year.apply(lambda x: int(x.replace('-', '')))

# %%

for te_date, tr_date, ens_vers, parl in [
                           ['2023-03-29', '2023-03-21', 'mae1_rsq1_fullstack_allstats_diff_grp_no_odds', False],
                           ['2023-03-29', '2023-03-21', 'mae1_rsq1_fullstack_allstats_diff_grp_no_odds', True],
                        #    ['2023-03-21', 'mae1_rsq1_fullstack_allstats_diff_grp', False],
                            # ['2023-03-17', 'mae1_rsq1_fullstack_allstats_diff_grp_no_odds', False],
                            # ['2023-03-17', 'mae1_rsq1_fullstack_allstats_diff_grp', False]

                        # ['2023-03-21', '2023-03-21', 'mae1_rsq1_fullstack_allstats_diff_grp_no_odds', True],
                        # ['2023-03-22', '2023-03-21', 'mae1_rsq1_fullstack_allstats_diff_grp_no_odds', True],
                        # ['2023-03-23', '2023-03-21', 'mae1_rsq1_fullstack_allstats_diff_grp_no_odds', True],
                        # ['2023-03-24', '2023-03-21', 'mae1_rsq1_fullstack_allstats_diff_grp_no_odds', True],
                        # ['2023-03-25', '2023-03-21', 'mae1_rsq1_fullstack_allstats_diff_grp_no_odds', True],
                        # ['2023-03-26', '2023-03-21', 'mae1_rsq1_fullstack_allstats_diff_grp_no_odds', True],

                          ]:
    
    run_params['train_date'] = int(tr_date.replace('-', ''))
    run_params['test_time_split'] = int(te_date.replace('-', ''))
    run_params['ensemble_vers'] = ens_vers
    run_params['parlay'] = parl

    for metric in run_params['metrics']:

        # load data and filter down
        pkey, model_output_path = create_pkey_output_path(metric, run_params)
        df, run_params = load_data(run_params)
        output_teams = df.loc[df.game_date==run_params['test_time_split'], ['player', 'team', 'opponent']]

        df = create_y_act(df, metric)
        df['week'] = 1
        df['year'] = df.game_date
        df['team'] = 0

        df_train, df_predict, output_start, min_samples = train_predict_split(df, run_params)

        # set up blank dictionaries for all metrics
        out_reg, out_class, out_quant, out_million = {}, {}, {}, {}

        #------------
        # Run the Stacking Models and Generate Output
        # #------------

        # get the training data for stacking and prediction data after stacking
        X_stack, y_stack, models_reg, models_quant = load_all_stack_pred(model_output_path)
        X_predict = get_stack_predict_data(df_train, df_predict, run_params, 
                                            models_reg, models_quant)

        X_stack_player = X_stack.copy()
        X_stack = X_stack.drop(['player', 'week', 'year'], axis=1).dropna(axis=0)
        y_stack = y_stack[y_stack.index.isin(X_stack.index)].y_act
        X_stack, y_stack = X_stack.reset_index(drop=True), y_stack.reset_index(drop=True)

        X_predict = X_predict.drop(['player', 'week' ,'year'], axis=1)

        # quantile regression metrics
        final_models = ['qr_q', 'gbm_q', 'lgbm_q', 'rf_q', 'knn_q']
        best_val_q25, best_pred_q25 = load_run_models(run_params, final_models, X_stack, y_stack, X_predict, 'quantile', alpha=0.25)
        best_val_q50, best_pred_q50 = load_run_models(run_params, final_models, X_stack, y_stack, X_predict, 'quantile', alpha=0.50)
        best_val_q75, best_pred_q75 = load_run_models(run_params, final_models, X_stack, y_stack, X_predict, 'quantile', alpha=0.75)        

        # create the stacking models
        final_models = ['ridge', 'lasso', 'huber', 'lgbm', 'xgb', 'rf', 'bridge', 'gbm', 'gbmh', 'knn']
        best_val_mean, best_pred_mean = load_run_models(run_params, final_models, X_stack, y_stack, X_predict, 'reg')

        preds = [best_pred_mean, best_pred_q25, best_pred_q50, best_pred_q75]
        labels = ['pred_mean', 'pred_q25', 'pred_q50', 'pred_q75']
        output = create_output(output_start, preds, labels)

        #-------------
        # Running the million dataset
        #-------------

        if run_params['parlay']: pickle_name = 'parlay_class'
        else: pickle_name = 'class'

        df_predict_prob, X_stack_prob, y_stack_prob, X_predict_prob = X_y_stack_class(df, metric, run_params, pickle_name)
        X_stack_prob, y_stack_prob = join_train_features(X_stack_player, X_stack_prob, y_stack_prob)
        X_predict_prob = join_predict_features(df_predict, X_predict, X_predict_prob)
        X_stack_prob = create_value_compare_col(X_stack_prob)
        X_predict_prob = create_value_compare_col(X_predict_prob)

        if 'no_odds' in run_params['ensemble_vers']:
            X_stack_prob = X_stack_prob.drop('decimal_odds', axis=1)
            X_predict_prob = X_predict_prob.drop('decimal_odds', axis=1)

        # class metrics
        final_models = ['lr_c', 'lgbm_c', 'rf_c', 'gbm_c', 'gbmh_c', 'xgb_c', 'knn_c']
        best_val_prob, best_pred_prob = load_run_models(run_params, final_models, X_stack_prob, y_stack_prob, X_predict_prob, pickle_name)
        if show_plot: show_calibration_curve(y_stack_prob, best_val_prob.mean(axis=1), n_bins=8)

        output_prob = create_output_class(df_predict_prob.assign(metric=metric), best_pred_prob, output_teams)
        output_prob = pd.merge(output_prob, output, on=['player', 'game_date'])
        output_prob = add_dk_lines(output_prob)

        output_prob = output_prob[['player', 'game_date', 'team', 'opponent', 'metric', 'decimal_odds', 'value', 
                                    'prob_over', 'pred_mean', 'pred_q25', 'pred_q50', 'pred_q75']]
        output_prob = output_prob.assign(pred_vers=run_params['pred_vers'], ens_vers=run_params['ensemble_vers'], 
                                        train_date=run_params['train_date'], parlay=run_params['parlay'])
        output_prob = np.round(output_prob,3).sort_values(by='prob_over', ascending=False)
        display(output_prob)

        del_str = f'''metric='{metric}'
                    AND game_date={run_params['test_time_split']} 
                    AND pred_vers='{run_params['pred_vers']}'
                    AND ens_vers='{run_params['ensemble_vers']}'
                    AND train_date={run_params['train_date']}
                    AND parlay={run_params['parlay']}
                    '''
        dm.delete_from_db('Simulation', 'Over_Probability', del_str, create_backup=False)
        dm.write_to_db(output_prob,'Simulation', 'Over_Probability', 'append')

#%%

def run_check(min_game_date, train_date, no_odds=True, yes_odds=True, parlay=0):

    if no_odds: ens_vers_no_odds = 'mae1_rsq1_fullstack_allstats_diff_grp_no_odds'
    else: ens_vers_no_odds = ''
    
    if yes_odds: ens_vers_yes_odds = 'mae1_rsq1_fullstack_allstats_diff_grp'
    else: ens_vers_yes_odds = ''

    print(f'Running check for train_date={train_date} with no_odds={no_odds} and yes_odds={yes_odds}')

    base_query = f''' 
            SELECT player, metric, game_date, team, opponent, decimal_odds, value,
                    AVG(prob_over) prob_over, 
                    AVG(pred_mean) pred_mean,
                    AVG(pred_q25) pred_q25,
                    AVG(pred_q50) pred_q50,
                    AVG(pred_q75) pred_q75
            FROM Over_Probability
            WHERE game_date >= {min_game_date}
                    AND train_date = {train_date}
                    AND ens_vers IN ('{ens_vers_no_odds}', '{ens_vers_yes_odds}')
                    AND parlay={parlay}
            GROUP BY player, metric, game_date, team, opponent, decimal_odds, value
    '''
    all_pred = dm.read(f'''

                    SELECT *, decimal_odds as weighting
                    FROM ({base_query})
                    WHERE prob_over >= 0.5
                     --   AND pred_q25 > value * 0.85
                        AND pred_mean > value
                        AND pred_q50 > value     
                    UNION  
                    SELECT *, decimal_odds/(decimal_odds-0.88) as weighting
                    FROM ({base_query})
                    WHERE prob_over < 0.5
                    --    AND pred_q75 < value * 1.15
                        AND pred_mean < value
                        AND pred_q50 < value 
                
        ''', 'Simulation')
    
    return all_pred

min_game_date = 20230321
train_date = 20230321
no_odds = True
yes_odds = False
parlay_chk = 1
all_pred = run_check(min_game_date, train_date, no_odds=no_odds, yes_odds=yes_odds, parlay=parlay_chk)

actual_pts, run_params = load_data(run_params)
actual_pts_stats = ['player', 'game_date', 'y_act_points', 'y_act_rebounds', 'y_act_assists', 
                    'y_act_three_pointers', 'y_act_steals', 'y_act_blocks']
actual_pts = actual_pts[actual_pts_stats] 
actual_pts.columns = [c.replace('y_act_', '') for c in actual_pts.columns]

actual_pts['points_rebounds'] = actual_pts.points + actual_pts.rebounds
actual_pts['points_assists'] = actual_pts.points + actual_pts.rebounds
actual_pts['points_rebounds_assists'] = actual_pts.points + actual_pts.rebounds + actual_pts.assists
actual_pts['steals_blocks'] = actual_pts.steals + actual_pts.blocks
actual_pts['assists_rebounds'] = actual_pts.assists + actual_pts.rebounds

actual_pts = pd.melt(actual_pts, id_vars=['player', 'game_date'], var_name=['metric'], value_name='actuals')

all_pred = pd.merge(all_pred, actual_pts, on=['player','game_date', 'metric'])
all_pred['is_over'] = np.where(all_pred.actuals > all_pred.value, 1, 0)
all_pred = all_pred.sort_values(by='prob_over', ascending=False).dropna(subset=['actuals']).reset_index(drop=True)

#%%

metrics = [
           'points', 
            'rebounds', 
            'assists',
            'three_pointers', 
         #   'points_rebounds',
         #    'points_assists', 
        #      'points_rebounds_assists',
         #     'assists_rebounds',
          'steals',
           'blocks',
         #   'steals_blocks'
]

all_pred_disp = all_pred[all_pred.metric.isin(metrics)]
print('number_samples:', all_pred_disp.shape[0])
skm, _, _ = get_skm(df_train, 'class', [])
_ = skm.test_scores(all_pred_disp.is_over, np.where(all_pred_disp.prob_over >= 0.5, 1, 0), sample_weight=all_pred_disp.weighting)
show_calibration_curve(all_pred_disp.is_over, all_pred_disp.prob_over, n_bins=8)
all_pred_disp.head(40).append(all_pred_disp.tail(40))

for m in metrics:
    
    all_pred_disp = all_pred[all_pred.metric.isin([m])]
    print(m, 'number_samples:', all_pred_disp.shape[0])
    skm, _, _ = get_skm(df_train, 'class', [])
    _ = skm.test_scores(all_pred_disp.is_over, np.where(all_pred_disp.prob_over >= 0.5, 1, 0), sample_weight=all_pred_disp.weighting)
    show_calibration_curve(all_pred_disp.is_over, all_pred_disp.prob_over, n_bins=8)
    # all_pred.head(40).append(all_pred.tail(40))

# %%


#%%
# test_dates = [
#     # ['2023-03-09', '2023-03-09'],
#     # ['2023-03-09', '2023-03-10'],
#     # ['2023-03-09', '2023-03-11'],
#     # ['2023-03-12', '2023-03-12'],
#     # ['2023-03-12', '2023-03-13'],
#     # ['2023-03-12', '2023-03-14'],
#     ['2023-03-17', '2023-03-21'],
# ]

# for tr_date, te_date in test_dates:

#     print(tr_date, te_date)
#     run_params['train_date_orig'] = tr_date
#     run_params['test_time_split_orig']= te_date

#     run_params['cv_time_input'] = int(run_params['cv_time_input_orig'].replace('-', ''))
#     run_params['train_date'] = int(run_params['train_date_orig'].replace('-', ''))
#     run_params['test_time_split'] = int(run_params['test_time_split_orig'].replace('-', ''))

#     for i, metric in enumerate(run_params['metrics']):

#         # load data and filter down
#         pkey, model_output_path = create_pkey_output_path(metric, run_params)
#         df, run_params = load_data(run_params)

#         df = create_y_act(df, metric)
#         df['week'] = 1
#         df['year'] = df.game_date
#         df['team'] = 0

#         df_train, df_predict, output_start, min_samples = train_predict_split(df, run_params)

#         # set up blank dictionaries for all metrics
#         out_reg, out_class, out_quant, out_million = {}, {}, {}, {}

#         #------------
#         # Run the Stacking Models and Generate Output
#         # #------------

#         # get the training data for stacking and prediction data after stacking
#         X_stack, y_stack, models_reg, models_quant = load_all_stack_pred(model_output_path)
#         X_predict = get_stack_predict_data(df_train, df_predict, run_params, 
#                                             models_reg, models_quant)

#         def update_col_names(df, metric):
#             df.columns = [f'{c}_{metric}' if c not in ('player', 'week', 'year') else c for c in df.columns]
#             return df
        
#         if i == 0: 
#             X_stack_all = update_col_names(X_stack.copy(), metric)
#             y_stack_all = update_col_names(y_stack.copy(), metric)
#             X_predict_all = update_col_names(X_predict.copy(), metric)
#         else: 
#             X_stack = update_col_names(X_stack, metric)
#             X_predict = update_col_names(X_predict, metric)
#             y_stack = update_col_names(y_stack, metric)

#             X_stack_all = pd.merge(X_stack_all, X_stack, on=['player', 'week', 'year'])
#             y_stack_all = pd.merge(y_stack_all, y_stack, on=['player', 'week', 'year'])
#             X_predict_all = pd.merge(X_predict_all, X_predict, on=['player', 'week', 'year'])

#     df_predict_prob_all = pd.DataFrame()
#     X_predict_prob_all = pd.DataFrame()
#     X_stack_prob_all = pd.DataFrame()
#     X_stack_prob_player_all = pd.DataFrame()
#     y_stack_prob_all = pd.DataFrame()
#     output_all = pd.DataFrame()

#     for metric in run_params['metrics']:

#         # load data and filter down
#         pkey, model_output_path = create_pkey_output_path(metric, run_params)
#         df, run_params = load_data(run_params)
#         output_teams = df.loc[df.game_date==run_params['test_time_split'], ['player', 'team', 'opponent']]

#         df = create_y_act(df, metric)
#         df['week'] = 1
#         df['year'] = df.game_date
#         df['team'] = 0

#         df_train, df_predict, output_start, min_samples = train_predict_split(df, run_params)

#         # set up blank dictionaries for all metrics
#         out_reg, out_class, out_quant, out_million = {}, {}, {}, {}
        
#         X_stack_player = X_stack_all.copy()
#         X_stack = X_stack_all.copy().drop(['player', 'week', 'year'], axis=1)
#         X_predict = X_predict_all.copy().drop(['player', 'week', 'year'], axis=1)
#         y_stack = y_stack_all[[f'y_act_{metric}']].copy().rename(columns={f'y_act_{metric}': 'y_act'})
#         y_stack = y_stack.y_act
#         stack_grp = get_group_col(X_stack_player, teams)

#         # # quantile regression metrics
#         # final_models = ['qr_q', 'gbm_q', 'lgbm_q', 'rf_q', 'knn_q']
#         # best_val_q25, best_pred_q25 = load_run_models(run_params, final_models, X_stack, y_stack, X_predict, 'quantile', alpha=0.25, grp=stack_grp)
#         # best_val_q50, best_pred_q50 = load_run_models(run_params, final_models, X_stack, y_stack, X_predict, 'quantile', alpha=0.50, grp=stack_grp)
#         # best_val_q75, best_pred_q75 = load_run_models(run_params, final_models, X_stack, y_stack, X_predict, 'quantile', alpha=0.75, grp=stack_grp)        

#         # # create the stacking models
#         # final_models = ['ridge', 'lasso', 'huber', 'lgbm', 'xgb', 'rf', 'bridge', 'gbm', 'gbmh', 'knn']
#         # best_val_mean, best_pred_mean = load_run_models(run_params, final_models, X_stack, y_stack, X_predict, 'reg', grp=stack_grp)

#         # preds = [best_pred_mean, best_pred_q25, best_pred_q50, best_pred_q75]
#         # labels = ['pred_mean', 'pred_q25', 'pred_q50', 'pred_q75']
#         # output = create_output(output_start, preds, labels)
#         # output_all = pd.concat([output_all, output.assign(metric=metric)], axis=0)

#         #-------------
#         # Running the million dataset
#         #-------------

#         df_predict_prob, X_stack_prob, y_stack_prob, X_predict_prob = X_y_stack_class(df, metric, run_params)
#         X_stack_prob_player = X_stack_prob[['player', 'year']].copy()
#         X_stack_prob, y_stack_prob = join_train_features(X_stack_player, X_stack_prob, y_stack_prob)
#         X_predict_prob = join_predict_features(df_predict, X_predict, X_predict_prob)
#         X_stack_prob = create_value_compare_col(X_stack_prob)
#         X_predict_prob = create_value_compare_col(X_predict_prob)

#         df_predict_prob_all = pd.concat([df_predict_prob_all, df_predict_prob.assign(metric=metric)], axis=0)
#         X_predict_prob_all = pd.concat([X_predict_prob_all, X_predict_prob.assign(metric=metric)], axis=0)
#         X_stack_prob_all = pd.concat([X_stack_prob_all, X_stack_prob.assign(metric=metric)], axis=0)
#         X_stack_prob_player_all = pd.concat([X_stack_prob_player_all, X_stack_prob_player.assign(metric=metric)], axis=0)
#         y_stack_prob_all = pd.concat([y_stack_prob_all, y_stack_prob], axis=0)

#     X_predict_prob_all = pd.concat([X_predict_prob_all, pd.get_dummies(X_predict_prob_all.metric)], axis=1).drop('metric', axis=1)
#     X_stack_prob_all = pd.concat([X_stack_prob_all, pd.get_dummies(X_stack_prob_all.metric)], axis=1).drop('metric', axis=1)

#     X_predict_prob_all = X_predict_prob_all.reset_index(drop=True)
#     X_stack_prob_all = X_stack_prob_all.reset_index(drop=True)
#     y_stack_prob_all = y_stack_prob_all.reset_index(drop=True).rename(columns={0:'y_act'}).y_act
#     X_stack_prob_player_all = X_stack_prob_player_all.reset_index(drop=True)
#     stack_prob_grp = get_group_col(pd.concat([X_stack_prob_player_all, X_stack_prob_all], axis=1), teams)

#     # class metrics
#     final_models = ['lr_c', 'lgbm_c', 'rf_c', 'gbm_c', 'gbmh_c', 'xgb_c', 'knn_c']
#     best_val_prob, best_pred_prob = load_run_models(run_params, final_models, X_stack_prob_all, 
#                                                     y_stack_prob_all, X_predict_prob_all, 'class', 
#                                                     grp=stack_prob_grp)
#     if show_plot: show_calibration_curve(y_stack_prob_all, best_val_prob.mean(axis=1), n_bins=8)

#     output_prob = create_output_class(df_predict_prob_all.reset_index(drop=True), best_pred_prob, output_teams)
#     output_prob = pd.merge(output_prob, output_all, on=['player', 'metric', 'game_date'])
#     output_prob = add_dk_lines(output_prob)

#     output_prob = output_prob[['player', 'game_date', 'team', 'opponent', 'metric', 'decimal_odds', 'value', 
#                                 'prob_over', 'pred_mean', 'pred_q25', 'pred_q50', 'pred_q75']]
#     output_prob = output_prob.assign(pred_vers=run_params['pred_vers'], ens_vers=run_params['ensemble_vers'],
#                                     train_date=run_params['train_date'])
#     output_prob = np.round(output_prob,3).sort_values(by='prob_over', ascending=False)
#     display(output_prob.head(40))
#     display(output_prob.tail(40))

#     del_str = f'''metric='{metric}'
#                 AND game_date={run_params['test_time_split']} 
#                 AND pred_vers='{run_params['pred_vers']}'
#                 AND ens_vers='{run_params['ensemble_vers']}'
#                 AND train_date={run_params['train_date']}
#                 '''
#     dm.delete_from_db('Simulation', 'Over_Probability', del_str, create_backup=False)
#     dm.write_to_db(output_prob,'Simulation', 'Over_Probability', 'append')


# #%%

# output_prob['mean_diff'] = output_prob.pred_mean - output_prob.value#) / np.sqrt(output_prob.value + 2.5)
# output_prob = output_prob.sort_values(by='mean_diff', ascending=False)
# # output_prob = np.round(output_prob,3).sort_values(by='prob_over', ascending=False)
# display(output_prob.head(40))
# display(output_prob.tail(40))

# #%%
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import cross_val_predict

# all_pred = dm.read(f'''
#                 SELECT player,
#                        game_date, 
#                        team, 
#                        opponent,
#                        metric,
#                        AVG(decimal_odds) decimal_odds,
#                        AVG(value) value,
#                        AVG(prob_over) prob_over,
#                        AVG(pred_mean) pred_mean,
#                        AVG(pred_q25) pred_q25,
#                        AVG(pred_q50) pred_q50,
#                        AVG(pred_q75) pred_q75
#                 FROM Over_Probability
#                 WHERE game_date >= 20230201
#                       AND decimal_odds <= 2.1
#                       AND decimal_odds >= 1.7
#                 GROUP BY player, game_date, team, opponent, metric
#                 ''', 'Simulation')

# actual_pts, _ = load_data(run_params)
# actual_pts = actual_pts[['player', 'game_date', 'y_act_points', 'y_act_rebounds', 
#                          'y_act_assists', 'y_act_three_pointers']] 
# actual_pts.columns = [c.replace('y_act_', '') for c in actual_pts.columns]
# actual_pts['points_rebounds'] = actual_pts.points + actual_pts.rebounds
# actual_pts['points_assists'] = actual_pts.points + actual_pts.rebounds
# actual_pts['points_rebounds_assists'] = actual_pts.points + actual_pts.rebounds + actual_pts.assists

# actual_pts = pd.melt(actual_pts, id_vars=['player', 'game_date'], var_name=['metric'], value_name='actuals')

# all_pred = pd.merge(all_pred, actual_pts, on=['player','game_date', 'metric'], how='left')
# all_pred['y_act'] = np.where(all_pred.actuals > all_pred.value, 1, 0)
# all_pred = all_pred.sort_values(by='prob_over', ascending=False).reset_index(drop=True)
# all_pred = all_pred[~(all_pred.actuals.isnull()) | (all_pred.game_date==run_params['test_time_split'])]

# train = all_pred.copy().sample(frac=1).reset_index(drop=True)

# train = train[
#     (train.metric.isin([
#                      'points', 
#                         'rebounds', 
#                         'assists',
#                        'three_pointers', 
#                         'points_rebounds',
#                         'points_assists', 
#                         'points_rebounds_assists'
#                    ]))
#             ].reset_index(drop=True)

# skm, _, _ = get_skm(train, 'class', [])

# def prepare_X(df, run_params):

#     X = df.drop(['player', 'opponent', 'actuals', 'y_act', 'team',
#                  ], axis=1)

#     for c in ['pred_mean' ,'pred_q25', 'pred_q50', 'pred_q75']:
#         X[f'{c}_vs_value'] = X[c] - X.value
#         X[f'{c}_over_value'] = X[c] / X.value


#     for c in ['metric',# 'team'#  'pred_vers', 'ens_vers','team',
#                 ]:
#         X = pd.concat([X, pd.get_dummies(X[c])], axis=1).drop(c, axis=1)

#     X_train = X[X.game_date < run_params['test_time_split']].reset_index(drop=True)
#     X_test = X[X.game_date == run_params['test_time_split']].reset_index(drop=True)

#     return X_train, X_test

# X_train, X_test = prepare_X(train, run_params)
# y_train = train.loc[train.game_date < run_params['test_time_split'], 'y_act'].reset_index(drop=True)

# full_stack_features=False
# std_model=True
# stack_model=False
# final_model, _, val_stats = run_stack_models('lr_c', 150, X_train, y_train, [], [], 
#                                     pd.DataFrame(), model_obj='class',
#                                         num_k_folds=3, print_coef=True, proba=True)
# show_calibration_curve(y_train, val_stats, n_bins=10)

# test = train[train.game_date==run_params['test_time_split']].reset_index(drop=True)
# test['final_prob'] = final_model[0].predict_proba(X_test)[:,1]
# test = test.sort_values(by='final_prob', ascending=False).reset_index(drop=True)
# test = test.drop([ 'actuals', 'y_act'], axis=1)
# display(test.head(40))
# display(test.tail(30))


