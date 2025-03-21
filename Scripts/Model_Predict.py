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
from hyperopt import hp, fmin, tpe, Trials
from sklearn.preprocessing import StandardScaler
from Fix_Standard_Dev import *
from joblib import Parallel, delayed
from skmodel import SciKitModel,ProbabilityEstimator, TruncatedNormalEstimator
from sklearn.pipeline import Pipeline
import re
from joblib import Parallel, delayed
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
from sklearn.model_selection import KFold

root_path = ffgeneral.get_main_path('NBA_SGP')
db_path = f'{root_path}/Data/'
dm = DataManage(db_path, timeout=200)

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
        if m in ('qr_q', 'gbmh_q'): pipe.set_params(**{f'{m}__quantile': alpha})
        elif m in ('rf_q', 'knn_q'): pipe.set_params(**{f'{m}__q': alpha})
        elif m == 'cb_q': pipe.set_params(**{f'{m}__loss_function': f'Quantile:alpha={alpha}'})
        else: pipe.set_params(**{f'{m}__alpha': alpha})


    if stack_model=='random_full_stack' and bayes_rand=='rand':
        params['random_sample__frac'] = np.arange(0.5, 1, 0.05)

    elif 'random_full_stack' in stack_model and bayes_rand=='optuna':
        params['random_sample__frac'] = ['real', 0.2, 1]
        params['feature_union__agglomeration__n_clusters'] = ['int', 3, 15]
        params['feature_union__pca__n_components'] = ['int', 3, 15]
        params[f'feature_union__{kb}_fu__k'] = ['int', 3, 50]
        params[f'{kb}__k'] = ['int', 5, 50]

    elif 'random_kbest' in stack_model and bayes_rand=='optuna':
        params['random_sample__frac'] = ['real', 0.2, 1]
        params[f'{kb}__k'] = ['int', 5, 50]
    
    else:
        pass
        

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

# Function to rename keys
def rename_dict_keys(key, metric):
    import re
    pattern = rf'{metric}_[^_]+_([^_]+)_(.+)$'
    match = re.search(pattern, key)
    if match:
        new_key = f'{match.group(1)}_{match.group(2)}'
        # Remove leading 'q_' or 'c_' if present
        new_key = re.sub(r'^(q_|c_)', '', new_key)
        return new_key
    return key


def load_all_stack_pred(model_output_path, metric):

    # load the regregression predictions
    models, full_hold = load_all_pickles(model_output_path, 'all')
    models = {rename_dict_keys(k, metric): v for k, v in models.items()}
    full_hold = {rename_dict_keys(k, metric): v for k, v in full_hold.items()}

    if model_output_path.split('/')[-2] == 'points_rebounds_assists_20241111_mse1_brier1':
        for k, v in full_hold.items():
            full_hold[k] = v[~((v.player=='Damian Lillard') & (v.year==20241104))].reset_index(drop=True)
        
    X_stack, X_stack_class, y_stack, y_stack_class = X_y_stack(full_hold)
    models_reg = {k: v for k, v in models.items() if 'reg' in k}
    models_class = {k: v for k, v in models.items() if 'class' in k}
    models_quant = {k: v for k, v in models.items() if 'quant' in k}

    return X_stack, X_stack_class, y_stack, y_stack_class, models_reg, models_class, models_quant

def fit_and_predict(m_label, m, df_predict, X, y, proba):

    # try:
    cols = m.named_steps['random_sample'].columns
    cols = [c for c in cols if c in X.columns]
    X = X[cols]
    X_predict = df_predict[cols]
    m.steps = [(name, trans) for (name, trans) in m.steps if name != "random_sample"]
    # except:
    #     X_predict = df_predict[X.columns]
        
    # try:
    m.fit(X,y)

    if proba: cur_predict = m.predict_proba(X_predict)[:,1]
    else: cur_predict = m.predict(X_predict)
    
    # except:
    #     cur_predict = []

    cur_predict = pd.DataFrame(cur_predict, columns=['pred'])
    cur_predict['model'] = m_label

    return cur_predict


def create_stack_predict(df_predict, models, X, y, proba=False):
    # create the full stack pipe with meta estimators followed by stacked model
    all_models = []
    for k, ind_models in models.items():
        for m in ind_models:
            all_models.append([k, m])

    predictions = Parallel(n_jobs=-1, verbose=0)(delayed(fit_and_predict)(model_name, m, df_predict, X, y, proba) for model_name, m in all_models)
    preds = pd.concat([p for p in predictions], axis=0)
    X_predict = pd.pivot_table(preds, values='pred', index=preds.index,columns='model', aggfunc='mean')
    X_predict = X_predict.rename_axis(None, axis=1)

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
    
    if model_obj == 'quantile': min_include -= 1
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
# Adding distribution features
#-----------------------

def group_quantile_columns(df):
    """
    Groups quantile columns by model type from a DataFrame.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing quantile columns
    
    Returns:
    dict: Dictionary with model types as keys and lists of column names as values
    """
    # Get all quantile columns
    quant_cols = [col for col in df.columns if col.startswith('quant_')]
    
    # Create a dictionary to store model-specific columns
    model_groups = {}
    
    # Define the pattern to match quantile columns
    pattern = r'quant_(\d+\.\d+)_(\w+)_q'
    
    # Group columns by model type
    for col in quant_cols:
        match = re.match(pattern, col)
        if match:
            quantile, model = match.groups()
            if model not in model_groups:
                model_groups[model] = []
            model_groups[model].append(col)
    
    # Sort columns within each group to ensure they're in order
    for model in model_groups:
        model_groups[model].sort()
    
    return model_groups

def get_quantile_values(df, model_type):
    """
    Gets the quantile columns for a specific model type.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing quantile columns
    model_type (str): Model type to extract (e.g., 'qr', 'lgbm', 'gbm', etc.)
    
    Returns:
    pandas.DataFrame: DataFrame containing only the specified model's quantile columns
    """
    groups = group_quantile_columns(df)
    if model_type in groups:
        return df[groups[model_type]]
    else:
        raise ValueError(f"No quantile columns found for model type: {model_type}")

def get_quantile_mapping(columns):
    """
    Creates a dictionary mapping column names to their quantile values.
    
    Parameters:
    columns (list): List of quantile column names
    
    Returns:
    dict: Dictionary mapping column names to their quantile values as floats
    """
    pattern = r'quant_(\d+\.\d+)_\w+_q'
    mapping = {}
    
    for col in columns:
        match = re.match(pattern, col)
        if match:
            quantile = float(match.group(1))
            mapping[col] = quantile
    
    return mapping

def fit_predict_truncnorm_kfold(X, y, groups, value_col, n_splits=5, random_state=42):
    """
    Performs k-fold cross validation for TruncatedNormalEstimator for multiple model groups.
    
    Parameters:
    X (pd.DataFrame): Input features dataframe
    y (pd.Series): Target values
    groups (dict): Dictionary of model groups and their quantile columns
    value_col (pd.Series): Values to use in predict_proba
    n_splits (int): Number of folds for cross-validation
    random_state (int): Random state for reproducibility
    
    Returns:
    pd.DataFrame: DataFrame with predictions for each model type
    dict: Dictionary of fitted models for each fold and model type
    """
    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Dictionary to store predictions and fitted models
    predictions = {}
    fitted_models = {model: [] for model in groups.keys()}
    
    # Perform k-fold for each model type
    for model, q_columns in groups.items():
        # Initialize array for predictions
        fold_predictions = np.zeros(len(X))
        
        # Perform k-fold cross validation
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            
            # Initialize and fit model on training data
            trunc_norm = TruncatedNormalEstimator(
                spline_type='spline',
                n_groups_min=0.03,
                n_groups_max=0.06
            )
            
            # Fit on training fold
            trunc_norm.fit(
                X.iloc[train_idx][q_columns],
                y.iloc[train_idx]
            )
            
            # Predict on validation fold
            fold_predictions[val_idx] = trunc_norm.predict_proba(
                X.iloc[val_idx][q_columns],
                value_col.iloc[val_idx]
            )
            
            # Store fitted model
            fitted_models[model].append(trunc_norm)
        
        # Store predictions for this model type
        predictions[f'{model}_trunc_prob'] = fold_predictions
    
    # Convert predictions to DataFrame
    predictions_df = pd.DataFrame(predictions, index=X.index)
    
    return predictions_df, fitted_models

def prod_dist_features(X_stack_class, X_predict_class, y_stack_trunc):

    for c in X_stack_class.columns:
        if 'quant' in c: 
            X_stack_class.loc[(X_stack_class[c].isnull()) | (X_stack_class[c]<0), c] = 0
            X_predict_class.loc[(X_predict_class[c].isnull()) | (X_predict_class[c]<0), c] = 0

    # Get all model groups
    groups = group_quantile_columns(X_stack_class)
    y_stack_trunc = pd.merge(X_stack_class[['player', 'year']], y_stack_trunc, on=['player', 'year'])

    start_time = time.time()
    # Run k-fold cross validation
    predictions_df, _ = fit_predict_truncnorm_kfold(X_stack_class, y_stack_trunc.y_act, groups=groups,
                                                    value_col=X_stack_class.value, n_splits=5)
    print(f"Time taken for k-fold cross validation: {time.time() - start_time} seconds")

    # Add predictions to original dataframe
    X_stack_class = X_stack_class.join(predictions_df)

    for model, q_columns in groups.items():
        start_time = time.time()
        trunc_norm = TruncatedNormalEstimator(spline_type='spline', n_groups_min=0.03, n_groups_max=0.06)
        trunc_norm.fit(X_stack_class[q_columns], y_stack_trunc.y_act)
        X_predict_class[f'{model}_trunc_prob'] = trunc_norm.predict_proba(X_predict_class[q_columns], X_predict_class.value)
        print(f"Time taken for TruncatedNormalEstimator {model}: {time.time() - start_time} seconds")


        start_time = time.time()
        # For each model type, get its quantile mapping
        quantile_map = get_quantile_mapping(q_columns)
        poisson_estimator = ProbabilityEstimator('poisson', quantile_map)
        X_stack_class[f'{model}_poisson'] = poisson_estimator.calc_exceedance_probs(X_stack_class, threshold_column='value')
        X_predict_class[f'{model}_poisson'] = poisson_estimator.calc_exceedance_probs(X_predict_class, threshold_column='value')
        print(f"Time taken for ProbabilityEstimator {model}: {time.time() - start_time} seconds")

    return X_stack_class, X_predict_class

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



def rename_existing(old_study_db, new_study_db, study_name):

    import datetime as dt
    new_study_name = study_name + '_' + dt.datetime.now().strftime('%Y%m%d%H%M%S')
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
    
    # try:
    next_study = optuna.create_study(
        study_name=new_name, 
        storage=new_storage, 
        load_if_exists=False
    )

    # except:
    #     rename_existing(old_storage, new_storage, new_name)
    #     next_study = optuna.create_study(
    #         study_name=new_name, 
    #         storage=new_storage, 
    #         load_if_exists=False
    #     )
    
    if old_name is not None and len(old_study.trials) > 0:
        print(f"Loaded {new_name} study with {old_name} {len(old_study.trials)} trials")
        next_study.add_trials(old_study.trials[-num_trials:])

    return next_study
    

def get_optuna_study(final_m, fname, run_params):
    time.sleep(1*np.random.random())
    old_name = f"{final_m}_{fname}_{run_params['last_train_date']}"
    new_name = f"{final_m}_{fname}_{run_params['train_date']}"
    next_study = get_new_study(f"{run_params['last_study_db']}_{fname}_{final_m}.sqlite3", 
                               f"{run_params['study_db']}_{fname}_{final_m}.sqlite3", 
                               old_name, new_name, run_params['num_past_trials'])
    return next_study


def run_stack_models(fname, final_m, i, model_obj, alpha, X_stack, y_stack, run_params, wt_col=None):

    print(f'\n{final_m}')

    min_samples = int(len(y_stack)/10)
    proba, _, print_coef = get_proba_adp_coef(model_obj, final_m, run_params)
    skm, _, _ = get_skm(pd.concat([X_stack, y_stack], axis=1), model_obj, to_drop=[])

    pipe, params = get_full_pipe(skm, final_m, stack_model=run_params['stack_model'], alpha=alpha, 
                                    min_samples=min_samples, bayes_rand=run_params['opt_type'])
    trials = get_optuna_study(final_m, fname, run_params)


    try:
        best_model, stack_scores, stack_pred, trial = skm.best_stack(pipe, params, X_stack, y_stack, 
                                                                    n_iter=run_params['n_iters'], alpha=alpha, wt_col=wt_col,
                                                                    bayes_rand=run_params['opt_type'],trials=trials,
                                                                    run_adp=False, print_coef=print_coef,
                                                                    proba=proba, num_k_folds=run_params['num_k_folds'],
                                                                    random_state=(i*2)+(i*7), 
                                                                    optuna_timeout=run_params['optuna_timeout'])
    except:
        print(f"Error with {final_m}. Running backup model")
        
        backup_models = {
            'reg': 'ridge',
            'class': 'lr_c',
            'quantile': 'qr_q'
        }
        
        backup_model = backup_models[model_obj]
        pipe, params = get_full_pipe(skm, backup_model, stack_model='kbest', alpha=alpha, 
                                    min_samples=min_samples, bayes_rand=run_params['opt_type'])
        trials = get_optuna_study(f'{backup_model}_backup_{final_m}', fname, run_params)

        best_model, stack_scores, stack_pred, trial = skm.best_stack(pipe, params, X_stack, y_stack, 
                                                                    n_iter=run_params['n_iters'], alpha=alpha, wt_col=wt_col,
                                                                    bayes_rand=run_params['opt_type'], trials=trials,
                                                                    run_adp=False, print_coef=print_coef,
                                                                    proba=proba, num_k_folds=run_params['num_k_folds'],
                                                                    random_state=(i*2)+(i*7), 
                                                                    optuna_timeout=run_params['optuna_timeout'])
        
    return best_model, stack_scores, stack_pred, trial

 

def get_func_params(model_obj, alpha):

    model_list = {
        'reg': ['rf', 'gbm', 'gbmh', 'mlp', 'cb', 'huber', 'xgb', 'lgbm', 'knn', 'ridge', 'lasso', 
                'bridge', 'enet', 'cb_p', 'cb_t', 'lgbm_p', 'lgbm_t', 'xgb_p', 'xgb_t', 'et'],
        'class': ['rf_c', 'gbm_c', 'gbmh_c', 'xgb_c','lgbm_c', 'knn_c', 'lr_c', 'mlp_c', 'cb_c', 'et_c'],
        'quantile': ['qr_q', 'gbm_q', 'lgbm_q', 'gbmh_q', 'rf_q', 'cb_q']#, 'knn_q']
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
    for cut in np.arange(0.2, 10, 0.2):
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

    model_list, func_params = remove_bad_models(model_obj, run_params, alpha, func_params, model_list)

    if os.path.exists(f"{path}/{fname}.p"):
        best_models, scores, stack_val_pred = load_stack_runs(path, fname)
    
    else:
        
        results = Parallel(n_jobs=-1, verbose=50)(
                        delayed(run_stack_models)
                        (fname, final_m, i, model_obj, alpha, X_stack, y_stack, run_params) 
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

    if metric in ('points_assists', 'points_rebounds', 'points_rebounds_assists', 'assists_rebounds'):
        metric_split = metric.split('_')
        df[f'y_act_{metric}'] = df[['y_act_' + c for c in metric_split]].sum(axis=1)
        df = create_metric_split_columns_backfill(df, metric_split)

    return df

def get_all_past_results(run_params):

    attach_pts, run_params = load_data(run_params)
    for metric in ['points_assists', 'points_rebounds', 'points_rebounds_assists', 'assists_rebounds']:
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
    
    # # set year and week to analyze

    'last_train_date_orig': '2025-02-01',
    'train_date_orig': '2025-03-13',
    'test_time_split_orig': '2025-03-13',

    # 'last_train_date_orig': '2025-01-10',
    # 'train_date_orig': '2025-02-01',
    # 'test_time_split_orig': dt.date.today().strftime('%Y-%m-%d'),
    
    'db_stack_predict_last': 'Stack_Predict_2025',

    'metrics':  [
                 'points', 'assists', 'rebounds', 'three_pointers', 
                 'assists_rebounds', 'points_assists', 'points_rebounds',
                # 'points_rebounds_assists', 'blocks', 'steals', 'steals_blocks', 
                   
    ],

    # opt params    
    'n_splits': 5,
    'num_k_folds': 3,
    'show_plot': True,
    'print_coef': True,
    'min_include': 3,

    # set version and iterations
    'pred_vers': 'mse1_brier1',
    #  'stack_model': 'random_full_stack_ind_cats',
    # 'stack_model_class': 'random_full_stack_ind_cats',
    # 'stack_model': 'random_kbest',
    # 'stack_model_class': 'random_kbest',
    'stack_model': 'random_full_stack',
    'stack_model_class': 'random_full_stack',
   
    'parlay': False,

    'opt_type': 'optuna',
    'hp_algo': 'tpe',
    'num_past_trials': 100,
    'optuna_timeout': 60*2.5,
    'n_iters': 50
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
#                             ['2024-11-18', '2024-11-11'],
#                             ['2024-11-19', '2024-11-11'],
#                             ['2024-11-20', '2024-11-11'],
#                             ['2024-11-21', '2024-11-11'],
#                             ['2024-11-22', '2024-11-11'],
#                             ['2024-11-23', '2024-11-11'],
#                             ['2024-11-24', '2024-11-11'],
#                             ]:

#     run_params['train_date'] = int(tr_date.replace('-', ''))
#     run_params['test_time_split'] = int(te_date.replace('-', ''))

individual_cats = {}
for metric in run_params['metrics']:

    # load data and filter down
    pkey, model_output_path = create_pkey_output_path(metric, run_params)
    df, run_params = load_data(run_params)
    run_params['cur_metric'] = metric

    run_params['last_study_db'] = f"sqlite:///optuna/{run_params['last_train_date']}/pred_"
    run_params['study_db'] = f"sqlite:///optuna/{run_params['train_date']}/pred_"

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
    X_stack, X_stack_class, y_stack, y_stack_class, models_reg, models_class, models_quant = load_all_stack_pred(model_output_path, metric)

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
    y_stack_trunc = y_stack.copy()
    y_stack = y_stack.drop(['player', 'year'], axis=1).y_act

    odds = pull_odds(metric, run_params['parlay'])
    X_stack_class = pd.merge(X_stack_class, odds, on=['player', 'year'])
    X_predict_class = pd.merge(X_predict_class, odds, on=['player', 'year'])

    print('Calculating Probabilities')
    X_stack_class, X_predict_class = prod_dist_features(X_stack_class, X_predict_class, y_stack_trunc)

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

if run_params['stack_model'] == 'random_full_stack_ind_cats':

    cut_time = (dt.datetime.strptime(run_params['train_date_orig'], '%Y-%m-%d') - dt.timedelta(50))
    cut_time = cut_time.strftime('%Y-%m-%d').replace('-', '')
    
    past_pred = dm.read(f'''SELECT * 
                           FROM Over_Probability_New 
                           WHERE game_date >= {cut_time}
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

    dm.delete_from_db('Simulation', 'Over_Probability_New', f"game_date >= {cut_time}", create_backup=True)
    dm.write_to_db(past_pred, 'Simulation', 'Over_Probability_New', 'append', create_backup=False)

#%%

#============
# Choice Dictionary Management
#============

def get_choices_dict():

    all_choices = {}
    for win_type in ['num_correct', 'num_wins', 'winnings', 'num_trials']:
        all_choices[win_type] = {}
        for start_spot in range(3):
            all_choices[win_type][start_spot] = {}
            for num_choices in range(1,7):
                all_choices[win_type][start_spot][num_choices] = []

    return all_choices

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

def predict_odds_reduce(cur_df, num_choices, lr_odds_reduce):
    num_games = list(cur_df.team.unique())
    num_games.extend(list(cur_df.opponent.unique()))
    num_games = int(len(set(num_games))/2)
    odds_reduce = lr_odds_reduce.predict([[num_games, num_choices]])[0][0]
    return odds_reduce

def fill_choices_dict(all_choices, preds):
    for start_spot in range(3):
        for num_choices in range(1,7):
            if preds.iloc[start_spot:start_spot+num_choices].shape[0] >= num_choices:
                wins = preds.iloc[start_spot:start_spot+num_choices].y_act.sum()
                odds = np.prod(preds.iloc[start_spot:start_spot+num_choices].decimal_odds)
                odds_reduce = predict_odds_reduce(preds.iloc[start_spot:start_spot+num_choices], num_choices, lr_odds_reduce)
                odds = odds * odds_reduce

                all_choices['num_correct'][start_spot][num_choices].append(wins)
                all_choices['num_trials'][start_spot][num_choices].append(1)

                if wins == num_choices:
                    all_choices['winnings'][start_spot][num_choices].append(odds)
                    all_choices['num_wins'][start_spot][num_choices].append(1)
                else:
                    all_choices['winnings'][start_spot][num_choices].append(-1)
                    all_choices['num_wins'][start_spot][num_choices].append(0)
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

def preprocess_X(df, wt_col, model, cv_time_input=None):
    X = df[['decimal_odds', 'value', 'prob_over', 'pred_mean', 'pred_q25', 'pred_q50', 'pred_q75']].copy()
    X = create_value_compare_col(X)

    if wt_col=='decimal_odds_twomax':
        X['decimal_odds_twomax'] = X.decimal_odds
        X.loc[X.decimal_odds_twomax > 2, 'decimal_odds_twomax'] = 2-(1/X.loc[X.decimal_odds_twomax > 2, 'decimal_odds_twomax'])

    if model != 'cb_c':
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
    df.loc[df[pred_col] < 0.5, pred_col] = 1-df.loc[df[pred_col] < 0.5, pred_col]
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

def rename_existing_stack(new_study_db, study_name):

    import datetime as dt
    new_study_name = str(study_name) + '_' + dt.datetime.now().strftime('%Y%m%d%H%M%S')
    optuna.copy_study(from_study_name=study_name, from_storage=new_study_db, to_storage=new_study_db, to_study_name=new_study_name)
    optuna.delete_study(study_name=study_name, storage=new_study_db)


def get_new_study_stack(old_db, new_db, old_name, new_name, num_trials):
    
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
        rename_existing_stack(new_storage, new_name)
        next_study = optuna.create_study(
            study_name=new_name, 
            storage=new_storage, 
            load_if_exists=False
        )
    
    try:
        if old_name is not None and len(old_study.trials) > 0:
            print(f"Loaded {new_name} study with {old_name} {len(old_study.trials)} trials")
            next_study.add_trials(old_study.trials[-num_trials:])
    except:
        pass

    return next_study
    

def get_optuna_study_stack(save_name, game_date, last_game_date):
    time.sleep(1*np.random.random())
    old_name = last_game_date
    new_name = game_date
    old_db = f"sqlite:///optuna/predict_optimize/{save_name}.sqlite3"
    new_db = f"sqlite:///optuna/predict_optimize/{save_name}.sqlite3"
    next_study = get_new_study_stack(old_db, new_db, old_name, new_name, num_trials=50)
    return next_study


#%%

i=20
test_date = run_params['test_time_split']
ens_vers = run_params['class_ens_vers']

query_cuts_df = dm.read(f"SELECT * FROM Best_Choices WHERE ens_vers='{run_params['class_ens_vers']}'", 'Simulation')
query_cuts_df.date_run = pd.to_datetime(query_cuts_df.date_run)
query_cuts_df = query_cuts_df[query_cuts_df.date_run==query_cuts_df.date_run.max()].reset_index(drop=True)

query_cuts = {}
for i, row in query_cuts_df.iterrows():
    query_cuts[row.label] = {}
    query_cuts[row.label]= row.to_dict()
    if query_cuts[row.label]['wt_col'] == 'None': query_cuts[row.label]['wt_col'] = None

print(run_params['class_ens_vers'])
for cut_name, cut_dict in query_cuts.items():

    decimal_cut_greater = cut_dict['decimal_cut_greater']
    decimal_cut_less = cut_dict['decimal_cut_less']
    val_greater = cut_dict['value_cut_greater']
    val_less = cut_dict['value_cut_less']
    wt_col = cut_dict['wt_col']
    include_under = bool(cut_dict['include_under'])
    rank_order = cut_dict['rank_order']
    bet_type = cut_dict['bet_type']
    remove_combos = bool(cut_dict['no_combos'])
    remove_threes = bool(cut_dict['remove_threes'])
    if bet_type=='sgp':
        matchup_rank = int(cut_dict['matchup_rank'])
        num_matchups = int(cut_dict['num_matchups'])

    print('\n=======\n', val_greater, val_less, wt_col, decimal_cut_greater, decimal_cut_less, include_under, '\n=======\n')
    print(f"Start spot {cut_dict['start_spot']}, num choices {cut_dict['num_choices']}")
    print(f'{cut_name}, Score:', np.round(cut_dict['winnings'],1))
    print(f'{cut_name}, Win Pct:', np.round(cut_dict['num_wins_pct'],2))
    print(f'{cut_name}, Num Correct Pct:', np.round(cut_dict['num_correct_pct'],2))

    q = f'''SELECT * 
            FROM Over_Probability_New 
            WHERE value {val_greater}
                AND value {val_less}
                AND decimal_odds {decimal_cut_greater}
                AND decimal_odds {decimal_cut_less}
                AND ens_vers = '{ens_vers}'
                AND metric IN ('points', 'assists', 'rebounds', 'three_pointers',
                               'assists_rebounds', 'points_assists', 'points_rebounds')
            ORDER BY game_date ASC
            '''

    model = 'lr_c'

    train_pred = dm.read(q, 'Simulation')
    train_pred = train_pred.drop('y_act', axis=1).rename(columns={'y_act_prob': 'y_act'})
    train_pred = get_date_info(train_pred)
    train_pred = train_pred.dropna().reset_index(drop=True)
    train_pred, test_pred, cv_time_input = train_split(train_pred, test_date=test_date, num_back_days=60)
    
    X_train = preprocess_X(train_pred, wt_col, model, cv_time_input)
    X_test = preprocess_X(test_pred, wt_col, model, cv_time_input)
    y_train = train_pred.y_act

    
    if rank_order in ['stack_model', 'avg']:
        save_name = create_save_path(decimal_cut_greater, decimal_cut_less, val_greater, val_less, wt_col, include_under='', foldername='pick_choices')
        last_run_date = find_last_run(ens_vers, tablename='Stack_Model_Predict', dbname=run_params['db_stack_predict_last'])

        skm, _, _ = get_skm(pd.concat([X_train, y_train], axis=1), model_obj='class', to_drop=[])
        pipe, params = get_full_pipe(skm,  model, stack_model='random_kbest', alpha=None, 
                                        min_samples=10, bayes_rand='optuna')
        
        params['random_sample__frac'] = ['real', 0.2, 1]
        params['k_best_c__k'] = ['int', 3, X_train.shape[1]]

        trials = get_optuna_study_stack(save_name, test_date, '')
        best_model, _, stack_pred, trial_obj = skm.best_stack(pipe, params, X_train, y_train, 
                                                                n_iter=25, alpha=None, wt_col=wt_col,
                                                                trials=trials, bayes_rand='optuna',
                                                                run_adp=False, print_coef=False,
                                                                proba=True, num_k_folds=run_params['num_k_folds'],
                                                                random_state=(i*2)+(i*7), optuna_timeout=180)
            
        show_calibration_curve(y_train, stack_pred['stack_pred'], n_bins=8)

        for c in X_train.columns:
            if c not in X_test.columns:
                X_test[c] = 0

        X_test = X_test[X_train.columns]
        best_model.fit(X_train, y_train)

        preds_stack = pd.Series(best_model.predict_proba(X_test)[:,1], name='final_pred')
        preds_stack = pd.concat([preds_stack, test_pred.reset_index(drop=True)], axis=1)

        preds_avg = preds_stack.copy()
        preds_avg['avg_prob'] = preds_avg[['final_pred', 'prob_over']].mean(axis=1)

    preds_orig = test_pred.reset_index(drop=True).copy()
    
    dummy_dict = {'xx': get_choices_dict()}
    if rank_order=='original':
        if include_under: preds_orig = flip_probs(preds_orig, pred_col='prob_over')
        if bet_type=='sgp': preds_orig, _ = top_sgp_choices('xx', preds_orig, 'prob_over', dummy_dict, matchup_rank, num_matchups, remove_combos, remove_threes)
        if bet_type=='all': preds_orig, _ = top_all_choices(preds_orig, 'prob_over', 'xx', dummy_dict, remove_combos, remove_threes)

    if include_under and rank_order in ['stack_model', 'avg']:
        preds_stack = flip_probs(preds_stack, pred_col='final_pred')
        preds_avg = flip_probs(preds_avg, pred_col='avg_prob')
        if bet_type=='sgp':
            preds_stack, _ = top_sgp_choices('xx', preds_stack, 'final_pred', dummy_dict, matchup_rank, num_matchups, remove_combos, remove_threes)
            preds_avg, _ = top_sgp_choices('xx', preds_avg, 'avg_prob', dummy_dict, matchup_rank, num_matchups, remove_combos, remove_threes)
        if bet_type=='all':
            preds_stack, _ = top_all_choices(preds_stack, 'final_pred', 'xx', dummy_dict, remove_combos, remove_threes)
            preds_avg, _ = top_all_choices(preds_avg, 'avg_prob', 'xx', dummy_dict, remove_combos, remove_threes)

    
    if rank_order=='stack_model': display(preds_stack.sort_values(by='final_pred', ascending=False).head(10))
    elif rank_order=='original': display(preds_orig.sort_values(by='prob_over', ascending=False).head(10))
    elif rank_order=='avg': display(preds_avg.sort_values(by='avg_prob', ascending=False).head(10))

# %%
