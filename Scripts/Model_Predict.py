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

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)


#====================
# Data Loading Functions
#====================

def create_pkey_output_path(metric, run_params, vers):

    pkey = f"{metric}_{run_params['train_date']}_{vers}"
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
    df_train = df[df.game_date < run_params['train_date']].reset_index(drop=True)
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
        'reg': SciKitModel(skm_df, model_obj='reg', r2_wt=r2_wt, sera_wt=sera_wt, mse_wt=mse_wt),
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
        params['feature_union__agglomeration__n_clusters'] = range(2, 10, 1)
        params[f'feature_union__{kb}__k'] = range(5, 20, 2)
        params['feature_union__pca__n_components'] = range(2, 10, 1)
        params[f'{kb}__k'] = range(1, 30)
    
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
                     stack_val_pred, model_obj='reg', alpha = None, show_plots=True,
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
                                                          random_state=(i*12)+(i*17))

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


def create_output(output_start, predictions, predictions_quantile=None):

    output = output_start.copy()
    output['pred_fp_per_game'] = predictions.mean(axis=1)

    if predictions_quantile is not None:
        output['pred_fp_per_game_quantile'] = predictions_quantile.mean(axis=1)

    output = output.sort_values(by='pred_fp_per_game', ascending=False).reset_index(drop=True)

    return output



def std_dev_features(cur_df, model_name, run_params, show_plot=True):

    skm, X, y = get_skm(cur_df, 'reg', to_drop=run_params['drop_cols'])
    pipe, params = get_full_pipe(skm, model_name, std_model=True)

    # fit and append the ADP model
    best_models, _, _ = skm.time_series_cv(pipe, X, y, params, n_iter=run_params['n_iters'], n_splits=run_params['n_splits'],
                                           col_split='game_date', time_split=run_params['cv_time_input'],
                                           bayes_rand='custom_rand', random_seed=1234)
    
    for bm in best_models: bm.fit(X, y)
    if show_plot:
        mf.shap_plot(best_models, X, model_num=0)
        plt.show()

    return best_models, X


def add_std_dev_max(df_train, df_predict, output, model_name, run_params, num_cols=10, iso_spline='iso'):

    std_dev_models, X = std_dev_features(df_train, model_name, run_params, show_plot=True)
    sd_cols, df_train, df_predict = mf.get_sd_cols(df_train, df_predict, X, std_dev_models, num_cols=num_cols)
    
    if iso_spline=='iso':
        sd_m, max_m, min_m = get_std_splines(df_train, sd_cols, show_plot=True, k=2, 
                                            min_grps_den=int(df_train.shape[0]*0.08), 
                                            max_grps_den=int(df_train.shape[0]*0.04),
                                            iso_spline=iso_spline)

    elif iso_spline=='spline':
        sd_m, max_m, min_m = get_std_splines(df_train, sd_cols, show_plot=True, k=2, 
                                            min_grps_den=int(df_train.shape[0]*0.1), 
                                            max_grps_den=int(df_train.shape[0]*0.05),
                                            iso_spline=iso_spline)

    output = assign_sd_max(output, df_predict, df_train, sd_cols, sd_m, max_m, min_m, iso_spline)

    return output


def assign_sd_max(output, df_predict, sd_df, sd_cols, sd_m, max_m, min_m, iso_spline):
    
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    sc.fit(sd_df[list(sd_cols.keys())])

    df_predict = df_predict.set_index('player')
    df_predict = df_predict.reindex(index=output['player'])
    df_predict = df_predict.reset_index()

    pred_sd_max = pd.DataFrame(sc.transform(df_predict[list(sd_cols.keys())])) * list(sd_cols.values())
    pred_sd_max = pred_sd_max.mean(axis=1)

    if iso_spline=='spline':
        output['std_dev'] = sd_m(pred_sd_max)
        output['max_score'] = max_m(pred_sd_max)
        output['min_score'] = min_m(pred_sd_max)
    elif iso_spline=='iso':
        output['std_dev'] = sd_m.predict(pred_sd_max)
        output['max_score'] = max_m.predict(pred_sd_max)
        output['min_score'] = min_m.predict(pred_sd_max)

    output.loc[(output.max_score < output.pred_fp_per_game), 'max_score'] = \
        output.loc[(output.max_score < output.pred_fp_per_game), 'pred_fp_per_game'] * 2
    
    return output

def val_std_dev(output, val_data, metrics={'pred_fp_per_game': 1}, iso_spline='iso', show_plot=True):
        
    sd_max_met = StandardScaler().fit(val_data[list(metrics.keys())]).transform(output[list(metrics.keys())])
    sd_max_met = np.mean(sd_max_met, axis=1)

    if iso_spline=='iso':
        sd_m, max_m, min_m = get_std_splines(val_data, metrics, show_plot=show_plot, k=2, 
                                            min_grps_den=int(val_data.shape[0]*0.1), 
                                            max_grps_den=int(val_data.shape[0]*0.05),
                                            iso_spline=iso_spline)
        output['std_dev'] = sd_m.predict(sd_max_met)
        output['max_score'] = max_m.predict(sd_max_met)
        output['min_score'] = min_m.predict(sd_max_met)

    elif iso_spline=='spline':
        sd_m, max_m, min_m = get_std_splines(val_data, metrics, show_plot=show_plot, k=2, 
                                            min_grps_den=int(val_data.shape[0]*0.1), 
                                            max_grps_den=int(val_data.shape[0]*0.05),
                                            iso_spline=iso_spline)
        output['std_dev'] = sd_m(sd_max_met)
        output['max_score'] = max_m(sd_max_met)
        output['min_score'] = min_m(sd_max_met)
 
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


def save_output_to_db(output, run_params):
    d = str(run_params['test_time_split'])
    d = dt.date(int(d[:4]), int(d[4:6]), int(d[6:]))

    output['metric'] = metric
    output['pred_vers'] = pred_vers
    output['ensemble_vers'] = ensemble_vers
    output['std_dev_type'] = std_dev_type
    output['game_date'] = d

    output = output[['player', 'game_date', 'metric', 'pred_fp_per_game', 'std_dev', 'min_score', 'max_score',
                     'pred_vers', 'ensemble_vers', 'std_dev_type']]

    del_str = f'''pred_vers='{pred_vers}'
                  AND metric='{metric}'
                  AND ensemble_vers='{ensemble_vers}' 
                  AND std_dev_type='{std_dev_type}'
                  AND game_date={d} 
                '''
    dm.delete_from_db('Simulation', 'Model_Predictions', del_str, create_backup=False)
    dm.write_to_db(output, 'Simulation', f'Model_Predictions', 'append')



#%%
#==========
# General Setting
#==========



#---------------
# Settings
#---------------

run_params = {
    
    # set year and week to analyze
    'cv_time_input_orig': '2023-02-03',
    'train_date_orig': '2023-02-13',
    'test_time_split_orig': '2023-02-14',
    'metrics': ['points', 'rebounds', 'assists', 'three_pointers'],

    'n_iters': 25,
    'n_splits': 5
}

run_params['cv_time_input'] = int(run_params['cv_time_input_orig'].replace('-', ''))
run_params['train_date'] = int(run_params['train_date_orig'].replace('-', ''))
run_params['test_time_split'] = int(run_params['test_time_split_orig'].replace('-', ''))

# set weights for running model
r2_wt = 1
sera_wt = 0
mse_wt = 5

# set version and iterations
pred_vers = 'mse5_rsq1_lowsample_perc'
ensemble_vers = 'mse5_rsq1_fullstack'
std_dev_type = 'pred_quant_spline'

full_stack_features = True

min_include = 2
show_plot= True
print_coef = False
num_k_folds = 3

r2_wt = 1
sera_wt = 0
mse_wt = 5
brier_wt = 1
matt_wt = 0


for metric in run_params['metrics']:

    # load data and filter down
    pkey, model_output_path = create_pkey_output_path(metric, run_params, pred_vers)
    df, run_params = load_data(run_params)

    df = df.drop([c for c in df.columns if 'y_act' in c and metric not in c], axis=1)
    df = df.rename(columns={f'y_act_{metric}': 'y_act'})
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


    # quantile regression metrics
    final_models = ['qr_q', 'gbm_q', 'lgbm_q', 'rf_q', 'knn_q']

    # def run_stack_models_load()

    for alpha in [0.8]:# [0.2, 0.5, 0.8]:
        if os.path.exists(f'{model_output_path}quant{alpha*100}_{ensemble_vers}.p'):
            best_models_quant, scores, stack_val_pred = load_stack_runs(model_output_path, f'quant{alpha*100}_' + ensemble_vers)
        else:
            stack_val_pred = pd.DataFrame(); scores = []; best_models_quant = []
            for i, fm in enumerate(final_models):
                best_models_quant, scores, stack_val_pred = run_stack_models(fm, i, X_stack, y_stack, best_models_quant, 
                                                                            scores, stack_val_pred, model_obj='quantile',
                                                                            alpha=alpha, show_plots=show_plot, 
                                                                            num_k_folds=num_k_folds, print_coef=print_coef)
            save_stack_runs(model_output_path, f'quant{alpha*100}_' + ensemble_vers, best_models_quant, scores, stack_val_pred)
            
        predictions = stack_predictions(X_predict, best_models_quant, final_models, model_obj='quantile')
        best_val_quant, best_predictions_quant, _ = average_stack_models(scores, final_models, y_stack, stack_val_pred, 
                                                                        predictions, model_obj='quantile', show_plot=show_plot, 
                                                                        min_include=min_include)


    # create the stacking models
    final_models = ['ridge', 'lasso', 'huber', 'lgbm', 'xgb', 'rf', 'bridge', 'gbm', 'gbmh', 'knn']
    if os.path.exists(f'{model_output_path}reg_{ensemble_vers}.p'):
        best_models_reg, scores, stack_val_pred = load_stack_runs(model_output_path, 'reg_' + ensemble_vers)
    else:
        stack_val_pred = pd.DataFrame(); scores = []; best_models_reg = []
        for i, fm in enumerate(final_models):
            best_models_reg, scores, stack_val_pred = run_stack_models(fm, i, X_stack, y_stack, best_models_reg, 
                                                                       scores, stack_val_pred, show_plots=show_plot,
                                                                       num_k_folds=num_k_folds, print_coef=print_coef)
        save_stack_runs(model_output_path, 'reg_' + ensemble_vers, best_models_reg, scores, stack_val_pred)

    predictions = stack_predictions(X_predict, best_models_reg, final_models)
    best_val_reg, best_predictions, best_score = average_stack_models(scores, final_models, y_stack, stack_val_pred, 
                                                                      predictions, model_obj='reg',
                                                                      show_plot=show_plot, min_include=min_include)
    
    # create the output and add standard devations / max scores
    output = create_output(output_start, best_predictions, best_predictions_quant)
    metrics = {'pred_fp_per_game': 1, 'pred_fp_per_game_quantile': 1}

    val_data = pd.concat([X_stack_player[['player', 'year']], 
                          pd.Series(best_val_reg.mean(axis=1), name='pred_fp_per_game'),
                          pd.Series(best_val_quant.mean(axis=1), name='pred_fp_per_game_quantile'),
                          pd.Series(y_stack, name='y_act')], axis=1)

    output = val_std_dev(output, val_data, metrics=metrics,  iso_spline='spline', show_plot=True)
    
    try:  
        output = add_actual(output)
        print(output.loc[:50, ['player', 'game_date', 'pred_fp_per_game', 'pred_fp_per_game_quantile', 
                               f'actual_{metric}', 'std_dev', 'min_score', 'max_score']])
                
        if show_plot: 
            show_scatter_plot(output.pred_fp_per_game, output[f'actual_{metric}'])
            skm_score, _, _ = get_skm(df_train, model_obj='reg', to_drop=[])
            print('Score:', np.round(skm_score.custom_score(output.pred_fp_per_game, output[f'actual_{metric}']),2))
        
        output = output.drop(f'actual_{metric}', axis=1)
    except:
        print(output.loc[:50, ['player', 'game_date', 'pred_fp_per_game', 'pred_fp_per_game_quantile', 
                               'std_dev', 'min_score', 'max_score']])

    save_output_to_db(output, run_params)


#%%

def pull_odds(metric):

    odds = dm.read(f'''SELECT player, stat_type, game_date year, value, over_under
                       FROM Draftkings_Odds 
                    ''', 'Player_Stats')
    odds.stat_type = odds.stat_type.apply(lambda x: x.lower())
    odds.year = odds.year.apply(lambda x: int(x.replace('-', '')))

    odds = odds[odds.stat_type==metric].reset_index(drop=True)
    odds = odds.loc[odds.over_under=='over', ['player', 'year', 'value']]

    return odds

def get_over_under_class(df, metric, run_params):
    
    odds = pull_odds(metric)
    df = pd.merge(df, odds, on=['player', 'year'])
    df['y_act'] = np.where(df.y_act > df.value, 1, 0)

    df_train, df_predict, output_start, min_samples = train_predict_split(df, run_params)
    
    return df_train, df_predict, output_start, min_samples

def X_y_stack_class(df, metric, run_params):

    df_train_class, df_predict_class, _, _ = get_over_under_class(df, metric, run_params)

    # load the regregression predictions
    pred, actual, models_class, _, full_hold_class = load_all_pickles(model_output_path, 'class')
    X_stack_class, y_stack_class, df_labels = X_y_stack('class', full_hold_class)
    X_stack_class = pd.concat([df_labels[['player', 'week', 'year']], X_stack_class], axis=1)

    _, X, y = get_skm(df_train_class, 'class', to_drop=run_params['drop_cols'])
    X_predict_class = create_stack_predict(df_predict_class, models_class, X, y, proba=True)
    X_predict_class = pd.concat([df_predict_class.player, X_predict_class], axis=1)
    return df_predict_class, X_stack_class, y_stack_class, X_predict_class

 #-------------
# Running the million dataset
#-------------

df_predict_mil, X_stack_mil, y_stack_mil, X_predict_mil = X_y_stack_class(df, metric, run_params)

X_stack_mil = pd.merge(X_stack_mil, X_stack_player, on=['player', 'week', 'year'], how='left')

X_predict_player = pd.concat([df_predict.player, X_predict.copy()], axis=1)
X_predict_player = X_predict_player.loc[:,~X_predict_player.columns.duplicated()].copy()
X_predict_mil = pd.merge(X_predict_mil, X_predict_player, on=['player'])

X_stack_mil = X_stack_mil.drop(['player', 'week', 'year'], axis=1).dropna(axis=0)
y_stack_mil = y_stack_mil[y_stack_mil.index.isin(X_stack_mil.index)]
X_stack_mil, y_stack_mil = X_stack_mil.reset_index(drop=True), y_stack_mil.reset_index(drop=True)

X_predict_mil = X_predict_mil.drop(['player'], axis=1)

# class metrics
final_models = ['lr_c', 'lgbm_c', 'rf_c', 'gbm_c', 'gbmh_c', 'xgb_c', 'knn_c']

if os.path.exists(f'{model_output_path}mil_{std_dev_type}.p'):
    best_models, scores, stack_val_pred_mil = load_stack_runs(model_output_path, 'class_' + std_dev_type)
else:
    stack_val_pred_mil = pd.DataFrame(); scores = []; best_models = []
    for i, fm in enumerate(final_models):
        best_models, scores, stack_val_pred_mil = run_stack_models(fm, i, X_stack_mil, y_stack_mil, best_models, 
                                                                   scores, stack_val_pred_mil, model_obj='class',
                                                                   show_plots=False, num_k_folds=num_k_folds,
                                                                   print_coef=print_coef)

    if show_plot: show_calibration_curve(y_stack_mil, stack_val_pred_mil[fm], n_bins=8)
    save_stack_runs(model_output_path, 'class_' + std_dev_type, best_models, scores, stack_val_pred_mil)

predictions = stack_predictions(X_predict_mil, best_models, final_models, model_obj='class')
best_val_mil, best_predictions_mil, _ = average_stack_models(scores, final_models, y_stack_mil, stack_val_pred_mil, 
                                                            predictions, model_obj='class', show_plot=show_plot, 
                                                            min_include=min_include)

if show_plot: show_calibration_curve(y_stack_mil, best_val_mil.mean(axis=1), n_bins=8)
# save_mil_data(df_predict_mil, best_predictions_mil, best_val_mil, run_params)

#%%

def pull_predictions(pull_date):
    df = dm.read(f'''SELECT player, metric, pred_fp_per_game, std_dev, min_score, max_score
                     FROM Model_Predictions 
                     WHERE game_date='{pull_date}'
                ''', 'Simulation')
    return df

def trunc_normal(mean_val, sdev, min_sc, max_sc, num_samples=500):

    import scipy.stats as stats

    # create truncated distribution
    lower_bound = (min_sc - mean_val) / sdev, 
    upper_bound = (max_sc - mean_val) / sdev
    trunc_dist = stats.truncnorm(lower_bound, upper_bound, loc=mean_val, scale=sdev)
    
    estimates = trunc_dist.rvs(num_samples)

    return estimates


def trunc_normal_dist(df, num_options=500):

    pred_list = []
    for pred, sdev, min_sc, max_sc in df[['pred_fp_per_game', 'std_dev', 'min_score', 'max_score']].values:
        pred_list.append(trunc_normal(pred, sdev, min_sc, max_sc, num_options))

    return pd.DataFrame(pred_list)

def create_dist(df, odds):
    points_dist = trunc_normal_dist(df)
    df = pd.concat([df[['player', 'metric']], points_dist], axis=1)
    df = pd.melt(df, id_vars=['player', 'metric'], var_name='iter', value_name='dist_pred')

    df = pd.merge(df, odds, on=['player', 'metric'])
    df['pred_over'] = np.where(df.dist_pred >= df.value, 1, 0)
    df['pred_under'] = np.where(df.dist_pred < df.value, 1, 0)
    df = df.groupby(['player', 'metric']).agg({'pred_over': 'sum', 'pred_under': 'sum', 'dist_pred': 'mean'}).reset_index()
    
    return df

def pull_odds(pull_date):

    odds = dm.read(f'''SELECT * 
                       FROM Draftkings_Odds 
                       WHERE game_date='{pull_date}'
                    ''', 'Player_Stats')
    odds.stat_type = odds.stat_type.apply(lambda x: x.lower())
    odds = odds.loc[odds.over_under=='over', ['player', 'stat_type', 'value']]
    odds = odds.reset_index(drop=True).rename(columns={'stat_type': 'metric'})

    return odds

def pull_actuals(pull_date):
    actuals = dm.read(f'''SELECT PLAYER_NAME player, PTS points, REB rebounds, AST assists, FG3M three_pointers
                                FROM Box_Score 
                                WHERE game_date='{pull_date}'
                                ''', 'Player_Stats')
    actuals = pd.melt(actuals, id_vars='player', var_name='metric', value_name='actual')
    return actuals

def add_over_under_labels(df, val_shift=0):
    df['value'] = df.value + val_shift
    df['is_over'] = np.where(df.actual > df.value, 1, 0)
    return df

#----------
# Modeling
#----------

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, RepeatedKFold

def create_X_y(df):
    df = df.sample(frac=1).reset_index(drop=True)
    df['pred_vs_value'] = df.dist_pred - df.value
    df['pred_over_value'] = df.dist_pred / df.value
 
    X_prob = df[['metric', 'pred_fp_per_game', 
                 'dist_pred', 'pred_over', 
                 'std_dev', 'min_score', 'max_score', 
                 'value', 'pred_vs_value', 'pred_over_value']]
    X_prob = pd.concat([X_prob.drop('metric', axis=1), pd.get_dummies(X_prob.metric)], axis=1)
    X_prob = pd.DataFrame(StandardScaler().fit_transform(X_prob), columns=X_prob.columns)
 
    y = df.is_over
    return df, X_prob, y

def get_proba_predictions(model, X, y):

    all_pred = pd.DataFrame()
    from sklearn.model_selection import RepeatedKFold

    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1234)
    for train_index, test_index in rkf.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        pred = pd.DataFrame(model.predict_proba(X_test)).set_index(X_test.index).reset_index()
        all_pred = pd.concat([all_pred, pred], axis=0)

    return all_pred.groupby('index').agg('mean')[1]




#%%

df_model = pd.DataFrame()
val_dates = ['2023-02-12', '2023-02-13', '2023-02-14', '2023-02-15']
for pull_date in val_dates:
    print(pull_date)
    df = pull_predictions(pull_date)
    odds = pull_odds(pull_date)
    df = pd.merge(df, odds, on=['player', 'metric'])

    dist_df = create_dist(df, odds)
    df = pd.merge(df, dist_df, on=['player', 'metric'])

    actuals = pull_actuals(pull_date)
    df = pd.merge(df, actuals, on=['player', 'metric'])
    df = add_over_under_labels(df, val_shift=-2)

    df_model = pd.concat([df_model, df], axis=0)

df_model, X_prob, y = create_X_y(df_model)

lr = LogisticRegression(C=10)
lr.fit(X_prob, y)
print(pd.Series(lr.coef_[0], index=X_prob.columns).sort_values(ascending=False))
print('\n', np.round(-np.mean(cross_val_score(lr, X_prob, y, scoring='neg_brier_score')),3))
# %%

proba_pred = get_proba_predictions(lr, X_prob, y)
df['pred_proba'] = proba_pred
show_calibration_curve(df.is_over, df.pred_proba, n_bins=5)

df.sort_values(by='pred_proba', ascending=False).iloc[:50]

# %%



# %%
