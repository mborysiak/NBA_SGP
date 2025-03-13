#%%
import pandas as pd 
import numpy as np
from ff.db_operations import DataManage
from ff import general as ffgeneral
import ff.data_clean as dc
from scipy.stats.morestats import shapiro
import datetime as dt
import optuna

from scipy.stats import poisson, truncnorm
from typing import Dict
from sklearn.preprocessing import StandardScaler
from Fix_Standard_Dev import *
from skmodel import SciKitModel
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import brier_score_loss
# from skmodel import SciKitModel
from hyperopt import Trials
from sklearn.metrics import r2_score
import datetime as dt
from skmodel import ProbabilityEstimator, TruncatedNormalEstimator

warnings.filterwarnings('ignore')
root_path = ffgeneral.get_main_path('NBA_SGP')
db_path = f'{root_path}/Data/'
dm = DataManage(db_path)

pd.set_option('display.max_columns', 999)


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

def create_y_act(df, metric):

    if metric in ('points_assists', 'points_rebounds', 'points_rebounds_assists', 'steals_blocks', 'assists_rebounds'):
        metric_split = metric.split('_')
        df[f'y_act_{metric}'] = df[['y_act_' + c for c in metric_split]].sum(axis=1)
        df = create_metric_split_columns(df, metric_split)

    df = df.drop([c for c in df.columns if 'y_act' in c and metric not in c], axis=1)
    df = df.rename(columns={f'y_act_{metric}': 'y_act'})
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

def train_predict_split(df, run_params):

    # # get the train / predict dataframes and output dataframe
    df_train = df[df.game_date < run_params['train_time_split']].reset_index(drop=True)
    df_train = df_train.dropna(subset=['y_act']).reset_index(drop=True)

    df_predict = df[df.game_date == run_params['train_time_split']].reset_index(drop=True)
    output_start = df_predict[['player', 'game_date']].copy().drop_duplicates()

    # get the minimum number of training samples for the initial datasets
    min_samples = 100#int(df_train[df_train.game_date < run_params['cv_time_input']].shape[0])  
    print('Shape of Train Set', df_train.shape)

    return df_train, df_predict, output_start, min_samples

def get_over_under_class(df, metric, run_params, model_obj='class'):
    
    odds = pull_odds(metric, run_params)
    df = pd.merge(df, odds, on=['player', 'year'])
    df = df.sort_values(by='game_date').reset_index(drop=True)

    if model_obj == 'class': df['y_act'] = np.where(df.y_act >= df.value, 1, 0)
    elif model_obj == 'reg': df['y_act'] = df.y_act - df.value

    df = create_value_columns(df, metric)
    df = remove_low_counts(df)
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




#%%


train_date = '20250201'
alpha = 0.5
use_vegas = 'choose'
model_obj = 'class'
m_obj = 'corr_class'

for metric in ['rebounds']:
    # for alpha in [0.35, 0.5, 0.65]:
        # m_obj = alpha
    # for m_obj in ['class', 'quantile', 'tweedie','poisson', 'regression']:
        # if m_obj != 'class': model_obj = 'reg'
        # else: model_obj = 'class'

    if model_obj =='class': proba = True
    else: proba = False

    print(train_date)
    print(metric)
    print(model_obj)
    print(m_obj)

    Xy = dm.read(f'SELECT * FROM Model_Data_{train_date}', f'Model_Features')
    # Xy2 = dm.read(f'SELECT * FROM Model_Data_{train_date}v2', f'Model_Features')
    # Xy = pd.concat([Xy1, Xy2], axis=1)
    Xy = Xy.sort_values(by=['game_date']).reset_index(drop=True)

    Xy.game_date = Xy.game_date.apply(lambda x: int(x.replace('-', '')))
    Xy['year'] = Xy.game_date
    Xy['week'] = 1

    run_params = {
        'parlay': False,
        'train_time_split': Xy.game_date.max()-1,
        'cv_time_input': int((dt.datetime.strptime(train_date, '%Y%m%d') - dt.timedelta(days=30)).strftime('%Y%m%d'))
    }

    Xy = create_y_act(Xy, metric)
    Xy['y_act'] = Xy.y_act + (np.random.random(size=len(Xy)) / 1000)
    if model_obj == 'class': Xy = get_over_under_class(Xy, metric, run_params, model_obj='class')

    # Xy = Xy.drop(f'y_act_{metric}', axis=1)
    # Xy = Xy.rename(columns={f'y_act_{metric}_over': 'y_act'})
    
    train, pred, _, _ = train_predict_split(Xy, run_params)

    preds = []
    actuals = []

    skm = SciKitModel(train, model_obj=model_obj, alpha=alpha, hp_algo='atpe')
    to_drop = list(train.dtypes[train.dtypes=='object'].index)
    X, y = skm.Xy_split('y_act', to_drop = to_drop)
    y = pd.Series(np.where(y<0, 0, y), name='y_act')

    if proba:
        p = 'select_perc_c'
        kb = 'k_best_c'
        m = 'cb_c'

    elif model_obj == 'reg':
        p = 'select_perc'
        kb = 'k_best'
        m = 'cb'

        if m_obj == 'poisson': m += '_p'
        elif m_obj == 'tweedie': m += '_t'

    elif model_obj == 'quantile':
        p = 'select_perc'
        kb = 'k_best'
        m = 'cb_q'
    

    trials = optuna.create_study(
        storage=f"sqlite:///optuna/experiments/{m}_{metric}_{model_obj}.sqlite3", 
    )

    pipe = skm.model_pipe([skm.piece('feature_drop'),
                            skm.piece('random_sample'),
                            skm.piece('std_scale'), 
                            skm.piece(p),
                            skm.feature_union([
                                            skm.piece('agglomeration'), 
                                            skm.piece(f'{kb}_fu'),
                                            skm.piece('pca')
                                            ]),
                            skm.piece(kb),
                            skm.piece(m)
                        ])
    
    if m == 'cb_q': pipe.set_params(**{f'{m}__loss_function': f'Quantile:alpha={alpha}'})

    params = skm.default_params(pipe, 'optuna')
    params['feature_drop__drop_cols'] = ['cat', [[c for c in X.columns if 'ev' in c], []]]

    best_models, oof_data, param_scores, _ = skm.time_series_cv(pipe, X, y, params, n_iter=10,
                                                                col_split='game_date',n_splits=5,
                                                                time_split=run_params['cv_time_input'], 
                                                                alpha=alpha, bayes_rand='optuna', proba=proba,
                                                                sample_weight=False, trials=trials,
                                                                random_seed=64893, optuna_timeout=120)
                                                                
    print('R2 score:', r2_score(oof_data['full_hold']['y_act'], oof_data['full_hold']['pred']))
    oof_data['full_hold'].plot.scatter(x='pred', y='y_act')
    plt.show()
    try: 
        show_calibration_curve(oof_data['full_hold'].y_act, oof_data['full_hold'].pred, n_bins=6)
        print('Brier score:', brier_score_loss(oof_data['full_hold']['y_act'], oof_data['full_hold']['pred']))
        plt.show()
    except: pass

    # display(oof_data['full_hold'].sort_values(by='pred', ascending=False).iloc[:50])
    # display(oof_data['full_hold'].sort_values(by='pred', ascending=True).iloc[:50])

    # for i in range(4):

    #     import matplotlib.pyplot as plt

    #     pipeline = best_models[i]
    #     pipeline.fit(X,y)
    #     # Extract the coefficients
    #     log_reg = pipeline.named_steps[m]

    #     try:
    #         log_reg.coef_.shape[1]
    #         coefficients = log_reg.coef_[0]
    #         cutoff = np.percentile(np.abs(coefficients), 80)
    #     except: 
    #         try:
    #             coefficients = log_reg.coef_
    #             cutoff = np.percentile(np.abs(coefficients), 80)
    #         except:
    #             coefficients = log_reg.feature_importances_
    #             cutoff = np.percentile(np.abs(coefficients), 80)

    #     # Get the feature names from SelectKBest
    #     rand_features = pipeline.named_steps['random_sample'].columns
    #     X_out = X[rand_features]
    #     selected_features = pipeline.named_steps[kb].get_support(indices=True)

    #     coef = pd.Series(coefficients, index=X_out.columns[selected_features])
    #     coef = coef[np.abs(coef) > cutoff].sort_values()
    #     coef.plot(kind = 'barh', figsize=(8, len(coef)/3))
    #     plt.show()

    train = oof_data['full_val'].assign(metric=metric, model=m, model_obj=model_obj, m_obj=m_obj)
    pred = oof_data['full_hold'].assign(metric=metric, model=m, model_obj=model_obj, m_obj=m_obj)
    
    dm.delete_from_db('Experiments', 'Train_Output', f"metric='{metric}' AND model='{m}' AND model_obj='{model_obj}' AND m_obj='{m_obj}'", create_backup=False)
    dm.write_to_db(train, 'Experiments', 'Train_Output', 'append')
    dm.write_to_db(pred, 'Experiments', 'Pred_Output', 'append')

#%%

metric = 'points_assists'
model = 'cb_q'
model_obj = 'quantile'

odds = dm.read(f'''SELECT player, game_date, value, decimal_odds
                FROM Draftkings_Odds 
                WHERE stat_type='{metric}' 
                        AND over_under='over' 
                        
            ''', 'Player_Stats')
odds.game_date = odds.game_date.apply(lambda x: int(x.replace('-', '')))
odds = odds.rename(columns={'game_date': 'year'})

train_df = dm.read('SELECT * FROM Train_Output', 'Experiments')
pred_df = dm.read('SELECT * FROM Pred_Output', 'Experiments')
train_df.y_act = train_df.y_act.astype('int')
pred_df.y_act = pred_df.y_act.astype('int')

train_df = pd.merge(train_df, odds, on=['player', 'year'])
pred_df = pd.merge(pred_df, odds, on=['player', 'year'])

train = train_df[(train_df.metric==metric) & (train_df.model==model) & (train_df.model_obj==model_obj)]
train = pd.pivot_table(train, index=['player', 'year', 'value', 'decimal_odds', 'y_act'], 
                       columns='m_obj', values='pred', aggfunc='mean')

train.columns = [f'pred_q{int(100*float(c))}' for c in train.columns]
train.loc[train.pred_q35<0, 'pred_q35'] = 0.0001
train.loc[train.pred_q50<train.pred_q35, 'pred_q50'] = train.loc[train.pred_q50<train.pred_q35, 'pred_q50'] + 0.001
train = train.reset_index()


pred = pred_df[(pred_df.metric==metric) & (pred_df.model==model) & (pred_df.model_obj==model_obj)]
pred = pd.pivot_table(pred, index=['player', 'year', 'value', 'decimal_odds', 'y_act'], 
                       columns='m_obj', values='pred', aggfunc='mean')
pred.columns = [f'pred_q{int(100*float(c))}' for c in pred.columns]
pred.loc[pred.pred_q35<0, 'pred_q35'] = 0.0001
pred.loc[pred.pred_q50<pred.pred_q35, 'pred_q50'] = pred.loc[pred.pred_q50<pred.pred_q35, 'pred_q35'] + 0.001

pred = pred.reset_index()

def get_accuracy_metrics(pred_acc, probs):
    pred = pred_acc.copy()
    pred['p_over'] = probs
    pred['y_act_over'] = np.where(pred.y_act > pred.value, 1, 0)

    r2_sc = r2_score(pred.y_act_over, pred.p_over)
    print('R2 score:', r2_sc)
    pred.plot.scatter(x='p_over', y='y_act_over')
    try: 
        br_sc = brier_score_loss(pred.y_act_over, pred.p_over)
        print('Brier score:', br_sc)
        show_calibration_curve(pred.y_act_over, pred.p_over, n_bins=6)
    except: pass

    return r2_sc, br_sc

# Create and fit the estimator
trunc_norm = TruncatedNormalEstimator(spline_type='spline', n_groups_min=0.03, n_groups_max=0.06)
metrics = ['pred_q35', 'pred_q50', 'pred_q65']
trunc_norm.fit(train[metrics], train.y_act)
probs = trunc_norm.predict_proba(pred[metrics], pred.value)
trunc_norm.plot_fits()

trunc_r2, trunc_brier = get_accuracy_metrics(pred, probs)

quantile_mapping = {
    'pred_q35': 0.35,
    'pred_q50': 0.50,
    'pred_q65': 0.65
}
poisson_estimator = ProbabilityEstimator('poisson', quantile_mapping)
probs = poisson_estimator.calc_exceedance_probs(pred, threshold_column='value')
poisson_r2, poisson_brier = get_accuracy_metrics(pred, probs)

#%%
# gamma_estimator = ProbabilityEstimator('gamma', quantile_mapping)
# probs = gamma_estimator.calc_exceedance_probs(pred, threshold_column='value')
# gamma_r2, gamma_brier = get_accuracy_metrics(pred, probs)

#%%

metrics = [m, metric, m_obj, use_vegas, r2_sc, br_sc]
save_df = pd.DataFrame(metrics).T
save_df.columns = ['pred_model','metric', 'model', 'use_vegas', 'r2', 'brier']
dm.write_to_db(save_df, 'Simulation', 'Experiments', 'append')
# output.sort_values(by='p_over', ascending=True).iloc[:50]

#%%
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

df = dm.read('SELECT * FROM Experiments', 'Simulation')

y = df.r2
X = df[['metric', 'model', 'use_vegas']]
X = pd.get_dummies(X, drop_first=True)
lr = LinearRegression()
lr.fit(X, y)
pd.Series(lr.coef_, index=X.columns).sort_values()

#%%

X = dm.read("SELECT * FROM X_Stack_Test", 'Experiments')
y = dm.read("SELECT * FROM y_Stack_Test", 'Experiments')

model_obj = 'class'
alpha=None
proba = True
metric = 'rebounds'

skm = SciKitModel(pd.concat([X,y], axis=1), model_obj=model_obj, alpha=alpha, hp_algo='tpe')
to_drop = []
X, y = skm.Xy_split('y_act', to_drop = to_drop)

if proba:
    p = 'select_perc_c'
    kb = 'k_best_c'
    m = 'rf_c'


trials = optuna.create_study(
    storage=f"sqlite:///optuna/experiments/{m}_{metric}_{model_obj}.sqlite3", 
)

pipe = skm.model_pipe([#skm.piece('feature_drop'),
                        skm.piece('random_sample'),
                        # skm.piece('std_scale'), 
                        # skm.piece(p),
                        # skm.feature_union([
                        #                 skm.piece('agglomeration'), 
                        #                 skm.piece(f'{kb}_fu'),
                        #                 skm.piece('pca')
                        #                 ]),
                        skm.piece(kb),
                        skm.piece(m)
                    ])


params = skm.default_params(pipe, 'optuna')

best_model, stack_scores, stack_pred, trial = skm.best_stack(pipe, params, X, y, 
                                                            n_iter=50, alpha=alpha, wt_col=None,
                                                            bayes_rand='optuna',trials=trials,
                                                            run_adp=False, print_coef=False,
                                                            proba=proba, num_k_folds=3,
                                                            random_state=12345, 
                                                            optuna_timeout=120)

params['random_sample__frac'] = ['real', 0.2, 1]
params[f'{kb}__k'] = ['int', 5, 50]

#%%

actuals = stack_pred['y']
preds = stack_pred['stack_pred']

print('R2 score:', r2_score(actuals, preds))
plt.scatter(preds, actuals)
plt.show()
try: 
    show_calibration_curve(actuals, preds, n_bins=6)
    print('Brier score:', brier_score_loss(actuals, preds))
    plt.show()
except: pass

import matplotlib.pyplot as plt

pipeline = best_model
pipeline.fit(X,y)
# Extract the coefficients
log_reg = pipeline.named_steps[m]

try:
    log_reg.coef_.shape[1]
    coefficients = log_reg.coef_[0]
    cutoff = np.percentile(np.abs(coefficients), 80)
except: 
    try:
        coefficients = log_reg.coef_
        cutoff = np.percentile(np.abs(coefficients), 80)
    except:
        coefficients = log_reg.feature_importances_
        cutoff = np.percentile(np.abs(coefficients), 80)

# Get the feature names from SelectKBest
rand_features = pipeline.named_steps['random_sample'].columns
X_out = X[rand_features]
selected_features = pipeline.named_steps[kb].get_support(indices=True)

coef = pd.Series(coefficients, index=X_out.columns[selected_features])
coef = coef[np.abs(coef) > cutoff].sort_values()
coef.plot(kind = 'barh', figsize=(8, len(coef)/3))
plt.show()
# %%
