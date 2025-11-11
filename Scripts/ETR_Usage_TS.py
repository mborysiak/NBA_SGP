#%%

import pandas as pd 
from ff.db_operations import DataManage
from ff import general as ffgeneral
from pyfixest import feols

root_path = ffgeneral.get_main_path('NBA_SGP')
db_path = f'{root_path}/Data/'
dm = DataManage(db_path)

def cleanup_minutes(df):
    df = df.dropna(subset=['minutes_played']).reset_index(drop=True)
    df = df[df.minutes_played != ''].reset_index(drop=True)
    df.minutes_played = df.minutes_played.apply(lambda x: float(x.split(':')[0]) + float(x.split(':')[1])/60)
    return df

def pull_game_odds():
    df = dm.read('''
                 SELECT
                    team,
                    game_date,
                    CASE
                        WHEN date(game_date) >= date('2025-10-01') THEN '2026'
                        WHEN date(game_date) >= date('2024-10-01') THEN '2025'
                        WHEN date(game_date) >= date('2023-10-01') THEN '2024'
                        WHEN date(game_date) >= date('2022-10-01') THEN '2023'
                    END AS season,
                    1.0 / moneyline_odds AS win_prob,
                    over as ou_total
                FROM Draftkings_Odds;
                 ''', 'Team_stats')
    
    # 2) center at 0.5 so >0 = favorite, <0 = dog
    df['expected_control'] = df['win_prob'] - 0.5

    return df

def pull_advanced_stats():
    df = dm.read(
        '''SELECT 
              TEAM_ABBREVIATION team,
              game_date,
              GAME_ID game_id,
              PLAYER_ID player_id,
              MIN minutes_played,
              TS_PCT true_shooting_pct,
              USG_PCT usage_pct
           FROM Advanced_Stats
           JOIN (
                SELECT
                    PLAYER_ID,
                    game_date
                FROM Box_Score
                WHERE FGA >= 5
          ) USING (PLAYER_ID, game_date)
        ''', 'Player_Stats')
    
    df = cleanup_minutes(df)

    return df

game_odds = pull_game_odds()
adv_stats = pull_advanced_stats()

df = pd.merge(
    adv_stats,
    game_odds,
    on=['team', 'game_date'],
    how='inner'
)

df['usage_pct_centered'] = df['usage_pct'] - df['usage_pct'].mean()
df['ou_centered'] = df['ou_total'] - df['ou_total'].mean()

df = df[df.minutes_played >= 10].reset_index(drop=True)
df["player_season"] = df["player_id"].astype(str) + "_" + df["season"].astype(str)

# ======================
# A) USAGE MODEL
# ======================
# "Does expected control move player usage?"
res_usg = feols(
    data=df,
    fml="usage_pct ~ expected_control + minutes_played + ou_centered | player_season",
    vcov={"CRV1": "game_id"}
)
print(res_usg.summary())

# ======================
# B) EFFICIENCY MODEL
# ======================
# "Does expected control move efficiency, holding usage?"
# we let TS depend on BOTH control and the actual usage in that game
res_ts = feols(
    data=df,
    fml="true_shooting_pct ~ expected_control * usage_pct_centered + ou_centered | player_season",
    vcov={"CRV1": "game_id"}
)

print("=== TRUE SHOOTING MODEL ===")
print(res_ts.summary())

#%%
