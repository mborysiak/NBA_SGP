#%%
import pandas as pd
import numpy as np

import requests
import pandas as pd
import numpy as np
import datetime as dt

from ff.db_operations import DataManage
from ff import general as ffgeneral
import ff.data_clean as dc

# set the root path and database management object
root_path = ffgeneral.get_main_path('NBA_SGP')
db_path = f'{root_path}/Data/'
dm = DataManage(db_path)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

#%%

class DKScraper():

    def __init__(self, base_url, event_group_id):
        self.base_url = base_url
        self.event_group_id = event_group_id
        self.base_api = requests.get(f"{base_url}/{event_group_id}?format=json").json()

    def nba_games_dk(self):
        """
        Scrapes current NFL game lines.
        Returns
        -------
        games : dictionary containing teams, spreads, totals, moneylines, home/away, and opponent.
        """
        
        dk_markets = self.base_api['eventGroup']['offerCategories'][0]['offerSubcategoryDescriptors'][0]['offerSubcategory']['offers']
      
        games = {}
        for i in dk_markets:
            if i[0]['outcomes'][0]['oddsDecimal'] == 0: # Skip this if there is no spread
                continue
            away_team = i[0]['outcomes'][0]['label']
            home_team = i[0]['outcomes'][1]['label']
            
            if away_team not in games: 
                # Gotta be a better way then a bunch of try excepts
                games[away_team] = {'is_home': 0}
                try:
                    games[away_team]['moneyline'] = i[2]['outcomes'][0]['oddsDecimal']
                except:
                    pass
                try:
                    games[away_team]['spread'] = [i[0]['outcomes'][0]['line'],
                                                   i[0]['outcomes'][0]['oddsDecimal']]
                except:
                    pass
                try:
                    games[away_team]['over'] = [i[1]['outcomes'][0]['line'],
                                                i[1]['outcomes'][0]['oddsDecimal']]
                except:
                    pass
                try:
                    games[away_team]['under'] = [i[1]['outcomes'][1]['line'],
                                                 i[1]['outcomes'][1]['oddsDecimal']]
                except:
                    pass
                games[away_team]['opponent'] = home_team
            
            if home_team not in games:
                games[home_team] = {'is_home': 1}
                try:
                    games[home_team]['moneyline'] = i[2]['outcomes'][1]['oddsDecimal']
                except:
                    pass
                try:
                    games[home_team]['spread'] = [i[0]['outcomes'][1]['line'],
                                                  i[0]['outcomes'][1]['oddsDecimal']]
                except:
                    pass
                try:
                    games[home_team]['over'] = [i[1]['outcomes'][0]['line'],
                                                i[1]['outcomes'][0]['oddsDecimal']]
                except:
                    pass
                try:
                    games[home_team]['under'] = [i[1]['outcomes'][1]['line'],
                                                 i[1]['outcomes'][1]['oddsDecimal']]
                except:
                    pass     
                games[home_team]['opponent'] = away_team

        game_list = []
        for k, v in games.items():
            try:
                cur_game = [k, v['opponent'], v['is_home'], v['moneyline']]
                cur_game.extend(v['spread'])
                cur_game.extend(v['over'])
                cur_game.extend(v['under'])
                game_list.append(cur_game)
            except:
                print(f"{k} vs {v['opponent']}", ' failed')

        game_df = pd.DataFrame(game_list)
        game_df.columns = ['team', 'opponent', 'is_home', 'moneyline_odds', 'spread', 
                           'spread_odds', 'over', 'over_odds', 'under', 'under_odds']
        
        game_df.loc[game_df.team.str.contains('Clipper'), 'team'] = 'LAC'
        game_df.loc[game_df.team.str.contains('Laker'), 'team'] = 'LAL'

        game_df.team = game_df.team.apply(lambda x: x.split(' ')[0])                
        game_df.opponent = game_df.opponent.apply(lambda x: x.split(' ')[0])
        return game_df


    def get_offer_cats(self):
        offer_cats = {}
        cat_json = self.base_api['eventGroup']['offerCategories']

        keep_cats = ['Player Points', 'Player Rebounds', 'Player Assists', 'Player Threes', 'Player Combos', 'Player Blocks/Steals', 'Player Defense']
        for i in range(1,len(cat_json)):
            if cat_json[i]['name'] in keep_cats:
                offer_cats[cat_json[i]['name']] = cat_json[i]['offerCategoryId']

        return offer_cats
    

    def nba_props_dk(self):
        props = {}
        offer_cats = self.get_offer_cats()

        for cat in offer_cats.values():
            dk_api = requests.get(f"{self.base_url}/{self.event_group_id}/categories/{cat}?format=json").json()
            for i in dk_api['eventGroup']['offerCategories']:
                if 'offerSubcategoryDescriptors' in i:
                    dk_markets = i['offerSubcategoryDescriptors']
            
            subcategoryIds = []# Need subcategoryIds first
            for i in dk_markets:
                subcategoryIds.append(i['subcategoryId'])
                        
            for ids in subcategoryIds:
                dk_api = requests.get(f"{self.base_url}/{self.event_group_id}/categories/{cat}/subcategories/{ids}?format=json").json()
                for i in dk_api['eventGroup']['offerCategories']:
                    if 'offerSubcategoryDescriptors' in i:
                        dk_markets = i['offerSubcategoryDescriptors']
                
                for i in dk_markets:
                    if 'offerSubcategory' in i:
                        market = i['name']
                        for j in i['offerSubcategory']['offers']:
                            for k in j:
                                if 'participant' in k['outcomes'][0]:
                                    player = k['outcomes'][0]['participant']
                                else:
                                    continue
                                
                                if player not in props:
                                    props[player] = {}
                                    
                                try:
                                    props[player][market] = {'over':[k['outcomes'][0]['line'],
                                                                     k['outcomes'][0]['oddsDecimal']],
                                                             'under':[k['outcomes'][1]['line'],
                                                                     k['outcomes'][1]['oddsDecimal']]}
                                except:
                                    pass
                
        return props
    
    @staticmethod
    def clean_stat_cat(df):
        stat_map = {
                'Points': 'points', 
                'Points - 1st Quarter': 'points_first_quarter', 
                'Rebounds': 'rebounds',
                'Rebounds - 1st Quarter': 'rebounds_first_quarter', 
                'Assists': 'assists', 
                'Pts + Reb + Ast': 'points_rebounds_assists',
                'Pts + Reb': 'points_rebounds', 
                'Pts + Ast': 'points_assists', 
                'Ast + Reb': 'assists_rebounds',
                'Assists + Rebounds': 'assists_rebounds', 
                'Blocks': 'blocks', 
                'Steals': 'steals',
                'Steals + Blocks': 'steals_blocks', 
                'Threes': 'three_pointers', 
                'Assists - 1st Quarter': 'assists_first_quarter'
            }
        df.stat_type = df.stat_type.apply(lambda x: x.lstrip().rstrip())
        df.stat_type = df.stat_type.map(stat_map)

        return df

    def dict_to_df(self, props):
        players = []
        stats_types = []
        over_unders = []
        ou_values = []
        decimal_odds = []

        for cur_player in props.keys():
            for cur_stat_type in props[cur_player].keys():
                for cur_ou, v in props[cur_player][cur_stat_type].items():
                    players.append(cur_player)
                    stats_types.append(cur_stat_type)
                    over_unders.append(cur_ou)
                    ou_values.append(v[0])
                    decimal_odds.append(v[1])

        df = pd.DataFrame({
                    'player': players,
                    'stat_type': stats_types,
                    'over_under': over_unders,
                    'value': ou_values,
                    'decimal_odds': decimal_odds
                    })
        df = self.clean_stat_cat(df)
        return df
    
#%%
fname = 'fantasy-basketball-projections.csv'

today_date = dt.datetime.now().date()
# today_date = dt.date(2023, 11, 15)
date_str = today_date.strftime('%Y%m%d')
try: os.replace(f"/Users/borys/Downloads/{fname}", 
                f"{root_path}/Data/OtherData/FantasyData/{date_str}_{fname}")
except: pass

df = pd.read_csv(f"{root_path}/Data/OtherData/FantasyData/{date_str}_{fname}").dropna(axis=0)

df = df.rename(columns={'Name': 'player',
                        'Rank': 'rank',
                        'Team': 'team',
                        'Position': 'position',
                        'Opponent': 'opponent',
                        'Points': 'points',
                        'Rebounds': 'rebounds',
                        'Assists': 'assists',
                        'Steals': 'steals',
                        'BlockedShots': 'blocks',
                        'FieldGoalsPercentage': 'fg_pct',
                        'FreeThrowsPercentage': 'ft_pct',
                        'ThreePointersPercentage': 'three_point_pct',
                        'FreeThrowsMade': 'ft_made',
                        'TwoPointersMade': 'two_point_made',
                        'ThreePointersMade': 'three_pointers',
                        'Turnovers': 'turnovers',
                        'Minutes': 'minutes',
                        'FantasyPoints': 'fantasy_points'})

df['game_date'] = today_date

df.player = df.player.apply(dc.name_clean)
df.team = df.team.apply(lambda x: x.lstrip().rstrip())
df[['fg_pct', 'ft_pct']] = df[['fg_pct', 'ft_pct']] / 100

dm.delete_from_db('Player_Stats', 'FantasyData', f"game_date='{today_date}'", create_backup=False)
dm.write_to_db(df, 'Player_Stats', 'FantasyData', if_exist='append')

#%%

player_teams = dm.read('''SELECT player, team
                          FROM (
                          SELECT player, team,
                                 row_number() OVER (PARTITION BY player ORDER BY game_date DESC) AS rn 
                          FROM FantasyData 
                          ) 
                          WHERE rn = 1''', 'Player_Stats')

schedule = dm.read("SELECT * FROM NBA_Schedule", 'Team_Stats')
schedule.game_time = pd.to_datetime(schedule.game_time)
today = dt.datetime.now() 
tomorrow = dt.datetime.now() + dt.timedelta(hours=12)
schedule = schedule[(schedule.game_time > today) & (schedule.game_time < tomorrow)]
teams = schedule[['game_time', 'home_team', 'away_team']].melt(id_vars='game_time', value_name='team')[['game_time', 'team']]
player_teams = pd.merge(player_teams, teams, on='team')


#%%
nba_scrape = DKScraper(base_url='https://sportsbook.draftkings.com//sites/US-NJ-SB/api/v5/eventgroups/', event_group_id=42648)
props = nba_scrape.nba_props_dk()
props_df = nba_scrape.dict_to_df(props)
props_df['game_date'] = dt.datetime.now().date()
props_df.player = props_df.player.apply(dc.name_clean)
props_df = pd.merge(props_df, player_teams, on='player')
props_df = props_df.drop(['team', 'game_time'], axis=1)
props_df.head(25)

#%%
games_df = nba_scrape.nba_games_dk()
games_df['game_date'] = dt.datetime.now().date()
games_df.head(16)

# %%

last_run = dm.read(f"SELECT * FROM Draftkings_Odds WHERE game_date='{dt.datetime.now().date()}'", 'Player_Stats')
last_run = last_run[~last_run.player.isin(props_df.player.unique())]
props_df = pd.concat([props_df, last_run], axis=0)

dm.delete_from_db('Player_Stats', 'Draftkings_Odds', f"game_date='{dt.datetime.now().date()}'", create_backup=True)
dm.write_to_db(props_df, 'Player_Stats', 'Draftkings_Odds', 'append')

dm.delete_from_db('Team_Stats', 'Draftkings_Odds', f"game_date='{dt.datetime.now().date()}'", create_backup=True)
dm.write_to_db(games_df, 'Team_Stats', 'Draftkings_Odds', 'append')


#%%

def name_extract(col):
    characters = ['III', 'II', '.', 'Jr', 'Sr']
    for c in characters:
        col = col.replace(c, '')
    col = col.split(' ')
    col = [c for c in col if c!='']
    return ' '.join(col[2:4])

df = pd.read_html('https://www.numberfire.com/nba/daily-fantasy/daily-basketball-projections')[3]
df.columns = [c[1] for c in df.columns]
df.Player = df.Player.apply(name_extract)
df.Player = df.Player.apply(dc.name_clean)

df = df.rename(columns={'Player': 'player',
                        'FP': 'fantasy_points',
                        'Salary': 'salary',
                        'Value': 'value',
                        'Min': 'minutes',
                        'Pts': 'points',
                        'Reb': 'rebounds',
                        'Ast': 'assists',
                        'Stl': 'steals',
                        'Blk': 'blocks',
                        'TO': 'turnovers',
                        '3PM': 'three_pointers'})

df['game_date'] = dt.datetime.now().date()
df.head(15)

# %%
dm.delete_from_db('Player_Stats', 'NumberFire_Projections', f"game_date='{dt.datetime.now().date()}'", create_backup=False)
dm.write_to_db(df, 'Player_Stats', 'NumberFire_Projections', 'append')


#%%
import os
today_month = dt.datetime.now().month
today_day = str(dt.datetime.now().day).zfill(2)
fname = f'FantasyPros_NBA_Daily_Fantasy_Basketball_Projections_({today_month}_{today_day})_.csv'

try: os.replace(f"/Users/borys/Downloads/{fname}", 
                f'{root_path}/Data/OtherData/Fantasy_Pros/{fname}')
except: pass

df = pd.read_csv(f'{root_path}/Data/OtherData/Fantasy_Pros/{fname}').dropna(axis=0)
df.columns = ['player', 'team', 'position', 'opponent', 'points', 'rebounds', 
              'assists', 'blocks', 'steals', 'fg_pct', 'ft_pct', 'three_pointers', 'games_played', 'minutes', 'turnovers']

df['game_date'] = dt.datetime.now().date()

df.player = df.player.apply(dc.name_clean)
df.team = df.team.apply(lambda x: x.lstrip().rstrip())

dm.delete_from_db('Player_Stats', 'FantasyPros', f"game_date='{dt.datetime.now().date()}'", create_backup=False)
dm.write_to_db(df, 'Player_Stats', 'FantasyPros', if_exist='append')

#%%


class NBAStats:
    
    def __init__(self):
        
        import nba_api.stats.endpoints as ep

        self.nba_teams = self.get_nba_teams()
        self.ep = ep
        self.all_games = self.game_finder()

    @staticmethod
    def get_datetime(game_date):
        return dt.datetime.strptime(game_date, '%Y-%m-%d').date()        

    def get_nba_teams(self):
        from nba_api.stats.static import teams
        nba_teams = teams.get_teams()
        return [t['abbreviation'] for t in nba_teams]

    def get_players(self):
        from nba_api.stats.static import players
        nba_players = players.get_players()
        return nba_players

    def game_finder(self):
        gamefinder = self.ep.leaguegamefinder.LeagueGameFinder()
        games = gamefinder.get_data_frames()[0]
        games.GAME_DATE = pd.to_datetime(games.GAME_DATE).apply(lambda x: x.date())
        return games
    
    def filter_games(self):
        return self.all_games[(self.all_games.GAME_DATE == self.game_date) &\
                              (self.all_games.TEAM_ABBREVIATION.isin(self.nba_teams))].reset_index(drop=True)

    @staticmethod
    def update_team_names(df):
        team_update = {
               'GSW': 'GS',
               'PHX': 'PHO',
               'NOP': 'NO',
               'NYK': 'NY',
               'SAS': 'SA'
               }
        for ot, nt in team_update.items(): 
            df.loc[df.TEAM_ABBREVIATION==ot, 'TEAM_ABBREVIATION'] = nt

        return df

    def player_team_df(self, ep_data, i_select=0):
        player_data = ep_data.get_data_frames()[0+i_select]
        team_data = ep_data.get_data_frames()[1+i_select]

        player_data['game_date'] = self.game_date
        team_data['game_date'] = self.game_date

        player_data = self.update_team_names(player_data)
        team_data = self.update_team_names(team_data)

        return player_data, team_data

    def get_box_score(self, game_id):
        box_score = self.ep.boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        box_score_player, box_score_team = self.player_team_df(box_score)
        return box_score_player, box_score_team

    def get_tracking_data(self, game_id):
        tracking = self.ep.boxscoreplayertrackv2.BoxScorePlayerTrackV2(game_id=game_id)
        tracking_player, tracking_team = self.player_team_df(tracking)
        return tracking_player, tracking_team

    def get_advanced_stats(self, game_id):
        adv_box_score = self.ep.boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id)
        adv_box_score_player, adv_box_score_team = self.player_team_df(adv_box_score)
        return adv_box_score_player, adv_box_score_team
    
    def get_hustle_stats(self, game_id):
        hustle = self.ep.hustlestatsboxscore.HustleStatsBoxScore(game_id=game_id)
        hustle_player, hustle_team = self.player_team_df(hustle, i_select=1)
        return hustle_player, hustle_team
    
    def get_usage_stats(self, game_id):
        usage = self.ep.boxscoreusagev2.BoxScoreUsageV2(game_id=game_id)
        usage_player, usage_team = self.player_team_df(usage)
        return usage_player, usage_team

    def pull_all_stats(self, stat_cat, game_date):

        self.game_date = game_date
        games = self.filter_games()

        stat_cats = {
            'box_score': 'self.get_box_score(game_id)',
            'tracking_data': 'self.get_tracking_data(game_id)',
            'advanced_stats': 'self.get_advanced_stats(game_id)',
            'hustle_stats': 'self.get_hustle_stats(game_id)',
            'usage_stats': 'self.get_usage_stats(game_id)'
        }

        print(f'Pulling {stat_cat}')
        players, teams = pd.DataFrame(), pd.DataFrame()
        for game_id in games.GAME_ID.unique():
            print(game_id)
            try: 
                player, team = eval(stat_cats[stat_cat])
                players = pd.concat([players, player], axis=0)
                teams = pd.concat([teams, team], axis=0)
            except: 
                print(f'Game Id {game_id} failed.')

        players.PLAYER_NAME = players.PLAYER_NAME.apply(dc.name_clean)

        return players.reset_index(drop=True), teams.reset_index(drop=True)

nba_stats = NBAStats()

#%%
import time

yesterday_date = dt.datetime.now().date()-dt.timedelta(1)
# yesterday_date = dt.datetime(2023, 12, 23).date()

box_score_players, box_score_teams = nba_stats.pull_all_stats('box_score', yesterday_date)
time.sleep(2)
tracking_players, tracking_teams = nba_stats.pull_all_stats('tracking_data', yesterday_date)
time.sleep(2)
adv_players, adv_teams = nba_stats.pull_all_stats('advanced_stats', yesterday_date)
time.sleep(2)
hustle_players, hustle_teams = nba_stats.pull_all_stats('hustle_stats', yesterday_date)
time.sleep(2)
usage_players, usage_teams = nba_stats.pull_all_stats('usage_stats', yesterday_date)


dfs = [box_score_players, tracking_players, adv_players, hustle_players, usage_players]
tnames = ['Box_Score', 'Tracking_Data', 'Advanced_Stats', 'Hustle_Stats', 'Usage_Stats']
for df, tname in zip(dfs, tnames):
    dm.delete_from_db('Player_Stats', tname, f"game_date='{yesterday_date}'")
    dm.write_to_db(df, 'Player_Stats', tname, 'append')

dfs = [box_score_teams, tracking_teams, adv_teams, hustle_teams]
tnames = ['Box_Score', 'Tracking_Data', 'Advanced_Stats', 'Hustle_Stats']
for df, tname in zip(dfs, tnames):
    dm.delete_from_db('Team_Stats', tname, f"game_date='{yesterday_date}'")
    dm.write_to_db(df, 'Team_Stats', tname, 'append')

#%%

nba_stats.ep.boxscoreusagev2.BoxScoreUsageV2(game_id='0022300566').get_data_frames()[0]

#%%

for game_date in dm.read("SELECT DISTINCT game_date FROM Box_Score", 'Player_Stats').values[1:]:
    
    print(game_date)
    game_date = dt.datetime.strptime(game_date[0], '%Y-%m-%d').date()
    usage_players, _ = nba_stats.pull_all_stats('usage_stats', game_date)

    # dm.write_to_db(hustle_teams, 'Team_Stats', 'Hustle_Stats', 'append')
    dm.write_to_db(usage_players, 'Player_Stats', 'Usage_Stats', 'append')
    time.sleep(5)



# %%

df = dm.read("SELECT * FROM Draftkings_Odds", 'Player_Stats')
df = df.groupby('game_date').agg({'player': 'count'}).reset_index()


# %%
team = dm.read("SELECT * FROM Draftkings_Odds", 'Team_stats')
team = team.groupby('game_date').agg({'team': 'count'}).reset_index()

pd.merge(df, team, on='game_date')

# %%

# # get NBA schedule data as JSON
# import requests
# year = '2023'
# r = requests.get('https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/' + year + '/league/00_full_schedule.json')
# json_data = r.json()

# # prepare output files
# data = []

# # loop through each month/game and write out stats to file
# for i in range(len(json_data['lscd'])):
#     for j in range(len(json_data['lscd'][i]['mscd']['g'])):
#         gamedate = json_data['lscd'][i]['mscd']['g'][j]['gdte']
#         etm = json_data['lscd'][i]['mscd']['g'][j]['etm']
#         stt = json_data['lscd'][i]['mscd']['g'][j]['stt']
#         game_id = json_data['lscd'][i]['mscd']['g'][j]['gid']
#         visiting_team = json_data['lscd'][i]['mscd']['g'][j]['v']['ta']
#         home_team = json_data['lscd'][i]['mscd']['g'][j]['h']['ta']
#         data.append([gamedate, etm, stt, game_id, home_team, visiting_team])

# df = pd.DataFrame(data, columns=['game_date', 'game_time', 'standard_time', 'game_id','home_team', 'away_team'])
# team_update = {
#                'GSW': 'GS',
#                'PHX': 'PHO',
#                'NOP': 'NO',
#                'NYK': 'NY',
#                'SAS': 'SA'
#                }
# for ot, nt in team_update.items():
#     df.loc[df.home_team==ot, 'home_team'] = nt
#     df.loc[df.away_team==ot, 'away_team'] = nt

# dm.write_to_db(df, 'Team_Stats', 'NBA_Schedule', 'replace')

# %%
