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
pd.set_option('display.max_rows', None)

#%%

class DKScraper():

    def __init__(self, base_url, event_group_id):
        self.base_url = base_url
        self.event_group_id = event_group_id
        self.base_api = requests.get(f"{base_url}/{event_group_id}?format=json").json()

    def nfl_games_dk(self, half=0):
        """
        Scrapes current NFL game lines.
        Returns
        -------
        games : dictionary containing teams, spreads, totals, moneylines, home/away, and opponent.
        """
        if half == 0:
            dk_api = requests.get("https://sportsbook.draftkings.com//sites/US-NJ-SB/api/v5/eventgroups/88808?format=json").json()
            dk_markets = dk_api['eventGroup']['offerCategories'][0]['offerSubcategoryDescriptors'][0]['offerSubcategory']['offers']
        elif half == 1:
            dk_api = requests.get("https://sportsbook.draftkings.com//sites/US-NJ-SB/api/v5/eventgroups/88808/categories/526?format=json").json()
            for i in dk_api['eventGroup']['offerCategories']:
                if i['name'] == 'Halves':
                        dk_markets = i['offerSubcategoryDescriptors'][0]['offerSubcategory']['offers']
        
        games = {}
        for i in dk_markets:
            if i[0]['outcomes'][0]['oddsDecimal'] == 0: # Skip this if there is no spread
                continue
            away_team = i[0]['outcomes'][0]['label']
            home_team = i[0]['outcomes'][1]['label']
            
            if away_team not in games: 
                # Gotta be a better way then a bunch of try excepts
                games[away_team] = {'location':0}
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
                games[home_team] = {'location':1}
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
                
        return games


    def get_offer_cats(self):
        offer_cats = {}
        cat_json = self.base_api['eventGroup']['offerCategories']

        keep_cats = ['Player Points', 'Player Rebounds', 'Player Assists', 'Player Threes', 'Player Combos', 'Player Blocks/Steals']
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
                                    print(player, market)
                                    pass
                
        return props


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

        return pd.DataFrame({
                    'player': players,
                    'stat_type': stats_types,
                    'over_under': over_unders,
                    'value': ou_values,
                    'decimal_odds': decimal_odds
                    })

#%%
nba_scrape = DKScraper(base_url='https://sportsbook.draftkings.com//sites/US-NJ-SB/api/v5/eventgroups/', event_group_id=42648)
props = nba_scrape.nba_props_dk()
props_df = nba_scrape.dict_to_df(props)
props_df['game_date'] = dt.datetime.now().date()
props_df.player = props_df.player.apply(dc.name_clean)
props_df.loc[props_df.stat_type=='Threes', 'stat_type'] = 'three_pointers'

props_df.head(10)

# %%
dm.delete_from_db('Player_Stats', 'Draftkings_Odds', f"game_date='{dt.datetime.now().date()}'", create_backup=False)
dm.write_to_db(props_df, 'Player_Stats', 'Draftkings_Odds', 'append')


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

try: os.replace(f"/Users/mborysia/Downloads/{fname}", 
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
fname = 'fantasy-basketball-projections.csv'

today_date = dt.datetime.now().date()
date_str = today_date.strftime('%Y%m%d')
try: os.replace(f"/Users/mborysia/Downloads/{fname}", 
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

    def player_team_df(self, ep_data):
        player_data = ep_data.get_data_frames()[0]
        team_data = ep_data.get_data_frames()[1]

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

    def pull_all_stats(self, stat_cat, game_date):

        self.game_date = game_date#self.get_datetime(game_date)
        games = self.filter_games()

        stat_cats = {
            'box_score': 'self.get_box_score(game_id)',
            'tracking_data': 'self.get_tracking_data(game_id)',
            'advanced_stats': ' self.get_advanced_stats(game_id)'
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
yesterday_date = dt.datetime(2023, 2, 24).date()

box_score_players, box_score_teams = nba_stats.pull_all_stats('box_score', yesterday_date)
time.sleep(5)
tracking_players, tracking_teams = nba_stats.pull_all_stats('tracking_data', yesterday_date)
time.sleep(5)
adv_players, adv_teams = nba_stats.pull_all_stats('advanced_stats', yesterday_date)

dfs = [box_score_players, tracking_players, adv_players]
tnames = ['Box_Score', 'Tracking_Data', 'Advanced_Stats']
for df, tname in zip(dfs, tnames):
    dm.delete_from_db('Player_Stats', tname, f"game_date='{yesterday_date}'")
    dm.write_to_db(df, 'Player_Stats', tname, 'append')

dfs = [box_score_teams, tracking_teams, adv_teams]
tnames = ['Box_Score', 'Tracking_Data', 'Advanced_Stats']
for df, tname in zip(dfs, tnames):
    dm.delete_from_db('Team_Stats', tname, f"game_date='{yesterday_date}'")
    dm.write_to_db(df, 'Team_Stats', tname, 'append')

# %%
# %%
