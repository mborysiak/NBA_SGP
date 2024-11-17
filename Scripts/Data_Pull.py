#%%
import pandas as pd
import numpy as np

import requests
import pandas as pd
import numpy as np
import datetime as dt
import pytz
import yaml

from ff.db_operations import DataManage
from ff import general as ffgeneral
import ff.data_clean as dc

# set the root path and database management object
root_path = ffgeneral.get_main_path('NBA_SGP')
db_path = f'{root_path}/Data/'
dm = DataManage(db_path)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)


def read_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Assuming the config file is in the same directory as the Python script
config_file = f'{root_path}/Scripts/config.yaml'
config = read_config(config_file)
          

#%%

api_key = config['api_key']
sport = 'basketball_nba' # use the sport_key from the /sports endpoint below, or use 'upcoming' to see the next 8 games across all sports
region = 'us' # uk | us | eu | au. Multiple can be specified if comma delimited
odds_format = 'decimal' # decimal | american
date_format = 'iso' # iso | unix

team_map = {
  'New York Knicks': 'NY',
  'Cleveland Cavaliers': 'CLE',
  'San Antonio Spurs': 'SA',
  'Houston Rockets': 'HOU',
  'Utah Jazz': 'UTA',
  'Dallas Mavericks': 'DAL',
  'Washington Wizards': 'WAS',
  'Atlanta Hawks': 'ATL',
  'Sacramento Kings': 'SAC',
  'Portland Trail Blazers': 'POR',
  'Orlando Magic': 'ORL',
  'Indiana Pacers': 'IND',
  'Milwaukee Bucks': 'MIL',
  'Boston Celtics': 'BOS',
  'Miami Heat': 'MIA',
  'Detroit Pistons': 'DET',
  'Memphis Grizzlies': 'MEM',
  'Chicago Bulls': 'CHI',
  'Toronto Raptors': 'TOR',
  'Denver Nuggets': 'DEN',
  'Phoenix Suns': 'PHO',
  'Los Angeles Lakers': 'LAL',
  'New Orleans Pelicans': 'NO',
  'Los Angeles Clippers': 'LAC',
  'Golden State Warriors': 'GS',
  'Brooklyn Nets': 'BKN',
  'Oklahoma City Thunder': 'OKC',
  'Philadelphia 76ers': 'PHI',
  'Charlotte Hornets': 'CHA',
  'Minnesota Timberwolves': 'MIN'
}

class OddsAPIPull:

    def __init__(self, api_key, base_url, sport, region, odds_format, date_format, historical=False):
        
        self.api_key = api_key
        self.sport = sport
        self.region = region
        self.odds_format = odds_format
        self.date_format = date_format
        self.historical = historical

        if self.historical: 
            self.base_url = f'{base_url}/historical/sports/'
            self.game_date = start_time.date().strftime('%Y-%m-%d')
        else: 
            self.base_url = f'{base_url}/sports/'
            self.game_date = dt.datetime.now().date().strftime('%Y-%m-%d')

    def get_response(self, r_pull):
        if r_pull.status_code != 200:
            print(f'Failed to get odds: status_code {r_pull.status_code}, response body {r_pull.text}')
        else:
            r_json = r_pull.json()
            print('Number of events:', len(r_json))
            print('Remaining requests', r_pull.headers['x-requests-remaining'])
            print('Used requests', r_pull.headers['x-requests-used'])

        return r_json

    @staticmethod
    def convert_utc_to_est(est_dt):
        
        # Define the EST timezone
        est = pytz.timezone('US/Eastern')

        # Localize the datetime object to EST
        local_time_est = est.localize(est_dt)

        # Convert the localized datetime to UTC
        utc_time = local_time_est.astimezone(pytz.utc)
        
        return utc_time.strftime("%Y-%m-%dT%H:%M:%SZ")



    def pull_events(self, start_time, end_time):

        if self.historical: self.game_date = start_time.date().strftime('%Y-%m-%d')
        else: self.game_date = dt.datetime.now().date().strftime('%Y-%m-%d') 
        
        if start_time is not None: start_time = self.convert_utc_to_est(start_time)
        if end_time is not None: end_time = self.convert_utc_to_est(end_time)

        get_params = {
                'api_key': self.api_key,
                'regions': self.region,
                'oddsFormat': self.odds_format,
                'dateFormat': self.date_format,
                'commenceTimeFrom': start_time,
                'commenceTimeTo': end_time
            }
        
        if self.historical:
            get_params['date'] = start_time
            self.start_time = start_time

        events = requests.get(
            f'{self.base_url}/{self.sport}/events',
            params=get_params
            )
        
        events_json = self.get_response(events)
        if self.historical: events_json = events_json['data']

        events_df = pd.DataFrame()
        for e in events_json:
            events_df = pd.concat([events_df, pd.DataFrame(e, index=[0])], axis=0)
        
        events_df['game_date'] = self.game_date
        events_df = events_df.rename(columns={'id': 'event_id'})

        return events_df.reset_index(drop=True)
    

    def pull_lines(self, markets, event_id):

        get_params={
                    'api_key': self.api_key,
                    'regions': self.region,
                    'markets': markets,
                    'oddsFormat': self.odds_format,
                    'dateFormat': self.date_format,
                }
       
        if self.historical:
            get_params['date'] = self.start_time

        odds = requests.get(
                f'{self.base_url}/{self.sport}/events/{event_id}/odds',
                params = get_params
            )
        
        odds_json = self.get_response(odds)
        if self.historical: odds_json = odds_json['data']

        props = pd.DataFrame()
        for odds in odds_json['bookmakers']:
            bookmaker = odds['key']
            market_props = odds['markets']
            for cur_prop in market_props:
                p = pd.DataFrame(cur_prop['outcomes'])
                p['bookmaker'] = bookmaker
                p['prop_type'] = cur_prop['key']
                p['event_id'] = event_id

                if cur_prop['key'] in ('spreads', 'h2h'):
                    p = p.rename(columns={'name': 'description'})

                props = pd.concat([props, p], axis=0)
                

        props = props.reset_index(drop=True)
        props['game_date'] = self.game_date

        return props

    def all_market_odds(self, markets, events_df):

        props = pd.DataFrame()
        for event_id in events_df.event_id.values:
            try:
                print(event_id)
                cur_props = self.pull_lines(markets, event_id)
                props = pd.concat([props, cur_props], axis=0)
            except:
                print(f'Failed to get data for event_id {event_id}')

        return props
    
# for i in range(270, 280):
hist_period = 0 #i*24 + 2
pull_historical = False

start_time = dt.datetime.now() - dt.timedelta(hours=hist_period)
end_time = (dt.datetime.now() + dt.timedelta(hours=12 - hist_period))

base_url = 'https://api.the-odds-api.com/v4/'
odds_api = OddsAPIPull(api_key, base_url, sport, region, odds_format, date_format, historical=pull_historical)

events_df = odds_api.pull_events(start_time=start_time, end_time=end_time)
events_df

#%%

dm.delete_from_db('Team_Stats', 'Game_Events', f"game_date='{odds_api.game_date}'", create_backup=False)
dm.write_to_db(events_df, 'Team_Stats', 'Game_Events', 'append')

#%%
stats = [
         'player_points', 'player_rebounds', 'player_assists', 'player_steals', 
         'player_blocks', 'player_threes', 'player_blocks_steals', 'player_turnovers', 
         'player_points_rebounds_assists', 'player_points_rebounds', 
         'player_points_assists', 'player_rebounds_assists'
         ]

markets = ','.join(stats)
player_props = odds_api.all_market_odds(markets, events_df)
player_props

#%%

stats = ['spreads', 'h2h', 'totals']
markets = ','.join(stats)
team_props = odds_api.all_market_odds(markets, events_df)
team_props

#%%

dm.delete_from_db('Team_Stats', 'Game_Odds', f"game_date='{odds_api.game_date}'", create_backup=False)
dm.write_to_db(team_props, 'Team_Stats', 'Game_Odds', 'append')

dm.delete_from_db('Player_Stats', 'Game_Odds', f"game_date='{odds_api.game_date}'", create_backup=False)
dm.write_to_db(player_props, 'Player_Stats', 'Game_Odds', 'append')

#%%

df = dm.read(f'''SELECT description player,
                        prop_type stat_type,
                        name over_under,
                        AVG(point) value,
                        AVG(price) decimal_odds,
                        game_date
                FROM Game_Odds 
                WHERE bookmaker='draftkings'
                      AND game_date='{odds_api.game_date}'
                GROUP BY description, 
                         prop_type, 
                         over_under,
                         game_date
             ''', 'Player_Stats')
df.player = df.player.apply(dc.name_clean)
df.stat_type = df.stat_type.apply(lambda x: x.replace('player_', ''))
df.loc[df.stat_type=='threes', 'stat_type'] = 'three_pointers'
df.loc[df.stat_type=='rebounds_assists', 'stat_type'] = 'assists_rebounds'
df.loc[df.stat_type=='blocks_steals', 'stat_type'] = 'steals_blocks'
df.loc[df.over_under=='Over', 'over_under'] = 'over'
df.loc[df.over_under=='Under', 'over_under'] = 'under'

dm.delete_from_db('Player_Stats', 'Draftkings_Odds', f"game_date='{odds_api.game_date}'", create_backup=False)
dm.write_to_db(df, 'Player_Stats', 'Draftkings_Odds', 'append')


#%%
df = dm.read(f'''SELECT * 
                FROM Game_Odds
                JOIN (
                     SELECT event_id, home_team, away_team, game_date
                     FROM Game_Events
                     WHERE game_date='{odds_api.game_date}' 
                     ) USING (event_id, game_date)
                WHERE bookmaker='draftkings'
                      AND prop_type IN ('spreads', 'h2h')
                ''', 'Team_Stats')

df = df.pivot_table(index=['description', 'game_date', 'home_team', 'away_team'], 
                    columns='prop_type', values=['price', 'point'], aggfunc='first').reset_index()

df.columns = [f"{c[0]}_{c[1]}" if c[1]!='' else c[0] for c in df.columns]
df['opponent'] = np.where(df.home_team==df.description, df.away_team, df.home_team)
df['is_home'] = np.where(df.home_team==df.description, 1, 0)
df = df.rename(columns={'description': 'team',
                        'point_spreads': 'spread',
                        'price_spreads': 'spread_odds',
                        'price_h2h': 'moneyline_odds'})

over_under = dm.read(f'''SELECT * 
                         FROM Game_Odds
                         JOIN (
                            SELECT event_id, home_team, away_team, game_date
                            FROM Game_Events
                            WHERE game_date='{odds_api.game_date}' 
                            ) USING (event_id, game_date)
                         WHERE bookmaker='draftkings'
                               and prop_type='totals'
                        ''', 'Team_Stats')

over_under = over_under.pivot_table(index=['game_date', 'home_team', 'away_team'],
                                    columns='name', values=['price', 'point'], aggfunc='first').reset_index()
over_under.columns = [f"{c[0]}_{c[1]}" if c[1]!='' else c[0] for c in over_under.columns]
over_under = over_under.rename(columns={'point_Over': 'over',
                                        'price_Over': 'over_odds',
                                        'point_Under': 'under',
                                        'price_Under': 'under_odds'})

df = df.merge(over_under, on=['game_date', 'home_team', 'away_team'], how='left')
df = df[['team', 'opponent', 'is_home', 'moneyline_odds', 'spread',
         'spread_odds', 'over', 'over_odds', 'under', 'under_odds', 'game_date']]

df.team = df.team.map(team_map)
df.opponent = df.opponent.map(team_map)

dm.delete_from_db('Team_Stats', 'Draftkings_Odds', f"game_date='{odds_api.game_date}'", create_backup=False)
dm.write_to_db(df, 'Team_Stats', 'Draftkings_Odds', 'append')

#%%




#%%
fname = 'fantasy-basketball-projections.csv'

today_date = dt.datetime.now().date()
# today_date = dt.date(2024, 10, 28)
date_str = today_date.strftime('%Y%m%d')
dl_files = os.listdir('/Users/borys/Downloads')
dl_files = [f for f in dl_files if 'fantasy-basketball-projections' in f]
os.rename(f"/Users/borys/Downloads/{dl_files[0]}", f"/Users/borys/Downloads/{fname}")

try: os.replace(f"/Users/borys/Downloads/{fname}", 
                f"{root_path}/Data/OtherData/FantasyData/{date_str}_{fname}")
except: pass

df = pd.read_csv(f"{root_path}/Data/OtherData/FantasyData/{date_str}_{fname}").dropna(axis=0)

df = df.rename(columns={'rank': 'rank',
                        'player': 'player',
                        'team': 'team',
                        'pos': 'position',
                        'opp': 'opponent',
                        'pts': 'points',
                        'reb': 'rebounds',
                        'ast': 'assists',
                        'stl': 'steals',
                        'blk': 'blocks',
                        'fg_pct': 'fg_pct',
                        'ft_pct': 'ft_pct',
                        'fg3_pct': 'three_point_pct',
                        'ftm': 'ft_made',
                        'fgm': 'two_point_made',
                        'fg3m': 'three_pointers',
                        'tov': 'turnovers',
                        'toc': 'minutes',
                        'fpts': 'fantasy_points'})

df['game_date'] = today_date

df.player = df.player.apply(dc.name_clean)
df.team = df.team.apply(lambda x: x.lstrip().rstrip())
df[['fg_pct', 'ft_pct']] = df[['fg_pct', 'ft_pct']] / 100
df = df.drop(['id', 'gs'], axis=1)

dm.delete_from_db('Player_Stats', 'FantasyData', f"game_date='{today_date}'", create_backup=False)
dm.write_to_db(df, 'Player_Stats', 'FantasyData', if_exist='append')

#%%

def name_extract(col):
    characters = ['III', 'II', '.', 'Jr', 'Sr']
    for c in characters:
        col = col.replace(c, '')
    col = col.split(' ')
    col = [c for c in col if c!='']
    return ' '.join(col[2:4])

import requests
from io import StringIO

df = pd.read_html(StringIO(requests.get('https://www.numberfire.com/nba/daily-fantasy/daily-basketball-projections', verify=False).text))[3]
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

df = pd.read_html("https://www.sportsline.com/nba/expert-projections/simulation/")[0]
sl_cols = {'PLAYER': 'player',
           'POS': 'position',
           'TEAM': 'team',
           'GAME': 'game',
           'FP': 'fantasy_points',
           'DK': 'draftkings',
           'FD': 'fanduel',
           'PROJ': 'projected',
           'PTS': 'points',
           'MIN': 'minutes',
           'FG': 'field_goals',
           'FGA': 'field_goals_attempted',
           'AST': 'assists',
           'TREB': 'rebounds',
           'DREB': 'defensive_rebounds',
           'OREB': 'offensive_rebounds',
           'ST': 'steals',
           'BK': 'blocks',
           'TO': 'turnovers',
           'FT': 'free_throws',
           'FGP': 'fg_pct',
           'FTP': 'ft_pct',
}
df = df.rename(columns=sl_cols)
df.columns = ['sl_' + c.lower() if c not in ('player', 'pos', 'team', 'game') else c for c in df.columns ]
df.player = df.player.apply(dc.name_clean)
df['game_date'] = dt.datetime.now().date()
df.head(15)

#%%

dm.delete_from_db('Player_Stats', 'SportsLine_Projections', f"game_date='{dt.datetime.now().date()}'", create_backup=False)
dm.write_to_db(df, 'Player_Stats', 'SportsLine_Projections', 'append')


#%%

import os
today_month = dt.datetime.now().month
today_day = str(dt.datetime.now().day).zfill(2)
today_year = dt.datetime.now().year

game_date = dt.datetime.now().date()
fname = f'FantasyPros_NBA_Daily_Fantasy_Basketball_Projections_({today_month}_{today_day})_.csv'

try: os.replace(f"/Users/borys/Downloads/{fname}", 
                f'{root_path}/Data/OtherData/Fantasy_Pros/{fname}')
except: pass

df = pd.read_csv(f'{root_path}/Data/OtherData/Fantasy_Pros/{fname}').dropna(axis=0)
df.columns = ['player', 'team', 'position', 'opponent', 'points', 'rebounds', 
              'assists', 'blocks', 'steals', 'fg_pct', 'ft_pct', 'three_pointers', 'games_played', 'minutes', 'turnovers']

df['game_date'] = game_date

df.player = df.player.apply(dc.name_clean)
df.team = df.team.apply(lambda x: x.lstrip().rstrip())

dm.delete_from_db('Player_Stats', 'FantasyPros', f"game_date='{game_date}'", create_backup=False)
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
# for i in range(11, 13):
#     yesterday_date = dt.datetime(2024, 11, i).date()

box_score_players, box_score_teams = nba_stats.pull_all_stats('box_score', yesterday_date)
time.sleep(1)
tracking_players, tracking_teams = nba_stats.pull_all_stats('tracking_data', yesterday_date)
time.sleep(1)
adv_players, adv_teams = nba_stats.pull_all_stats('advanced_stats', yesterday_date)
time.sleep(1)
hustle_players, hustle_teams = nba_stats.pull_all_stats('hustle_stats', yesterday_date)
time.sleep(1)
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

# Build new features
# Number of games in x number of rolling days
# Number of miles traveled in x number of rolling days
# add usage stats

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

# get NBA schedule data as JSON
import requests
year = '2023'
r = requests.get('https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/' + year + '/league/00_full_schedule.json')
json_data = r.json()

# prepare output files
data = []

# loop through each month/game and write out stats to file
for i in range(len(json_data['lscd'])):
    for j in range(len(json_data['lscd'][i]['mscd']['g'])):
        gamedate = json_data['lscd'][i]['mscd']['g'][j]['gdte']
        etm = json_data['lscd'][i]['mscd']['g'][j]['etm']
        stt = json_data['lscd'][i]['mscd']['g'][j]['stt']
        game_id = json_data['lscd'][i]['mscd']['g'][j]['gid']
        visiting_team = json_data['lscd'][i]['mscd']['g'][j]['v']['ta']
        home_team = json_data['lscd'][i]['mscd']['g'][j]['h']['ta']
        data.append([gamedate, etm, stt, game_id, home_team, visiting_team])

df = pd.DataFrame(data, columns=['game_date', 'game_time', 'standard_time', 'game_id','home_team', 'away_team'])
team_update = {
               'GSW': 'GS',
               'PHX': 'PHO',
               'NOP': 'NO',
               'NYK': 'NY',
               'SAS': 'SA'
               }
for ot, nt in team_update.items():
    df.loc[df.home_team==ot, 'home_team'] = nt
    df.loc[df.away_team==ot, 'away_team'] = nt

dm.write_to_db(df, 'Team_Stats', 'NBA_Schedule', 'replace')

# %%


# import sqlite3

# for t in ['FantasyData', 'NumberFire_Projections', 'FantasyPros', 'Draftkings_Odds']:
#     conn = sqlite3.connect('c:/Users/borys/Downloads/Player_Stats.sqlite3')
#     df = pd.read_sql_query(f"SELECT * FROM {t} WHERE game_date >= '2024-02-20' ", conn)
#     dm.write_to_db(df, 'Player_Stats', t, 'append')

# conn = sqlite3.connect('c:/Users/borys/Downloads/Team_Stats.sqlite3')
# df = pd.read_sql_query(f"SELECT * FROM Draftkings_Odds WHERE game_date >= '2024-02-20' ", conn)
# dm.write_to_db(df, 'Team_Stats', 'Draftkings_Odds', 'append')
# %%
