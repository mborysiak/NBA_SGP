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

event_ids = tuple(list(events_df.event_id.values) + [0])
dm.delete_from_db('Team_Stats', 'Game_Events', f"game_date='{odds_api.game_date}' AND event_id IN {event_ids}", create_backup=False)
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
print('\n==========\nNumber draftkings props:', len(player_props[player_props.bookmaker=='draftkings']))
player_props

#%%

stats = ['spreads', 'h2h', 'totals']
markets = ','.join(stats)
team_props = odds_api.all_market_odds(markets, events_df)
team_props

#%%

dm.delete_from_db('Team_Stats', 'Game_Odds', f"game_date='{odds_api.game_date}' AND event_id IN {event_ids} ", create_backup=False)
dm.write_to_db(team_props, 'Team_Stats', 'Game_Odds', 'append')

dm.delete_from_db('Player_Stats', 'Game_Odds', f"game_date='{odds_api.game_date}' AND event_id IN {event_ids}", create_backup=False)
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
df = df[df.team!='FA']

dm.delete_from_db('Player_Stats', 'FantasyPros', f"game_date='{game_date}'", create_backup=False)
dm.write_to_db(df, 'Player_Stats', 'FantasyPros', if_exist='append')

#%%
fname = 'fantasy-basketball-projections.csv'

today_date = dt.datetime.now().date()
# today_date = dt.date(2024, 11, 20)
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

# fantasy data is forgetting to update teams after trades, so this corrects the duplicates
fp_teams = dm.read(f"SELECT player, team FROM FantasyPros WHERE game_date='{today_date}'", 'Player_Stats')
team_update = {
               'GSW': 'GS',
               'NOR': 'NO',
               'NYK': 'NY',
               'SAS': 'SA',
               'UTH': 'UTA'
               }
for ot, nt in team_update.items(): 
    fp_teams.loc[fp_teams.team==ot, 'team'] = nt

df2 = pd.merge(df, fp_teams, on=['player', 'team'])
print('In FantasyData, but not FantasyPros:', [d for d in df.player.values if d not in df2.player.values])
print('In FantasyPros, but not FantasyData:', [d for d in df2.player.values if d not in df.player.values])

dm.delete_from_db('Player_Stats', 'FantasyData', f"game_date='{today_date}'", create_backup=False)
dm.write_to_db(df2, 'Player_Stats', 'FantasyData', if_exist='append')

#%%

# def name_extract(col):
#     characters = ['III', 'II', '.', 'Jr', 'Sr']
#     for c in characters:
#         col = col.replace(c, '')
#     col = col.split(' ')
#     col = [c for c in col if c!='']
#     return ' '.join(col[2:4])

# import requests
# from io import StringIO

# df = pd.read_html(StringIO(requests.get('https://www.numberfire.com/nba/daily-fantasy/daily-basketball-projections', verify=False).text))[3]
# df.columns = [c[1] for c in df.columns]
# df.Player = df.Player.apply(name_extract)
# df.Player = df.Player.apply(dc.name_clean)

# df = df.rename(columns={'Player': 'player',
#                         'FP': 'fantasy_points',
#                         'Salary': 'salary',
#                         'Value': 'value',
#                         'Min': 'minutes',
#                         'Pts': 'points',
#                         'Reb': 'rebounds',
#                         'Ast': 'assists',
#                         'Stl': 'steals',
#                         'Blk': 'blocks',
#                         'TO': 'turnovers',
#                         '3PM': 'three_pointers'})

# game_date = dt.datetime.now().date()
# df['game_date'] = game_date
# df.head(15)

# if df.player[0] != 'No Data':
#     dm.delete_from_db('Player_Stats', 'NumberFire_Projections', f"game_date='{game_date}'", create_backup=False)
#     dm.write_to_db(df, 'Player_Stats', 'NumberFire_Projections', 'append')

#%%

today_date = dt.datetime.now().date()
date_str = today_date.strftime('%Y%m%d')

fname = 'DAILY.csv'
try: os.replace(f"/Users/borys/Downloads/{fname}", 
                f"{root_path}/Data/OtherData/FanDuelResearch/{date_str}_{fname}")
except: pass

df = pd.read_csv(f"{root_path}/Data/OtherData/FanDuelResearch/{date_str}_{fname}")
df.player = df.player.apply(dc.name_clean)

df = df.drop(['team', 'gameInfo', 'fieldGoalShootingPercentage', 'threePointsShootingPercentage', 
              'freeThrowShootingPercentage', 'positionRank', 'overallRank', 'gamesPlayed'], axis=1)

df = df.rename(columns={'player': 'player',
                        'team': 'team',
                        'salary': 'salary',
                        'value': 'value',
                        'min': 'minutes',
                        'fieldGoalsMade': 'fg_made',
                        'fieldGoalsAttempted': 'fg_attempted',
                        'threePointsMade': 'three_pointers',
                        'threePointsAttempted': 'three_pointers_attempted',
                        'freeThrowsMade': 'ft_made',
                        'freeThrowsAttempted': 'ft_attempted',
                        'assists': 'assists',
                        'steals': 'steals',
                        'blocks': 'blocks',
                        'turnovers': 'turnovers',
                        'points': 'points',
                        'rebounds': 'rebounds',
                        'fantasy': 'fantasy_points',
                        })

df['fg_pct'] = df.fg_made / df.fg_attempted
df['ft_pct'] = df.ft_made / df.ft_attempted
df['three_point_pct'] = df.three_pointers / df.three_pointers_attempted
df = df.fillna(0)

game_date = dt.datetime.now().date()
df['game_date'] = game_date
df.head(15)

if df.player[0] != 'No Data':
    dm.delete_from_db('Player_Stats', 'NumberFire_Projections', f"game_date='{game_date}'", create_backup=False)
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

df = df[df.game!='WAS@UTA'] #duplicated in data over time
df2 = pd.merge(df, fp_teams, on=['player', 'team'])
print('In Sportsline, but not FantasyPros:', [d for d in df.player.values if d not in df2.player.values])
print('In FantasyPros, but not Sportsline:', [d for d in df2.player.values if d not in df.player.values])

df2.head(15)

dm.delete_from_db('Player_Stats', 'SportsLine_Projections', f"game_date='{dt.datetime.now().date()}'", create_backup=False)
dm.write_to_db(df2, 'Player_Stats', 'SportsLine_Projections', 'append')


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

    def player_team_df(self, ep_data, i_select=0, is_tracking=False, 
                       is_advanced=False, is_hustle=False, is_usage=False, is_defensive=False):
        player_data = ep_data.get_data_frames()[0+i_select]
        team_data = ep_data.get_data_frames()[1+i_select]

        if is_tracking:
            player_data = self.player_tracking_rename(player_data)
            team_data = self.team_tracking_rename(team_data)
        elif is_advanced:
            player_data = self.player_advanced_rename(player_data)
            team_data = self.team_advanced_rename(team_data)
        elif is_hustle:
            player_data = self.player_hustle_rename(player_data)
            team_data = self.team_hustle_rename(team_data)
        elif is_usage:
            player_data = self.player_usage_rename(player_data)
            team_data = self.team_usage_rename(team_data)
        elif is_defensive:
            player_data = self.player_defensive_rename(player_data)
            team_data = None

        player_data['game_date'] = self.game_date
        player_data = self.update_team_names(player_data)

        if team_data is not None:
            team_data['game_date'] = self.game_date
            team_data = self.update_team_names(team_data)

        return player_data, team_data

    def get_box_score(self, game_id):
        box_score = self.ep.boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id, timeout=5)
        box_score_player, box_score_team = self.player_team_df(box_score)
        return box_score_player, box_score_team

    def get_tracking_data(self, game_id):
        tracking = self.ep.boxscoreplayertrackv3.BoxScorePlayerTrackV3(game_id=game_id, timeout=5)
        tracking_player, tracking_team = self.player_team_df(tracking, is_tracking=True)
        return tracking_player, tracking_team

    def get_advanced_stats(self, game_id):
        adv_box_score = self.ep.boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=game_id, timeout=5)
        adv_box_score_player, adv_box_score_team = self.player_team_df(adv_box_score, is_advanced=True)
        return adv_box_score_player, adv_box_score_team
    
    def get_hustle_stats(self, game_id):
        hustle = self.ep.boxscorehustlev2.BoxScoreHustleV2(game_id=game_id, timeout=5)
        hustle_player, hustle_team = self.player_team_df(hustle, is_hustle=True)
        return hustle_player, hustle_team
    
    def get_usage_stats(self, game_id):
        usage = self.ep.boxscoreusagev3.BoxScoreUsageV3(game_id=game_id, timeout=5)
        usage_player, usage_team = self.player_team_df(usage, is_usage=True)
        return usage_player, usage_team
    
    def get_defensive_stats(self, game_id):
        defensive = self.ep.boxscoredefensivev2.BoxScoreDefensiveV2(game_id=game_id, timeout=5)
        defensive_player, defensive_team = self.player_team_df(defensive, is_defensive=True)
        return defensive_player, defensive_team

    @staticmethod
    def player_tracking_rename(df):
        col_rename = {
            'gameId': 'GAME_ID',
            'teamId': 'TEAM_ID',
            'teamTricode': 'TEAM_ABBREVIATION',
            'firstName': 'PLAYER_NAME',
            'teamCity': 'TEAM_CITY', 
            'personId': 'PLAYER_ID',
            'position': 'START_POSITION',
            'comment': 'COMMENT',
            'minutes': 'MIN',
            'speed': 'SPD',
            'distance': 'DIST',
            'reboundChancesOffensive': 'ORBC',
            'reboundChancesDefensive': 'DRBC',
            'reboundChancesTotal': 'RBC',
            'touches': 'TCHS',
            'secondaryAssists': 'SAST',
            'freeThrowAssists': 'FTAST',
            'passes': 'PASS',
            'assists': 'AST',
            'contestedFieldGoalsMade': 'CFGM',
            'contestedFieldGoalsAttempted': 'CFGA',
            'contestedFieldGoalPercentage': 'CFG_PCT',
            'uncontestedFieldGoalsMade': 'UFGM',
            'uncontestedFieldGoalsAttempted': 'UFGA',
            'uncontestedFieldGoalsPercentage': 'UFG_PCT',
            'fieldGoalPercentage': 'FG_PCT',
            'defendedAtRimFieldGoalsMade': 'DFGM',
            'defendedAtRimFieldGoalsAttempted': 'DFGA',
            'defendedAtRimFieldGoalPercentage': 'DFG_PCT'
        }

        df['firstName'] = df.firstName + ' ' + df.familyName
        df = df.rename(columns=col_rename)
        df = df[col_rename.values()]
        return df
    
    @staticmethod
    def team_tracking_rename(df):
        col_rename = {
            'gameId': 'GAME_ID',
            'teamId': 'TEAM_ID',
            'teamName': 'TEAM_NAME',
            'teamTricode': 'TEAM_ABBREVIATION',
            'teamCity': 'TEAM_CITY',
            'minutes': 'MIN',
            'distance': 'DIST',
            'reboundChancesOffensive': 'ORBC',
            'reboundChancesDefensive': 'DRBC',
            'reboundChancesTotal': 'RBC',
            'touches': 'TCHS',
            'secondaryAssists': 'SAST',
            'freeThrowAssists': 'FTAST',
            'passes': 'PASS',
            'assists': 'AST',
            'contestedFieldGoalsMade': 'CFGM',
            'contestedFieldGoalsAttempted': 'CFGA',
            'contestedFieldGoalPercentage': 'CFG_PCT',
            'uncontestedFieldGoalsMade': 'UFGM',
            'uncontestedFieldGoalsAttempted': 'UFGA',
            'uncontestedFieldGoalsPercentage': 'UFG_PCT',
            'fieldGoalPercentage': 'FG_PCT',
            'defendedAtRimFieldGoalsMade': 'DFGM',
            'defendedAtRimFieldGoalsAttempted': 'DFGA',
            'defendedAtRimFieldGoalPercentage': 'DFG_PCT'
        }

        df = df.rename(columns=col_rename)
        df = df[col_rename.values()]
        return df
    
    @staticmethod
    def player_advanced_rename(df):
        col_rename = {
            'gameId': 'GAME_ID',
            'teamId': 'TEAM_ID',
            'teamTricode': 'TEAM_ABBREVIATION',
            'firstName': 'PLAYER_NAME',
            'teamCity': 'TEAM_CITY', 
            'personId': 'PLAYER_ID',
            'position': 'START_POSITION',
            'comment': 'COMMENT',
            'minutes': 'MIN',
            "estimatedOffensiveRating": 'E_OFF_RATING', 
            "offensiveRating": 'OFF_RATING', 
            "estimatedDefensiveRating": 'E_DEF_RATING', 
            "defensiveRating": 'DEF_RATING', 
            "estimatedNetRating": 'E_NET_RATING', 
            "netRating": 'NET_RATING', 
            "assistPercentage": 'AST_PCT', 
            "assistToTurnover": 'AST_TOV', 
            "assistRatio": 'AST_RATIO', 
            "offensiveReboundPercentage": 'OREB_PCT', 
            "defensiveReboundPercentage": 'DREB_PCT', 
            "reboundPercentage": 'REB_PCT', 
          #  "turnoverRatio": 'TOV_RATIO', 
            "effectiveFieldGoalPercentage": 'EFG_PCT', 
            "trueShootingPercentage": 'TS_PCT', 
            "usagePercentage": 'USG_PCT', 
            "estimatedUsagePercentage": 'E_USG_PCT', 
            "estimatedPace": 'E_PACE', 
            "pace": 'PACE', 
            "pacePer40": 'PACE_PER40', 
            "possessions": 'POSS', 
            "PIE": 'PIE'
        }

        df['firstName'] = df.firstName + ' ' + df.familyName
        df = df.rename(columns=col_rename)
        df = df[col_rename.values()]
        return df
    
    @staticmethod
    def team_advanced_rename(df):
        col_rename = {
            'gameId': 'GAME_ID',
            'teamId': 'TEAM_ID',
            'teamName': 'TEAM_NAME',
            'teamTricode': 'TEAM_ABBREVIATION',
            'teamCity': 'TEAM_CITY',
            'minutes': 'MIN',
            "estimatedOffensiveRating": 'E_OFF_RATING', 
            "offensiveRating": 'OFF_RATING', 
            "estimatedDefensiveRating": 'E_DEF_RATING', 
            "defensiveRating": 'DEF_RATING', 
            "estimatedNetRating": 'E_NET_RATING', 
            "netRating": 'NET_RATING', 
            "assistPercentage": 'AST_PCT', 
            "assistToTurnover": 'AST_TOV', 
            "assistRatio": 'AST_RATIO', 
            "offensiveReboundPercentage": 'OREB_PCT', 
            "defensiveReboundPercentage": 'DREB_PCT', 
            "reboundPercentage": 'REB_PCT', 
            "estimatedTeamTurnoverPercentage": 'TM_TOV_PCT', 
         #   "turnoverRatio": 'TOV_RATIO', 
            "effectiveFieldGoalPercentage": 'EFG_PCT', 
            "trueShootingPercentage": 'TS_PCT', 
            "usagePercentage": 'USG_PCT', 
            "estimatedUsagePercentage": 'E_USG_PCT', 
            "estimatedPace": 'E_PACE', 
            "pace": 'PACE', 
            "pacePer40": 'PACE_PER40', 
            "possessions": 'POSS', 
            "PIE": 'PIE'
        }

        df = df.rename(columns=col_rename)
        df = df[col_rename.values()]
        return df
    
    @staticmethod
    def player_hustle_rename(df):
        col_rename = {
            'gameId': 'GAME_ID',
            'teamId': 'TEAM_ID',
            'teamTricode': 'TEAM_ABBREVIATION',
            'firstName': 'PLAYER_NAME',
            'teamCity': 'TEAM_CITY', 
            'personId': 'PLAYER_ID',
            'position': 'START_POSITION',
            'comment': 'COMMENT',
            'minutes': 'MINUTES',
            "points": 'PTS', 
            "contestedShots": 'CONTESTED_SHOTS', 
            "contestedShots2pt": 'CONTESTED_SHOTS_2PT',
            "contestedShots3pt": 'CONTESTED_SHOTS_3PT', 
            "deflections": 'DEFLECTIONS', 
            "chargesDrawn": 'CHARGES_DRAWN',
            "screenAssists": 'SCREEN_ASSISTS',
            "screenAssistPoints": 'SCREEN_AST_PTS',
            "looseBallsRecoveredOffensive": 'OFF_LOOSE_BALLS_RECOVERED',
            "looseBallsRecoveredDefensive": 'DEF_LOOSE_BALLS_RECOVERED',
            "looseBallsRecoveredTotal": 'LOOSE_BALLS_RECOVERED',
            "offensiveBoxOuts": 'OFF_BOXOUTS',
            "defensiveBoxOuts": 'DEF_BOXOUTS',
            "boxOutPlayerTeamRebounds": 'BOX_OUT_PLAYER_TEAM_REBS',
            "boxOutPlayerRebounds": 'BOX_OUT_PLAYER_REBS',
            "boxOuts": 'BOX_OUTS'
        }

        df['firstName'] = df.firstName + ' ' + df.familyName
        df = df.rename(columns=col_rename)
        df = df[col_rename.values()]
        return df
    
    @staticmethod
    def team_hustle_rename(df):
        col_rename = {
            'gameId': 'GAME_ID',
            'teamId': 'TEAM_ID',
            'teamName': 'TEAM_NAME',
            'teamTricode': 'TEAM_ABBREVIATION',
            'teamCity': 'TEAM_CITY',
            'minutes': 'MINUTES',
            "points": 'PTS',
            "contestedShots": 'CONTESTED_SHOTS',
            "contestedShots2pt": 'CONTESTED_SHOTS_2PT',
            "contestedShots3pt": 'CONTESTED_SHOTS_3PT',
            "deflections": 'DEFLECTIONS',
            "chargesDrawn": 'CHARGES_DRAWN',
            "screenAssists": 'SCREEN_ASSISTS',
            "screenAssistPoints": 'SCREEN_AST_PTS',
            "looseBallsRecoveredOffensive": 'OFF_LOOSE_BALLS_RECOVERED',
            "looseBallsRecoveredDefensive": 'DEF_LOOSE_BALLS_RECOVERED',
            "looseBallsRecoveredTotal": 'LOOSE_BALLS_RECOVERED',
            "offensiveBoxOuts": 'OFF_BOXOUTS',
            "defensiveBoxOuts": 'DEF_BOXOUTS',
            "boxOutPlayerTeamRebounds": 'BOX_OUT_PLAYER_TEAM_REBS',
            "boxOutPlayerRebounds": 'BOX_OUT_PLAYER_REBS',
            "boxOuts": 'BOX_OUTS'
        }

        df = df.rename(columns=col_rename)
        df = df[col_rename.values()]
        return df
    
    @staticmethod
    def player_usage_rename(df):
        col_rename = {
            'gameId': 'GAME_ID',
            'teamId': 'TEAM_ID',
            'teamTricode': 'TEAM_ABBREVIATION',
            'firstName': 'PLAYER_NAME',
            'teamCity': 'TEAM_CITY', 
            'personId': 'PLAYER_ID',
            'position': 'START_POSITION',
            'comment': 'COMMENT',
            'minutes': 'MIN',
            "usagePercentage": 'USG_PCT',
            "percentageFieldGoalsMade": 'PCT_FGM',
            "percentageFieldGoalsAttempted": 'PCT_FGA',
            "percentageThreePointersMade": 'PCT_FG3M',
            "percentageThreePointersAttempted": 'PCT_FG3A',
            "percentageFreeThrowsMade": 'PCT_FTM',
            "percentageFreeThrowsAttempted": 'PCT_FTA',
            "percentageReboundsOffensive": 'PCT_OREB',
            "percentageReboundsDefensive": 'PCT_DREB',
            "percentageReboundsTotal": 'PCT_REB',
            "percentageAssists": 'PCT_AST',
            "percentageTurnovers": 'PCT_TOV',
            "percentageSteals": 'PCT_STL',
            "percentageBlocks": 'PCT_BLK',
            "percentageBlocksAllowed": 'PCT_BLKA',
            "percentagePersonalFouls": 'PCT_PF',
            "percentagePersonalFoulsDrawn": 'PCT_PFD',
            "percentagePoints": 'PCT_PTS'
        }

        df['firstName'] = df.firstName + ' ' + df.familyName
        df = df.rename(columns=col_rename)
        df = df[col_rename.values()]
        return df
    
    @staticmethod
    def team_usage_rename(df):
        col_rename = {
            'gameId': 'GAME_ID',
            'teamId': 'TEAM_ID',
            'teamName': 'TEAM_NAME',
            'teamTricode': 'TEAM_ABBREVIATION',
            'teamCity': 'TEAM_CITY',
            'minutes': 'MIN',
            "usagePercentage": 'USG_PCT',
            "percentageFieldGoalsMade": 'PCT_FGM',
            "percentageFieldGoalsAttempted": 'PCT_FGA',
            "percentageThreePointersMade": 'PCT_FG3M',
            "percentageThreePointersAttempted": 'PCT_FG3A',
            "percentageFreeThrowsMade": 'PCT_FTM',
            "percentageFreeThrowsAttempted": 'PCT_FTA',
            "percentageReboundsOffensive": 'PCT_OREB',
            "percentageReboundsDefensive": 'PCT_DREB',
            "percentageReboundsTotal": 'PCT_REB',
            "percentageAssists": 'PCT_AST',
            "percentageTurnovers": 'PCT_TOV',
            "percentageSteals": 'PCT_STL',
            "percentageBlocks": 'PCT_BLK',
            "percentageBlocksAllowed": 'PCT_BLKA',
            "percentagePersonalFouls": 'PCT_PF',
            "percentagePersonalFoulsDrawn": 'PCT_PFD',
            "percentagePoints": 'PCT_PTS'
        }

        df = df.rename(columns=col_rename)
        df = df[col_rename.values()]
        return df
    
    @staticmethod
    def player_defensive_rename(df):
        col_rename = {
            'gameId': 'GAME_ID',
            'teamId': 'TEAM_ID',
            'teamTricode': 'TEAM_ABBREVIATION',
            'firstName': 'PLAYER_NAME',
            'teamCity': 'TEAM_CITY', 
            'personId': 'PLAYER_ID',
            'position': 'START_POSITION',
            'comment': 'COMMENT',
        }

        df['firstName'] = df.firstName + ' ' + df.familyName
        df = df.rename(columns=col_rename)
        df = df[col_rename.values()]
        return df
    

    def pull_all_stats(self, stat_cat, game_date):

        self.game_date = game_date
        games = self.filter_games()

        stat_cats = {
            'box_score': 'self.get_box_score(game_id)',
            'tracking_data': 'self.get_tracking_data(game_id)',
            'advanced_stats': 'self.get_advanced_stats(game_id)',
            'hustle_stats': 'self.get_hustle_stats(game_id)',
            'usage_stats': 'self.get_usage_stats(game_id)',
            "matchupMinutes": 'matchupMinutes', 
            "partialPossessions": 'partialPossessions', 
            "switchesOn": 'switchesOn', 
            "playerPoints": 'playerPoints', 
            "defensiveRebounds": 'defensiveRebounds', 
            "matchupAssists": 'matchupAssists', 
            "matchupTurnovers": 'matchupTurnovers', 
            "steals": 'steals', 
            "blocks": 'blocks', 
            "matchupFieldGoalsMade": 'matchupFieldGoalsMade', 
            "matchupFieldGoalsAttempted": 'matchupFieldGoalsAttempted', 
            "matchupFieldGoalPercentage": 'matchupFieldGoalPercentage', 
            "matchupThreePointersMade": 'matchupThreePointersMade', 
            "matchupThreePointersAttempted": 'matchupThreePointersAttempted', 
            "matchupThreePointerPercentage": 'matchupThreePointerPercentage'
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
# yesterday_date = dt.datetime(2025, 2, 13).date()

box_score_players, box_score_teams = nba_stats.pull_all_stats('box_score', yesterday_date)
time.sleep(10)
tracking_players, tracking_teams = nba_stats.pull_all_stats('tracking_data', yesterday_date)
time.sleep(10)
adv_players, adv_teams = nba_stats.pull_all_stats('advanced_stats', yesterday_date)
time.sleep(10)
hustle_players, hustle_teams = nba_stats.pull_all_stats('hustle_stats', yesterday_date)
time.sleep(10)
usage_players, usage_teams = nba_stats.pull_all_stats('usage_stats', yesterday_date)
# time.sleep(5)
# defensive_players, _ = nba_stats.pull_all_stats('defensive_stats', yesterday_date)

dfs = [box_score_players, tracking_players, adv_players, hustle_players, usage_players]#, defensive_players]
tnames = ['Box_Score', 'Tracking_Data', 'Advanced_Stats', 'Hustle_Stats', 'Usage_Stats']#, 'Defensive_Stats']
for df, tname in zip(dfs, tnames):
    dm.delete_from_db('Player_Stats', tname, f"game_date='{yesterday_date}'")
    dm.write_to_db(df, 'Player_Stats', tname, 'append')

dfs = [box_score_teams, tracking_teams, adv_teams, hustle_teams, usage_teams]
tnames = ['Box_Score', 'Tracking_Data', 'Advanced_Stats', 'Hustle_Stats']
for df, tname in zip(dfs, tnames):
    dm.delete_from_db('Team_Stats', tname, f"game_date='{yesterday_date}'")
    dm.write_to_db(df, 'Team_Stats', tname, 'append')


#%%

# Build new features
# Number of games in x number of rolling days
# Number of miles traveled in x number of rolling days
# add usage stats

# %%


# for t in ['Advanced_Stats', 'Box_Score', 'Draftkings_Odds', 'Hustle_Stats', 'Tracking_Data','Usage_Stats',
#           'FantasyData', 'NumberFire_Projections', 'FantasyPros', 'SportsLine_Projections']:
    
#     df = dm.read(f"SELECT * FROM {t}", 'Player_Stats')
#     try: df.player = df.player.apply(dc.name_clean)
#     except: df.PLAYER_NAME = df.PLAYER_NAME.apply(dc.name_clean)

#     dm.write_to_db(df, 'Player_Stats', t, 'replace', True)

#%%

