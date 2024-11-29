'''Charlie Wyman and Jillian Berry
CPSC322 Final Project

This file contains code used to preprocess data
TODO: find a way for Las Vegas raiders to be compatible with Oakland Raiders (for 2018/2019)
        And the same goes for the Washington Football Team/Commanders/Redskins
'''

import importlib

import mysklearn.myutils
import mysklearn.myutils as myutils

import mysklearn.mypytable
from mysklearn.mypytable import MyPyTable 

# Before running data-cleaning algorithm, change first TO to TOL and second TO to TOG (turnovers gained/lost)
# Change second RushY to DRushY and PassY to DPassY
# Change second Opp to OppScore
# add row of ,,,,,,,,, between seasons
# All numbers should be divided by the number of weeks played in that season to find the aggregate average
# Turnover margin is just the sum of all previous turnovers lost and gained
# FG% data not available week by week, so manually enter team FG% for whole season each season
# Turning WinLoss Record in Win Percentage (float)

'''
#set to new team name each run
teamname = "ArizonaCardinals"

my_raw_data = MyPyTable().load_from_file(f"input_data/NFL_teamdata/raw_data/{teamname}.csv")
my_raw_data.convert_to_numeric()
#my_raw_data.pretty_print()
header = ['Season', 'Week', 'HomeTeam', 'AwayTeam', 'WinPercentage', 'RushYards', 'PassYards', 'Scoring', 
          'RushYardsAllowed', 'PassYardsAllowed', 'DefenseScoringAllowed', 'KickingPercentage', 'TurnoverMargin', 'Winner']
# NOTE:: all yardage and scoring numbers measured as *per game that season* avg
my_cleaned_data = MyPyTable(column_names=header)

season = 2018
total_weeks = 1 # keep track of number of games played in that season (for averaging)
total_rush_gained = 0
total_pass_gained = 0
total_scoring = 0
total_rush_allowed = 0
total_pass_allowed = 0
total_scoring_allowed = 0
turnover_margin = 0
kick_percentage = 0
wins = 0
losses = 0
for row in range(len(my_raw_data.data)):
    if season == 2018: # manually set kicking pct
        kick_percentage = 0.706
    elif season == 2019:
        kick_percentage = 0.886
    elif season == 2020:
        kick_percentage = 0.763
    elif season == 2021:
        kick_percentage = 0.811
    elif season == 2022:
        kick_percentage = 0.875
    elif season == 2023:
        kick_percentage = 0.848
    elif season == 2024:
        kick_percentage = 0.905



    if my_raw_data.data[row][0] == "":
        season += 1
        total_weeks = 1
        total_rush_gained = 0
        total_pass_gained = 0
        total_scoring = 0
        total_rush_allowed = 0
        total_pass_allowed = 0
        total_scoring_allowed = 0
        turnover_margin = 0
        wins = 0
        losses = 0
    else:
        new_row = []
        new_row.append(season) #add Season
        new_row.append(my_raw_data.data[row][my_raw_data.column_names.index("Week")]) #add Week
        new_row.append(teamname) #add hometeam
        new_row.append(my_raw_data.data[row][my_raw_data.column_names.index("Opp")]) # add awayteam
        if wins + losses == 0:
            win_pct = 0
        else:
            win_pct = round(wins/(wins + losses), 3)
        new_row.append(win_pct) # add WinPercentage
        total_rush_gained += (my_raw_data.data[row][my_raw_data.column_names.index("RushY")])
        new_row.append(round(total_rush_gained/total_weeks, 2)) # add rush yard gained average
        total_pass_gained += (my_raw_data.data[row][my_raw_data.column_names.index("PassY")])
        new_row.append(round(total_pass_gained/total_weeks, 2)) # add pass yard gained average
        total_scoring += (my_raw_data.data[row][my_raw_data.column_names.index("Tm")])
        new_row.append(round(total_scoring/total_weeks, 2)) # add scoring per game average
        total_rush_allowed += (my_raw_data.data[row][my_raw_data.column_names.index("DRushY")])
        new_row.append(round(total_rush_allowed/total_weeks, 2)) # add rush yards allowed per game
        total_pass_allowed += (my_raw_data.data[row][my_raw_data.column_names.index("DPassY")])
        new_row.append(round(total_pass_allowed/total_weeks, 2)) # add pass yards allowed per game
        total_scoring_allowed += (my_raw_data.data[row][my_raw_data.column_names.index("OppScore")])
        new_row.append(round(total_scoring_allowed/total_weeks, 2)) # add scoring allowed per game
        new_row.append(kick_percentage) # add kicking percentage
        if (my_raw_data.data[row][my_raw_data.column_names.index("TOG")]) == '':
            (my_raw_data.data[row][my_raw_data.column_names.index("TOG")]) = 0
        if (my_raw_data.data[row][my_raw_data.column_names.index("TOL")]) == '':
            (my_raw_data.data[row][my_raw_data.column_names.index("TOL")]) = 0
        turnover_margin += int(my_raw_data.data[row][my_raw_data.column_names.index("TOG")]) - int(my_raw_data.data[row][my_raw_data.column_names.index("TOL")])
        new_row.append(turnover_margin) # add turnover margin
        if (int(my_raw_data.data[row][my_raw_data.column_names.index("Tm")]) > int(my_raw_data.data[row][my_raw_data.column_names.index("OppScore")])):
            new_row.append('HomeTeam') # add winner (home team)
            wins += 1
        elif (int(my_raw_data.data[row][my_raw_data.column_names.index("Tm")]) < int(my_raw_data.data[row][my_raw_data.column_names.index("OppScore")])):
            new_row.append('AwayTeam') # add winner (away team)
            losses += 1
        else:
            new_row.append(None) # game ended in a tie
        total_weeks += 1
        my_cleaned_data.data.append(new_row)


#my_cleaned_data.pretty_print()
my_cleaned_data.save_to_file(f"input_data/NFL_teamdata/cleaned_data/{teamname}_cleaned.csv")

'''


my_raw_data = MyPyTable().load_from_file("input_data/NFL_regseason_data.csv")
header = ['Season', 'Week', 'HomeTeam', 'AwayTeam', 'WinPercentage', 'RushYards', 'PassYards', 'Scoring', 
          'RushYardsAllowed', 'PassYardsAllowed', 'DefenseScoringAllowed', 'KickingPercentage', 'TurnoverMargin', 'Winner']
my_clean_data = MyPyTable(column_names=header)

for row in range(len(my_raw_data.data)):
    new_row = []
    new_row.append(my_raw_data.data[row][my_raw_data.column_names.index("Season")])
    new_row.append(my_raw_data.data[row][my_raw_data.column_names.index("Week")])

    if my_raw_data.data[row][my_raw_data.column_names.index("Away")] == "@": # finding home and away teams
        home_team = my_raw_data.data[row][my_raw_data.column_names.index("Loser/tie")]
        away_team = my_raw_data.data[row][my_raw_data.column_names.index("Winner/tie")]
        winner = "A"
    else:
        home_team = my_raw_data.data[row][my_raw_data.column_names.index("Winner/tie")]
        away_team = my_raw_data.data[row][my_raw_data.column_names.index("Loser/tie")]
        winner = "H"
    
    if (home_team == "Washington Redskins") or (home_team == "Washington Football Team"):
        home_team = "Washington Commanders"
    if (away_team == "Washington Redskins") or (away_team == "Washington Football Team"):
        away_team = "Washington Commanders"
    if home_team == "Oakland Raiders":
        home_team = "Las Vegas Raiders"
    if away_team == "Oakland Raiders":
        away_team = "Las Vegas Raiders"
    new_row.append(home_team)
    new_row.append(away_team)

    # finding who has the greater values of the following categories, and labeling them with H (home) or A (away)
    home_team = home_team.replace(" ", "")
    home_team_data = MyPyTable().load_from_file(f"input_data/NFL_teamdata/cleaned_data/{home_team}_cleaned.csv")
    away_team = away_team.replace(" ", "")
    away_team_data = MyPyTable().load_from_file(f"input_data/NFL_teamdata/cleaned_data/{away_team}_cleaned.csv")

    home_team_game_data = [] # first finding the corresponding games from the team data files
    for myrow in range(len(home_team_data.data)):
        if (home_team_data.data[myrow][home_team_data.column_names.index("Season")] == new_row[0]) and (home_team_data.data[myrow][home_team_data.column_names.index("Week")] == new_row[1]):
            home_team_game_data = home_team_data.data[myrow]
    away_team_game_data = []
    for myrow in range(len(away_team_data.data)):
        if (away_team_data.data[myrow][away_team_data.column_names.index("Season")] == new_row[0]) and (away_team_data.data[myrow][away_team_data.column_names.index("Week")] == new_row[1]):
            away_team_game_data = away_team_data.data[myrow]
    
    # Win percentage
    print(home_team_game_data)
    print(away_team_game_data)
    if home_team_game_data[home_team_data.column_names.index("WinPercentage")] >= away_team_game_data[away_team_data.column_names.index("WinPercentage")]:
        new_row.append("H") # home team has higher win percentage
    else:
        new_row.append("A")

    # RushYards
    if home_team_game_data[home_team_data.column_names.index("RushYards")] >= away_team_game_data[away_team_data.column_names.index("RushYards")]:
        new_row.append("H") # home team has higher Rush Yards
    else:
        new_row.append("A")
    
    # PassYards
    if home_team_game_data[home_team_data.column_names.index("PassYards")] >= away_team_game_data[away_team_data.column_names.index("PassYards")]:
        new_row.append("H") # home team has higher Pass Yards
    else:
        new_row.append("A")

    # Scoring
    if home_team_game_data[home_team_data.column_names.index("Scoring")] >= away_team_game_data[away_team_data.column_names.index("Scoring")]:
        new_row.append("H") # home team has higher Scoring
    else:
        new_row.append("A")
    
    # RushYardsAllowed
    if home_team_game_data[home_team_data.column_names.index("RushYardsAllowed")] >= away_team_game_data[away_team_data.column_names.index("RushYardsAllowed")]:
        new_row.append("H") # home team has higher Rush Yards Allowed
    else:
        new_row.append("A")
    
    # PassYardsAllowed
    if home_team_game_data[home_team_data.column_names.index("PassYardsAllowed")] >= away_team_game_data[away_team_data.column_names.index("PassYardsAllowed")]:
        new_row.append("H") # home team has higher Pass Yards Alowed
    else:
        new_row.append("A")

    # Defense Scoring Allowed
    if home_team_game_data[home_team_data.column_names.index("DefenseScoringAllowed")] >= away_team_game_data[away_team_data.column_names.index("DefenseScoringAllowed")]:
        new_row.append("H") # home team has higher Defense Scoring Allowed
    else:
        new_row.append("A")

    # KickingPercentage
    if home_team_game_data[home_team_data.column_names.index("KickingPercentage")] >= away_team_game_data[away_team_data.column_names.index("KickingPercentage")]:
        new_row.append("H") # home team has higher Kicking Percentage
    else:
        new_row.append("A")
    
    # Turnover Margin
    if home_team_game_data[home_team_data.column_names.index("TurnoverMargin")] >= away_team_game_data[away_team_data.column_names.index("TurnoverMargin")]:
        new_row.append("H") # home team has higher Turnover Margin
    else:
        new_row.append("A")

    # Winner
    new_row.append(winner)

    my_clean_data.data.append(new_row)

my_clean_data.save_to_file(f"input_data/NFL_regseason_data_clean.csv")

