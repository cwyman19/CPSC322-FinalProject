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

'''
my_bills_data = MyPyTable().load_from_file("input_data/NFL_teamdata/BuffaloBills_cleaned.csv")
#my_bills_data.pretty_print()

print(my_bills_data.column_names)
#my_bills_data.column_names.append("Season")

## adding season column

season = 2018
for row in range(len(my_bills_data.data)):
    if my_bills_data.data[row][0] == "":
        season += 1
    else:
        my_bills_data.data[row].append(season)

# removing boxscore column and OT column
season_index = my_bills_data.column_names.index("Season")
my_bills_data.remove_column(season_index)

my_bills_data.pretty_print()


my_bills_data.save_to_file("input_data/NFL_teamdata/BuffaloBills_cleaned.csv")
'''

# Before running data-cleaning algorithm, change first TO to TOL and second TO to TOG (turnovers gained/lost)
# Change second RushY to DRushY and PassY to DPassY
# Change second Opp to OppScore
# add row of ,,,,,,,,, between seasons
# All numbers should be divided by the number of weeks played in that season to find the aggregate average
# Turnover margin is just the sum of all previous turnovers lost and gained
# FG% data not available week by week, so manually enter team FG% for whole season each season
# Turning WinLoss Record in Win Percentage (float)

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



