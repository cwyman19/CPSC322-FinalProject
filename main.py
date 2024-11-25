'''
Just using this file to create mypytable objects that help me preprocess data

'''


# some useful mysklearn package import statements and reloads
import importlib

import mysklearn.myutils
importlib.reload(mysklearn.myutils)
import mysklearn.myutils as myutils

# uncomment once you paste your mypytable.py into mysklearn package
import mysklearn.mypytable
importlib.reload(mysklearn.mypytable)
from mysklearn.mypytable import MyPyTable 

my_bills_data = MyPyTable().load_from_file("input_data/NFL_teamdata/BuffaloBills_cleaned.csv")
#my_bills_data.pretty_print()

print(my_bills_data.column_names)
#my_bills_data.column_names.append("Season")

## adding season column
'''
season = 2018
for row in range(len(my_bills_data.data)):
    if my_bills_data.data[row][0] == "":
        season += 1
    else:
        my_bills_data.data[row].append(season)
'''
# removing boxscore column and OT column
season_index = my_bills_data.column_names.index("Season")
my_bills_data.remove_column(season_index)

my_bills_data.pretty_print()


my_bills_data.save_to_file("input_data/NFL_teamdata/BuffaloBills_cleaned.csv")




