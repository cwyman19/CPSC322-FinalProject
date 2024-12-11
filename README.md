# README

## To See Classifier Performance
* Run all cells in the Technical-Report.ipynb file.
    * This will print information on Knn, Bayes, Decision Tree, and Random Forest Classifiers using our NFL Dataset
    * There is a code cell that will print out the Decision Tree ruleset if the user desires. It is commented out by default to keep the notebook clean as the decision tree ruleset is incredibly long.

## To See the Full NFL Dataset
* Our Classifiers are using the data in input_data/NFL_regseason_data_clean.csv
    * This file was created by combining NFL_regseason_data.csv and all the data in input_data/NFL_teamdata.  
    * This data was all process through an algorithm in data_processing.py to create the cleaned dataset.

## To Run the API
* Run the API_Stuff/NFLdata_webapp.py file
    * After the file is run, the API will be open. 
    * This can be navigated to through the link at the top of the NFLdata_webapp.py file.
    * All attributes in the query string are set to H by default. The user can change these to A to test different predictions

## To Run Unit Tests
Run `pytest --verbose`
* This command runs all the discovered tests in the project
* You can run individual test modules with
    * `pytest --verbose test_myclassifiers.py`
* Note: the `-s` flag can be helpful because it will show print statement output from test execution
