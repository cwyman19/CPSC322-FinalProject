import requests 
import json 

url = "http://127.0.0.1:5001/predict?WinPercentage=H&RushYards=H&PassYards=H&Scoring=H&RushYardsAllowed=H&PassYardsAllowed=H&DefenseScoringAllowed=H&KickingPercentage=H&TurnoverMargin=H"

response = requests.get(url=url)

