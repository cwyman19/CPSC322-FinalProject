import pickle

NFL_header = ["WinPercentage", "RushYards", "PassYards", "Scoring", "RushYardsAllowed", "PassYardsAllowed", "DefenseScoringAllowed", "KickingPercentage", "TurnoverMargin"]
NFL_Decision_tree = ['Attribute', 'Scoring', ['Value', 'A', ['Attribute', 'DefenseScoringAllowed', ['Value', 'A', ['Attribute', 'TurnoverMargin', ['Value', 'A', ['Attribute', 'WinPercentage', ['Value', 'A', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Attribute', 'RushYards', ['Value', 'A', ['Attribute', 'PassYards', ['Value', 'A', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Leaf', 'A', 7, 11]], ['Value', 'H', ['Leaf', 'A', 1, 11]]]], ['Value', 'H', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Leaf', 'H', 2, 3]], ['Value', 'H', ['Leaf', 'A', 1, 3]]]]]], ['Value', 'H', ['Attribute', 'PassYards', ['Value', 'A', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Leaf', 'H', 4, 7]], ['Value', 'H', ['Leaf', 'A', 1, 7]]]], ['Value', 'H', ['Leaf', 'A', 6, 13]]]]]], ['Value', 'H', ['Attribute', 'PassYards', ['Value', 'A', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Attribute', 'RushYards', ['Value', 'A', ['Leaf', 'A', 5, 15]], ['Value', 'H', ['Leaf', 'A', 5, 15]]]], ['Value', 'H', ['Attribute', 'RushYards', ['Value', 'A', ['Leaf', 'A', 3, 7]], ['Value', 'H', ['Leaf', 'A', 2, 7]]]]]], ['Value', 'H', ['Leaf', 'A', 2, 24]]]]]], ['Value', 'H', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Attribute', 'PassYards', ['Value', 'A', ['Attribute', 'RushYards', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'H', 3, 6]], ['Value', 'H', ['Leaf', 'A', 1, 6]]]], ['Value', 'H', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'H', 6, 11]], ['Value', 'H', ['Leaf', 'H', 2, 11]]]]]], ['Value', 'H', ['Attribute', 'RushYards', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'A', 2, 5]], ['Value', 'H', ['Leaf', 'H', 2, 5]]]], ['Value', 'H', ['Leaf', 'A', 1, 6]]]]]], ['Value', 'H', ['Attribute', 'RushYards', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'A', 3, 5]], ['Value', 'H', ['Leaf', 'H', 2, 5]]]], ['Value', 'H', ['Leaf', 'A', 3, 8]]]]]]]], ['Value', 'H', ['Attribute', 'PassYards', ['Value', 'A', ['Attribute', 'RushYards', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Leaf', 'A', 4, 6]], ['Value', 'H', ['Leaf', 'H', 1, 6]]]], ['Value', 'H', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Leaf', 'H', 1, 6]], ['Value', 'H', ['Leaf', 'A', 4, 6]]]]]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Leaf', 'A', 4, 9]], ['Value', 'H', ['Leaf', 'A', 1, 9]]]], ['Value', 'H', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Leaf', 'A', 2, 5]], ['Value', 'H', ['Leaf', 'H', 1, 5]]]]]]]], ['Value', 'H', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'A', 2, 7]], ['Value', 'H', ['Leaf', 'A', 2, 7]]]], ['Value', 'H', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'A', 1, 7]], ['Value', 'H', ['Leaf', 'A', 4, 7]]]]]], ['Value', 'H', ['Leaf', 'A', 2, 16]]]]]], ['Value', 'H', ['Attribute', 'RushYards', ['Value', 'A', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'A', 5, 11]], ['Value', 'H', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'A', 2, 5]], ['Value', 'H', ['Leaf', 'A', 1, 5]]]], ['Value', 'H', ['Leaf', 'H', 1, 6]]]]]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'H', 2, 5]], ['Value', 'H', ['Leaf', 'A', 1, 5]]]], ['Value', 'H', ['Leaf', 'A', 2, 7]]]], ['Value', 'H', ['Leaf', 'A', 7, 14]]]]]]]]]], ['Value', 'H', ['Attribute', 'RushYards', ['Value', 'A', ['Attribute', 'WinPercentage', ['Value', 'A', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'H', 6, 14]], ['Value', 'H', ['Leaf', 'H', 8, 14]]]], ['Value', 'H', ['Attribute', 'PassYards', ['Value', 'A', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'H', 4, 15]], ['Value', 'H', ['Leaf', 'H', 6, 15]]]], ['Value', 'H', ['Leaf', 'A', 1, 16]]]]]], ['Value', 'H', ['Leaf', 'H', 8, 38]]]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Attribute', 'PassYards', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'A', 4, 17]], ['Value', 'H', ['Leaf', 'H', 6, 17]]]], ['Value', 'H', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'H', 2, 5]], ['Value', 'H', ['Leaf', 'A', 2, 5]]]]]], ['Value', 'H', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'H', 1, 6]], ['Value', 'H', ['Attribute', 'PassYards', ['Value', 'A', ['Leaf', 'H', 2, 5]], ['Value', 'H', ['Leaf', 'A', 1, 5]]]]]]]], ['Value', 'H', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Attribute', 'PassYards', ['Value', 'A', ['Leaf', 'H', 3, 7]], ['Value', 'H', ['Leaf', 'H', 2, 7]]]], ['Value', 'H', ['Leaf', 'H', 2, 9]]]], ['Value', 'H', ['Attribute', 'PassYards', ['Value', 'A', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Leaf', 'H', 7, 12]], ['Value', 'H', ['Leaf', 'A', 1, 12]]]], ['Value', 'H', ['Leaf', 'A', 1, 13]]]]]]]]]], ['Value', 'H', ['Attribute', 'WinPercentage', ['Value', 'A', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Attribute', 'PassYards', ['Value', 'A', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Leaf', 'H', 3, 8]], ['Value', 'H', ['Leaf', 'A', 2, 8]]]], ['Value', 'H', ['Leaf', 'H', 1, 9]]]], ['Value', 'H', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Leaf', 'H', 6, 9]], ['Value', 'H', ['Leaf', 'H', 3, 9]]]]]], ['Value', 'H', ['Attribute', 'PassYards', ['Value', 'A', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'A', 5, 11]], ['Value', 'H', ['Leaf', 'A', 3, 11]]]], ['Value', 'H', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'H', 2, 4]], ['Value', 'H', ['Leaf', 'A', 1, 4]]]]]], ['Value', 'H', ['Leaf', 'A', 2, 17]]]]]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Attribute', 'PassYards', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'A', 5, 16]], ['Value', 'H', ['Leaf', 'A', 8, 16]]]], ['Value', 'H', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'A', 1, 4]], ['Value', 'H', ['Leaf', 'A', 1, 4]]]]]], ['Value', 'H', ['Attribute', 'PassYards', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'A', 1, 5]], ['Value', 'H', ['Leaf', 'H', 3, 5]]]], ['Value', 'H', ['Leaf', 'H', 1, 6]]]]]], ['Value', 'H', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Attribute', 'PassYards', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'H', 2, 6]], ['Value', 'H', ['Leaf', 'H', 3, 6]]]], ['Value', 'H', ['Leaf', 'A', 2, 8]]]], ['Value', 'H', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'A', 2, 7]], ['Value', 'H', ['Leaf', 'A', 5, 7]]]]]]]]]]]]]], ['Value', 'H', ['Attribute', 'TurnoverMargin', ['Value', 'A', ['Attribute', 'WinPercentage', ['Value', 'A', ['Attribute', 'RushYards', ['Value', 'A', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Attribute', 'PassYards', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'A', 3, 11]], ['Value', 'H', ['Leaf', 'H', 3, 11]]]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'A', 8, 17]], ['Value', 'H', ['Leaf', 'A', 8, 17]]]]]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'A', 5, 14]], ['Value', 'H', ['Leaf', 'H', 3, 14]]]], ['Value', 'H', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'H', 3, 7]], ['Value', 'H', ['Leaf', 'H', 3, 7]]]]]]]], ['Value', 'H', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Attribute', 'PassYards', ['Value', 'A', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'A', 16, 27]], ['Value', 'H', ['Leaf', 'A', 6, 27]]]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'A', 7, 14]], ['Value', 'H', ['Leaf', 'A', 5, 14]]]]]], ['Value', 'H', ['Attribute', 'PassYards', ['Value', 'A', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'H', 3, 14]], ['Value', 'H', ['Leaf', 'A', 5, 14]]]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'A', 5, 11]], ['Value', 'H', ['Leaf', 'A', 4, 11]]]]]]]]]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Attribute', 'PassYards', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'A', 4, 10]], ['Value', 'H', ['Leaf', 'A', 4, 10]]]], ['Value', 'H', ['Leaf', 'A', 8, 18]]]], ['Value', 'H', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Attribute', 'PassYards', ['Value', 'A', ['Leaf', 'A', 11, 23]], ['Value', 'H', ['Leaf', 'A', 5, 23]]]], ['Value', 'H', ['Attribute', 'PassYards', ['Value', 'A', ['Leaf', 'A', 8, 22]], ['Value', 'H', ['Leaf', 'A', 10, 22]]]]]]]], ['Value', 'H', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Leaf', 'A', 18, 57]], ['Value', 'H', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Attribute', 'PassYards', ['Value', 'A', ['Leaf', 'A', 13, 22]], ['Value', 'H', ['Leaf', 'A', 6, 22]]]], ['Value', 'H', ['Attribute', 'PassYards', ['Value', 'A', ['Leaf', 'A', 12, 17]], ['Value', 'H', ['Leaf', 'A', 4, 17]]]]]]]]]]]], ['Value', 'H', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Attribute', 'RushYards', ['Value', 'A', ['Leaf', 'A', 10, 19]], ['Value', 'H', ['Attribute', 'PassYards', ['Value', 'A', ['Leaf', 'A', 4, 9]], ['Value', 'H', ['Leaf', 'A', 5, 9]]]]]], ['Value', 'H', ['Leaf', 'A', 22, 41]]]], ['Value', 'H', ['Attribute', 'RushYards', ['Value', 'A', ['Attribute', 'PassYards', ['Value', 'A', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'A', 2, 4]], ['Value', 'H', ['Leaf', 'A', 1, 4]]]], ['Value', 'H', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'A', 6, 14]], ['Value', 'H', ['Leaf', 'A', 6, 14]]]]]], ['Value', 'H', ['Leaf', 'A', 4, 22]]]], ['Value', 'H', ['Leaf', 'A', 12, 34]]]]]]]], ['Value', 'H', ['Attribute', 'RushYards', ['Value', 'A', ['Attribute', 'WinPercentage', ['Value', 'A', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Attribute', 'PassYards', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'A', 1, 6]], ['Value', 'H', ['Leaf', 'A', 3, 6]]]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'A', 1, 2]], ['Value', 'H', ['Leaf', 'H', 1, 2]]]]]], ['Value', 'H', ['Leaf', 'A', 2, 10]]]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Attribute', 'PassYards', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'H', 4, 13]], ['Value', 'H', ['Leaf', 'A', 3, 13]]]], ['Value', 'H', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'A', 1, 3]], ['Value', 'H', ['Leaf', 'A', 1, 3]]]]]], ['Value', 'H', ['Attribute', 'PassYards', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'A', 2, 10]], ['Value', 'H', ['Leaf', 'A', 3, 10]]]], ['Value', 'H', ['Leaf', 'H', 2, 12]]]]]]]], ['Value', 'H', ['Attribute', 'PassYards', ['Value', 'A', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Leaf', 'H', 1, 4]], ['Value', 'H', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'A', 1, 3]], ['Value', 'H', ['Leaf', 'H', 1, 3]]]]]], ['Value', 'H', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Leaf', 'A', 1, 9]], ['Value', 'H', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'A', 2, 8]], ['Value', 'H', ['Leaf', 'A', 4, 8]]]]]]]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'A', 4, 15]], ['Value', 'H', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Leaf', 'A', 2, 5]], ['Value', 'H', ['Leaf', 'A', 1, 5]]]], ['Value', 'H', ['Leaf', 'A', 6, 11]]]]]]]]]], ['Value', 'H', ['Attribute', 'PassYards', ['Value', 'A', ['Attribute', 'WinPercentage', ['Value', 'A', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'A', 2, 4]], ['Value', 'H', ['Leaf', 'A', 1, 4]]]], ['Value', 'H', ['Leaf', 'A', 5, 9]]]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'A', 8, 14]], ['Value', 'H', ['Leaf', 'A', 2, 14]]]], ['Value', 'H', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'A', 4, 13]], ['Value', 'H', ['Leaf', 'A', 4, 13]]]]]]]], ['Value', 'H', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'A', 1, 3]], ['Value', 'H', ['Leaf', 'A', 2, 3]]]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'A', 6, 13]], ['Value', 'H', ['Leaf', 'A', 5, 13]]]], ['Value', 'H', ['Leaf', 'A', 2, 15]]]]]]]], ['Value', 'H', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Attribute', 'WinPercentage', ['Value', 'A', ['Leaf', 'A', 1, 8]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'A', 3, 7]], ['Value', 'H', ['Leaf', 'A', 1, 7]]]]]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Attribute', 'WinPercentage', ['Value', 'A', ['Leaf', 'A', 2, 7]], ['Value', 'H', ['Leaf', 'A', 3, 7]]]], ['Value', 'H', ['Leaf', 'A', 3, 10]]]]]], ['Value', 'H', ['Attribute', 'WinPercentage', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'A', 2, 7]], ['Value', 'H', ['Leaf', 'H', 3, 7]]]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'H', 2, 3]], ['Value', 'H', ['Leaf', 'A', 1, 3]]]]]], ['Value', 'H', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'H', 1, 3]], ['Value', 'H', ['Leaf', 'A', 2, 3]]]]]]]]]]]]]]]], ['Value', 'H', ['Attribute', 'DefenseScoringAllowed', ['Value', 'A', ['Attribute', 'TurnoverMargin', ['Value', 'A', ['Attribute', 'RushYards', ['Value', 'A', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Attribute', 'PassYards', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'A', 5, 8]], ['Value', 'H', ['Leaf', 'H', 3, 8]]]], ['Value', 'H', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Attribute', 'WinPercentage', ['Value', 'A', ['Leaf', 'A', 1, 17]], ['Value', 'H', ['Leaf', 'H', 12, 17]]]], ['Value', 'H', ['Attribute', 'WinPercentage', ['Value', 'A', ['Leaf', 'H', 1, 15]], ['Value', 'H', ['Leaf', 'H', 12, 15]]]]]], ['Value', 'H', ['Attribute', 'WinPercentage', ['Value', 'A', ['Leaf', 'H', 1, 20]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'H', 8, 19]], ['Value', 'H', ['Leaf', 'H', 5, 19]]]]]]]]]], ['Value', 'H', ['Attribute', 'WinPercentage', ['Value', 'A', ['Attribute', 'PassYards', ['Value', 'A', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'H', 3, 4]], ['Value', 'H', ['Leaf', 'H', 1, 4]]]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'H', 1, 2]], ['Value', 'H', ['Leaf', 'A', 1, 2]]]]]], ['Value', 'H', ['Attribute', 'PassYards', ['Value', 'A', ['Leaf', 'H', 8, 20]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'H', 5, 12]], ['Value', 'H', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'H', 2, 7]], ['Value', 'H', ['Leaf', 'H', 3, 7]]]]]]]]]]]], ['Value', 'H', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Attribute', 'PassYards', ['Value', 'A', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'H', 4, 9]], ['Value', 'H', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Attribute', 'WinPercentage', ['Value', 'A', ['Leaf', 'H', 1, 3]], ['Value', 'H', ['Leaf', 'A', 1, 3]]]], ['Value', 'H', ['Leaf', 'H', 2, 5]]]]]], ['Value', 'H', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Attribute', 'WinPercentage', ['Value', 'A', ['Leaf', 'H', 2, 15]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'H', 5, 13]], ['Value', 'H', ['Leaf', 'H', 4, 13]]]]]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Attribute', 'WinPercentage', ['Value', 'A', ['Leaf', 'A', 1, 8]], ['Value', 'H', ['Leaf', 'H', 5, 8]]]], ['Value', 'H', ['Attribute', 'WinPercentage', ['Value', 'A', ['Leaf', 'A', 1, 6]], ['Value', 'H', ['Leaf', 'A', 2, 6]]]]]]]]]], ['Value', 'H', ['Attribute', 'PassYards', ['Value', 'A', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'A', 3, 6]], ['Value', 'H', ['Leaf', 'A', 3, 6]]]], ['Value', 'H', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'A', 2, 3]], ['Value', 'H', ['Leaf', 'H', 1, 3]]]], ['Value', 'H', ['Leaf', 'H', 5, 8]]]]]]]]]], ['Value', 'H', ['Attribute', 'WinPercentage', ['Value', 'A', ['Attribute', 'RushYards', ['Value', 'A', ['Leaf', 'H', 13, 24]], ['Value', 'H', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Attribute', 'PassYards', ['Value', 'A', ['Leaf', 'A', 1, 2]], ['Value', 'H', ['Leaf', 'H', 1, 2]]]], ['Value', 'H', ['Leaf', 'H', 2, 4]]]], ['Value', 'H', ['Leaf', 'H', 7, 11]]]]]], ['Value', 'H', ['Attribute', 'RushYards', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Attribute', 'PassYards', ['Value', 'A', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Leaf', 'A', 2, 11]], ['Value', 'H', ['Leaf', 'H', 6, 11]]]], ['Value', 'H', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Leaf', 'H', 7, 22]], ['Value', 'H', ['Leaf', 'H', 10, 22]]]]]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Leaf', 'H', 12, 26]], ['Value', 'H', ['Leaf', 'H', 11, 26]]]], ['Value', 'H', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Leaf', 'H', 8, 14]], ['Value', 'H', ['Leaf', 'H', 3, 14]]]]]]]], ['Value', 'H', ['Attribute', 'PassYards', ['Value', 'A', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'H', 11, 32]], ['Value', 'H', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Leaf', 'H', 3, 21]], ['Value', 'H', ['Leaf', 'H', 15, 21]]]]]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Leaf', 'H', 18, 35]], ['Value', 'H', ['Leaf', 'H', 10, 35]]]], ['Value', 'H', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Leaf', 'H', 22, 34]], ['Value', 'H', ['Leaf', 'H', 8, 34]]]]]]]]]], ['Value', 'H', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Attribute', 'PassYards', ['Value', 'A', ['Leaf', 'H', 11, 49]], ['Value', 'H', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'H', 13, 38]], ['Value', 'H', ['Leaf', 'H', 22, 38]]]]]], ['Value', 'H', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Attribute', 'PassYards', ['Value', 'A', ['Leaf', 'H', 4, 16]], ['Value', 'H', ['Leaf', 'H', 11, 16]]]], ['Value', 'H', ['Attribute', 'PassYards', ['Value', 'A', ['Leaf', 'H', 6, 26]], ['Value', 'H', ['Leaf', 'H', 15, 26]]]]]]]], ['Value', 'H', ['Attribute', 'PassYards', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'H', 10, 18]], ['Value', 'H', ['Leaf', 'H', 6, 18]]]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'H', 7, 17]], ['Value', 'H', ['Leaf', 'H', 5, 17]]]]]], ['Value', 'H', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'A', 1, 13]], ['Value', 'H', ['Leaf', 'H', 10, 13]]]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'H', 6, 23]], ['Value', 'H', ['Leaf', 'H', 15, 23]]]]]]]]]]]]]]]], ['Value', 'H', ['Attribute', 'PassYards', ['Value', 'A', ['Attribute', 'WinPercentage', ['Value', 'A', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Leaf', 'H', 2, 32]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Attribute', 'TurnoverMargin', ['Value', 'A', ['Attribute', 'RushYards', ['Value', 'A', ['Leaf', 'A', 3, 6]], ['Value', 'H', ['Leaf', 'A', 3, 6]]]], ['Value', 'H', ['Attribute', 'RushYards', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'A', 1, 5]], ['Value', 'H', ['Leaf', 'H', 3, 5]]]], ['Value', 'H', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'A', 1, 4]], ['Value', 'H', ['Leaf', 'A', 1, 4]]]]]]]], ['Value', 'H', ['Attribute', 'RushYards', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Attribute', 'TurnoverMargin', ['Value', 'A', ['Leaf', 'H', 1, 3]], ['Value', 'H', ['Leaf', 'A', 1, 3]]]], ['Value', 'H', ['Leaf', 'H', 4, 7]]]], ['Value', 'H', ['Attribute', 'TurnoverMargin', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'A', 1, 4]], ['Value', 'H', ['Leaf', 'A', 1, 4]]]], ['Value', 'H', ['Leaf', 'H', 4, 8]]]]]]]]]], ['Value', 'H', ['Attribute', 'TurnoverMargin', ['Value', 'A', ['Attribute', 'RushYards', ['Value', 'A', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'H', 1, 2]], ['Value', 'H', ['Leaf', 'A', 1, 2]]]], ['Value', 'H', ['Leaf', 'A', 5, 7]]]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Leaf', 'H', 1, 4]], ['Value', 'H', ['Leaf', 'A', 3, 4]]]], ['Value', 'H', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Leaf', 'A', 2, 8]], ['Value', 'H', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'A', 2, 6]], ['Value', 'H', ['Leaf', 'H', 2, 6]]]]]]]]]], ['Value', 'H', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Attribute', 'RushYards', ['Value', 'A', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'A', 2, 7]], ['Value', 'H', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Leaf', 'A', 1, 5]], ['Value', 'H', ['Leaf', 'A', 2, 5]]]]]], ['Value', 'H', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Leaf', 'H', 1, 8]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'A', 3, 7]], ['Value', 'H', ['Leaf', 'H', 2, 7]]]]]]]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Attribute', 'RushYards', ['Value', 'A', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Leaf', 'H', 1, 5]], ['Value', 'H', ['Leaf', 'H', 3, 5]]]], ['Value', 'H', ['Leaf', 'A', 6, 11]]]], ['Value', 'H', ['Attribute', 'RushYards', ['Value', 'A', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Leaf', 'A', 1, 6]], ['Value', 'H', ['Leaf', 'A', 3, 6]]]], ['Value', 'H', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Leaf', 'H', 1, 6]], ['Value', 'H', ['Leaf', 'A', 3, 6]]]]]]]]]]]]]], ['Value', 'H', ['Attribute', 'TurnoverMargin', ['Value', 'A', ['Attribute', 'WinPercentage', ['Value', 'A', ['Attribute', 'RushYards', ['Value', 'A', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'H', 2, 5]], ['Value', 'H', ['Leaf', 'A', 1, 5]]]], ['Value', 'H', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'A', 1, 3]], ['Value', 'H', ['Leaf', 'H', 1, 3]]]]]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'A', 2, 8]], ['Value', 'H', ['Leaf', 'A', 4, 8]]]], ['Value', 'H', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Leaf', 'H', 3, 8]], ['Value', 'H', ['Leaf', 'A', 3, 8]]]]]]]], ['Value', 'H', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'A', 1, 4]], ['Value', 'H', ['Leaf', 'A', 2, 4]]]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'H', 4, 11]], ['Value', 'H', ['Leaf', 'A', 2, 11]]]]]], ['Value', 'H', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Leaf', 'H', 2, 9]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'H', 2, 7]], ['Value', 'H', ['Leaf', 'H', 3, 7]]]]]]]]]], ['Value', 'H', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Attribute', 'RushYards', ['Value', 'A', ['Leaf', 'A', 2, 4]], ['Value', 'H', ['Leaf', 'A', 1, 4]]]], ['Value', 'H', ['Attribute', 'RushYards', ['Value', 'A', ['Leaf', 'A', 1, 5]], ['Value', 'H', ['Leaf', 'H', 2, 5]]]]]], ['Value', 'H', ['Attribute', 'RushYards', ['Value', 'A', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'H', 5, 12]], ['Value', 'H', ['Leaf', 'H', 4, 12]]]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'H', 3, 14]], ['Value', 'H', ['Leaf', 'H', 7, 14]]]]]]]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Attribute', 'RushYards', ['Value', 'A', ['Leaf', 'H', 3, 10]], ['Value', 'H', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Leaf', 'H', 2, 7]], ['Value', 'H', ['Leaf', 'A', 2, 7]]]]]], ['Value', 'H', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Attribute', 'RushYards', ['Value', 'A', ['Leaf', 'H', 2, 4]], ['Value', 'H', ['Leaf', 'A', 2, 4]]]], ['Value', 'H', ['Attribute', 'RushYards', ['Value', 'A', ['Leaf', 'A', 4, 12]], ['Value', 'H', ['Leaf', 'A', 4, 12]]]]]]]]]]]], ['Value', 'H', ['Attribute', 'RushYards', ['Value', 'A', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Leaf', 'A', 1, 9]], ['Value', 'H', ['Attribute', 'WinPercentage', ['Value', 'A', ['Leaf', 'H', 1, 8]], ['Value', 'H', ['Leaf', 'H', 4, 8]]]]]], ['Value', 'H', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Attribute', 'WinPercentage', ['Value', 'A', ['Leaf', 'A', 1, 7]], ['Value', 'H', ['Leaf', 'H', 4, 7]]]], ['Value', 'H', ['Attribute', 'WinPercentage', ['Value', 'A', ['Leaf', 'H', 3, 12]], ['Value', 'H', ['Leaf', 'H', 5, 12]]]]]]]], ['Value', 'H', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'H', 3, 9]], ['Value', 'H', ['Attribute', 'WinPercentage', ['Value', 'A', ['Leaf', 'H', 1, 6]], ['Value', 'H', ['Leaf', 'H', 4, 6]]]]]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Attribute', 'WinPercentage', ['Value', 'A', ['Leaf', 'H', 3, 14]], ['Value', 'H', ['Leaf', 'H', 8, 14]]]], ['Value', 'H', ['Attribute', 'WinPercentage', ['Value', 'A', ['Leaf', 'H', 3, 16]], ['Value', 'H', ['Leaf', 'H', 8, 16]]]]]]]]]], ['Value', 'H', ['Attribute', 'KickingPercentage', ['Value', 'A', ['Attribute', 'WinPercentage', ['Value', 'A', ['Leaf', 'H', 2, 25]], ['Value', 'H', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'H', 1, 5]], ['Value', 'H', ['Leaf', 'A', 3, 5]]]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Leaf', 'H', 6, 18]], ['Value', 'H', ['Leaf', 'H', 7, 18]]]]]]]], ['Value', 'H', ['Attribute', 'RushYardsAllowed', ['Value', 'A', ['Attribute', 'WinPercentage', ['Value', 'A', ['Leaf', 'H', 1, 11]], ['Value', 'H', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Leaf', 'A', 3, 10]], ['Value', 'H', ['Leaf', 'A', 3, 10]]]]]], ['Value', 'H', ['Attribute', 'WinPercentage', ['Value', 'A', ['Leaf', 'A', 1, 13]], ['Value', 'H', ['Attribute', 'PassYardsAllowed', ['Value', 'A', ['Leaf', 'H', 2, 12]], ['Value', 'H', ['Leaf', 'H', 5, 12]]]]]]]]]]]]]]]]]]]

#TODO: implement forest with fit() for nfl data
#then pickle forest

packaged_obj = (NFL_header, NFL_Decision_tree)
outfile = open("tree.p", 'wb')
pickle.dump(packaged_obj,outfile)
outfile.close()
