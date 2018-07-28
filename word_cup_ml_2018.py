# -*- coding: utf-8 -*-
""" 
Created on Thu Jul 26 15:18:46 2018

@author: asalkanovic1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.ticker as plticker
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Loading the data sets
world_cup = pd.read_csv('C:\\Users\\asalkanovic1\\Downloads\\WC\\World Cup 2018 Dataset.csv')
results = pd.read_csv('C:\\Users\\asalkanovic1\\Downloads\\WC\\results.csv')

# Ensuring data sets are laoded
world_cup.head()
results.head()

# Feature engineering - match winner
winner = []
for i in range(len(results['home_team'])):
    if results ['home_score'][i] > results['away_score'][i]:
        winner.append(results['home_team'][i])
    elif results ['home_score'][i] < results['away_score'][i]:
        winner.append(results['away_team'][i])
    else:
        winner.append('Draw')
results['winning_team'] = winner

# Feature engineering - goal difference
results['goal_difference'] = np.absolute(results['home_score'] - results['away_score'])

results.head()

# Examine Nigeria specifically to see glean more about useful features
df = results[(results['home_team'] == 'Nigeria') | (results['away_team'] == 'Nigeria')]

nigeria = df.iloc[:]

# Creating a column for year. And we want matches after the first WC year in 1930.
year = []
for row in nigeria['date']:
    year.append(int(row[:4]))
nigeria['match_year'] = year
nigeria_1930 = nigeria[nigeria.match_year >= 1930]
nigeria_1930.count()

# Re-creating the plot in the demo project that doesn't have source code provided
X = np.arange(3)
objects = ('Win', 'Loss', 'Draw')
win = len(nigeria_1930[nigeria_1930['winning_team'] == 'Nigeria'])
loss = len(nigeria_1930[(nigeria_1930['winning_team'] != 'Nigeria') & (nigeria_1930['winning_team'] != 'Draw')])
draw = len(nigeria_1930[nigeria_1930['winning_team'] == 'Draw'])
Y = [win, loss, draw]
plt.bar(X, Y, align = 'center', alpha = 0.5, color = 'grb')
plt.xticks(X, objects)
plt.xlabel('Nigeria_Result')
plt.ylabel('Count')
plt.show()

worldcup_teams = ['Australia', 'Iran', 'Japan', 'Korea Republic',
                  'Saudi Arabia', 'Egypt', 'Morocco', 'Nigeria',
                  'Senegal', 'Tunisia', 'Costa Rica', 'Mexico',
                  'Panama', 'Argentina', 'Brazil', 'Columbia',
                  'Peru', 'Uruguay', 'Belgium', 'Croatia',
                  'Denmark', 'England', 'France', 'Germany',
                  'Iceland', 'Poland', 'Portugal', 'Russia',
                  'Servia', 'Spain', 'Sweden', 'Switzerland']

df_teams_home = results[results['home_team'].isin(worldcup_teams)]
df_teams_away = results[results['away_team'].isin(worldcup_teams)]
df_teams = pd.concat((df_teams_home, df_teams_away))
df_teams.drop_duplicates()
df_teams.count()

"""
Create a year column and drop games before 1930 as well as columns that won’t 
affect match outcome for example date, home_score, away_score, tournament, city, 
country, goal_difference and match_year.
"""
year = []
for row in df_teams['date']:
    year.append(int(row[:4]))
df_teams['match_year'] = year
df_teams_1930 = df_teams[df_teams.match_year >= 1930]
df_teams_1930.head()

# Drop columns that do not affect match outcome
df_teams_1930 = df_teams_1930.drop(['date', 'home_score', 'away_score', 'tournament', 'city', 'country', 'goal_difference', 'match_year'], 1)
