# -*- coding: utf-8 -*-
""" 
Created on Thu Jul 26 15:18:46 2018

@author: asalkanovic1

Tutorial: https://blog.goodaudience.com/predicting-fifa-world-cup-2018-using-machine-learning-dc07ad8dd576
https://github.com/itsmuriuki/FIFA-2018-World-cup-predictions/blob/master/Predicting%20Fifa%202018.ipynb
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.ticker as plticker
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

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

# Let's begin the main effort
worldcup_teams = ['Australia', 'Iran', 'Japan', 'Korea Republic',
                  'Saudi Arabia', 'Egypt', 'Morocco', 'Nigeria',
                  'Senegal', 'Tunisia', 'Costa Rica', 'Mexico',
                  'Panama', 'Argentina', 'Brazil', 'Colombia',
                  'Peru', 'Uruguay', 'Belgium', 'Croatia',
                  'Denmark', 'England', 'France', 'Germany',
                  'Iceland', 'Poland', 'Portugal', 'Russia',
                  'Serbia', 'Spain', 'Sweden', 'Switzerland']

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
df_teams_1930 = df_teams_1930.drop(['date', 'home_score', 'away_score', 'tournament', 'city', 'country', 'neutral','goal_difference', 'match_year'], 1)
df_teams_1930.head()


### Building the model ###

# The prediction label: The winning_team will show '2' if home team has won,
# a '1' if there was a draw, and a '0' if the home team has lost
df_teams_1930 = df_teams_1930.reset_index(drop=True)
df_teams_1930.loc[df_teams_1930.winning_team == df_teams_1930.home_team, 'winning_team']=2
df_teams_1930.loc[df_teams_1930.winning_team == df_teams_1930.away_team, 'winning_team']=0
df_teams_1930.loc[df_teams_1930.winning_team == 'Draw', 'winning_team']=1

df_teams_1930.head()

# Convert home team and away team from categorical variables to continuous inputs
# Get dummy variables
final = pd.get_dummies(df_teams_1930, prefix = ['home_team','away_team'], columns = ['home_team','away_team'])

# Separate X and Y sets
X = final.drop(['winning_team'], axis = 1)
Y = final["winning_team"]
Y = Y.astype('int')

# Separate train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 42)

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
score = logreg.score(X_train, Y_train)
score2 = logreg.score(X_test, Y_test)
print("LogReg - Training set accuracy: ", '%.3f'%(score))
print("LogReg - Test set accuracy: ", '%.3f'%(score2))
print("")

#svm = SVC()
#svm.fit(X_train, Y_train)
#score3 = svm.score(X_train, Y_train)
#score4 = svm.score(X_test, Y_test)
#print("SVM - Training set accuracy: ", '%.3f'%(score3))
#print("SVM - Test set accuracy: ", '%.3f'%(score4))

# Let's consider Fifa Rankings
# The team which is positioned high in the FIFA Rankings wil lbe considered the 
# "favourite" for the match and therefore, will be positioned under the "home_teams"
# column since there are no "home" or "away" teams in the World Cup fixtures.

# Loading new datasets
ranking = pd.read_csv('C:\\Users\\asalkanovic1\\Downloads\\WC\\fifa_rankings.csv')
fixtures = pd.read_csv('C:\\Users\\asalkanovic1\\Downloads\\WC\\fixtures.csv')

# List for storing the group stage games
pred_set = []

# Create new columns with ranking position of each team
fixtures.insert(1, 'first_position', fixtures['Home Team'].map(ranking.set_index('Team')['Position']))
fixtures.insert(2, 'second_position', fixtures['Away Team'].map(ranking.set_index('Team')['Position']))

# We only need the group stage games, so we have to slice the fixtures dataset
fixtures = fixtures.iloc[:48, :]
fixtures.tail()

# Loop to add teams to new prediction dataset based on the ranking position of each team
for index, row in fixtures.iterrows():
    if row['first_position'] < row['second_position']:
        pred_set.append({'home_team': row['Home Team'], 'away_team': row['Away Team'], 'winning_team': None})
    else:
        pred_set.append({'home_team': row['Away Team'], 'away_team': row['Home Team'], 'winning_team': None})

pred_set = pd.DataFrame(pred_set)
backup_pred_set = pred_set
pred_set.head()

# Get dummy variables and drop winning_team column
pred_set = pd.get_dummies(pred_set, prefix = ['home_team','away_team'], columns = ['home_team','away_team'])

# Add missing columns comparted to te model's training dataset
missing_cols = set(final.columns) - set(pred_set.columns)
for c in missing_cols:
    pred_set[c] = 0
pred_set = pred_set[final.columns]

# Remove winning team column
pred_set = pred_set.drop(['winning_team'], axis = 1)

pred_set.head()

# Group stage matches
predictions = logreg.predict(pred_set)
for i in range(fixtures.shape[0]):
    print(backup_pred_set.iloc[i, 1] + " and " + backup_pred_set.iloc[i, 0])
    if predictions[i] == 2:
        print("Winner: " + backup_pred_set.iloc[i, 1])
    elif predictions[i] == 1:
        print("Draw")
    elif predictions[i] == 0:
        print("Winner: " + backup_pred_set.iloc[i, 0])
    print('Probability of ' + backup_pred_set.iloc[i, 1] + ' winning: ', '%.3f'%(logreg.predict_proba(pred_set)[i][2]))
    print('Probability of Draw: ', '%.3f'%(logreg.predict_proba(pred_set)[i][1]))
    print('Probability of ' + backup_pred_set.iloc[i, 0] + ' winning: ', '%.3f'%(logreg.predict_proba(pred_set)[i][0]))
    print("")

def clean_and_predict(matches, ranking, final, logreg):

    # Initialization of auxiliary list for data cleaning
    positions = []

    # Loop to retrieve each team's position according to FIFA ranking
    for match in matches:
        positions.append(ranking.loc[ranking['Team'] == match[0],'Position'].iloc[0])
        positions.append(ranking.loc[ranking['Team'] == match[1],'Position'].iloc[0])
    
    # Creating the DataFrame for prediction
    pred_set = []

    # Initializing iterators for while loop
    i = 0
    j = 0

    # 'i' will be the iterator for the 'positions' list, and 'j' for the list of matches (list of tuples)
    while i < len(positions):
        dict1 = {}

        # If position of first team is better, he will be the 'home' team, and vice-versa
        if positions[i] < positions[i + 1]:
            dict1.update({'home_team': matches[j][0], 'away_team': matches[j][1]})
        else:
            dict1.update({'home_team': matches[j][1], 'away_team': matches[j][0]})

        # Append updated dictionary to the list, that will later be converted into a DataFrame
        pred_set.append(dict1)
        i += 2
        j += 1

    # Convert list into DataFrame
    pred_set = pd.DataFrame(pred_set)
    backup_pred_set = pred_set

    # Get dummy variables and drop winning_team column
    pred_set = pd.get_dummies(pred_set, prefix=['home_team', 'away_team'], columns=['home_team', 'away_team'])

    # Add missing columns compared to the model's training dataset
    missing_cols2 = set(final.columns) - set(pred_set.columns)
    for c in missing_cols2:
        pred_set[c] = 0
    pred_set = pred_set[final.columns]

    # Remove winning team column
    pred_set = pred_set.drop(['winning_team'], axis=1)

    # Predict!
    predictions = logreg.predict(pred_set)
    for i in range(len(pred_set)):
        print(backup_pred_set.iloc[i, 1] + " and " + backup_pred_set.iloc[i, 0])
        if predictions[i] == 2:
            print("Winner: " + backup_pred_set.iloc[i, 1])
        elif predictions[i] == 1:
            print("Draw")
        elif predictions[i] == 0:
            print("Winner: " + backup_pred_set.iloc[i, 0])
        print('Probability of ' + backup_pred_set.iloc[i, 1] + ' winning: ' , '%.3f'%(logreg.predict_proba(pred_set)[i][2]))
        print('Probability of Draw: ', '%.3f'%(logreg.predict_proba(pred_set)[i][1])) 
        print('Probability of ' + backup_pred_set.iloc[i, 0] + ' winning: ', '%.3f'%(logreg.predict_proba(pred_set)[i][0]))
        print("")
    
# List of Round of 16 match ups 
group_16 = [('Uruguay', 'Portugal'),
            ('France', 'Croatia'),
            ('Brazil', 'Mexico'),
            ('England', 'Colombia'),
            ('Spain', 'Russia'),
            ('Argentina', 'Peru'),
            ('Germany', 'Switzerland'),
            ('Poland', 'Belgium')]

clean_and_predict(group_16, ranking, final, logreg)

quarter_finals = [('Portugal', 'France'),
            ('Spain', 'Argentina'),
            ('Brazil', 'England'),
            ('Germany', 'Belgium')]

clean_and_predict(quarter_finals, ranking, final, logreg)

semi_finals = [('Portugal', 'Brazil'),
        ('Argentina', 'Germany')]

clean_and_predict(semi_finals, ranking, final, logreg)

wc_final = [('Brazil', 'Germany')]

clean_and_predict(wc_final, ranking, final, logreg)
