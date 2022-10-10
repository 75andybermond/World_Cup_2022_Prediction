import pandas as pd 
import streamlit as st 
import numpy as np 
import pickle
from PIL import Image
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler



df = pd.read_csv('df_for_ML_countries_not_numeric.csv', index_col='date')

model = pickle.load(open('finalized_model.pkl', 'rb'))


st.title("Qui ramènera la coupe à la maison ? ")

image = Image.open('cup_image.jpg')

st.image(image, caption='World_cup_2022')

# Home part 

home_team = st.selectbox('Home team', ["Qatar" ,"Ecuador", "Senegal", "Netherlands" ,"England", "IR Iran" , "USA" , "Wales",  "Argentina" , "Saudi Arabia",  "Mexico", "Poland",  "France" ,"Australia", "Denmark" ,"Tunisia" , "Spain" , "Costa Rica",  "Germany", "Japan" , "Belgium",  "Canada",  "Morocco",  "Croatia",  "Brazil",  "Serbia",  "Switzerland",  "Cameroon",  "Portugal",  "Ghana",  "Uruguay", "Korea Republic"])

home_fifa_rank =  st.slider('Home_team FIFA rank ', 0, 100)


#Function to get average of the last 10 maths played by the home_team
def get_average_home(df, country):
    df = df.drop(columns=['away_team',  'home_team_fifa_rank','away_team_fifa_rank', 'away_team_goalkeeper_score', 'away_team_mean_defense_score',	'away_team_mean_offense_score'	, 'away_team_mean_midfield_score', 'result' ], axis=1)
    l = df.loc[(df['home_team'] == country ) & (df.index.isin(df.index[:-10])) ].tail(10) 
    l = df.loc['mean'] = l.mean()
    l = df.tail(1)
    
    return l

res = get_average_home(df, home_team)

st.write("Result:", res)

home_team_goalkeeper = st.number_input('home_team_goalkeeper_score', )

home_team_defense = st.number_input('home_team_defense_score', )

home_team_offense =  st.number_input('home_team_offense_score', )

home_team_midfield = st.number_input('home_team_midfield_score', )

## Away part 


away_team = st.selectbox('Away team', ["Qatar" ,"Ecuador", "Senegal", "Netherlands" ,"England", "IR Iran" , "USA" , "Wales",  "Argentina" , "Saudi Arabia",  "Mexico", "Poland",  "France" ,"Australia", "Denmark" ,"Tunisia" , "Spain" , "Costa Rica",  "Germany", "Japan" , "Belgium",  "Canada",  "Morocco",  "Croatia",  "Brazil",  "Serbia",  "Switzerland",  "Cameroon",  "Portugal",  "Ghana",  "Uruguay", "Korea Republic"])

away_fifa_rank =  st.slider('Away_team FIFA rank ', 0, 100)

def get_average_away(df, country):
    df = df.drop(columns=['home_team',  'home_team_fifa_rank','away_team_fifa_rank', 'home_team_goalkeeper_score', 'home_team_mean_defense_score',	'home_team_mean_offense_score'	, 'home_team_mean_midfield_score', 'result' ], axis=1)
    l = df.loc[(df['away_team'] == country ) & (df.index.isin(df.index[:-10])) ].tail(10) # loc Qatar and get the last 10 matchs played 
    l = df.loc['mean'] = l.mean()
    l = df.tail(1)
    
    return l

res_away = get_average_away(df, away_team)

st.write("Result:", res_away)

away_team_goalkeeper = st.number_input('away_team_goalkeeper_score', )

away_team_defense = st.number_input('away_team_defense_score', )

away_team_offense =  st.number_input('away_team_offense_score', )

away_team_midfield = st.number_input('away_team_midfield_score', )

#### Labelling countries 

le = preprocessing.LabelEncoder()
home_team = le.fit_transform([home_team])
away_team =  le.fit_transform([away_team])

columns = ['home_team', 'away_team', 'home_team_fifa_rank', 'away_team_fifa_rank', 'home_team_goalkeeper_score',
       'away_team_goalkeeper_score', 'home_team_mean_defense_score',
       'home_team_mean_offense_score', 'home_team_mean_midfield_score',
       'away_team_mean_defense_score', 'away_team_mean_offense_score',
       'away_team_mean_midfield_score']

### Prediction 
from sklearn.preprocessing import StandardScaler

def predict():
    row = np.array([home_team, away_team, home_fifa_rank, away_fifa_rank,home_team_goalkeeper, away_team_goalkeeper, home_team_defense, home_team_offense, home_team_midfield,
                    away_team_defense, away_team_offense, away_team_midfield])
    X = pd.DataFrame([row], columns=columns)
    Scaller = StandardScaler()
    # standardization 
    X = Scaller.fit_transform(X) 
    prediction = model.predict(X)[0]
    
    if prediction == 3:
        st.success('Nice victory', icon="✅")
    elif prediction == 1:
        st.error('It is a draw')
    else:
        st.error('Sorry it is a loss :thumbsdown:')

st.button('Prediction for the team at home', on_click=predict)

