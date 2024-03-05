import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
# import numpy as np

# create a title and sub-title
st.write("""
#Covid Detection
Detect if someone is  Covid Positive using Machine Learning and python!
""")
# Get the dataean
ds = pd.read_csv('C:\\Users\\sabyasachi\\Desktop\\final project\\Final.csv')
# ds.drop(columns=['Unnamed: 0','Unnamed: 0.1'])
# set a subheader
st.subheader('Data Information:')
# show data as table
st.dataframe(ds.head(20))
# ds.drop(columns=['Unnamed: 0','Unnamed: 0.1'])
# Show statistics on the data

st.write(ds.describe())
# Show data as chart
# chart = st.bar_chart(ds)
# split the data
X = ds.loc[:, ['head_ache', 'fever', 'cough', 'sore_throat', 'shortness_of_breath', 'age_60_and_above']].values

y = ds['corona_result'].values
# Split into 75 % traiming and 25%testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# Get the feature input from the user
def get_user_input():
    fever = st.sidebar.slider('fever', 0, 1, 0)
    cough = st.sidebar.slider('cough', 0, 1, 0)
    sore_throat = st.sidebar.slider('sore_throat', 0, 1, 0)
    shortness_of_breath = st.sidebar.slider('shortness_of_breath', 0, 1, 0)
    head_ache = st.sidebar.slider('head_ache', 0, 1, 0)
    age_60_and_above = st.sidebar.slider('age_60_and_above', 0, 1, 0)
    # gender = st.sidebar.slider('gender', 0, 199, 117)
    # test_indication = st.sidebar.slider('test_indication', 0, 199, 117)


# store a dictionary into a variable
    user_data = {'fever': fever, 'cough': cough, 'sore-throat': sore_throat, 'shortness_of_breath': shortness_of_breath,
              'head_ache': head_ache, 'age_60_and_above': age_60_and_above}

    #transfrom  the data into a dataframe
    features = pd.DataFrame(user_data, index=[0])
    return features
#Store  user input into a variable
user_input = get_user_input()

#set a subheader
st.subheader('Symptoms:')
st.write(user_input)

#create and train the model

RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, y_train)

#show module metrics
st.subheader('Test Accuracy Score:')
st.write(str(accuracy_score(y_test, RandomForestClassifier.predict(X_test) * 100)*100) +'%')

#store the models predictions in a variable
prediction = RandomForestClassifier.predict(user_input)

# Set a subheader and display the classification
st.subheader('Result:')
st.write(prediction)

# st.subheader('Result')
st.write("0 indicates COVID NEGATIVE and 2 indicates COVID POSITIVE")



