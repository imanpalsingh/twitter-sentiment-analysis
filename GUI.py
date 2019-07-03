'''
Author - Imanpal Singh <imanpalsingh@gmail.com>
GUI application for twitter sentiment analysis
Date created :  - 02-07-2019
Date modified : - 03-07-2019
'''

#importing requierd libraries
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 8000)

import re

import tkinter as tk
from tkinter import Text

import pyperclip

#Global variable to hold score
score=0


#Machine Learning part

def Algorithm(file):
    
    global score
    #Loading the dataset
    dataset = pd.read_csv(file)
    
    #Cleaning tweets
    clean_tweets = []
    
    for i in range(len(dataset)):
        tw = re.sub('[^a-zA-Z]', ' ', dataset['tweet'][i])
        tw = re.sub('@[\w]*',' ',tw)
        tw = tw.lower()
        tw = tw.split()
        tw = [ps.stem(token) for token in tw if not token in set(stopwords.words('english'))]
        tw = ' '.join(tw)
        clean_tweets.append(tw)
    
    #textual encoding
    X = cv.fit_transform(clean_tweets)
    X = X.toarray()
    y = dataset.iloc[:, 1].values
    
    #splitting the data
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y)
    #training the data
    nb.fit(X_train,y_train)
    score = nb.score(X_test,y_test)
    print("Score is : - ",score)
    

#Function to handle Go button event
def forward():
    tw = tweet.get("1.0","end")
    tw = re.sub('[^a-zA-Z]', ' ', tw)
    tw = re.sub('@[\w]*',' ',tw)
    tw = tw.lower()
    tw = tw.split("\n")
    tw = [ps.stem(token) for token in tw if not token in set(stopwords.words('english'))]
    tw = cv.transform(tw)
   
    tw = tw.toarray()
    
    
    y_pred = nb.predict(tw)
    
    tweet.delete("1.0","end")
    
    if y_pred[0] == 0:
        tweet.insert("1.0","The tweet entered is normal ( model's accuracy : {}% )".format(score*100))
    else :
        tweet.insert("1.0","The tweet entered is negative ( model's accuracy : {}% )".format(score*100))

#Function to handle Paste from clipboard button event
def clippaste():
    
    tweet.insert("1.0",pyperclip.paste())
    




    

#Initialising algorithm
Algorithm('train.csv')
    
#GUI part
    
#Creating a window
Main = tk.Tk()
Main.configure(background='white')
Main.title("Twitter Sentiment analysis")
Main.geometry("1000x400+400+300")

#Adding the heading
one = tk.Label(Main,text="Twitter Sentiment analysis",fg="white",width="100",height="2")
one.configure(background="#6E97ED",font=(20))
    
#Adding the textbox
tweet = tk.Text(Main,height="10",width="60")
tweet.insert("1.0","Paste tweet here..")
tweet.configure(bd=0,fg="#6E97ED")

#Adding buttons   
button_frame = tk.Frame(Main)
button_frame.configure(background="white")
go= tk.Button(button_frame,text="GO !",width="10",height="5",command=forward)
go.configure(background="#6E97ED",fg="white",bd=0)
paste = tk.Button(button_frame,text="Paste from clipboard",width="20",height="5",command=clippaste)
paste.configure(background="#6E97ED",fg="white",bd=0)
    
#Finishing up
one.pack(pady=30)
tweet.pack(side="top",padx=10,pady=20)
go.pack(side="left")
paste.pack(side="left",padx="30")
button_frame.pack(side="bottom")
    
#Removing resizeable feature
Main.resizable(0,0)
tk.mainloop()
    