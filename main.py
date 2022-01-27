from cProfile import label
from fileinput import filename
from tkinter import font
from _plotly_utils.utils import find_closest_string
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import datetime
import networkx as nx
from icecream import ic
from nltk.util import bigrams 
import os
from bokeh.palettes import Spectral4
from bokeh.models import (BoxSelectTool, Circle, EdgesAndLinkedNodes, HoverTool,
                          MultiLine, NodesAndLinkedEdges, Range1d, TapTool,
                          Range1d, ColumnDataSource, LabelSet)
from bokeh.plotting import from_networkx

from bokeh.palettes import Blues8, Reds8, Purples8, Oranges8, Viridis8, Spectral8
from bokeh.transform import linear_cmap

from functions import *



#################################
# Set up the upper navigate bar #
#################################

st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #3498DB;">
  <a class="navbar-brand" href="https://hope-project.dk/#/" target="_blank">HOPE Project</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
      <li class="nav-item">
        <a class="nav-link" href= "https://github.com/stinenyhus/HOPE-keyword-query-Twitter" target="_blank">GitHub</a>
      </li>
    </ul>
  </div>
</nav>
""", unsafe_allow_html = True)


############################################
# Hide the menu button (on the right side) #
############################################

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True) 


#############################################################################
# Condense the layout (fit on the screen to reduce the amount of scrolling) #
#############################################################################

padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)


##################################
# Define palettes for bar charts #
##################################


palette0 =[[0, "#F0E442"],    #yellow
           [0.01, "#d62728"], #brick red
           [0.03, "#800080"], #purple
           [1.0, "#000000"]]  #black 
 
# Blues 
palette1 =[[0, "#f8fcfd"],   
           [0.01, "#e9f5f8"], 
           [0.02, "#cbe6ef"], 
           [0.04, "#bcdfeb"],
           [0.07, "#62b4cf"],
           [1.0, "#00008b"]]  


######################
# Pipeline for plots #
######################


def main():

    menu = st.sidebar.selectbox('MENU', ['Restriktion', 
                                         'Genåb', 
                                         'Corona', 
                                         'Coronapas',
                                         'Lockdown',
                                         'Mettef',
                                         'Mundbind',
                                         'Omicron',
                                         'Pressekonference',
                                         'Vaccin'])

    navigator = st.sidebar.radio('NAVIGATE TO', ['Tweet Frequency', 
                                                 'Sentiment', 
                                                 'Hashtag Frequency', 
                                                 'Word Frequency', 
                                                 'Bigrams', 
                                                 'WordCloud'])
    

    if navigator == 'Tweet Frequency':
       # st.title('Tweet Frequency Over Time') 

        df = read_pkl(label = menu, 
                       path = 'data/',
                       data_prefix = '.pkl') 
        
        if menu == 'Omicron':
          fig = plot_tweet_freq(data = df,
                                x = 'date',
                                y = 'nr_of_tweets',
                                y_smoothed = 's500_nr_of_tweets',
                                line_name = 'Number of tweets',
                                line_smoothed_name = 'Smoothed values',
                                title = 'Frequency of Mentions', 
                                xaxis_range = [datetime.datetime(2021, 11, 1),
                                               datetime.datetime(2022, 1, 31)])

        else:  
          fig = plot_tweet_freq(data = df,
                                x = 'date',
                                y = 'nr_of_tweets',
                                y_smoothed = 's500_nr_of_tweets',
                                line_name = 'Number of tweets',
                                line_smoothed_name = 'Smoothed values',
                                title = 'Frequency of Mentions', 
                                xaxis_range = [datetime.datetime(2020, 11, 1),
                                               datetime.datetime(2022, 1, 31)])
                                  
        st.plotly_chart(fig, use_container_width = True)

    elif navigator == 'Sentiment':
        #   st.title('Sentiment Over Time')
           st.subheader('Centered sentiment score')

           df0 = read_pkl(label = menu, 
                          path = 'data/',
                          data_prefix = '.pkl') 

           df1 = read_pkl(label = menu, 
                          path = 'data/',
                          data_prefix = '_sentiment.pkl') 
        
           if menu == 'Omicron':

             fig = plot_sentiment(df0, 
                                  df1,
                                  x = 'date',
                                  y = 'compound_mean', 
                                  y_smoothed = 's500_compound',                              
                                  upper = 'compound_upper',
                                  lower = 'compound_lower',
                                  line_name = 'Centered sentiment score',
                                  line_smoothed_name = 'Smoothed',
                                  title = 'Sentiment',
                                  xaxis_range = [datetime.datetime(2021, 11, 1),
                                                datetime.datetime(2022, 1, 31)]) 

           elif (menu == 'Vaccin') or (menu == 'Corona'):
             fig = plot_sentiment(df0, 
                                  df1,
                                  x = 'date',
                                  y = 'compound_mean', 
                                  y_smoothed = 's5000_compound',                              
                                  upper = 'compound_upper',
                                  lower = 'compound_lower',
                                  line_name = 'Centered sentiment score',
                                  line_smoothed_name = 'Smoothed',
                                  title = 'Sentiment',
                                  xaxis_range = [datetime.datetime(2020, 12, 15),
                                                datetime.datetime(2022, 1, 25)]) 

           else:
             fig = plot_sentiment(df0, 
                                  df1,
                                  x = 'date',
                                  y = 'compound_mean', 
                                  y_smoothed = 's500_compound',                              
                                  upper = 'compound_upper',
                                  lower = 'compound_lower',
                                  line_name = 'Centered sentiment score',
                                  line_smoothed_name = 'Smoothed',
                                  title = 'Sentiment',
                                  xaxis_range = [datetime.datetime(2020, 12, 15),
                                                datetime.datetime(2022, 1, 25)]) 


           st.plotly_chart(fig, use_container_width = True)

           st.subheader('Z polarity score')

           if menu == 'Omicron':
                        fig1 = plot_sentiment(df0, 
                                              df1,
                                              x = 'date',
                                              y = 'z_mean', 
                                              y_smoothed = 's500_polarity_score_z', 
                                              upper = 'z_upper',
                                              lower = 'z_lower',
                                              line_name = 'z(Polarity score)',
                                              line_smoothed_name = 'Smoothed',
                                              title = 'Sentiment',
                                              xaxis_range = [datetime.datetime(2021, 11, 1),
                                                             datetime.datetime(2022, 1, 31)]) 
           elif (menu == 'Vaccin') or (menu == 'Corona'):
             fig1 = plot_sentiment(df0, 
                                   df1,
                                   x = 'date',
                                   y = 'z_mean', 
                                   y_smoothed = 's5000_polarity_score_z', 
                                   upper = 'z_upper',
                                   lower = 'z_lower',
                                   line_name = 'z(Polarity score)',
                                   line_smoothed_name = 'Smoothed',
                                   title = 'Sentiment',
                                   xaxis_range = [datetime.datetime(2020, 12, 15),
                                                  datetime.datetime(2022, 1, 25)])

           else:
             fig1 = plot_sentiment(df0, 
                                   df1,
                                   x = 'date',
                                   y = 'z_mean', 
                                   y_smoothed = 's500_polarity_score_z', 
                                   upper = 'z_upper',
                                   lower = 'z_lower',
                                   line_name = 'z(Polarity score)',
                                   line_smoothed_name = 'Smoothed',
                                   title = 'Sentiment',
                                   xaxis_range = [datetime.datetime(2020, 12, 15),
                                                  datetime.datetime(2022, 1, 25)]) 



           st.plotly_chart(fig1, use_container_width = True)



    elif navigator == 'Hashtag Frequency':
         # st.title('Hashtags Frequency') 

          df = read_pkl(label = menu, 
                        path = 'data/',
                        data_prefix = '_hash.pkl') 

          df = df.sort_values('nr_of_hashtags', ascending = True)


          fig = plot_bar_freq(data = df,
                              x = 'nr_of_hashtags',
                              y = 'hashtag',
                              title = 'Most Frequent Hashtags', 
                              colourscale = palette0)

          st.plotly_chart(fig, use_container_width = True)


    elif navigator == 'Word Frequency':
       # st.title('Word Frequency')

        df = read_pkl(label = menu, 
                       path = 'data/',
                       data_prefix = '_w_freq.pkl') 

        df= df.sort_values('Frequency', ascending = True)

        fig = plot_bar_freq(data = df,
                            x = 'Frequency',
                            y_ = 'word',
                            title = 'Most Frequent Words', 
                            colourscale = palette1)

        st.plotly_chart(fig, use_container_width = True)
     

    elif navigator == 'Bigrams':
      #   st.title('Bigrams')

         df = read_pkl(label = menu, 
                        path = 'data/',
                        data_prefix = '_bigrams.pkl') 

         plot = plot_bigrams(data = df,
                             title = 'Bigrams')

         st.bokeh_chart(plot, use_container_width=True)

    elif navigator == 'WordCloud':
       #  st.title('WordCloud')

         label = menu.lower()
         filename = label + '_wordcloud.png'
         path = 'data/' +  label + '/' + filename
        

         st.image(path)
             


    else:
             raise ValueError('Invalid input data!')
                

if __name__  == '__main__':
    main()   