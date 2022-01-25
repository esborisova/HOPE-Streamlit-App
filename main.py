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


############################################################################
# Condense the layout (fit on the screen to reduce the amount of scrolling)#
############################################################################

padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)




pallette0 =[[0, "#F0E442"],    #yellow
            [0.01, "#d62728"], #brick red
            [0.03, "#800080"], #purple
            [1.0, "#000000"]]  #black 
 
# Blues 
pallette1 =[[0, "#f8fcfd"],   
            [0.01, "#e9f5f8"], 
            [0.02, "#cbe6ef"], 
            [0.04, "#bcdfeb"],
            [0.07, "#62b4cf"],
            [1.0, "#00008b"]]  



def main():

    menu = st.sidebar.selectbox('MENU', ['Restriktion', 'Genåb'])
    navigator = st.sidebar.radio('NAVIGATE TO', ['Tweet Frequency', 
                                                 'Sentiment', 
                                                 'Hashtag Frequency', 
                                                 'Word Frequency', 
                                                 'Bigrams', 
                                                 'WordCloud'])
    

    if navigator == 'Tweet Frequency':
       # st.title('Tweet Frequency Over Time') 

        df = read_file(label = menu, 
                       folder = '_streamlit/',
                       data_prefix = '.pkl') 
        

        fig = plot_tweet_freq(data = df,
                              x_column = 'date',
                              y1_column = 'nr_of_tweets',
                              y2_column = 's500_nr_of_tweets',
                              line1_name = 'Number of tweets',
                              line2_name = 'Smoothed values',
                              title = 'Frequency of Mentions')
                                  
        st.plotly_chart(fig, use_container_width = True)

    elif navigator == 'Sentiment':
        #   st.title('Sentiment Over Time')
           st.subheader('Centered sentiment score')
           df0 = read_file(label = menu, 
                         folder = '_streamlit/',
                         data_prefix = '.pkl') 

           df1 = read_file(label = menu, 
                         folder = '_streamlit/',
                         data_prefix = '_sentiment.pkl') 
        
           
           fig = plot_sentiment(df0, 
                                df1,
                                x_column = 'date',
                                y_column = 's200_compound', 
                                x1_column = 'date',
                                mean = 'compound_mean',
                                upper = 'compound_upper',
                                lower = 'compound_lower',
                                line_name = 'Centered sentiment score',
                                line1_name = 'Smoothed',
                                title = 'Sentiment') 


           st.plotly_chart(fig, use_container_width = True)

           st.subheader('Z polarity score')
           fig1 = plot_sentiment(df0, 
                                 df1,
                                 x_column = 'date',
                                 y_column = 'polarity_score_z_smoothed', 
                                 x1_column = 'date',
                                 mean = 'z_mean',
                                 upper = 'z_upper',
                                 lower = 'z_lower',
                                 line_name = 'z(Polarity score)',
                                 line1_name = 'Smoothed',
                                 title = 'Sentiment') 


           st.plotly_chart(fig1, use_container_width = True)



    elif navigator == 'Hashtag Frequency':
         # st.title('Hashtags Frequency') 

          df = read_file(label = menu, 
                         folder = '_streamlit/',
                         data_prefix = '_hash.pkl') 

          df = df.sort_values('nr_of_hashtags', ascending = True)


          fig = plot_bar_freq(data = df,
                               x_column = 'nr_of_hashtags',
                               y_column = 'hashtag',
                               title_text = 'Most Frequent Hashtags', 
                               colourscale = pallette0)

          st.plotly_chart(fig, use_container_width = True)


    elif navigator == 'Word Frequency':
       # st.title('Word Frequency')

        df = read_file(label = menu, 
                         folder = '_streamlit/',
                         data_prefix = '_w_freq.pkl') 

        df= df.sort_values('Frequency', ascending = True)

        fig = plot_bar_freq(data = df,
                            x_column = 'Frequency',
                            y_column = 'word',
                            title_text = 'Most Frequent Words', 
                            colourscale = pallette1)

        st.plotly_chart(fig, use_container_width = True)
     

    elif navigator == 'Bigrams':
      #   st.title('Bigrams')

         df1 = read_file(label = menu, 
                         folder = '_streamlit/',
                         data_prefix = '.pkl') 

         w_freq = word_freq(df1)

         df2 = read_file(label = menu, 
                         folder = '_streamlit/',
                         data_prefix = '_bigrams.pkl') 


         plot = plot_bigrams(data = df2 ,
                             w_freq = w_freq,
                             title = 'Bigrams')

         st.bokeh_chart(plot, use_container_width=True)

    elif navigator == 'WordCloud':
       #  st.title('WordCloud')

         name = menu.lower()
         path = name + '_streamlit/' + name + '_wordcloud.png'

         st.image(path)
             


    else:
             raise ValueError('Invalid input data!')
                

if __name__  == '__main__':
    main()   