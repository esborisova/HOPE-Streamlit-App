from _plotly_utils.utils import find_closest_string
import streamlit as st
import math
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import datetime
import networkx as nx
from icecream import ic
from nltk.util import bigrams
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

# Inferno
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


######################################
# Define palettes for networks graph #
######################################

# Greens
color_palette_nodes = ['#E0FFFF',  '#bcdfeb', '#62b4cf', '#1E90FF']
color_palette_edges = ['#8fbc8f', '#3cb371', '#2e8b57', '#006400']


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
        df = read_pkl(label = menu, 
                      path = 'data/',
                      data_prefix = '_tweet_freq.pkl')  

        if menu == 'Omicron':
          fig = plot_line(x = df['date'],
                          y = df['nr_of_tweets'],
                          y2 = df['s500_nr_of_tweets'],
                          line_name = 'Number of tweets',
                          line2_name = 'Smoothed values',
                          title = 'Frequency of Mentions', 
                          xaxis_range = [datetime.datetime(2021, 11, 1),
                                         datetime.datetime(2022, 1, 31)])

        elif (menu == 'Vaccin') or (menu == 'Corona'):
          fig = plot_line(x = df['date'],
                          y = df['nr_of_tweets'],
                          y2 = df['s5000_nr_of_tweets'],
                          line_name = 'Number of tweets',
                          line2_name = 'Smoothed values',
                          title = 'Frequency of Mentions', 
                          xaxis_range = [datetime.datetime(2020, 11, 1),
                                         datetime.datetime(2022, 1, 31)])

        else:  
          fig = plot_line(x = df['date'],
                          y = df['nr_of_tweets'],
                          y2 = df['s500_nr_of_tweets'],
                          line_name = 'Number of tweets',
                          line2_name = 'Smoothed values',
                          title = 'Frequency of Mentions', 
                          xaxis_range = [datetime.datetime(2020, 11, 1),
                                         datetime.datetime(2022, 1, 31)])
                                  
        st.plotly_chart(fig, use_container_width = True)


    elif navigator == 'Sentiment':
           st.subheader('Centered sentiment score')

           df = read_pkl(label = menu, 
                          path = 'data/',
                          data_prefix = '_sentiment.pkl') 
        
           if menu == 'Omicron':
             fig = plot_mean(x = df['date'],
                             y = df['compound_mean'], 
                             y2 = df['s500_compound'],                              
                             upper_bound = df['compound_upper'],
                             lower_bound = df['compound_lower'],
                             line_name = 'Centered sentiment score',
                             line2_name = 'Smoothed',
                             title = 'Sentiment',
                             xaxis_range = [datetime.datetime(2021, 11, 1),
                                            datetime.datetime(2022, 1, 31)]) 

           elif (menu == 'Vaccin') or (menu == 'Corona'):
             fig = plot_mean(x = df['date'],
                             y = df['compound_mean'], 
                             y2 = df['s5000_compound'],                              
                             upper_bound = df['compound_upper'],
                             lower_bound = df['compound_lower'],
                             line_name = 'Centered sentiment score',
                             line2_name = 'Smoothed',
                             title = 'Sentiment',
                             xaxis_range = [datetime.datetime(2020, 12, 15),
                                            datetime.datetime(2022, 1, 25)]) 

           else:
             fig = plot_mean(x = df['date'],
                             y = df['compound_mean'], 
                             y2 = df['s500_compound'],                              
                             upper_bound = df['compound_upper'],
                             lower_bound = df['compound_lower'],
                             line_name = 'Centered sentiment score',
                             line2_name = 'Smoothed',
                             title = 'Sentiment',
                             xaxis_range = [datetime.datetime(2020, 12, 15),
                                            datetime.datetime(2022, 1, 25)]) 


           st.plotly_chart(fig, use_container_width = True)

           st.subheader('Z polarity score')
           if menu == 'Omicron':
                        fig1 = plot_mean(x = df['date'],
                                         y = df['z_mean'], 
                                         y2 = df['s500_polarity_score_z'], 
                                         upper_bound = df['z_upper'],
                                         lower_bound = df['z_lower'],
                                         line_name = 'z(Polarity score)',
                                         line2_name = 'Smoothed',
                                         title = 'Sentiment',
                                         xaxis_range = [datetime.datetime(2021, 11, 1),
                                                        datetime.datetime(2022, 1, 31)]) 
           elif (menu == 'Vaccin') or (menu == 'Corona'):
             fig1 = plot_mean(x = df['date'],
                              y = df['z_mean'], 
                              y2 = df['s5000_polarity_score_z'], 
                              upper_bound = df['z_upper'],
                              lower_bound = df['z_lower'],
                              line_name = 'z(Polarity score)',
                              line2_name = 'Smoothed',
                              title = 'Sentiment',
                              xaxis_range = [datetime.datetime(2020, 12, 15),
                                             datetime.datetime(2022, 1, 25)])

           else:
             fig1 = plot_mean(x = df['date'],
                              y = df['z_mean'], 
                              y2 = df['s500_polarity_score_z'], 
                              upper_bound = df['z_upper'],
                              lower_bound = df['z_lower'],
                              line_name = 'z(Polarity score)',
                              line2_name = 'Smoothed',
                              title = 'Sentiment',
                              xaxis_range = [datetime.datetime(2020, 12, 15),
                                             datetime.datetime(2022, 1, 25)]) 

           st.plotly_chart(fig1, use_container_width = True)

    elif navigator == 'Hashtag Frequency':
          df = read_pkl(label = menu, 
                        path = 'data/',
                        data_prefix = '_hash.pkl') 

          df = df.sort_values('nr_of_hashtags', ascending = True)

          fig = plot_bar(x = df['nr_of_hashtags'],
                         y = df['hashtag'],
                         title = 'Most Frequent Hashtags', 
                         colourscale = palette0)

          st.plotly_chart(fig, use_container_width = True)


    elif navigator == 'Word Frequency':
        df = read_pkl(label = menu, 
                       path = 'data/',
                       data_prefix = '_w_freq.pkl') 

        df0 = df.nlargest(30, columns=['Frequency'])
        df0 = df0.sort_values('Frequency', ascending = True)

        fig = plot_bar(x = df0['Frequency'],
                       y = df0['Word'],
                       title = 'Most Frequent Words', 
                       colourscale = palette1)

        st.plotly_chart(fig, use_container_width = True)
     

    elif navigator == 'Bigrams':

         df = read_pkl(label = menu, 
                       path = 'data/',
                       data_prefix = '_bigrams.pkl') 

         w_freq = read_pkl(label = menu, 
                           path = 'data/',                        
                           data_prefix = '_w_freq.pkl') 

         value = st.slider('Select the number of bigrams', min_value = 1, max_value = 30,  value = 30, step = 1)
         
         # create a dict of words and their frequencies
         freq_dict = w_freq.to_dict(orient = 'split')['data']  

         # create a dict of bigrams and their co-occurence values      
         co_occurence = dict(df.values)                       
                            
         G = nx.Graph()

         for key, value in co_occurence.items():    
            G.add_edge(key[0], key[1], weight=(value * 5))

         pos = nx.spring_layout(G, k = 4)

         # scale co_occurence values 
         scaled_co_occurence = scale(co_occurence)              
        
         # create a dict of words (from bigrams) and their frequencies across the dataset
         freq = bigram_freq(freq_dict, G, scale = True)        

         fig = plot_bigrams(G, freq, scaled_co_occurence, pos, color_palette_nodes, color_palette_edges, 'Bigrams')

         st.bokeh_chart(fig, use_container_width = True)


    elif navigator == 'WordCloud':
         label = menu.lower()
         filename = label + '_wordcloud.png'
         path = 'data/' +  label + '/' + filename
      
         st.image(path)
             

    else:
             raise ValueError('Invalid input data!')
                

if __name__  == '__main__':
    main()   