from _plotly_utils.utils import find_closest_string
import streamlit as st
import math
import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
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




def read_pkl(label: str,
             path: str,
             data_prefix: str) -> pd.DataFrame:

    """Reads a pickle file
    
    
    Args:
        label (str): The name of the selectbox (f.ex., Restriktion, Omicron, etc.)

        path (str): The path to the folder with files 

        data_prefix (str): The file name prefix (f.ex., restriktion_hash.pkl: Label — restriktion, prefix — _hash.pkl)
    
    
    Returns:
          pd.DataFrame: The pandas dataframe with data
    """

    label = label.lower()

    filename =  label + data_prefix
    new_path = path + label + '/' + filename

    df = pd.read_pickle(new_path)  

    return df



def plot_line(x: str,
              y: str,
              y2: str,
              line_name: str,
              line2_name: str, 
              title: str, 
              xaxis_range: list):

    """Plots the line graph with two lines. It depics the distribution of two Y values across the same X values.
    
    
    Args:
        x (str): X values 

        y (str): The first Y values 

        y2 (str): The second Y values 

        line_name (str): The name of the line showing the distribution of the first Y values

        line2_name (str): The name of the line showing the distribution of the second Y values

        title: The title of the line plot
       
        xaxis_range: The range of X axis 
    
    
    Returns:
          fig: The line plot
    """

    fig = go.Figure(data = go.Scatter(x = x.astype(dtype=str), 
                                      y = y,
                                      name = line_name,
                                      mode = 'lines',
                                      line = dict(color = 'black', width = 5)))
        
    fig.add_trace(go.Scatter(x = x.astype(dtype=str),
                             y = y2,
                             mode = 'lines',
                             line = dict(color = 'rgb(49,130,189)', width = 5),
                             name = line2_name))      

    fig.update_layout(height = 600, 
                      width = 700, 
                      title = title,
                      title_font_size = 50,
                      title_font = dict(color = 'grey'),
                      title_x = 0.5, 
                      title_y = 0.98,
                      showlegend = True,
                      legend = dict(x = 0.99,
                                   yanchor = 'top',
                                   xanchor = 'right',
                                   font = dict(size = 22)),
                      yaxis = dict(tickfont = dict(size = 25), color = 'grey'),
                      xaxis = dict(tickfont = dict(size = 25), color = 'grey'),
                      xaxis_range = xaxis_range)
     
    fig.update_xaxes(tickformat = '%b %d')

    return fig



def plot_mean(x: str, 
              y: str, 
              y2: str,  
              upper_bound: str, 
              lower_bound: str,
              line_name: str, 
              line2_name: str, 
              title: str,
              xaxis_range: str):

    """Plots the line graph with two lines. It depics the distribution of two mean Y values across the same X values. It also shows the confidence interval. 
    
    
    Args:
        x (str): X values 

        y (str): The first mean Y values 

        y2 (str): The second mean Y values

        upper_bound (str): The values for the upper bound of the confidence interval

        lower_bound (str): The values for the lower bound of the confidence interval

        line_name (str): The name of the line showing the distribution of the first Y values

        line2_name (str): The name of the line showing the distribution of the second Y values

        title: The title of the line plot
       
        xaxis_range: The range of X axis 
    
    
    Returns:
          fig: The line plot
    """
    
    fig = go.Figure()   

    fig.add_trace(go.Scatter(x = x,
                             y = y,
                             name = line_name,
                             line = dict(color = 'black', width = 4)))
                       
    fig.add_trace(go.Scatter(name = 'Upper Bound',
                             x = x,
                             y = upper_bound,
                             mode = 'lines',
                             marker = dict(color="#444"),
                             fillcolor = 'rgba(68, 68, 68, 0.3)',
                             fill = 'tonexty',
                             line = dict(width = 0),
                             showlegend = False))

    fig.add_trace(go.Scatter(name = 'Lower Bound',
                             x = x,
                             y = lower_bound,
                             marker = dict(color="#444"),
                             line = dict(width=0),
                             mode = 'lines',
                             fillcolor = 'rgba(68, 68, 68, 0.3)',
                             fill = 'tonexty',
                             showlegend = False))

    fig.add_trace(go.Scatter(x = x,
                             y = y2,
                             mode = 'lines',
                             line = dict(color = 'orange', width = 4.5),
                             name = line2_name))
 
    fig.update_layout(height = 900, 
                      width = 800, 
                      title = title,
                      title_font_size = 50,
                      title_font = dict(color = 'grey'),
                      title_x = 0.5, 
                      title_y = 0.98,
                      showlegend = True,
                      hovermode = "x",
                      legend = dict(x = 0.99,
                                    yanchor = 'top',
                                    xanchor = 'right',
                                    font = dict(size = 20)),
                      yaxis = dict(tickfont = dict(size = 20), color = 'grey'),
                      xaxis = dict(tickfont = dict(size = 25), color = 'grey'),
                      xaxis_range = xaxis_range)
           

    fig.update_yaxes(autorange=False, range = [-2.00, 2.00], dtick = 0.25)

      
    fig.add_hrect(y0 = 0, y1 = -2.0, line_width = 0, fillcolor = 'red', opacity = 0.05)
    fig.add_hrect(y0 = 0, y1 = 2.0, line_width = 0, fillcolor = 'green', opacity = 0.05)

    return fig



def plot_bar(x: str,
             y: str,
             title: str, 
             colourscale: list):

    """Plots the bar chart 
    
    
    Args:
        x (str): The X values 

        y (str): The Y values 

        title: The title of the bar chart
       
        colourscale: The colour palette for the bars
    
    
    Returns:
          fig: The bar chart
    """
    
    fig = go.FigureWidget(data=[go.Bar(x = x, 
                                       y = y, 
                                       orientation='h', 
                                       marker = dict(color = x, colorscale = colourscale))])

    fig.update_layout(title_text = title, 
                      title_font_size = 50,
                      title_font = dict(color='grey'),
                      title_x = 0.5, 
                      title_y = 0.98,
                      yaxis = dict(tickfont = dict(size=18), color = 'grey'),
                      xaxis = dict(tickfont = dict(size=20), color = 'grey'))

    return fig



def bigram_freq(freq:dict,
                G,
                scale: bool) -> dict:
    """Creates a dictionary of words (present in bigrams) and their frequencies
    
    Args:
        freq (dict): A dictionary with all words from the data corpus and their frequency values  

        G: A Networkx graph

        scale: A boolean value. If true, the logarithm of frequency values will be calculated  
    
    Returns:
          freq_dict (dict): The dictionary containg words from bigrams and their frequencies
    """

    freq_dict = {}
    
    for node in G.nodes():
        for word in freq:
                 if word[0] == node:
                     if scale == True:
                         freq_dict[word[0]] = (math.log2(word[1]))*3 
                     else:
                         freq_dict[word[0]] = word[1]
    return freq_dict



def scale(data: dict) -> dict:
    """Calculates logarithm base 2 and multiplies the result by 3
    
    Args:
        data (dict): A ditctionary with values

    Returns:
          data (dict): The input dictonary with updated values
    
    """

    for key, value in data.items():
        data[key] = (math.log2(value))*3

    return data



def plot_bigrams(G,
                 word_freq: dict,
                 co_occurence: dict,
                 pos, 
                 palette_nodes: list,
                 palette_edges: list,  
                 title: str):

    """Plots node graph
    
    Args:
        G: A Networkx graph

        word_freq (dict):  A dictionary of bigram members and their frequencies across the dataset

        co_occurence (dict): A dictionaty of bigrams and their co-occurence values

        pos (dict): A dictionary of positions keyed by node  
        
        palette_nodes (list): A colour palette for the nodes 
        
        palette_edges (list):  A colour palette for the edges
                 
        title (str): The title of the node graph

    Returns:
          plot: The node graph
    
    """    

    from bokeh.plotting import figure
    

    nx.set_node_attributes(G, name = 'freq', values = word_freq)
    nx.set_edge_attributes(G, name = 'co_occurence', values = co_occurence)

    node_highlight_color = Spectral4[1]
    edge_highlight_color = Spectral4[2]   

    color_nodes = 'freq'
    color_edges = 'co_occurence'


    plot = figure(tools = 'pan,wheel_zoom,save,reset', 
                  active_scroll ='wheel_zoom',
                  title = title)

    plot.title.text_font_size = '20px'

    plot.add_tools(HoverTool(tooltips = None), TapTool(), BoxSelectTool())
   
    network_graph = from_networkx(G, pos, scale = 10, center = (0, 0))


    min_col_val_node = min(network_graph.node_renderer.data_source.data[color_nodes])
    max_col_val_node = max(network_graph.node_renderer.data_source.data[color_nodes])


    min_col_val_edge = min(network_graph.edge_renderer.data_source.data[color_edges])
    max_col_val_edge = max(network_graph.edge_renderer.data_source.data[color_edges])
    
    network_graph.node_renderer.glyph = Circle(size = 40, 
                                               fill_color =  linear_cmap(color_nodes, 
                                                                         palette_nodes, 
                                                                         min_col_val_node, 
                                                                         max_col_val_node),
                                               fill_alpha = 1)  

    network_graph.node_renderer.hover_glyph = Circle(size = 5, 
                                                     fill_color = node_highlight_color,
                                                     line_width = 3)

    network_graph.node_renderer.selection_glyph = Circle(size = 5, 
                                                         fill_color = node_highlight_color, 
                                                         line_width = 5)

    network_graph.edge_renderer.glyph = MultiLine(line_alpha = 1, 
                                                  line_color = linear_cmap(color_edges, 
                                                                           palette_edges, 
                                                                           min_col_val_edge, 
                                                                           max_col_val_edge),
                                                  line_width = 4)


    network_graph.edge_renderer.selection_glyph = MultiLine(line_color = edge_highlight_color, line_width = 4)
    network_graph.edge_renderer.hover_glyph = MultiLine(line_color = edge_highlight_color, line_width = 4)


    network_graph.selection_policy = NodesAndLinkedEdges()
    network_graph.inspection_policy = NodesAndLinkedEdges()
       
    plot.renderers.append(network_graph)
    
    x, y = zip(*network_graph.layout_provider.graph_layout.values())
    node_labels = list(G.nodes())
    source = ColumnDataSource({'x': x, 'y': y, 'name': [node_labels[i] for i in range(len(x))]})
    labels = LabelSet(x = 'x', 
                      y ='y', 
                      text = 'name', 
                      source = source, 
                      background_fill_color = 'pink', 
                      text_font_size = '24px', 
                      background_fill_alpha = .3)

    plot.renderers.append(labels)

    return plot


def smoothing(label: str) -> int:
    """Defines the smoothing value depending on the dataset

    Args:
       label (str): The name of the dataset (from the menu)

    Returns: 
          smoothing_value (int): The smoothing value 
    """

    smoothing_value = 0

    if (label == 'Vaccin') or (label == 'Corona'):
        smoothing_value = 5000
    else:
        smoothing_value = 500
        
    return smoothing_value



def set_lab_freq(label: str):
    """Defines the name of the dataframe column with smoothed tweet frequency values and the (date) range of X axis

    Args:
       label (str): The name of the dataset (from the menu)
    
    Returns:
          y2_name (str): The name of the dataframe column with smoothed Y values

          xaxis_range: The range of X axis  
    """

    y2_name = None
    xaxis_range = None

    if label == 'Omicron':
        y2_name = 's500_nr_of_tweets'
        xaxis_range = [datetime.datetime(2021, 11, 1),
                       datetime.datetime(2022, 1, 31)]

    elif (label == 'Vaccin') or (label == 'Corona'):
        y2_name = 's5000_nr_of_tweets'
        xaxis_range = [datetime.datetime(2020, 11, 1),
                       datetime.datetime(2022, 1, 31)]
        
    else:
        y2_name = 's500_nr_of_tweets'
        xaxis_range = [datetime.datetime(2020, 11, 1),
                       datetime.datetime(2022, 1, 31)]

    return y2_name, xaxis_range



def set_lab_vader(label):
    """Defines the name of the dataframe column with smoothed vader sentiment scores and the (date) range of X axis

    Args:
       label (str): The name of the dataset (from the menu)
    
    Returns:
          y2_name (str): The name of the dataframe column with smoothed Y values

          xaxis_range: The range of X axis  
    """

    y2_name = None
    xaxis_range = None

    if label == 'Omicron':
        y2_name = 's500_compound'
        xaxis_range = [datetime.datetime(2021, 11, 1),
                       datetime.datetime(2022, 1, 31)]

    elif (label == 'Vaccin') or (label == 'Corona'):
        y2_name = 's5000_compound'
        xaxis_range = [datetime.datetime(2020, 12, 15),
                       datetime.datetime(2022, 1, 25)]
        
    else:
        y2_name = 's500_compound'
        xaxis_range = [datetime.datetime(2020, 12, 15),
                       datetime.datetime(2022, 1, 25)]
    return y2_name, xaxis_range



def set_lab_bert(label):
    """Defines the name of the dataframe column with smoothed bert sentiment scores and the (date) range of X axis

    Args:
       label (str): The name of the dataset (from the menu)
    
    Returns:
          y2_name (str): The name of the dataframe column with smoothed Y values

          xaxis_range: The range of X axis  
    """

    y2_name = None
    xaxis_range = None

    if label == 'Omicron':
        y2_name = 's500_polarity_score_z'
        xaxis_range = [datetime.datetime(2021, 11, 1),
                       datetime.datetime(2022, 1, 31)]

    elif (label == 'Vaccin') or (label == 'Corona'):
        y2_name = 's5000_polarity_score_z'
        xaxis_range = [datetime.datetime(2020, 12, 15),
                       datetime.datetime(2022, 1, 25)]
        
    else:
        y2_name = 's500_polarity_score_z'
        xaxis_range = [datetime.datetime(2020, 12, 15),
                       datetime.datetime(2022, 1, 25)]

    return y2_name, xaxis_range

