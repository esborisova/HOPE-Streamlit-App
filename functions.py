from _plotly_utils.utils import find_closest_string
import streamlit as st
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

    """Reads pickle file
    
    
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



def plot_tweet_freq(data: pd.DataFrame,
                    x: str,
                    y: str,
                    y_smoothed: str,
                    line_name: str,
                    line_smoothed_name: str, 
                    title: str, 
                    xaxis_range: list):

    """Plots the line graph showing tweet frequency per day 
    
    
    Args:
        data (pd.DataFrame): The pandas dataframe with data

        x (str): The dataframe column with x values (dates)

        y (str): The dataframe column with y values (number of tweets)

        y_smoothed (str): The dataframe column with smoothed y values 

        line_name (str): The name of the line showing number of tweets per day

        line_smoothed_name (str): The name of the line showing smoothed number of tweets values per day

        title: The title of the line plot
       
        xaxis_range: The range of X axis 
    
    
    Returns:
          fig: The line plot
    """

    fig = go.Figure(data = go.Scatter(x = data[x].astype(dtype=str), 
                                      y = data[y],
                                      name = line_name,
                                      line = dict(color = 'black', width = 4)))
        
    fig.add_trace(go.Scatter(x = data[x],
                             y = data[y_smoothed],
                             mode = 'lines',
                             line = dict(color = 'rgb(49,130,189)', width = 4.5),
                             name = line_smoothed_name))      

    fig.update_layout(height = 600, 
                      width = 700, 
                      title = title,
                      title_font_size = 50,
                      title_font = dict(color = 'black'),
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




def plot_sentiment(data: pd.DataFrame,  
                   data_smoothed: pd.DataFrame,
                   x: str, 
                   y: str, 
                   y_smoothed: str,  
                   upper: str, 
                   lower: str,
                   line_name: str, 
                   line_smoothed_name: str, 
                   title: str,
                   xaxis_range: str):

    """Plots the line graph showing sentiment per day 
    
    
    Args:
        data (pd.DataFrame): The pandas dataframe with sentiment compound/z scores 
        
        data_smothed (pd.DataFrame): The pandas dataframe with smoothed compound/z sentiment scores

        x (str): The dataframe column with x values (dates)

        y (str): The dataframe column with y values (sentiment compound/z scores)

        y_smoothed (str): The dataframe column with smoothed y values for the second line

        line_name (str): The name of the line showing the average of sentimnet per day

        line_smoothed_name (str): The name of the line showing smoothed sentiment scores per day 

        title: The title of the line plot
       
        xaxis_range: The range of X axis 
    
    
    Returns:
          fig: The line plot
    """
    
    fig = go.Figure()   

    fig.add_trace(go.Scatter(x = data_smoothed[x],
                             y = data_smoothed[y],
                             name = line_name,
                             line = dict(color = 'black', width = 4)))
                       
    fig.add_trace(go.Scatter(name='Upper Bound',
                             x = data_smoothed[x],
                             y = data_smoothed[upper],
                             mode = 'lines',
                             marker = dict(color="#444"),
                             fillcolor = 'rgba(68, 68, 68, 0.3)',
                             fill = 'tonexty',
                             line = dict(width=0),
                             showlegend = False))

    fig.add_trace(go.Scatter(name = 'Lower Bound',
                             x = data_smoothed[x],
                             y = data_smoothed[lower],
                             marker = dict(color="#444"),
                             line = dict(width=0),
                             mode = 'lines',
                             fillcolor = 'rgba(68, 68, 68, 0.3)',
                             fill = 'tonexty',
                             showlegend = False))

    fig.add_trace(go.Scatter(x = data[x],
                             y = data[y_smoothed],
                             mode = 'lines',
                             line = dict(color = 'orange', width = 4.5),
                             name = line_smoothed_name))
 
    fig.update_layout(height = 900, 
                      width = 800, 
                      title = title,
                      title_font_size = 50,
                      title_font = dict(color = 'black'),
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



def plot_bar_freq(data: pd.DataFrame,
                  x: str,
                  y: str,
                  title: str, 
                  colourscale: list):

    """Plots the bar chart showing frequency of words/hashtags across the dataset
    
    
    Args:
        data (pd.DataFrame): The pandas dataframe with data

        x (str): The dataframe column with x values (frequency)

        y (str): The dataframe column with y values (words/hashtags)

        title: The title of the bar chart
       
        colourscale: The colour palette for the bars
    
    
    Returns:
          fig: The bar chart
    """
    
    fig = go.FigureWidget(data=[go.Bar(x = data[x], 
                                       y = data[y], 
                                       orientation='h', 
                                       marker = dict(color = data[x], colorscale = colourscale))])

    fig.update_layout(title_text = title, 
                      title_font_size = 50,
                      title_font = dict(color='grey'),
                      title_x = 0.5, 
                      title_y = 0.98,
                      yaxis = dict(tickfont = dict(size=18), color = 'grey'),
                      xaxis = dict(tickfont = dict(size=20), color = 'grey'))

    return fig





def word_freq(data: pd.DataFrame) -> pd.DataFrame:
    """Calculates word frequency
    
    Args:
        data (pd.DataFrame): The pandas dataframe with data
  
    
    Returns:
          w_freq (pd.DataFrame): The dataframe with words and their frequency values
    """
    
    w_freq = data.tokens_string.str.split(expand = True).stack().value_counts()
    w_freq = w_freq.to_frame().reset_index().rename(columns={'index': 'word', 0: 'Frequency'})

    return w_freq


# Plot bigrams

def plot_bigrams(data: pd.DataFrame,
                 title: str ):


    from bokeh.plotting import figure

    # Create dictionary of bigrams and their counts
    d = data.set_index('bigram').T.to_dict('records')
    k = 4

    # Create network plot 
    G = nx.Graph()

    # Create connections between nodes
    for key, value in d[0].items():
        G.add_edge(key[0], key[1], weight=(value * 5))

    pos = nx.spring_layout(G, k=k)

    # Nodes
    d = data.to_dict(orient = 'split')['data']
    d = [(int(word[1]))*2 for node in G.nodes() for word in d if word[0] == node]

    # Calculate degree for each node and add as node attribute
    degrees = dict(nx.degree(G))
    nx.set_node_attributes(G, name = 'degree', values = degrees)

    #Slightly adjust degree so that the nodes with very small degrees are still visible
    number_to_adjust_by = 10
    adjusted_node_size = dict([(node, degree+number_to_adjust_by) for node, degree in nx.degree(G)])
    nx.set_node_attributes(G, name = 'adjusted_node_size', values = adjusted_node_size)

      
    #Choose colors for node and edge highlighting
    node_highlight_color = Spectral4[1]
    edge_highlight_color = Spectral4[2]

    #Choose attributes from G network to size and color by — setting manual size (e.g. 10) or color (e.g. 'skyblue') 
    
    size_by_this_attribute = 'adjusted_node_size'
    color_by_this_attribute = 'adjusted_node_size'

    #Pick a color palette — Blues8, Reds8, Purples8, Oranges8, Viridis8
    color_palette = Blues8
      
    #Create a plot — set dimensions, toolbar, and title
    plot = figure(tools = 'pan,wheel_zoom,save,reset', 
                  active_scroll ='wheel_zoom',
                  title = title)

    plot.title.text_font_size = '20px'

    plot.add_tools(HoverTool(tooltips=None), TapTool(), BoxSelectTool())

    #Create a network graph object
    network_graph = from_networkx(G, pos, scale = 10, center = (0, 0))

    #Set node sizes and colors according to node degree (color as spectrum of color palette)
    minimum_value_color = min(network_graph.node_renderer.data_source.data[color_by_this_attribute])
    maximum_value_color = max(network_graph.node_renderer.data_source.data[color_by_this_attribute])

    #Set node sizes and colors according to node degree (color as category from attribute)

    network_graph.node_renderer.glyph = Circle(size = size_by_this_attribute, 
                                               fill_color = linear_cmap(color_by_this_attribute, 
                                                                        color_palette, 
                                                                        minimum_value_color, 
                                                                        maximum_value_color))  

    #Set node highlight colors
    network_graph.node_renderer.hover_glyph = Circle(size = size_by_this_attribute, 
                                                     fill_color = node_highlight_color,
                                                     line_width = 3)

    network_graph.node_renderer.selection_glyph = Circle(size = size_by_this_attribute, 
                                                         fill_color = node_highlight_color, 
                                                         line_width = 3)

    #Set edge opacity and width
    network_graph.edge_renderer.glyph = MultiLine(line_alpha = 0.5, 
                                                  line_width = 3)

    #Set edge highlight colors
    network_graph.edge_renderer.selection_glyph = MultiLine(line_color = edge_highlight_color, line_width = 3)
    network_graph.edge_renderer.hover_glyph = MultiLine(line_color = edge_highlight_color, line_width = 3)
    
    #Highlight nodes and edges
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
                      text_font_size = '22px', 
                      background_fill_alpha = .3)

    plot.renderers.append(labels)

    return plot
