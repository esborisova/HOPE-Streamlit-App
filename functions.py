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



def read_file(label: str,
              folder: str,
              data_prefix: str):

    label = label.lower()
    df = pd.read_pickle(label + folder + label + data_prefix)  

    return df

# Plot tweet frequency over time
def plot_tweet_freq(data: pd.DataFrame,
                    x_column: str,
                    y1_column: str,
                    y2_column: str,
                    line1_name: str,
                    line2_name: str,
                    title: str):

  fig = go.Figure(data = go.Scatter(x = data[x_column].astype(dtype=str), 
                                    y = data[y1_column],
                                    name = line1_name,
                                    line = dict(color = 'black', width = 3)))
        
  fig.add_trace(go.Scatter(x = data[x_column],
                           y = data[y2_column],
                           mode = 'lines',
                           line = go.scatter.Line(color = 'rgb(49,130,189)', width = 3),
                           name = line2_name))      

  fig.update_layout(height = 500, 
                    width = 600, 
                    title = title,
                    title_font_size = 50,
                    title_font = dict(color = 'black'),
                    title_x = 0.5, 
                    title_y = 0.95,
                    showlegend = True,
                    legend = dict(x = 0.99,
                                  yanchor = 'top',
                                  xanchor = 'right',
                                  font = dict(size = 22)),
                    yaxis = dict(tickfont = dict(size = 25), color = 'grey'),
                    xaxis = dict(tickfont = dict(size = 25), color = 'grey'),
                    xaxis_range = [datetime.datetime(2021, 1, 1),
                                   datetime.datetime(2021, 12, 31)])
     
  fig.update_xaxes(tickformat = '%b %d')

  return fig



# plot hashtags frequency
def plot_bar_freq(data: pd.DataFrame,
                  x_column: str,
                  y_column: str,
                  title_text: str,
                  colorscale: str):
    
    fig = go.FigureWidget(data=[go.Bar(x = data[x_column], 
                                       y = data[y_column], 
                                       orientation='h', 
                                       marker={'color': data[x_column], 'colorscale': colorscale})]) 

    fig.update_layout(title_text = title_text, 
                      title_font_size = 50,
                      title_font = dict(color='grey'),
                      title_x = 0.5, 
                      title_y = 0.95,
                      yaxis = dict(tickfont = dict(size=18), color = 'grey'),
                      xaxis = dict(tickfont = dict(size=20), color = 'grey'))

    return fig




# Calculate word frequency
def word_freq(data: pd.DataFrame) -> pd.DataFrame:
    
    w_freq = data.tokens_string.str.split(expand = True).stack().value_counts()
    w_freq = w_freq.to_frame().reset_index().rename(columns={'index': 'word', 0: 'Frequency'})

    return w_freq


# Plot bigrams

def plot_bigrams(data: pd.DataFrame,
                 w_freq: pd.DataFrame,
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
    d = w_freq.to_dict(orient = 'split')['data']
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
