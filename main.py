import pandas as pd
import numpy as np

import geopandas as gpd
import os

# os methods for manipulating paths
from os.path import dirname, join

from datetime import date, datetime
import shapely.wkt
# Bokeh Libraries
from bokeh.io import show
from bokeh.plotting import figure
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar
from bokeh.models import ColumnDataSource, CategoricalColorMapper, NumeralTickFormatter
from bokeh.models import Patches                                                ### *** IMPORTANT *** ###
from bokeh.palettes import brewer
from bokeh.transform import transform
#from bokeh.models.Tools import LassoSelectTool, WheelZoomTool

from bokeh.models import BasicTicker,LinearColorMapper, PrintfTickFormatter
# Bokeh basics 
from bokeh.io import curdoc
from bokeh.models.widgets import Tabs
from bokeh.models import DateSlider, Slider, HoverTool, WheelZoomTool,BoxZoomTool, ResetTool, PanTool, DateRangeSlider
from bokeh.models import Select, TapTool, CustomJS, CDSView, GroupFilter        ### *** IMPORTANT *** ###
from bokeh.layouts import widgetbox, row, column, gridplot

from bokeh.models.selections import Selection                                   ### *** IMPORTANT *** ###

from bokeh.palettes import Category20_16                                        ## Not necessary, because each state has been assigned with unique color

from bokeh.models.widgets import CheckboxGroup                                  ## Not necessary, because multiple states can be selected on US map

# ==================== inport data and USA map ==================== #

us_states_covid19_daily = pd.read_csv(join(dirname(__file__), 'data', 'us_states_covid19_daily.csv'), parse_dates=["date"])#['date']

gdf_usa_state_map = gpd.read_file(join(dirname(__file__), 'data', 'cb_2018_us_state_20m'))[['STUSPS', 'NAME', 'geometry']]
#Rename columns
gdf_usa_state_map.columns = ['state', 'stateName', 'geometry']

# Get rid of islands and Alaska
gdf_usa_state_map = gdf_usa_state_map[~gdf_usa_state_map.stateName.isin(['Hawaii', 'Alaska', 'Guam', 'Virgin Islands', 'American Samoa', 'Puerto Rico', 'Northern Mariana Islands'])]
# print(gdf_usa_state_map)

# import airport traffic volume data
airport_traffic = pd.read_csv(join(os.getcwd(), 'data', 'airport_traffic_US.csv'))
#airport_traffic = pd.read_csv(r"data\airport_traffic_US.csv")

# ========================================================= #

df_us_states_covid19_daily = (us_states_covid19_daily.sort_values(['state', 'date']))


us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

#Preprocess Airport Data --------------------------------------------------------------
airport_traffic['Date'] =airport_traffic['Date'].astype(str)
airport_traffic['state'] = airport_traffic['State'].map(us_state_abbrev)
airport_traffic = airport_traffic.rename(columns = {'State' : 'stateName'}, inplace=False)
airport_traffic = airport_traffic.rename(columns = {'Centroid' : 'geometry'}, inplace=False)
airport_traffic = airport_traffic[airport_traffic.stateName != "Hawaii" ]
columns=["state", "stateName", "geometry","Date","AirportName", "PercentOfBaseline"]
airport_traffic = airport_traffic[columns]
airport_traffic["geometry"] = airport_traffic["geometry"].map(shapely.wkt.loads)
airport_traffic["DateTime"] = airport_traffic["Date"].map(lambda x: datetime.strptime(x, '%Y-%m-%d'))
def get_x(point):
    #point is like 'POINT(151.180087713813 -33.9459774986125)'
    return point.x

def get_y(point):
    #point is like 'POINT(151.180087713813 -33.9459774986125)'
    return point.y

airport_traffic["x"] = airport_traffic["geometry"].map(get_x)
airport_traffic["y"] = airport_traffic["geometry"].map(get_y)
airport_traffic = airport_traffic[["Date","DateTime","x","y","stateName","AirportName","PercentOfBaseline"]]
airport_traffic = airport_traffic.sort_values(by=['AirportName'])

airport_unique = list(set(airport_traffic["AirportName"]))
airport_unique.sort()

df_stateName_airportName = airport_traffic.drop(columns=["Date","DateTime","x","y","PercentOfBaseline"])
stateName=[]
airportName=[]
for i, airport in enumerate(airport_unique):
    indexNoAirport = airport_traffic.loc[airport_traffic['AirportName']==airport].index[0]
    airportName.append(airport)
    stateName.append(df_stateName_airportName._get_value(indexNoAirport, 'stateName'))
df_stateName_airportName_unique = pd.DataFrame({'AirportName': airportName, 'StateName': stateName})
df_stateName_airportName_unique = df_stateName_airportName_unique.sort_values(by=['AirportName'])
df_stateName_airportName_unique = df_stateName_airportName_unique.reset_index()
df_stateName_airportName_unique = df_stateName_airportName_unique.drop(columns=['index'])
#print(df_stateName_airportName_unique.head(20))

unique_airports = list(set(airport_traffic["AirportName"].values))

def update_airport_data(selected_date,selected_states):
    traffic_selected_date = airport_traffic.loc[airport_traffic["Date"] == selected_date]
    traffic_selected_date["CircleSize"] = np.round(traffic_selected_date["PercentOfBaseline"].values/2)
    if(len(selected_states)>0):
        frames = []
        for i in range(len(selected_states)):
            frames.append(traffic_selected_date.loc[traffic_selected_date["stateName"] == selected_states[i],:])
        traffic_selected_date = pd.concat(frames)
    # For just hover tool, to show "No data"
    n =np.empty(traffic_selected_date["stateName"].values.shape)
    n[:] = np.nan
    traffic_selected_date["positiveIncrease"] = n    
    traffic_selected_date.fillna({'positiveIncrease' : 'No data'}, inplace = True)
    #------------
    return ColumnDataSource(data={'x': traffic_selected_date["x"].values,
                                 'y': traffic_selected_date["y"].values,
                                 "stateName": traffic_selected_date["stateName"].values,
                                 "AirportName":traffic_selected_date["AirportName"].values, 
                                 "PercentOfBaseline":traffic_selected_date["PercentOfBaseline"].values, 
                                 "CircleSize": traffic_selected_date["CircleSize"].values, 
                                 "positiveIncrease":traffic_selected_date["positiveIncrease"].values})


def create_data_for_heatmap(df):
    columns = ["Date","DateTime","AirportName", "PercentOfBaseline"]
    df1 = df[columns]
    df1 = df1.sort_values(["DateTime"])
    return ColumnDataSource(df1)

def update_heatmap_period(attr, old, new):
  
    date1 = datetime.fromtimestamp(dateRange.value[0]/1000)
    date_str1 = date1.strftime("%Y-%m-%d")

    date2 = datetime.fromtimestamp(dateRange.value[1]/1000)
    date_str2 = date2.strftime("%Y-%m-%d")

    date1 = datetime.strptime(date_str1, '%Y-%m-%d')
    date2 = datetime.strptime(date_str2, '%Y-%m-%d')
    columns = ["Date","DateTime","AirportName", "PercentOfBaseline"]
    heatmap_df = airport_traffic[columns]
    heatmap_df = heatmap_df.sort_values(["DateTime"])
    heatmap_df= heatmap_df[(heatmap_df["DateTime"] >= date1) & (heatmap_df["DateTime"] <= date2) ]

    selected_airports = []
    indices = airport_src.selected.indices
    
    #print("INDICES {}".format(indices))
    for i in indices:
        selected_airports.append(airport_src.data["AirportName"][i])
    if(len(selected_airports)>0):
        frames = []
        for i in range(len(selected_airports)):
            frames.append(heatmap_df.loc[heatmap_df["AirportName"] == selected_airports[i],:])
        heatmap_df = pd.concat(frames)

    new_heatmap_src = create_data_for_heatmap(heatmap_df)
    heatmap_src.data.update(new_heatmap_src.data)
    
#Preprocess Airport Data END--------------------------------------------------------------


USA_states = {state: abbrev for abbrev, state in us_state_abbrev.items()}

df_us_states_covid19_daily['stateName'] = df_us_states_covid19_daily['state'].map(USA_states)
# df_us_states_covid19_daily["date"] = df_us_states_covid19_daily["date"].astype(str)
df_us_states_covid19_daily["DateTime"] = df_us_states_covid19_daily["date"].astype(str)
#.map(lambda x: datetime.strptime(x, '%Y-%m-%d'))
df_us_states_covid19_daily_LinePlot = df_us_states_covid19_daily.copy()
#df_us_states_covid19_daily_Choropleth = df_us_states_covid19_daily.copy()

us_state_unique = list(set(df_us_states_covid19_daily['stateName']))
toBeRemoveList = ['Hawaii', 'Alaska', 'Guam', 'Virgin Islands', 'American Samoa', 'Puerto Rico', 'Northern Mariana Islands']
for i, name in enumerate(toBeRemoveList):
    us_state_unique.remove(name)

us_state_unique.sort(reverse = False)
#print(us_state_unique[:])
us_state_color = ['#393b79', '#5254a3', '#6b6ecf', '#9c9ede', '#637939', '#8ca252', '#b5cf6b', '#cedb9c', '#8c6d31', '#bd9e39',
                '#e7ba52', '#e7cb94', '#843c39', '#ad494a', '#d6616b', '#e7969c', '#7b4173', '#a55194', '#ce6dbd', '#de9ed6',
                '#3182bd', '#6baed6', '#9ecae1', '#c6dbef', '#e6550d', '#fd8d3c', '#fdae6b', '#fdd0a2', '#31a354', '#74c476',
                '#a1d99b', '#c7e9c0', '#756bb1', '#9e9ac8', '#bcbddc', '#dadaeb', '#636363', '#969696', '#bdbdbd', '#d9d9d9',
                '#0072B2', '#E69F00', '#F0E442', '#009E73', '#56B4E9', '#D55E00', '#CC79A7', '#000000', '#EC1557']
us_state_color_temp = {'state':us_state_unique, 'color':us_state_color}
us_state_unique_color = pd.DataFrame(us_state_color_temp)
#print(us_state_unique_color.head(60))

# ========================================================= #

import json

#initial_state_selection = ['Rhode Island', 'Arizona', 'Tennessee', 'Oklahoma', 'South Carolina', 'California', 'Utah', 'New York']
#initial_state_selection = ['Alabama', 'Arizona']

initial_selected_date = "2020-11-11"

df_state_Choropleth = (df_us_states_covid19_daily
            .loc[:,['date', 'state', 'positive', 'death', 'positiveCasesViral', 'positiveIncrease', 'stateName']])
df_state_Choropleth = df_state_Choropleth.dropna(axis=0, how='any')     # delete/drop NaN rows

def json_data(selectedDate, state_list):
    date = selectedDate
    df_state_date = df_state_Choropleth[df_state_Choropleth['date'] == date]
    
    #state_not_in_list = diff(us_state_unique, state_list)
    state_not_in_list = list(set(us_state_unique).difference(set(state_list)))
    #df_state_date.drop(state_not_in_list, axis=0, inplace=True) # this does not work because the stateName is not the index (primary key / key attribute)
    
    for i, state in enumerate(state_not_in_list):
        df_state_date.drop(df_state_date.index[(df_state_date['stateName'] == state)], axis=0, inplace=True)

    merged = gdf_usa_state_map.merge(df_state_date, on='state', how='left')
    merged.fillna({'positiveIncrease' : 'No data'}, inplace = True)
    merged.fillna({'date' : 'No data'}, inplace = True)
    merged.fillna({'positive' : 'No data'}, inplace = True)
    merged.fillna({'death' : 'No data'}, inplace = True)
    merged.fillna({'positiveCasesViral' : 'No data'}, inplace = True)
    del merged['stateName_y']
    merged = merged.rename(columns = {'stateName_x' : 'stateName'}, inplace=False)
    merged['date'] = merged['date'].astype(str)
    merged = merged.sort_values(by=['stateName'])       # sort the stateName, so that the indices is matched with sequence in us_state_unique, important for state selection on US map 
    merged = merged.reset_index()                       # reset the sorted stateName indices, so that the indices is matched with sequence in us_state_unique, important for state selection on US map 

    #Read data to json
    merged_json = json.loads(merged.to_json())

    #Convert to str like object
    json_data = json.dumps(merged_json)

    return json_data

# ============================================= #

from datetime import date, datetime

# Bokeh Libraries
from bokeh.io import show
from bokeh.plotting import figure
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar
from bokeh.models import CategoricalColorMapper, NumeralTickFormatter
from bokeh.models import BasicTicker,LinearColorMapper, PrintfTickFormatter
from bokeh.palettes import brewer

# Bokeh basics 
from bokeh.io import curdoc
from bokeh.models.widgets import Tabs
from bokeh.models import DateSlider, Slider, HoverTool, WheelZoomTool,BoxZoomTool, ResetTool, PanTool, DateRangeSlider 
from bokeh.layouts import widgetbox, row, column, gridplot

from bokeh.models import LassoSelectTool, BoxSelectTool             ## *** SELECTION TOOL FOR HEATMAP *** ##

from bokeh.models.widgets import CheckboxGroup

from bokeh.palettes import Category20_16            ## Not necessary, because each state has been assigned with unique color
from bokeh.models import Circle
from bokeh.models import Select, TapTool, CustomJS, CDSView, GroupFilter       ### *** IMPORTANT *** ###

from bokeh.models.selections import Selection       ### *** IMPORTANT *** ###

from bokeh.models import ColumnDataSource, Patches  ### *** IMPORTANT *** ###
from bokeh.transform import transform

from bokeh.models import FixedTicker

from sklearn.linear_model import LinearRegression
from bokeh.models import Slope

from bokeh.models import Div

# initial selected states for line plot
initial_state_selection_linePlot = ["California", "New York"]

# state colors
# state_colors = Category20_16        # Not necessary because unique color has been assigned to each state

# create Column Data Source (CDS) from the Dataframe
#source = ColumnDataSource(df_state_Choropleth)                 ### *** Not necessary to create ColumnDataSource

# ================================================================================ #

airport_traffic_scatterplot = airport_traffic.copy()
df_us_states_covid19_daily_scatterplot = df_us_states_covid19_daily.copy()

def update_state_airport_data(selected_states):
    airport_selected_states = []
    selectedState = []
    for i, statename in enumerate(selected_states):
        df_traffic = (airport_traffic_scatterplot[airport_traffic_scatterplot["stateName"]==statename]
                        .loc[:,['DateTime', 'stateName', 'PercentOfBaseline']])

        df_traffic = df_traffic.rename(columns={"DateTime": "date"})

        df_state = (df_us_states_covid19_daily_scatterplot[df_us_states_covid19_daily_scatterplot['stateName']==statename]
                    .loc[:,['date','positiveIncrease']])

        df_airport_states = pd.merge(df_traffic, df_state, how="left", on=["date"])
        df_airport_states = df_airport_states.sort_values(["date"]).reset_index(drop=True)

    print(df_airport_states)

    return ColumnDataSource(df_airport_states)

def make_scatter_plot(src):
    pScatter = figure(title="Scatter plot of airport(s) traffic volume vs COVID-19 positive cases in selected state (with regression line)",
                        x_axis_label="COVID-19 positive cases", 
                        y_axis_label="Airport traffic volume (%)",
                        plot_width=1500, plot_height=1000)

    pScatter.scatter(x='positiveIncrease', y='PercentOfBaseline', 
                    source=state_airport_src, marker="circle", 
                    fill_alpha=1.0, size=10, fill_color="grey",
                    legend='stateName')

    x = src.data['positiveIncrease']
    x_array = np.array(x)
    y = src.data['PercentOfBaseline']
    y_array = np.array(y)

    #par = np.polyfit(x_array, y_array, 1, full=True)
    #slope=par[0][0]
    #intercept=par[0][1]
    #y_predicted = [-slope*i + intercept  for i in x]
    #pScatter.line(x_array, y_predicted, color='red', line_width=5)
    
    model = LinearRegression().fit(x_array.reshape(-1,1), y)
    slope = model.coef_[0]
    intercept = model.intercept_
    regression_line = Slope(gradient=-slope, y_intercept=intercept, line_color="red", line_width=5)
    pScatter.add_layout(regression_line)


    # title
    pScatter.title.align = 'center'
    pScatter.title.text_font_size = '20pt'
    pScatter.title.text_font = 'serif'
    # Axis titles
    pScatter.xaxis.axis_label_text_font_size = '14pt'
    pScatter.xaxis.axis_label_text_font_style = 'bold'
    pScatter.yaxis.axis_label_text_font_size = '14pt'
    pScatter.yaxis.axis_label_text_font_style = 'bold'

    pScatter.legend.label_text_font_size = '20pt'

    return pScatter

# ================================================================================ #

# generate dataframe and columndatasource
# for line plot
def make_dataset(state_list):
    xs = []
    ys = []
    datetime = []
    colors = []
    labels = []
    y_c = []

    #For circle in the line graph
    
    for i, state in enumerate(state_list):

        # Isolate relevant data (DataFrame)
        df_state_linePlot = (df_us_states_covid19_daily_LinePlot[df_us_states_covid19_daily_LinePlot['stateName']==state]
                                                .loc[:,['DateTime', 'date','state', 'positive', 'death', 'positiveCasesViral', 'positiveIncrease', 'stateName']])
        
        df_state_linePlot = df_state_linePlot[df_state_linePlot['positiveIncrease'] >= 0]

        x = df_state_linePlot['date']
        y = df_state_linePlot['positiveIncrease']
        d = df_state_linePlot["DateTime"]
        xs.append(list(x))
        ys.append(list(y))
        datetime.append(list(d))
        #colors.append(state_colors[i])          # Not necessary, because each state has been assigned with unique color
        indexNO = list(us_state_unique_color.loc[us_state_unique_color['state']==state].index.values)
        for i in indexNO:
            colors.append(us_state_unique_color._get_value(i, 'color'))
        labels.append(state)

    new_src = ColumnDataSource(data={'x': xs, 'y': ys,'color': colors, 'label': labels, "DateTime":datetime})
    return new_src

def create_line_circle(selected_states):
    date_indices = heatmap_src.selected.indices
    if(len(date_indices)>0):
        #print("Date indices", date_indices)
        #print("Date", heatmap_src.data["Date"][date_indices[0]])
        #selected dates from heatmap
        selected_date = heatmap_src.data["Date"][date_indices[:]]
    else:
        selected_date = []
    frames = []
    for state in selected_states:
        df_state_linePlot = (df_us_states_covid19_daily_LinePlot[df_us_states_covid19_daily_LinePlot['stateName']==state]
                                                    .loc[:,['date','DateTime', 'positiveIncrease']])
        if(len(selected_date)>0):
            for date in selected_date:
                #df_state_linePlot = df_state_linePlot[df_state_linePlot['positiveIncrease'] >= 0]
                frames.append(df_state_linePlot.loc[df_state_linePlot["date"]==date,["date","DateTime","positiveIncrease"]])
    
    if(len(frames)>0):
        df = pd.concat(frames)
    else:
        df = pd.DataFrame({"date":[],"DateTime":[],"positiveIncrease":[]})

    new_src = ColumnDataSource(data={'x': df["date"].values, 'y': df["positiveIncrease"].values,"DateTime":df["DateTime"].values})
    return new_src

# function of making the line plot
def make_plot(src, circle_src):

    hoverp2 = HoverTool(tooltips = [('Date','@DateTime'),('New cases','@y')])
    wheelzoomp2 = WheelZoomTool()
    panp2 = PanTool()
    boxzoomp2 = BoxZoomTool() # not needed if wheelZoom is provided
    resettoolp2 = ResetTool()

    p2 = figure(x_axis_type='datetime',
                title='Daily COVID-19 new cases in USA',
                plot_height=600, plot_width=1200,
                x_axis_label='Date', y_axis_label='number of new cases',tools=[hoverp2, wheelzoomp2, panp2, boxzoomp2, resettoolp2])
    
    p2.multi_line('x', 'y', color='color', legend='label',
                    line_width=3, source=src)
    
    
    p2.circle(x="x",y="y",size = 15, color = "red",source= circle_src)
    p2 = style(p2)

    return p2

# function for styling the line plot
def style(p2):
    # Title 
    p2.title.align = 'center'
    p2.title.text_font_size = '20pt'
    p2.title.text_font = 'serif'

    # Axis titles
    p2.xaxis.axis_label_text_font_size = '14pt'
    p2.xaxis.axis_label_text_font_style = 'bold'
    p2.yaxis.axis_label_text_font_size = '14pt'
    p2.yaxis.axis_label_text_font_style = 'bold'

    # Tick labels
    p2.xaxis.major_label_text_font_size = '12pt'
    p2.yaxis.major_label_text_font_size = '12pt'

    p2.legend.label_text_font_size = '20pt'

    return p2   

# Define the callback function: update_plot
# for choropleth map plot and line plot
def update_plot(attr, old, new):
    #List of selected states
    state_to_plot = []

    date = datetime.fromtimestamp(date_slider.value/1000)
    date_str = date.strftime("%Y-%m-%d")
    #print(date_str)

    # List of state to plot
    # state_to_plot = [state_selection.labels[i] for i in state_selection.active]       # Not necessary, because multiple states can be selected on US map
    indices = geosource.selected.indices
    for i in indices:
        state_to_plot.append(us_state_unique[i])
    
    new_data = json_data(date_str, state_to_plot)
    geosource.geojson = new_data
    p1.title.text = 'Number of Covid-19 cases: %s' %date_str
    p1.title.align = 'center'
    p1.title.text_font_size = '20pt'
    p1.title.text_font = 'serif'


    # call function of line plot
    new_src = make_dataset(state_to_plot)
    line_src.data.update(new_src.data)
    new_circle_src = create_line_circle(state_to_plot)
    circle_src.data.update(new_circle_src.data)

    #Call  function for airports
    new_airport_src = update_airport_data(date_str,state_to_plot)
    airport_src.data.update(new_airport_src.data)

    #Call function for scatter plot
    new_state_airport_src = update_state_airport_data(state_to_plot)
    state_airport_src.data.update(new_state_airport_src.data)


#state_selection = CheckboxGroup(labels=us_state_unique, active=[0,1])                  # Not necessary, because multiple states can be selected on US map
#state_selection.on_change('active', update_plot)

#initial_state_selection = [state_selection.labels[i] for i in state_selection.active]  # Not necessary, because multiple states can be selected on US map
initial_state_selection = us_state_unique


#Input GeoJSON source that contains features for plotting.
geosource = GeoJSONDataSource(geojson = json_data(initial_selected_date, initial_state_selection))

#Define a sequential multi-hue color palette.
palette = brewer['Reds'][9]
palette = palette[::-1]     #Reverse color order so that dark red is highest obesity.

#Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.
color_mapper = LinearColorMapper(palette = palette, low = 0, high = 5000, nan_color = '#d9d9d9')

#Define custom tick labels for color bar.
tick_labels = {'5000': '>5000'}

#Add hover tool
hover = HoverTool(tooltips = [('State','@stateName'),('Airport','@AirportName'),('New cases','@positiveIncrease')])

#Create color bar. 
color_bar = ColorBar(color_mapper=color_mapper, label_standoff=8,width = 500, height = 20,
border_line_color=None,location = (0,0), orientation = 'horizontal', major_label_overrides = tick_labels)

#Create figure object.
#for Choropleth map plot
## *** IMPORTANT : remember to add the tap in tools *** ##
wheelzoom = WheelZoomTool()
pan = PanTool()
# boxzoom = BoxZoomTool() # not needed if wheelZoom is provided
resettool = ResetTool()
tap = TapTool()
lasso_select = LassoSelectTool()
box_select = BoxSelectTool()

p1 = figure(title = 'Number of Covid-19 cases: %s' %initial_selected_date, 
            plot_height = 600 , plot_width = 1000, toolbar_location = "right", tools=[hover,wheelzoom,resettool,pan, tap])         ## *** IMPORTANT : remember to add the tap in tools *** ##
p1.xgrid.grid_line_color = None
p1.ygrid.grid_line_color = None

#Add patch renderer to figure.
#for Choropleth map plot
p1.patches('xs','ys', source = geosource, fill_color = {'field' :'positiveIncrease', 'transform' : color_mapper},
          line_color = 'black', line_width = 0.35, fill_alpha = 1)

airport_src = update_airport_data(initial_selected_date,selected_states = [])
initial_circle = Circle(x='x', y='y', size="CircleSize", fill_color='blue')
#Specify figure layout.
#for Choropleth map plot
selected_circle = Circle(fill_alpha=1, fill_color="blue", line_color="blue")
nonselected_circle = Circle(fill_alpha=0.2, fill_color="blue", line_color=None)

p1.add_glyph(airport_src,
            initial_circle,
            selection_glyph=selected_circle,
            nonselection_glyph=nonselected_circle)

p1.add_layout(color_bar, 'below')
p1.title.text = 'Number of Covid-19 cases: {date}'.format(date=initial_selected_date)
p1.title.align = 'center'
p1.title.text_font_size = '20pt'
p1.title.text_font = 'serif'

# Make a slider object: slider
#for Choropleth map plot
date_slider = DateSlider(title='Date', value=date(2020, 11, 11), start=date(2020, 3, 15), end=date(2020, 12, 1), step=1, format = "%Y-%m-%d")
date_slider.on_change('value', update_plot)

# call update_plot function when multiple states are selected on US map
geosource.selected.on_change('indices', update_plot)
airport_src.selected.on_change('indices', update_heatmap_period)
#---------------------------Create a heatmap----------------------
heatmap_src = create_data_for_heatmap(airport_traffic)

colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
colors = colors[::-1]
mapper = LinearColorMapper(palette=colors, low=np.array(list(set(heatmap_src.data["PercentOfBaseline"]))).min(), high=np.array(list(set(heatmap_src.data["PercentOfBaseline"]))).max())
unique_dates = list(set(heatmap_src.data["Date"]))
unique_dates = sorted(unique_dates, key=lambda x: datetime.strptime(x, '%Y-%m-%d'))

toolList = ['lasso_select', 'box_select', 'tap', 'reset', 'save']

unique_airports = list(set(heatmap_src.data["AirportName"]))            # Sort the airport name in the heatmap in Alphabetical order (This fix the airport random order in the heatmap every re-run the application)
unique_airports.sort()
unique_airports.reverse()

hover_heatmap = HoverTool(tooltips = [('Airport','@AirportName'),('Date','@Date'),('Traffic Volume (%)','@PercentOfBaseline')])
p3 = figure(plot_width=1700, plot_height=800, title="Heatmap of airports in US (2020 March-December)",
           x_range=unique_dates, y_range=unique_airports,
           toolbar_location="left", tools=[lasso_select,box_select,tap,pan,wheelzoom,resettool,hover_heatmap], x_axis_location="above")

p3.rect(x="Date", y="AirportName", width=1, height=1, source=heatmap_src,
       line_color=None, fill_color=transform('PercentOfBaseline', mapper))

color_bar = ColorBar(color_mapper=mapper, location=(0, 0),
                     ticker=BasicTicker(desired_num_ticks=len(colors)),
                     formatter=PrintfTickFormatter(format="%d%%"))

p3.title.align = 'center'
p3.title.text_font_size = '20pt'
p3.title.text_font = 'serif'

#---------------------------Custom x axis formatter (For Heatmap)-------------------------

from bokeh.models import TickFormatter
from bokeh.util.compiler import TypeScript

TS_CODE = """
import {TickFormatter} from "models/formatters/tick_formatter"

export class MyFormatter extends TickFormatter {
  // TickFormatters should implement this method, which accepts a list
  // of numbers (ticks) and returns a list of strings
  doFormat(ticks: string[] | number[]) {
    // format the first tick as-is
    const formatted = [`${ticks[0]}`]
    for (let i = 1, len = ticks.length; i < len; i++) {
        if(i%4==0){
      formatted.push(`${ticks[i]}`)
      }
      else
      {
          formatted.push(`${""}`)
      }
    }
    return formatted
  }
}
"""

class MyFormatter(TickFormatter):

    __implementation__ = TypeScript(TS_CODE)


#----------------------------------------------------------------------------
p3.add_layout(color_bar, 'right')

p3.xaxis.formatter = MyFormatter()

# p3.axis.axis_line_color = None
p3.axis.major_tick_line_color = "red"
p3.axis.major_label_text_font_size = "14px"
p3.axis.major_label_standoff = 0
p3.xaxis.major_label_orientation = 1.0
p3.yaxis.axis_label_text_font_size = "30pt"
p3.xaxis.axis_label_text_font_size = "7pt"


dateRange = DateRangeSlider(title='Period', value=(date(2020, 4, 1),date(2020, 6, 1)), start=date(2020, 3, 15), end=date(2020, 12, 1), step=1, format = "%Y-%m-%d")
dateRange.on_change('value', update_heatmap_period)

#---------


# call function of making the dataset for line plot
line_src = make_dataset(initial_state_selection_linePlot)
circle_src = create_line_circle(initial_state_selection_linePlot)
# call the function of making the line plot 
p2 = make_plot(line_src,circle_src)
heatmap_src.selected.on_change("indices",update_plot)

# ---------------------------------------------------------- #

#selected_states = ['Texas']
#selected_states = ['New York']
selected_states = ['California']

#selected_states = ['New York', 'California']
#state_airport_src = update_state_airport_data(selected_states=[])
state_airport_src = update_state_airport_data(selected_states)
pScatter = make_scatter_plot(state_airport_src)

# ---------------------------------------------------------- #
# show the airport and state table in the dashboard

from bokeh.models.widgets import DataTable, DateFormatter, TableColumn

Columns = [TableColumn(field=Ci, title=Ci) for Ci in df_stateName_airportName_unique.columns]
data_table = DataTable(columns=Columns, source=ColumnDataSource(df_stateName_airportName_unique), width=1000, height=500)

# ==================== dashboard layout ==================== #

# Add a title for the entire visualization using Div
html = """<h1>Correlation between Airport Traffic Volume and COVID-19 Cases in USA</h1>
<h2><i>Please hold the SHIFT key when selecting multiple states on the US map</i></h2>
<h2><i>The scatter plot shows only the final selected state</i></h2>
"""
sup_title = Div(text=html)

#layout = column(p, controls)
#layout = gridplot([[p, state_selection],[date_slider, None]])

layout1 = column(sup_title, p1, date_slider, p2, data_table)
layout2 = column(dateRange, p3, pScatter)
layout = row(layout1,layout2)

curdoc().add_root(layout)

#Display figure.
show(layout)