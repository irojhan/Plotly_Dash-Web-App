#from google.colab import files

#uploaded = files.upload()
import io


# importing required libraries
import dash
#from jupyter_dash import JupyterDash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup as BS
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import dash_table
import time

#sns.set()

app = dash.Dash(__name__, suppress_callback_exceptions=True)

# importing data and preprocessing
data = pd.read_csv('kc_house_data.csv')
# omitting the extra and meaningless part of the date column
data['date'] = data.apply(lambda x: x.date[0:8], axis=1)
# Creating an "age" column based on date of selling and date of built
data['age'] = data.apply(lambda x: int(x.date[0:4]) - x.yr_built, axis=1)
# Creating an "age_renovated" column based on date of selling and date of renovation
data['age_renovated'] = data.apply(
    lambda row: row.age if row.yr_renovated == 0 else int(row.date[0:4]) - row.yr_renovated, axis=1)
data = data[['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view',
             'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'age', 'yr_renovated', 'age_renovated',
             'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']]
df = data.copy()
# Creating a new column which includes the price of each square meter of the property
df['persqft'] = df['price'] / df['sqft_living']
df = df[['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living',
         'sqft_lot', 'persqft', 'floors', 'waterfront', 'view', 'condition', 'grade',
         'sqft_above', 'sqft_basement', 'yr_built', 'age', 'yr_renovated',
         'age_renovated', 'zipcode', 'lat', 'long', 'sqft_living15',
         'sqft_lot15']]
# requesting the data for zip codes and their related city from a website
url = "https://www.ciclt.net/sn/clt/capitolimpact/gw_ziplist.aspx?FIPS=53033"
data = requests.get(url)
soup = BS(data.text, 'html.parser')
# scraping the website and saving the zipcode and city pairs in a dictionary
ZipCode_dict = {}
for i in soup.find_all(border='3'):

    for j in i.find_all('tr'):
        k = j.find(align='left')
        if k != None:
            zip = int(k.get_text())
            city = k.next_sibling.next_sibling.get_text()
            ZipCode_dict[zip] = city
# creating a new column based on the zipcode column and the dictionary
df['city'] = df["zipcode"].map(ZipCode_dict)
df = df[['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living',
         'sqft_lot', 'persqft', 'floors', 'waterfront', 'view', 'condition',
         'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'age',
         'yr_renovated', 'age_renovated', 'zipcode', 'city', 'lat', 'long',
         'sqft_living15', 'sqft_lot15']]
df = df[df['sqft_living'] < 5000]
df = df[(df['persqft'] < np.quantile(df['persqft'], .95)) & (df['persqft'] > np.quantile(df['persqft'], .05))]
df.reset_index(drop=True, inplace=True)
df_to_plot = df.copy()
df_to_plot['bedrooms'] = df_to_plot['bedrooms'].apply(str)

# bedroom - bathroom price- groupby
bedroom_bathroom_mean_df = df.groupby(['bedrooms', 'waterfront'])['price'].mean().reset_index()
pd.set_option('display.float_format', lambda x: f'{x:,.3f}')
df2 = df.copy()
df2.loc[:, "price"] = df["price"].map('{:,.0f}'.format)
df3 = df['city'].value_counts().copy()


# defining a function for determining the upper and lower threshold for
def lower_upper(x):
    Q1 = np.percentile(x, 25)
    Q3 = np.percentile(x, 75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return lower, upper


lower_area, upper_area = lower_upper(df['sqft_living'])
lower_price, upper_price = lower_upper(df['price'])

# omitting outliers

area_outliers = np.where(df['sqft_living'] > upper_area)
price_outliers = np.where(df['price'] > upper_price)
# Return the unique, sorted array of values that are in either of the two input arrays.
total_outliers = np.union1d(area_outliers, price_outliers)

df = df.copy()
df.drop(total_outliers, inplace=True)
df.reset_index(drop=True, inplace=True)

# create bar chart
bar_ = go.Bar(x=df3.values, y=df3.index, orientation='h')

# set layout
layout = go.Layout(
    title='Number of flats in location',
    xaxis=dict(title='numerical amount of flats'),
    yaxis=dict(title='location'),
    height=600,
    margin=dict(l=100, r=50, b=100, t=100),
)

# create figure
fig2 = go.Figure(data=bar_, layout=layout)

# plotting price versus some of other independent variables
fig1 = px.scatter_matrix(df,
                         dimensions=['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'waterfront', 'view',
                                     'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'age',
                                     'age_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15'])

# Define the available variables for the dropdown
available_vars = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'waterfront', 'view', 'condition',
                  'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'age', 'age_renovated', 'lat', 'long',
                  'sqft_living15', 'sqft_lot15']

# Create initial histogram figure
variable = 'price'  # Initial variable to plot
fig = go.Figure(
    go.Histogram(
        x=df[variable],
        nbinsx=30,
        histnorm='probability density',
        marker=dict(color='blue'),
        opacity=0.75
    ),
    layout=go.Layout(
        title=f"{variable.capitalize()} Distribution",
        xaxis=dict(title=variable.capitalize()),
        yaxis=dict(title='Probability Density'),
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode='closest'
    )
)



# Define the layout
app.layout = html.Div(children=[
    html.Div(children=[
        html.H1(children='DSE 6000 Web App project'),
        html.H2(children='Group 2: Roohollah Jahanmahin, Sam Darabi, Sina Dehbashi'),
        html.H3(children='House Price Prediction'),
    ], style={'text-align': 'center', 'background-color': 'lavender', 'margin': '10px 3px 3px 3px',
              'padding': '3px 3px 3px 3px'}),

    html.Div(children=[
        html.H4(children='Pairplot', style={'font-size': '20px', 'font-weight': 'bold','text-align': 'center'}),
        html.P('what is the relationship between different features and price?', style={'font-size': '20px', 'font-weight': 'bold','text-align': 'center'}),
        dcc.Dropdown(
            id='pairplot-dropdown',
            options=[{'label': var, 'value': var} for var in available_vars],
            value=['price', 'bedrooms'],
            multi=True
        ),

        dcc.Graph(id='pairplot-graph'),
        html.P('1. Based on the resulted pairplot between number of bedrooms, living square, age with price of the house, we can conclude that as the number of bedrooms are so close to each other except some outliers such as a house with 33 bedrooms, so the price does not change significantly.', style={'font-size': '17px','text-align': 'center'}),
        html.P('2. By increasing living square, we can conclude that the price will increase.', style={'font-size': '17px', 'text-align': 'center'}),
        html.P('3. We have some houses with higher ages but that house have the value 1.081 M with the age of 93, on the other side we can have a house with the age of 36 and the price of 118 K. so there is not high correlation between age and price.',style={'font-size': '17px', 'text-align': 'center'}),

    ], style={'text-align': 'center', 'background': 'linear-gradient(45deg, #2193b0, #6dd5ed)', 'height': '100vh', 'margin': '10px 3px 3px 3px',
              'padding': '3px 3px 3px 3px'}),

    html.Div(children=[
        html.Hr(),
        html.H5(children='Scatter Plot', style={'font-size': '20px', 'font-weight': 'bold','text-align': 'center'}),
        html.P('Here we want to check how the price changes with the number of bedrooms. Logically the more bedrooms a house does have, the higher the price should be.', style={'font-size': '17px','text-align': 'center'}),

        dcc.Graph(
            id='scatter-plot',
            figure=px.scatter(df, x="long", y="lat", color='bedrooms', size='price', hover_name='city',
                              labels={"bedrooms": "Number of Bedrooms"})
        ),
        html.P('Based on the figures we have above, the price goes up as the number of its bedrooms does.However, there are some cases where a 10-bedroom house is less expensive compared to 7 or 8- bedroom ones. In fact , we can say that the the number of bedrooms in a house is an important factor affecting the price, but there are other factors which we need to consider to compare two houses values.', style={'font-size': '17px','text-align': 'center'})

    ], style={'width': '50%', 'float': 'right', 'text-align': 'center', 'height': '100vh'}),

    html.Div([
        html.Hr(),
        html.H6(children='Select a variable to plot', style={'font-size': '20px', 'font-weight': 'bold','text-align': 'center'}),
        html.P('We are interested to see the price distribution in the house market. It can help us understand what the main customer are target. From our experience, since the number of rich people are way much less than people with average income, there must be more affordable houses.', style={'font-size': '17px','text-align': 'center'}),

        dcc.Dropdown(
            id='variable-dropdown',
            options=[{'label': var.capitalize(), 'value': var} for var in available_vars],
            value=variable
        ),
        dcc.Graph(id='histogram-graph', figure=fig),
        html.P('As figure shows and we expected, most of the houses have average price and number of houses with higher price are low. It does make sense, since most of the buyers have average income and there should be more houses in that price range for them to be able to buy. Also, the market can keep its balance in this way.', style={'font-size': '17px','text-align': 'center'})

    ], style={'width': '50%', 'float': 'left', 'text-align': 'center', 'height': '100vh'}),

    html.Div([
        html.Hr(),
        html.H6("Correlation Heatmap", style={'font-size': '20px', 'font-weight': 'bold','text-align': 'center'}),
        html.Div([
            html.P('To study the correlation between factors affecting a house price, we decided to use a Heatmap. It can help us see the possible correlation between them.', style={'font-size': '17px','text-align': 'center'}),

            dcc.Dropdown(
                id='var-dropdown',
                options=[{'label': var, 'value': var} for var in available_vars],
                value=['price', 'bedrooms', 'bathrooms'],
                multi=True
            ),
            dcc.Graph(id='correlation-heatmap'),
            html.P('The result that this figure is showing us is interesting. We did not expect that there is a higher correlation between number of bathrooms and a house price, compared to the number of its bedrooms and the price. Also, we can see that there is a correlation between the number of bathrooms and the number of bedrooms, which it was expected.', style={'font-size': '17px','text-align': 'center'})

        ])
    ], style={'width': '50%', 'float': 'left', 'text-align': 'center', 'height': '100vh'}),

    html.Div([
        html.Hr(),
        html.H1('Boxplot', style={'font-size': '20px', 'font-weight': 'bold','text-align': 'center'}),
        html.Div([
            dcc.Dropdown(
                id='feature-dropdown',
                options=[{'label': col, 'value': col} for col in df.columns],
                value='price'
            )

        ]),
        dcc.Graph(id='boxplot'),
        html.P('This box plot confirms our finding in the price distribution graph. The price mean is closer to the average value of the houses. It is simply because the number of houses are relatively high and the mean will be in that price range. Aslo we can see the expensive houses are the outliers in the blox figure.',
            style={'font-size': '17px', 'text-align': 'center'})

    ], style={'width': '50%', 'float': 'left', 'text-align': 'center', 'height': '100vh'}),
    html.Div(children=[
        html.Hr(),
        html.H3(children='Number of flats in location', style={'font-size': '20px', 'font-weight': 'bold','text-align': 'center'}),
        dcc.Graph(figure=fig2),
        html.P('The main purpose of plotting this figure is to see what is the number of flats distribution in cities. As we can see, majority of them are located in Seattle with more than 6000 flats, followed by Renton which has around 1500 ones. Redmond is ranked 3rd with roughly 1400 flats.', style={'font-size': '17px', 'text-align': 'center'})

    ], style={'width': '100%', 'float': 'left', 'text-align': 'center', 'height': '100vh'})

    ,

    html.Div([
        html.Br(),
        html.H1('Prediction Section', style={'font-size': '20px', 'font-weight': 'bold','text-align': 'center'}),
        html.Hr(),
        html.Br(),
        html.P('In this section, some different ML models are applied on the dataset, and the results are compared at the final step', style={'font-size': '20px', 'text-align': 'center', 'background': 'lavender'})
    ]),

    html.Div([
        html.Hr(),
        html.H2('Linear Regression Model',  style={'font-size': '24px', 'font-weight': 'bold', 'text-align': 'center'}),
        html.Br(),
        html.P('Enter your data',  style={'font-size': '20px', 'font-weight': 'bold', 'text-align': 'center'}),
        html.Br(),
        html.P('Age:',  style={'font-size': '17px', 'font-weight': 'bold', 'text-align': 'center'}),
        dcc.Input(id='age_input_id', value='', type='text',  style={'font-size': '17px', 'font-weight': 'bold', 'text-align': 'center'}),
        html.P('Living Sqrt:',  style={'font-size': '17px', 'font-weight': 'bold', 'text-align': 'center'}),
        dcc.Input(id='living_input_id', value='', type='text',  style={'font-size': '17px', 'font-weight': 'bold', 'text-align': 'center'}),
        html.P('Number of bedrooms:',  style={'font-size': '17px', 'font-weight': 'bold', 'text-align': 'center'}),
        dcc.Input(id='bedrooms_input_id', value='', type='text',  style={'font-size': '17px', 'font-weight': 'bold', 'text-align': 'center'}),
        html.P('Number of bathrooms:',  style={'font-size': '17px', 'font-weight': 'bold', 'text-align': 'center'}),
        dcc.Input(id='bathrooms_input_id', value='', type='text',  style={'font-size': '17px', 'font-weight': 'bold', 'text-align': 'center'}),
        html.Br(),
        html.Button('Run LR model', id='btn_predict', n_clicks=0, style={'font-size': '17px', 'font-weight': 'bold', 'text-align': 'center', 'margin-top': '25px'}),
        html.Br(),
        html.Div(id='Output_id',
                 children='Enter a value and press submit'),
        html.Br(),
        html.Div(id='pred_output_lr_id')

    ],style={'text-align': 'center', 'background': 'linear-gradient(45deg, #2193b0, #6dd5ed)', 'height': '70vh',
              'padding': '3px 3px 3px 3px', 'margin': 'auto'}),

    html.Div([
        html.Hr(),
        html.H2('Ridge, Lasso regression Models',  style={'font-size': '24px', 'font-weight': 'bold', 'text-align': 'center'}),
        html.Br(),
        html.Button('Run models', id='btn_models', n_clicks=0, style={'font-size': '17px', 'font-weight': 'bold', 'text-align': 'center', 'margin-top': '25px'}),
        html.Br(),
        html.Div(id='Output_models_id',
                 children='Enter a value and press submit'),
    ],style={'text-align': 'center', 'background': 'lavender', 'height': '22vh',
              'padding': '3px 3px 3px 3px', 'margin': 'auto'}),

    html.Div([
        html.Hr(),
        html.H2('Decision Tree and Random forest Models',  style={'font-size': '24px', 'font-weight': 'bold', 'text-align': 'center'}),
        html.Br(),
        html.Button('Run models', id='btn_DT_RF_id', n_clicks=0, style={'font-size': '17px', 'font-weight': 'bold', 'text-align': 'center', 'margin-top': '25px'}),
        html.Br(),
        html.Div(id='Output_DT_RF_id',
                 children='Enter a value and press submit'),
    ],style={'text-align': 'center', 'background': 'linear-gradient(45deg, #2193b0, #6dd5ed)', 'height': '22vh',
              'padding': '3px 3px 3px 3px', 'margin': 'auto'}),

    html.Div([
        html.Hr(),
        html.H2('KNN Model',  style={'font-size': '24px', 'font-weight': 'bold', 'text-align': 'center'}),
        html.Br(),
        html.Button('Run KNN model', id='btn_knn_id', n_clicks=0, style={'font-size': '17px', 'font-weight': 'bold', 'text-align': 'center', 'margin-top': '25px'}),
        html.Br(),
        html.Div(id='Output_knn_id',
                 children='Enter a value and press submit'),
    ],style={'text-align': 'center', 'background': 'lavender', 'height': '22vh',
              'padding': '3px 3px 3px 3px', 'margin': 'auto'}),

    html.Div([
        html.Hr(),
        html.H2('XGBoost Model',  style={'font-size': '24px', 'font-weight': 'bold', 'text-align': 'center'}),
        html.Br(),
        html.Button('Run XGBoost model', id='btn_xgboost_id', n_clicks=0, style={'font-size': '17px', 'font-weight': 'bold', 'text-align': 'center', 'margin-top': '25px'}),
        html.Br(),
        html.Div(id='Output_xgboost_id',
                 children='Enter a value and press submit'),
    ],style={'text-align': 'center', 'background': 'linear-gradient(45deg, #2193b0, #6dd5ed)', 'height': '22vh',
              'padding': '3px 3px 3px 3px', 'margin': 'auto'}),

    html.Div([
        html.Hr(),
        html.H2('Compare Models',  style={'font-size': '24px', 'font-weight': 'bold', 'text-align': 'center'}),
        html.Br(),
        html.Button('compare', id='btn_compare_id', n_clicks=0, style={'font-size': '17px', 'font-weight': 'bold', 'text-align': 'center', 'margin-top': '25px'}),
        html.Br(),
        html.Div(id='Output_compare_id',
                 children='Enter a value and press submit'),
    ],style={'text-align': 'center', 'background': 'lavender', 'height': '50vh',
              'padding': '3px 3px 3px 3px', 'margin': 'auto'})

])


# Define the callback to update the pairplot
@app.callback(
    dash.dependencies.Output('pairplot-graph', 'figure'),
    [dash.dependencies.Input('pairplot-dropdown', 'value')]
)
def update_pairplot(selected_vars):
    fig = px.scatter_matrix(df, dimensions=selected_vars)
    return fig


# Define app callback
@app.callback(
    dash.dependencies.Output('scatter-plot', 'figure'),
    [dash.dependencies.Input('dropdown', 'value')]
)
def update_scatter_plot(selected_bedrooms):
    filtered_df = df[df['bedrooms'] == selected_bedrooms]
    fig = px.scatter(filtered_df, x="long", y="lat", color='bedrooms', size='price',
                     hover_name='city', labels={"bedrooms": "Number of Bedrooms"})
    return fig


@app.callback(
    dash.dependencies.Output('histogram-graph', 'figure'),
    [dash.dependencies.Input('variable-dropdown', 'value')]
)
def update_histogram(variable):
    fig = go.Figure(
        go.Histogram(
            x=df[variable],
            nbinsx=30,
            histnorm='probability density',
            marker=dict(color='blue'),
            opacity=0.75
        ),
        layout=go.Layout(
            title=f"{variable.capitalize()} Distribution",
            xaxis=dict(title=variable.capitalize()),
            yaxis=dict(title='Probability Density'),
            margin=dict(l=50, r=50, t=50, b=50),
            hovermode='closest'
        )
    )
    return fig


@app.callback(Output('correlation-heatmap', 'figure'),
              [Input('var-dropdown', 'value')])
def update_heatmap(selected_vars):
    # Select data for selected variables
    selected_data = df[selected_vars]

    # Calculate correlation matrix
    corr = selected_data.corr()

    # Define heatmap figure
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='RdYlGn',
        zmin=-1, zmax=1
    ))

    fig.update_layout(
        #title={'text':"Correlation Heatmap", 'x': 0.5, 'y': 0.95, 'xanchor': 'center','yanchor': 'top'},
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode='closest'
    )

    return fig


@app.callback(
    Output('boxplot', 'figure'),
    [Input('feature-dropdown', 'value')
    ]
)
def update_boxplot(feature):
    fig = go.Figure()
    fig.add_trace(go.Box(y=df[feature], name='{}'.format(feature)))
    fig.update_layout(title={
        'text': 'Boxplot of {}'.format(feature),
        'x': 0.5,
        'y': 0.9,
        'xanchor': 'center',
        'yanchor': 'top'})
    fig.update_yaxes(title_text='Value')
    return fig


"""
for prediction we use 4 features with high correlation which are 
 1- number of bedrooms
 2- number of bathrooms
 3- basement sqrt
 4- age 

"""

X = df[['age', 'sqft_living', 'bedrooms', 'bathrooms']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

score_List = []
"""
linear regression
"""


@app.callback(
    Output('Output_id', 'children'),
    [Input('btn_predict', 'n_clicks'),
     Input('age_input_id', 'value'),
     Input('living_input_id', 'value'),
     Input('bedrooms_input_id', 'value'),
     Input('bathrooms_input_id', 'value')]
)
def update_lr(n_clicks, age_input_id, living_input_id, bedrooms_input_id, bathrooms_input_id):
    if n_clicks == 0:
        return 'Press button to run the linear regression model'
    else:
        # print(f"shape of x train: {X_train.shape}")
        # print(f"shape of y train: {y_train.shape}")
        # print(f"shape of x test: {X_test.shape}")
        # print(f"shape of y train: {y_test.shape}")
        lr = LinearRegression(n_jobs=-1)
        lr_train_score, lr_test_score, lr_RMSE, lr_model = parameter_finder(lr, {})

        score_List.append(lr_train_score)
        score_List.append(lr_test_score)
        score_List.append(lr_RMSE)

        X_new = np.array(
            [float(age_input_id), float(living_input_id), float(bedrooms_input_id), float(bathrooms_input_id)]).reshape(
            1, -1)
        lr_pred = lr_model.predict(X_new)
        # sum1=float(age_input_id)+float(living_input_id)+float(bedrooms_input_id)+float(bathrooms_input_id)

        return '[Train Score : {} -- Test Score : {} -- RMSE : {} -- Run Number : {} --- Price Prediction is : {}]'.format(
            '{:.3f}'.format(lr_train_score), '{:.3f}'.format(lr_test_score), '{:.3f}'.format(lr_RMSE), '{:.3f}'.format(n_clicks), '{:.3f}'.format(lr_pred.item()))


"""
Ridge and Lasso regression
"""


@app.callback(
    Output('Output_models_id', 'children'),
    Input('btn_models', 'n_clicks')
)
def update_regress(n_clicks):
    if n_clicks == 0:
        return 'Press button to run the linear regression model'
    else:
        ridge = Ridge(random_state=1)  # Linear least squares with l2 regularization.
        param_ridge = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
        ridge_train_score, ridge_test_score, ridge_RMSE, ridge_model = parameter_finder(ridge, param_ridge)

        lasso = Lasso(random_state=1)  # Linear Model trained with L1 prior as regularizer.
        param_lasso = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
        lasso_train_score, lasso_test_score, lasso_RMSE, lasso_model = parameter_finder(lasso, param_lasso)

        score_List.append(ridge_train_score)
        score_List.append(ridge_test_score)
        score_List.append(ridge_RMSE)
        score_List.append(lasso_train_score)
        score_List.append(lasso_test_score)
        score_List.append(lasso_RMSE)

        return '[Ridge Train Score : {} -- Ridge Test Score : {} -- Ridge RMSE : {} --- Lasso Train Score : {} -- Lasso Test Score : {} -- Lasso RMSE : {}'.format(
            '{:.3f}'.format(ridge_train_score), '{:.3f}'.format(ridge_test_score), '{:.3f}'.format(ridge_RMSE), '{:.3f}'.format(lasso_train_score), '{:.3f}'.format(lasso_test_score), '{:.3f}'.format(lasso_RMSE))


"""
Decision Tree and Random forest
"""


@app.callback(
    Output('Output_DT_RF_id', 'children'),
    Input('btn_DT_RF_id', 'n_clicks')
)
def update_DT_RF(n_clicks):
    if n_clicks == 0:
        return 'Press button to run models'
    else:
        dtr = DecisionTreeRegressor(random_state=1)
        param_dtr = {'min_samples_split': [2, 3, 4, 5],
                     'min_samples_leaf': [1, 2, 3]}
        dtr_train_score, dtr_test_score, dtr_RMSE, DT_model = parameter_finder(dtr, param_dtr)

        rfr = RandomForestRegressor(random_state=1, n_jobs=-1)
        param_rfr = {'min_samples_split': [2, 3, 4, 5],
                     'min_samples_leaf': [1, 2, 3]}
        rfr_train_score, rfr_test_score, rfr_RMSE, RF_model = parameter_finder(rfr, param_rfr)

        score_List.append(dtr_train_score)
        score_List.append(dtr_test_score)
        score_List.append(dtr_RMSE)
        score_List.append(rfr_train_score)
        score_List.append(rfr_test_score)
        score_List.append(rfr_RMSE)

        return 'Decision Tree Train Score : {} -- Decision Tree Test Score : {} -- Decision Tree RMSE : {} --- Random Forest Train Score : {} -- Random Forest Test Score : {} -- Random Forest RMSE : {}'.format('{:.3f}'.format(dtr_train_score), '{:.3f}'.format(dtr_test_score), '{:.3f}'.format(dtr_RMSE), '{:.3f}'.format(rfr_train_score),
                                                    '{:.3f}'.format(rfr_test_score), '{:.3f}'.format(rfr_RMSE))


"""
KNN model
"""


@app.callback(
    Output('Output_knn_id', 'children'),
    Input('btn_knn_id', 'n_clicks')
)
def update_KNN(n_clicks):
    if n_clicks == 0:
        return 'Press button to run KNN model'

    else:
        knr = KNeighborsRegressor(n_jobs=-1)
        param_knr = {'n_neighbors': [5, 10, 15, 20],
                     'weights': ['uniform', 'distance']}
        knr_train_score, knr_test_score, knr_RMSE, knn_model = parameter_finder(knr, param_knr)

        score_List.append(knr_train_score)
        score_List.append(knr_test_score)
        score_List.append(knr_RMSE)

        # return '[KNN train score : {} , KNN test score : {} , KNN RMSE : {} , Run-time : {}]'.format(2,4, 6)
        return 'Train Score : {} -- Test Score : {} -- RMSE : {}'.format('{:.3f}'.format(knr_train_score), '{:.3f}'.format(knr_test_score), '{:.3f}'.format(knr_RMSE))


"""
XGBoost model
"""


@app.callback(
    Output('Output_xgboost_id', 'children'),
    Input('btn_xgboost_id', 'n_clicks')
)
def update_XGB(n_clicks):
    if n_clicks == 0:
        return 'Press button to run Xgboost model'
    else:
        xgboost = XGBRegressor(n_jobs=-1)
        param_xgboost = {'n_estimators': [100, 300],
                         'learning_rate': [0.1, 0.05],
                         'subsample': [0.75],
                         'colsample_bytree': [1],
                         'max_depth': [3, 4, 5],
                         'gamma': [0]}
        xgboost_train_score, xgboost_test_score, xgboost_RMSE, xgboost_model = parameter_finder(xgboost, param_xgboost)

        score_List.append(xgboost_train_score)
        score_List.append(xgboost_test_score)
        score_List.append(xgboost_RMSE)

        return 'Train score : {} -- Test score : {} -- RMSE : {}'.format('{:.3f}'.format(xgboost_train_score), '{:.3f}'.format(xgboost_test_score),
                                                                       '{:.3f}'.format(xgboost_RMSE))


"""
Define the function for tunning
"""


def parameter_finder(model, parameters):
    start = time.time()

    grid = GridSearchCV(model,
                        param_grid=parameters,
                        refit=True,
                        cv=KFold(shuffle=True, random_state=1),
                        n_jobs=-1)
    grid_fit = grid.fit(X_train, y_train)
    y_train_pred = grid_fit.predict(X_train)
    y_pred = grid_fit.predict(X_test)

    train_score = grid_fit.score(X_train, y_train)
    test_score = grid_fit.score(X_test, y_test)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))

    model_name = str(model).split('(')[0]
    end = time.time()

    # print(f"The best parameters for {model_name} model is: {grid_fit.best_params_}")
    # print("--" * 10)
    # print(f"(R2 score) in the training set is {train_score:0.2%} for {model_name} model.")
    # print(f"(R2 score) in the testing set is {test_score:0.2%} for {model_name} model.")
    # print(f"RMSE is {RMSE:,} for {model_name} model.")
    # print("--" * 10)
    # print(f"Runtime of the program is: {end - start:0.2f}")

    return train_score, test_score, RMSE, grid_fit


"""
Define Compare function
"""


@app.callback(
    Output('Output_compare_id', 'children'),
    Input('btn_compare_id', 'n_clicks')
)
def compare(n_clicks):
    if n_clicks == 0:
        return 'Click Compare button!'
    else:
        try:
            models_score = pd.DataFrame({'  Training Score  ': ['{:.3f}'.format(score_List[0]), '{:.3f}'.format(score_List[3]), '{:.3f}'.format(score_List[6]), '{:.3f}'.format(score_List[9]), '{:.3f}'.format(score_List[12]), '{:.3f}'.format(score_List[15]), '{:.3f}'.format(score_List[18])],
                                         '  Testing Score  ': ['{:.3f}'.format(score_List[1]), '{:.3f}'.format(score_List[4]), '{:.3f}'.format(score_List[7]), '{:.3f}'.format(score_List[10]), '{:.3f}'.format(score_List[13]), '{:.3f}'.format(score_List[16]), '{:.3f}'.format(score_List[19])],
                                         '  RMSE  ': ['{:.3f}'.format(score_List[2]), '{:.3f}'.format(score_List[5]), '{:.3f}'.format(score_List[8]), '{:.3f}'.format(score_List[11]), '{:.3f}'.format(score_List[14]), '{:.3f}'.format(score_List[17]), '{:.3f}'.format(score_List[20])]},
                                        index=['LinearRegression', 'Ridge', 'Lasso', 'Decision Tree', 'Random Forest','KNeighborsRegressor' , 'XGBoost', ])

            # Create an HTML table to display the DataFrame
            table = html.Table([
                html.Thead(html.Tr([html.Th('Model')] + [html.Th(col) for col in models_score.columns])),
                html.Tbody([
                    html.Tr([html.Td(idx)] + [html.Td(val) for val in row.values],
                            style={'border-bottom': '3px solid #ddd'}) for idx, row in models_score.iterrows()
                ],style={'font-size': '17px', 'font-weight': 'bold','text-align': 'center', 'margin': 'auto'})
            ],style={'text-align': 'center', 'margin': 'auto'})

            # table = dash_table.DataTable(
            # id='table',
            # columns=[{"name": i, "id": i} for i in models_score.columns],
            # data=models_score.to_dict('records', index=True),
            # style_table={'height': '300px', 'overflowY': 'auto'}
            # )
            # table = html.Table([
            #   html.Thead(html.Tr([html.Th(col) for col in models_score.columns])),
            #  html.Tbody([
            #     html.Tr([
            #        html.Td(models_score.iloc[i][col]) for col in models_score.columns
            #   ]) for i in range(len(models_score))
            # ])
            # ])
            return table
            # return 'Done'
        except:
            return 'Not'


def get_score_list():
    return


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)


def parameter_finder(model, parameters):
    start = time.time()

    grid = GridSearchCV(model,
                        param_grid=parameters,
                        refit=True,
                        cv=KFold(shuffle=True, random_state=1),
                        n_jobs=-1)
    grid_fit = grid.fit(X_train, y_train)
    y_train_pred = grid_fit.predict(X_train)
    y_pred = grid_fit.predict(X_test)

    train_score = grid_fit.score(X_train, y_train)
    test_score = grid_fit.score(X_test, y_test)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))

    model_name = str(model).split('(')[0]
    end = time.time()

    print(f"The best parameters for {model_name} model is: {grid_fit.best_params_}")
    print("--" * 10)
    print(f"(R2 score) in the training set is {train_score:0.2%} for {model_name} model.")
    print(f"(R2 score) in the testing set is {test_score:0.2%} for {model_name} model.")
    print(f"RMSE is {RMSE:,} for {model_name} model.")
    print("--" * 10)
    print(f"Runtime of the program is: {end - start:0.2f}")

    return train_score, test_score, RMSE


def prediction(df_final=None):
    X = df_final.drop(columns=['price', 'id', 'date', 'zipcode'])
    y = df_final['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # print(f"shape of x train: {X_train.shape}")
    # print(f"shape of y train: {y_train.shape}")
    # print(f"shape of x test: {X_test.shape}")
    # print(f"shape of y train: {y_test.shape}")
    lr = LinearRegression(n_jobs=-1)
    lr_train_score, lr_test_score, lr_RMSE = parameter_finder(lr, {})

    ridge = Ridge(random_state=1)  # Linear least squares with l2 regularization.
    param_ridge = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
    ridge_train_score, ridge_test_score, ridge_RMSE = parameter_finder(ridge, param_ridge)

    lasso = Lasso(random_state=1)  # Linear Model trained with L1 prior as regularizer.
    param_lasso = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
    lasso_train_score, lasso_test_score, lasso_RMSE = parameter_finder(lasso, param_lasso)

    eln = ElasticNet(random_state=1)  # Linear regression with combined L1 and L2 priors as regularizer.
    param_eln = {'alpha': [0.001, 0.01, 0.1, 1, 10],
                 'l1_ratio': [0.3, 0.4, 0.5, 0.6, 0.7]}
    eln_train_score, eln_test_score, eln_RMSE = parameter_finder(eln, param_eln)

    dtr = DecisionTreeRegressor(random_state=1)
    param_dtr = {'min_samples_split': [2, 3, 4, 5],
                 'min_samples_leaf': [1, 2, 3]}
    dtr_train_score, dtr_test_score, dtr_RMSE = parameter_finder(dtr, param_dtr)

    rfr = RandomForestRegressor(random_state=1, n_jobs=-1)
    param_rfr = {'min_samples_split': [2, 3, 4, 5],
                 'min_samples_leaf': [1, 2, 3]}
    rfr_train_score, rfr_test_score, rfr_RMSE = parameter_finder(rfr, param_rfr)

    knr = KNeighborsRegressor(n_jobs=-1)
    param_knr = {'n_neighbors': [5, 10, 15, 20],
                 'weights': ['uniform', 'distance']}
    knr_train_score, knr_test_score, knr_RMSE = parameter_finder(knr, param_knr)

    xgboost = XGBRegressor(n_jobs=-1)
    param_xgboost = {'n_estimators': [100, 300],
                     'learning_rate': [0.1, 0.05],
                     'subsample': [0.75],
                     'colsample_bytree': [1],
                     'max_depth': [3, 4, 5],
                     'gamma': [0]}
    xgboost_train_score, xgboost_test_score, xgboost_RMSE = parameter_finder(xgboost, param_xgboost)

    models_score = pd.DataFrame({'Training score': [lr_train_score, ridge_train_score, lasso_train_score,
                                                    eln_train_score, dtr_train_score, rfr_train_score, knr_train_score,
                                                    xgboost_train_score],
                                 'Testing score': [lr_test_score, ridge_test_score, lasso_test_score, eln_test_score,
                                                   dtr_test_score, rfr_test_score, knr_test_score, xgboost_test_score],
                                 'RMSE': [lr_RMSE, ridge_RMSE, lasso_RMSE, eln_RMSE, dtr_RMSE, rfr_RMSE, knr_RMSE,
                                          xgboost_RMSE]},
                                index=['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'DecisionTreeRegressor',
                                       'RandomForestRegressor', 'KNeighborsRegressor', 'XGBoostRegressor'])
    models_score

    fig, ax = plt.subplots(figsize=(20, 10))
    sns.set(style='white')
    ax.set_title("Camparison", fontsize=20)
    ax = sns.barplot(x=list(models_score.index), y=models_score['RMSE'] / 1000000000, alpha=0.7, palette='Greens_r')
    ax.set_ylabel("RMSE\n(billion tomans)", fontsize=20)
    sec_ax = ax.twinx()
    sec_ax = sns.lineplot(x=list(models_score.index), y=models_score['Training score'], linewidth=3, color='blue')
    sec_ax = sns.scatterplot(x=list(models_score.index), y=models_score['Training score'], s=200)
    sec_ax = sns.lineplot(x=list(models_score.index), y=models_score['Testing score'], linewidth=3, color='red')
    sec_ax = sns.scatterplot(x=list(models_score.index), y=models_score['Testing score'], s=200)
    sec_ax.set_ylabel("R2 scores", fontsize=20)

    sec_ax.legend(labels=['Training score', 'Testing score'], fontsize=20)
    sns.despine(offset=10)
    plt.show()


# X= df.drop(columns = ['price','id', 'date', 'zipcode','city','lat','long','sqft_living15','yr_built','yr_renovated','waterfront','condition','grade','view'])
X = df[
    ['age', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement', 'sqft_lot15',
     'sqft_living15', 'grade', 'condition']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
rfr = RandomForestRegressor(random_state=1, n_jobs=-1)
param_rfr = {'min_samples_split': [2, 3, 4, 5],
             'min_samples_leaf': [1, 2, 3]}
rfr_train_score, rfr_test_score, rfr_RMSE, ll = parameter_finder(rfr, param_rfr)

print('{}-{}-{}'.format('{:.2f}'.format(rfr_train_score), '{:.2f}'.format(rfr_test_score), '{:.2f}'.format(rfr_RMSE)))

from dash import Dash, dcc, html, Input, Output, State
#from jupyter_dash import JupyterDash

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div(dcc.Input(id='input_id1', type='text')),
    html.Button('Submit', id='btn_id1', n_clicks=0),
    html.Div(id='output_id1',
             children='Enter a value and press submit')
])


@app.callback(
    Output('output_id1', 'children'),
    Input('btn_id1', 'n_clicks'),
)
def update_output(n_clicks):
    return 'The input value was "{}" and the button has been clicked {} times'.format(
        'vvvvv',
        n_clicks
    )


if __name__ == '__main__':
    app.run_server(debug=True)


my_string = """This is the first line 
This is the second line
third -
\********************
{} """.format('Click Compare button!')
print(my_string)