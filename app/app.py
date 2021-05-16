import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import dash
import dash_table
import dash_daq as daq
import pandas as pd
import plotly.express as px


external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets = external_stylesheets)

modal_clicks = 0

df = pd.read_pickle('df_app.pkl')

def find_products(data_frame: pd.DataFrame, 
                  branded_food_cat: str, 
                  nutrient_prefs: list, 
                  desc_kw: str, 
                  ingred_kw: str) -> pd.DataFrame:

    """This function filters out products that don't match the provided preferences"""

    data_frame = data_frame[data_frame.branded_food_category == branded_food_cat] # filter the food category

    for nutrient_condition in nutrient_prefs: # nutrient conditions are provided in a tuple format: (<nutrient name>, <min or max>, <amount>)
        
        if nutrient_condition[1] == 'max':
            data_frame = data_frame[data_frame['nutr_amnt'].apply(lambda x: x[nutrient_condition[0]][0] <= nutrient_condition[2] if nutrient_condition[0] in x.keys() else False)]
        
        elif nutrient_condition[1] == 'min':
            data_frame = data_frame[data_frame['nutr_amnt'].apply(lambda x: x[nutrient_condition[0]][0] >= nutrient_condition[2] if nutrient_condition[0] in x.keys() else False)]
    
    if desc_kw is not None: # keeping only the products that contain the provided keyword in their description
        data_frame = data_frame[data_frame['description'].apply(lambda x: desc_kw in x.lower() if not pd.isnull(x) else False)]

    if ingred_kw is not None: # keeping only the products that contain the provided keyword in their ingredients list
        data_frame = data_frame[data_frame['ingredients'].apply(lambda x: ingred_kw in x.lower() if not pd.isnull(x) else False)]
    
    return data_frame

# Creating a dictionary with the Main Categories as keys and their lists of Sub-Categories as values
all_options = {}
for i in df.MainCategory.dropna().unique():
    all_options[i] = [j for j in df[df.MainCategory == i]['SubCategory'].unique()]

# Creating a dictionary with the Sub-Categories as keys and their lists of Categories as values
all_options_sub = {}
for i in df.SubCategory.dropna().unique():
    all_options_sub[i] = [j for j in df[df.SubCategory == i]['branded_food_category'].unique()]

# The above dictionaries are used in the callbacks of the category dropdown menus.


app.layout = html.Div(
    [  
        dbc.Row( # The first row of the layout, contains the image of the logo and the header.
            [

                dbc.Col(
                    html.Div(
                        [
                            html.Img(src='/assets/HeaderB.png')
                        ]
                    ),
                    width={'size':12, 'offset':0},
                )
            ]
        ),

        dbc.Row( # The second row of the layout contains the images of the instructions.
            [
                dbc.Col(
                   html.Div( 
                        [
                            html.Img(src='/assets/BannerA.jpg'),
                        ]
                    ),
                    width={'size':2, 'offset':0}
                ),

                dbc.Col(
                    html.Div(
                        [
                            html.Img(src='/assets/BannerB.jpg')
                            
                        ]
                    ),
                    width={'size':2, 'offset':4}
                )
            ]
        ),

        dbc.Row( #The third row contains all the inputs, plus the "See Results" button at the end.
            [
                dbc.Col(
                    html.Div(
                        [
                            html.H5('Menu'),
                            dcc.RadioItems( # Radio items input for the Main categories.
                                id='radio_menu',
                                options=[{'label': k, 'value': k} for k in all_options.keys()],
                                labelStyle={'display': 'block','margin-left':'20px'},
                                inputStyle={"margin-right": "10px"}
                            )
                        ],
                        style={'background-color':'#eaeec6'}
                    ),
                    width={"size": 2, "order": 1, "offset": 0},    
                ),

                dbc.Col(
                    html.Div(
                        [
                            html.H5('Select Category'),
                            dcc.Dropdown(
                                id='cat_dropdown', 
                                options=[{'label': i, 'value': i} for i in df['SubCategory'].dropna().unique()]
                                # The available options are changed after a Main Category is selected. See first callback.
                            ),
                            html.Br(),
                            html.Br(),

                            html.H5('Select Sub-Categoty'),
                            dcc.Dropdown(
                                id='subcat_dropdown', 
                                options=[{'label': i, 'value': i} for i in df['branded_food_category'].dropna().unique()]
                                # The available options are changed after a Sub-Category is selected. See second callback.
                            ),
                            html.Br(style={'margin': '4px'})
                        ],
                        style={'background-color':'#eaeec6'}
                    ),
                    width={"size": 2, "order": 2, "offset": 0},
                ),

                dbc.Col(
                    html.Div(
                        [
                            html.H5(id='table_title'),
                            html.Table( # A table containing some statistics for the selected category. Controlled by the second callback.
                                id='cat_stats',
                                style={'border': '1px solid brown', 'background':'#eaeec6'}
                            ),
                            html.Br()
                        ],
                        style={'background-color':'#eaeec6'}
                    ),
                    width={'size':2, 'order':3}
                ),

                dbc.Col(
                    html.Div( # Contains three dropdown menus for nutrient preferences.
                        [
                            html.H5('Filter Nutrients'),

                            dcc.Dropdown(
                                id='dropdown_nutrient', 
                                options=[
                                    {'label': 'Calories', 'value': 'Energy'},
                                    {'label': 'Sugars', 'value': 'Sugars, total including NLEA'},
                                    {'label': 'Fat', 'value': 'Total lipid (fat)'},
                                    {'label': 'Protein', 'value': 'Protein'},
                                    {'label': 'Fiber', 'value': 'Fiber, total dietary'},
                                    {'label': 'Folic acid', 'value': 'Folic acid'}
                                ]
                            ),

                            html.Br(),

                            dcc.Dropdown(
                                id='dropdown_nutrient_2', 
                                options=[
                                    {'label': 'Calories', 'value': 'Energy'},
                                    {'label': 'Sugars', 'value': 'Sugars, total including NLEA'},
                                    {'label': 'Fat', 'value': 'Total lipid (fat)'},
                                    {'label': 'Protein', 'value': 'Protein'},
                                    {'label': 'Fiber', 'value': 'Fiber, total dietary'},
                                    {'label': 'Folic acid', 'value': 'Folic acid'}
                                ]
                            ),

                            html.Br(),

                            dcc.Dropdown(
                                id='dropdown_nutrient_3', 
                                options=[
                                    {'label': 'Calories', 'value': 'Energy'},
                                    {'label': 'Sugars', 'value': 'Sugars, total including NLEA'},
                                    {'label': 'Fat', 'value': 'Total lipid (fat)'},
                                    {'label': 'Protein', 'value': 'Protein'},
                                    {'label': 'Fiber', 'value': 'Fiber, total dietary'},
                                    {'label': 'Folic acid', 'value': 'Folic acid'}
                                ]
                            ),

                            html.Br(style={'margin': '3px'})
                        ],
                        style={'background-color':'#eaeec6'}
                    ),
                    width={"size": 1, "order": 4, "offset": 0},
                ),
                
                dbc.Col(
                    html.Div( # Contains three min/max radio items, one for each nutrient dropdown menu.
                        [
                            html.H5('Method'),

                            dcc.RadioItems(
                                id='radio_min_max',
                                options=[
                                    {'label': 'Min', 'value': 'min'},
                                    {'label': 'Max', 'value': 'max'}
                                ],
                                value='min',
                                labelStyle={'display': 'inline-block', 'margin-left':'20px'},
                                inputStyle={"margin-right": "5px"}
                            ),

                            html.Br(style={'margin': '3px'}),

                            dcc.RadioItems(
                                id='radio_min_max_2',
                                options=[
                                    {'label': 'Min', 'value': 'min'},
                                    {'label': 'Max', 'value': 'max'}
                                ],
                                value='min',
                                labelStyle={'display': 'inline-block', 'margin-left':'20px'},
                                inputStyle={"margin-right": "5px"}
                            ),

                            html.Br(style={'margin': '6px'}),

                            dcc.RadioItems(
                                id='radio_min_max_3',
                                options=[
                                    {'label': 'Min', 'value': 'min'},
                                    {'label': 'Max', 'value': 'max'}
                                ],
                                value='min',
                                labelStyle={'display': 'inline-block', 'margin-left':'20px'},
                                inputStyle={"margin-right": "5px"}
                            ),

                            html.Br()                           

                        ],
                        style={'background-color':'#eaeec6'}
                    ),
                    width={"size": 1, "order": 5, "offset": 0},
                ),

                dbc.Col(
                    html.Div( # Contains three numeric inputs, one for each nutrient dropdown.
                        [
                            html.H5('Set amounts'),

                            daq.NumericInput(
                                id='min_max_amount',
                                min=0,
                                max=1000,
                                size=80,
                                value=0
                            ),

                            html.Br(style={'margin': '1px'}),

                            daq.NumericInput(
                                id='min_max_amount_2',
                                min=0,
                                max=1000,
                                size=80,
                                value=0
                            ),

                            html.Br(style={'margin': '1px'}),

                            daq.NumericInput(
                                id='min_max_amount_3',
                                min=0,
                                max=1000,
                                size=80,
                                value=0
                            ),

                            html.Br()

                        ],
                        style={'background-color':'#eaeec6'}
                    ),
                    width={"size": 1, "order": 6, "offset": 0},
                ),

                dbc.Col(
                    html.Div( # Contains the ingredient keyword input and the description keyword input.
                        [
                            html.H5('Ingredient keyword'),

                            dcc.Input(
                                id='ingred_kw'
                            ),

                            html.Br(),
                            html.Br(),
                            html.Br(),
                            html.Br(),

                            html.H5('Description keyword'),

                            dcc.Input(
                                id='desc_kw'
                            ),

                            html.Br(),
                            html.Br()                            
                        ],
                        style={'background-color':'#eaeec6'}
                    ),
                    width={'size':2, 'order':7}
                ),

                dbc.Col(
                    html.Div(
                        [
                            html.Button( 
                                id='button', 
                                n_clicks = 0,
                                style={
                                    'background-color': 'transparent',
                                    'height': '218px',
                                    'width': '175px',
                                    'font-size': '26px'
                                },
                                hidden=True
                            )
                        ],
                        style={'background-image': 'url(/assets/SeeResults.png)'}
                    ),
                    width={'size':1, 'order':8}
                )

            ],
            no_gutters=True, # reduces the space between the columns.
            align='start',
        ),
                
        dbc.Row( # The fourth row contains the presentation of the results: one table and one graph.
            [
                dbc.Col(
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H4('Results'), 
                                    html.H6('(ordered by lowest amount of unfavourable nutrients | nutrient amounts are per 100g)')
                                ]
                            ),

                            dash_table.DataTable( # The table contains all the products that match the provided preferences.
                                id='table_data',
                                style_cell={
                                    'whiteSpace': 'normal',
                                    'height': 'auto',
                                    'textAlign': 'left', 
                                    'border': '1px solid brown'
                                },
                                style_header={
                                    'backgroundColor':'#eaeec6',
                                    'fontWeight': 'bold'
                                },
                                style_table={
                                    'height':'500px', 
                                    'overflowY': 'auto'
                                },
                                style_cell_conditional=[
                                    {'if': {'column_id': 'Ingredients'},
                                     'width': '35%'}
                                ]
                            )
                        ]
                    ),
                    width={'size':6, 'offset':0}
                ),
                dbc.Col(
                    html.Div(
                        [
                            html.H4('Comparison of the Top Results'),

                            html.Br(),

                            dcc.Graph( # The graph compares the top results.
                                id='main_graph')
                        ]
                    ),
                    width={'size':6, 'offset':0}
                ),

                dbc.Modal( # This modal pops up when there are no products that match the given preferences.
                    [
                        dbc.ModalHeader("Alert!"),
                        dbc.ModalBody("No products match the specified filters."),
                        dbc.ModalFooter(
                            dbc.Button("Close", id="close_modal", className="ml-auto", n_clicks=0)
                        ),
                    ],
                    id="modal",
                ),
            ]
            
        )
        
    ],
)

@app.callback( # Adjusts the available options of the Sub-Category dropdown based on the selection of the Main Category.
    Output('cat_dropdown', 'options'),
    Input('radio_menu', 'value')
)

def update_subcat_options(main_cat):

    if not pd.isnull(main_cat):

        return [{'label': i, 'value': i} for i in all_options[main_cat]]

    else:

        return [{'label':j, 'value':j} for j in df.SubCategory.dropna().unique()]

@app.callback( # Adjusts the available options of the Category dropdown based on the selection of the Sub-Category.
    Output('subcat_dropdown', 'options'),
    Input('cat_dropdown', 'value')
)

def update_subcat_options(sub_cat):

    if not pd.isnull(sub_cat):

        return [{'label': i, 'value': i} for i in all_options_sub[sub_cat]]

    else:

        return [{'label':j, 'value':j} for j in df.branded_food_category.dropna().unique()]

@app.callback( # Enables the "See Results" button and fills the category analysis table.
    Output('button', 'hidden'),
    Output('cat_stats', 'children'),
    Output('table_title', 'children'),
    Input('subcat_dropdown', 'value')
)

def enable_button(sbcat_selection):

    if not pd.isnull(sbcat_selection): # Means a category has been selected, therefore the button should be shown, and the table filled.

        # First, preparing the statistics that will be included in the table of the Category analysis.
        df_subcat = df.loc[(df.branded_food_category == sbcat_selection), :]

        df_subcat.loc[:, 'Calories'] = df_subcat['nutr_amnt'].apply(lambda x: x['Energy'][0] if 'Energy' in x.keys() else 0)
        df_subcat.loc[:, 'Protein'] = df_subcat['nutr_amnt'].apply(lambda x: x['Protein'][0] if 'Protein' in x.keys() else 0)
        df_subcat.loc[:, 'Fiber'] = df_subcat['nutr_amnt'].apply(lambda x: x['Fiber, total dietary'][0] if 'Fiber, total dietary' in x.keys() else 0)
        df_subcat.loc[:, 'Sugars'] = df_subcat['nutr_amnt'].apply(lambda x: x['Sugars, total including NLEA'][0] if 'Sugars, total including NLEA' in x.keys() else 0)
        df_subcat.loc[:, 'Fat'] = df_subcat['nutr_amnt'].apply(lambda x: x['Total lipid (fat)'][0] if 'Total lipid (fat)' in x.keys() else 0)
        
        df_subcat = df_subcat[['Calories', 'Protein', 'Fiber', 'Sugars', 'Fat']]
        df_subcat_agg = df_subcat.agg(['mean', 'median', 'std', 'max', 'min']).round(1).reset_index().rename(columns={'index':'stat'})

        # Creating the children of the html.Table element.
        table_children = [
                            html.Tr(
                                [html.Th(col, style={'border': '1px solid brown'}) for col in df_subcat_agg.columns]
                            )
                        ] 
        table_children.extend(
                        [
                            html.Tr(
                                [html.Td(df_subcat_agg.iloc[i][col], style={'border': '1px solid brown'}) for col in df_subcat_agg.columns]
                            ) for i in range(len(df_subcat_agg))
                        ]
        )

        return False, table_children, sbcat_selection

    else: # Means a category has not been selected yet, therefore the button should be hidden. The table is filled with zeros.
        empty_table = df.loc[:2, :]

        empty_table.loc[:, 'Calories'] = empty_table['nutr_amnt'].apply(lambda x: x['Energy'][0] if 'Energy' in x.keys() else 0)
        empty_table.loc[:, 'Protein'] = empty_table['nutr_amnt'].apply(lambda x: x['Protein'][0] if 'Protein' in x.keys() else 0)
        empty_table.loc[:, 'Fiber'] = empty_table['nutr_amnt'].apply(lambda x: x['Fiber, total dietary'][0] if 'Fiber, total dietary' in x.keys() else 0)
        empty_table.loc[:, 'Sugars'] = empty_table['nutr_amnt'].apply(lambda x: x['Sugars, total including NLEA'][0] if 'Sugars, total including NLEA' in x.keys() else 0)
        empty_table.loc[:, 'Fat'] = empty_table['nutr_amnt'].apply(lambda x: x['Total lipid (fat)'][0] if 'Total lipid (fat)' in x.keys() else 0)

        empty_table = empty_table.loc[:,['Calories', 'Protein', 'Fiber', 'Sugars', 'Fat']]
        empty_table = empty_table.agg(['mean', 'median', 'std', 'max', 'min']).round().reset_index().rename(columns={'index':'stat'})

        empty_table_children= [
                            html.Tr(
                                [html.Th(col, style={'border': '1px solid brown'}) for col in empty_table.columns]
                            )
                        ] 
        empty_table_children.extend(
                        [
                            html.Tr(
                                [html.Td(0, style={'border': '1px solid brown'}) for col in empty_table.columns]
                            ) for i in range(len(empty_table))
                        ]
        )

        return True, empty_table_children, 'Category Analysis'

@app.callback( # Main callback, controls the results table, the graph and the modal. It's activated when the "See Results" button is clicked.
    Output('table_data', 'data'),
    Output('table_data', 'columns'),
    Output('main_graph', 'figure'),
    Output('modal', 'is_open'),
    [Input('button', 'n_clicks'), Input('close_modal', 'n_clicks')],
    [
        State('subcat_dropdown', 'value'),
        State('dropdown_nutrient', 'value'),
        State('dropdown_nutrient_2', 'value'),
        State('dropdown_nutrient_3', 'value'),
        State('radio_min_max', 'value'),
        State('radio_min_max_2', 'value'),
        State('radio_min_max_3', 'value'),
        State('min_max_amount', 'value'),
        State('min_max_amount_2', 'value'),
        State('min_max_amount_3', 'value'),
        State('ingred_kw', 'value'),
        State('desc_kw', 'value')
    ]
)

def update_table(n_clicks, n_clicks_close_modal, category, nutr1, nutr2, nutr3,  min_max1, min_max2, min_max3, amnt1, amnt2, amnt3, ingred_kw, desc_kw):
    
    cols = ['description', 'ingredients', 'brand_owner', 'calories', 'sugars', 'fat', 'protein', 'fiber', 'folic_acid', 'bad_nutrients']
    formal_cols = ['Description', 'Ingredients', 'Company', 'Calories (kcal)', 'Sugars (g)',
                    'Fat (g)', 'Protein (g)', 'Fiber (g)', 'Folic acid (μg)', 'Unfavourable nutrients']

    if n_clicks != 0: # The button is clicked.

        # The following if statements check how many of the three nutrient preference inputs have been filled.
        # If the second is filled, then the third is checked. If the third is not filled, only the first two are considered.
        # If the second is not filled, then the first is checked. If neither the first is filled, no nutrient preferences are considered.
        # The trys and excepts are used to catch errors of processing an empty dataframe in case of no matching products.

        if not pd.isnull(nutr2):
            
            if not pd.isnull(nutr3): # All three are filled.
                test = find_products(
                data_frame=df, 
                branded_food_cat = category, 
                nutrient_prefs=[(nutr1, min_max1, amnt1), (nutr2, min_max2, amnt2), (nutr3, min_max3, amnt3)],
                desc_kw=desc_kw,
                ingred_kw=ingred_kw)

                try:
                    test = test[cols]
                    test = test.rename(columns=dict(zip(cols, formal_cols))).sort_values(by='Unfavourable nutrients', ascending=True)

                    data = test.to_dict('records')
                    columns = [{"name": i, "id": i} for i in test.columns]

                except:
                    pass

            else: # First and second are filled.

                test = find_products(
                    data_frame=df,
                    branded_food_cat = category, 
                    nutrient_prefs=[(nutr1, min_max1, amnt1), (nutr2, min_max2, amnt2)],
                    desc_kw=desc_kw,
                    ingred_kw=ingred_kw)

                try:
                    test = test[cols]
                    test = test.rename(columns=dict(zip(cols, formal_cols))).sort_values(by='Unfavourable nutrients', ascending=True)


                    data = test.to_dict('records')
                    columns = [{"name": i, "id": i} for i in test.columns]

                except:
                    pass

        elif not pd.isnull(nutr1): # Only the first is filled.

            test = find_products(
                data_frame=df, 
                branded_food_cat = category, 
                nutrient_prefs=[(nutr1, min_max1, amnt1)],
                desc_kw=desc_kw,
                ingred_kw=ingred_kw)
            try:
                test = test[cols]
                test = test.rename(columns=dict(zip(cols, formal_cols))).sort_values(by='Unfavourable nutrients', ascending=True)


                data = test.to_dict('records')
                columns = [{"name": i, "id": i} for i in test.columns]

            except:
                pass

        else:

            test = df[df.branded_food_category == category]

            try:
                test = test[cols]
                test = test.rename(columns=dict(zip(cols, formal_cols))).sort_values(by='Unfavourable nutrients', ascending=True)


                data = test.to_dict('records')
                columns = [{"name": i, "id": i} for i in test.columns]

            except:
                pass            

        if not len(test) > 0: # No matching products. Modal must pop up.

            modal_open = True

            global modal_clicks

            if n_clicks_close_modal > modal_clicks: # Making the "close" button of the modal functional.

                modal_open = False
                modal_clicks += 1

            return data, columns, {}, modal_open

        n_products = min(5, len(test)) # Comparing 5 products at most.
        avg_cat = df.groupby('branded_food_category')[['calories', 'protein', 'fiber', 'sugars', 'fat', 'folic_acid']].mean()

        # Creating a dictionary that will serve as the source of the visualization dataframe.
        # The 'names' control are prepared in such a way as to be suitable for the plot legend.
        d = {'names': [f'{i}. ' + name + ' - ' + brand for i, name, brand in zip(range(1, n_products+1), test.Description.fillna('-').values[:n_products], test.Company.fillna('-').values[:n_products])] + [category + ' average'],
             'Calories (kcal)': [calorie for calorie in test['Calories (kcal)'].values[:n_products]] + [avg_cat.loc[category, :].calories],
             'Protein (g)': [prot for prot in test['Protein (g)'].values[:n_products]] + [avg_cat.loc[category, :].protein],
             'Fiber (g)': [fiber for fiber in test['Fiber (g)'].values[:n_products]] + [avg_cat.loc[category, :].fiber],
             'Sugars (g)': [sugar for sugar in test['Sugars (g)'].values[:n_products]] + [avg_cat.loc[category, :].sugars],
             'Fat (g)' : [fat for fat in test['Fat (g)'].values[:n_products]] + [avg_cat.loc[category, :].fat],
             'Folic acid (μg)': [folic for folic in test['Folic acid (μg)'].values[:n_products]] + [avg_cat.loc[category, :]['folic_acid']]
             }

        # Creating the visualization dataframe from the previous dictionary.
        # Using pd.melt() to unpivot the data and make it more suitable for the px.bar function.
        viz_df = pd.melt(pd.DataFrame(d), id_vars='names', value_vars=['Calories (kcal)', 'Protein (g)', 'Fiber (g)', 'Sugars (g)', 'Fat (g)', 'Folic acid (μg)'])
        col_seq = ['#333131', '#615F5F', '#878686', '#A29F9F', '#C4C2C2', '#FF6666'] # Color sequences of the barplot.

        fig = px.bar(data_frame= viz_df, x='names', y='value', facet_col='variable', 
                     color='names', labels={'names':'', 'value':''}, 
                     #title= 'Product Comparison' + ' - ' ,
                     facet_col_spacing=0.05, height=600,
                     facet_col_wrap=3, facet_row_spacing=0.15, color_discrete_sequence=col_seq[:n_products]+[col_seq[-1]])

        fig.update_yaxes(matches=None, showticklabels=True)
        fig.update_xaxes(showticklabels=False)
        fig.update_layout(legend = dict(yanchor='top', y=1.1+n_products/7, xanchor='left', x=0))
        fig.for_each_annotation(lambda a: a.update(text=a.text.replace("variable=", "")))

        modal_open = False 

        return data, columns, fig, modal_open

    else: # The initial state of the table and the graph. When the app opens, they are both empty.

        empty_cols = [{"name": i, "id": i} for i in df[cols].rename(columns=dict(zip(cols, formal_cols))).columns]
        d = [dict(zip(formal_cols, ['' for i in range(len(formal_cols))]))]

        return d, empty_cols, {}, False

if __name__ == "__main__":
    app.run_server(debug=True)