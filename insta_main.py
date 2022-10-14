import numpy as np
import pandas as pd
import streamlit as st
import home_page,view_dataset,visualise,prediction
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from random import randint
from streamlit_option_menu import option_menu

st.set_page_config(page_title = 'InstagramReachAnalysis' , page_icon = 'random' , layout = "centered"  ,initial_sidebar_state = 'auto')  

pages_dict = {"Home" : home_page , "View Data": view_dataset , "Visualise Data": visualise , "Predict" : prediction} 

df = pd.read_csv('Instagram_data.csv')


page_selected = option_menu(menu_title = "Main Menu" , options = list(pages_dict.keys()) , default_index = 0 , orientation = 'horizontal')
pages_dict[page_selected].app(df)

def aggrid_interactive_table(df: pd.DataFrame):

    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True
    )

    options.configure_side_bar()

    options.configure_selection("single")
    selection = AgGrid(
        df,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        theme="material",
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
    )


