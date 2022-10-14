import numpy as np
import pandas as pd
import streamlit as st
from insta_main import aggrid_interactive_table
import io
def app(df):
	st.title("Display Data ")

	st.subheader("View the dataframe")
	with st.expander(label = "View Dataset"):
		aggrid_interactive_table(df)
	'''
	st.subheader("View Data Description")


	with st.expander(label = "View Dataset Descriptive Statistics"):

		st.write(df.describe())

		st.write("Conclusion : We can conclude that there are highest number of views on instagram and the least number of people are interested in  ")
	'''
	#with col1:
	st.subheader("View Data Description")
	with st.expander(label = "View Dataset Information"):

		st.write(df.describe())
		buffer = io.StringIO()
		df.info(buf=buffer)
		s = buffer.getvalue()

		st.text(s)
		st.info("Conclusions ")
		st.write('1. All the columns in the dataframe are numeric except Caption and Hashtags.')
		st.write("2. There are no null values in the dataframe")
	'''					

	with col2:
		st.subheader("Particular column values")
		column= st.selectbox('Select the columns', list(df.columns))
		st.write(df[column])'''

	st.subheader('Pivot Table')

	ind = st.multiselect("Enter the columns you want to group by row" , list(df.columns)) 
	col = st.multiselect("Enter the columns you want to group by column" , list(df.columns)) 
	if len(ind)!=0 or len(col)!=0 :
		piv_tab = pd.pivot_table(data = df , index = ind ,columns = col , aggfunc = "median" )

		st.write(piv_tab)