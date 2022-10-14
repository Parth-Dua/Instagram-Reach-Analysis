import numpy as np
import pandas as pd
import streamlit as st
import home_page,view_dataset,visualise
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import seaborn as sns


def app(df):
	st.title('Visualise Dataframe')

	st.subheader('Percentage of impressions from various sources')

	with st.expander(label = "Pie Chart of sources with respect to Impresssions"):
		d = {}
		for i in df.columns:
			if i[:4] == 'From':
				d[i] = df[i].sum()
		st.set_option('deprecation.showPyplotGlobalUse', False)
		plt.pie(list(d.values()) , labels=  list(d.keys()) , shadow = True ,autopct='%1.1f%%')
		st.pyplot()
		st.markdown(" <b style = color:red;font-size:20px>Conclusion </b>:<p style = color:green;font-size:20px> Most impresssions on Instagram come from people sitting at home which is closely followed from hastags. </p>",unsafe_allow_html = True )



	st.subheader("Relationsip between variables")
	with st.expander(label = "Scatterplot and Line plot "):
		f1 = st.sidebar.selectbox("Select the first feature for correlation" , list(df.columns))
		f2 = st.sidebar.selectbox("Select the second feature for correlation ", list(df.columns))

		col1 , col2 = st.columns(2)

		with col1:
			st.write('Scatterplot ')
			st.set_option('deprecation.showPyplotGlobalUse', False)
			sns.scatterplot(data  = df , x = f1 , y = f2)
			st.pyplot()

		with col2:
			st.write('		Line Plot ')
			st.set_option('deprecation.showPyplotGlobalUse', False)
			plt.plot(df[f1], df[f2])
			
			st.pyplot()

		c = df.corr().loc[f1, f2]

		a= 'not'
		if c>0.7	: a = 'very strong positive' 
		elif c>0.4 and c<0.69 : a = 'strong positive'
		elif c>0.3 and c<0.39 : a = 'Moderate positive'
		elif c>0.2 and c<0.29 : a = 'weak positive'
		elif c>0.1 and c<0.19 : a = 'negligible'
		elif c==0 : a = 'not'
		elif c>-0.19 and c<-0.1 : a = 'negligible'
		elif c>-0.29 and c<-0.2 : a = 'weak negative'
		elif c>-0.39 and c<-0.3 : a = 'Moderate negative'
		elif c>-0.69 and c<-0.40 : a = 'Strong negative'
		elif c<-0.7 : a = 'Very strong negative'

		st.markdown(f"<b style = color:red;font-size:20px>Conclusion </b>: <p style = color:green;font-size:20px>{f1} and {f2} are {a} correlated with correlation coefficient of {round(c,2)}</p>"  ,unsafe_allow_html = True)


	st.subheader("See distributions of various sources of Impresssions")
	with st.expander(label = "Histogram"):
		f = st.sidebar.selectbox("Select the feature for distribution" , list(df.columns)[:-2])
		plt.figure(figsize = (13,10))
		st.set_option('deprecation.showPyplotGlobalUse', False)
		sns.distplot(df[f] )
		st.pyplot()

		plt.figure(figsize = (13,10))

		st.set_option('deprecation.showPyplotGlobalUse', False)
		sns.boxplot(df[f] )
		st.pyplot()

		st.markdown(f"<b style = color:red;font-size:20px>Conclusion </b>: <p style = color:green;font-size:20px>1.The IQR range (75th percentile - 25th percentile ) is {df[f].describe()['75%'] - df[f].describe()['25%']}</p>"  ,unsafe_allow_html = True)
		st.markdown(f"<p style = color:green;font-size:20px>2.The distribution is approximately normal. So we can use the Central limit Theorem to make predictions for the whole population with this sample.</p>"  ,unsafe_allow_html = True)
		st.markdown(f"<p style = color:green;font-size:20px>3.The skewness of '{f}' is {round(3*(df[f].describe()['mean'] - df[f].describe()['50%'])/df[f].describe()['std'] , 2)}</p>"  ,unsafe_allow_html = True)

	st.subheader("See The most frequently used hashtags ")
	with st.expander(label = 'Hastags distribution'):
		text = " ".join(i for i in df.Hashtags)
		stopwords = set(STOPWORDS)
		wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
		plt.imshow(wordcloud, interpolation='bilinear')
		plt.axis("off")
		st.pyplot()