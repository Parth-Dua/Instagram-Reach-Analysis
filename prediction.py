import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score , mean_squared_error , mean_absolute_error , r2_score

# ML classifier Python modules
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression


def app(df):

	st.title("Predict Impressions !")
	selected_feat = st.sidebar.multiselect("Select the features or independent variables : " ,df.columns[1:-2] )
	feature_values = []
	for i in selected_feat:
		feature_values.append(st.sidebar.slider(i , int(df[i].min()) , int(df[i].max()) ))
	try:
		
		X = df[selected_feat]
		y = df.iloc[:,0]
		X_train,X_test,y_train,y_test = train_test_split(X,y , test_size = 0.3 , random_state = 42 )


		
		def linear_prediction(features):
			global model
			model = LinearRegression()
			model.fit(X_train,y_train)	
			
			y_pred = model.predict([features])
			st.markdown(f'<p style = color:red;font-size:20px>The predicted impressions are {round(y_pred[0])} according to Linear Regression model</p>',unsafe_allow_html = True)
			st.markdown(f'<p style = color:red;font-size:20px>The Linear Regression model accuracy rate is {round(model.score(X_train ,y_train)*100 ,2)}</p>',unsafe_allow_html = True)
			return model


		
		def svr(features):

			global kernel,c,gamma
			kernel  = st.selectbox("Kernel:"  , ['linear', 'poly', 'rbf', 'sigmoid'])
			c = st.slider('Regularisation : ' , 1 , 10)
			gamma= st.selectbox("gamma value :"  , ['scale' , 'auto'])
			model = SVR( kernel = kernel , C = c , gamma = gamma )
			model.fit(X_train , y_train )

			y_pred = model.predict([features])

			st.markdown(f'<p style = color:red;font-size:20px>The predicted impressions are {round(y_pred[0])} according to Support Vector Regressor model</p>',unsafe_allow_html = True)
			st.markdown(f'<p style = color:red;font-size:20px>The Support Vector Regressor model accuracy rate is {round(model.score(X_train ,y_train)*100 ,2)}</p>',unsafe_allow_html = True)

			return model
		
		def rfr(features):

			
			global est,depth,min_samples
			est = st.slider("Enter number of trees" , 5,500 , step =10)
			depth = st.slider("Enter maximum depth of the decision tree" , 5, 500 , step =10)
			min_samples = 	st.slider("Enter minimum number of samples required to split an internal node " ,  5, 50 , 10)


			model = RandomForestRegressor(n_estimators = est  ,max_depth  = depth , min_samples_split = min_samples)
			model.fit(X_train , y_train )

			y_pred = model.predict([features])

			st.markdown(f'<p style = color:red;font-size:20px>The predicted impressions are {round(y_pred[0])} according to Random Forest Regressor model</p>',unsafe_allow_html = True)
			st.markdown(f'<p style = color:red;font-size:20px>The Random Forest Regressor model accuracy rate is {round(model.score(X_train ,y_train)*100 ,2)}</p>' ,unsafe_allow_html = True)

			return model

			
		st.subheader("Linear Regression")
		with st.expander(label = 'Linear Regression Model'):
			linear_prediction(feature_values)


		st.subheader("Support Vector Regressor")
		with st.expander(label = 'Support Vector Regressor'):
			svr(feature_values)

		st.subheader("Random Forest Regressor")
		with st.expander(label = 'Random Forest Regressor'):
			rfr(feature_values)

		st.subheader("Models Evaluation")
		with st.expander("Evaluate and Compare the models "):

			lin_reg = LinearRegression()
			lin_reg.fit(X_train,y_train)	

			rfr1 = RandomForestRegressor(n_estimators = est  ,max_depth  = depth , min_samples_split = min_samples)
			rfr1.fit(X_train , y_train )
			
			svr1 = SVR( kernel = kernel , C = c , gamma = gamma )
			svr1.fit(X_train , y_train)

			y_pred_l = lin_reg.predict(X_train) 
			y_pred_s = svr1.predict(X_train)
			y_pred_r = rfr1.predict(X_train)

			col1, col2, col3 = st.columns(3)

			with col1:
				st.subheader('Support Vector Regressor')

				st.write(f'R squared value : {round(r2_score(y_pred_s,y_train), 2)}')

				st.write(f"Mean Squared Error : {round(mean_squared_error(y_train , y_pred_s) ,2)}")

				st.write("Residual Analysis : ")
				
				residual = y_train - y_pred_s
				plt.xlabel("Real Values")
				plt.ylabel("predicted values")
				sns.distplot(residual)
				st.pyplot()
				sns.scatterplot(y_train,y_pred_s)
				plt.plot(y_train,y_train)
				plt.xlabel("Real Values")
				plt.ylabel("predicted values")
				st.pyplot()

			with col2:
				st.subheader('Linear Regression Model')

				st.write(f'R squared value : {round(r2_score(y_pred_l,y_train),2)}')

				st.write(f"Mean Squared Error : {round(mean_squared_error(y_train , y_pred_l) ,2)}")

				st.write("Residual Analysis : ")
				
				residual = y_train - y_pred_l
				plt.xlabel("Real Values")
				plt.ylabel("predicted values")
				sns.distplot(residual)
				st.pyplot()
				sns.scatterplot(y_train,y_pred_l)
				plt.plot(y_train,y_train)
				plt.xlabel("Real Values")
				plt.ylabel("predicted values")
				st.pyplot()

			with col3:
				st.subheader('Random Forest Regressor')

				st.write(f'R squared value : {round(r2_score(y_pred_r,y_train),2)}')

				st.write(f"Mean Squared Error : {round(mean_squared_error(y_train , y_pred_r) ,2)}")

				st.write("Residual Analysis : ")
				
				residual = y_train - y_pred_r
				plt.xlabel("Real Values")
				plt.ylabel("predicted values")
				sns.distplot(residual)
				st.pyplot()
				sns.scatterplot(y_train,y_pred_r)
				plt.plot(y_train,y_train)
				plt.xlabel("Real Values")
				plt.ylabel("predicted values")
				st.pyplot()


			st.markdown(f'<p style = color:green;font-size:20px>The best model is Random Forest Classifier because it has no homoscadesiticity, has least mean squared error and has highest value of correlation coefficient squared .</p>',unsafe_allow_html = True)



	except:
		pass