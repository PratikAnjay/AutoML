import pandas as pd
#import pickle
import streamlit as st 
import base64
def main():
   
    #st.title("Personal Loan Authenticator")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Automatic Machine Learning </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    from PIL import Image
    image_loan=Image.open("ml.jpg")
    st.sidebar.title("Pick Your Algorithm") 
    choose_model=st.sidebar.selectbox(label=' ',options=['Random Forest','Logistic Regression'])
    #st.sidebar.title("Auto Machine Learning") 
    st.sidebar.image(image_loan,use_column_width=True)
    if (choose_model=='Random Forest'):
        file_upload=st.file_uploader("Upload input csv file for Predictions",type=["csv"])
        try:
         if file_upload is not None:
            f1=pd.read_csv(file_upload)
            f1.isna().sum()
            f1=f1.dropna()
            X=f1[['age','previous_year_rating','length_of_service','KPI_Met','awards_won']]
            y=f1['is_promoted']
            d2=f1[['age','previous_year_rating','length_of_service','KPI_Met','awards_won','is_promoted']]
            if st.checkbox('Show Input Data'):
                st.write(d2)
            from sklearn.model_selection import train_test_split
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
            from sklearn.ensemble import RandomForestClassifier
            classifier=RandomForestClassifier()
            classifier.fit(X_train, y_train)
            
            y_pred=classifier.predict(X_test)
            from sklearn.metrics import accuracy_score
            score=accuracy_score(y_test,y_pred)
            from sklearn.metrics import confusion_matrix
            c1=confusion_matrix(y_test,y_pred)
            
            
            
            #st.write("Confusion Matrix : ", c1)
            st.write("Model Accuracy : ", score)
            
            from sklearn.model_selection import cross_val_score
            cv=cross_val_score(classifier,X_train, y_train,cv=5,scoring='accuracy')
            st.write("Cross Calidation of Model : ", cv)
            
            import matplotlib.pyplot as plt
            from sklearn import metrics
            y_pred_proba = classifier.predict_proba(X_test)[::,1]
            fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
            auc = metrics.roc_auc_score(y_test, y_pred_proba)
            plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
            plt.legend(loc=4)
            st.pyplot(plt)
            
            
            
            file_upload=st.file_uploader("Upload csv file for Predictions",type=["csv"])
        #except:
            #st.header("An Error occured")
            
        
        #try:
         
         if file_upload is not None:
            data=pd.read_csv(file_upload)
            data=data[['age','previous_year_rating','length_of_service','KPI_Met','awards_won']]
            data=data.dropna()
            #st.write(data)
            predictions=classifier.predict(data)
            data['Prediction'] = predictions
            st.subheader("Find the Predicted Results below :")
            st.write(data)
            st.text("0 : Not Eligible for Promotion")
            st.text("1 : Eligible for Promotion")
            
        #except:
            #st.header("An Error occured")
            
            
            csv = data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
            href = f'<a href="data:file/csv;base64,{b64}">Download The Prediction Results CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
            st.markdown(href, unsafe_allow_html=True)
            display_df = st.checkbox(label='Visualize the Predicted Value')
            
            if display_df:
                st.bar_chart(data['Prediction'].value_counts())
                st.text(data['Prediction'].value_counts())  
            
            
            
            hide_st_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
"""
            st.markdown(hide_st_style, unsafe_allow_html=True)
            
        except:
            st.header("An Error occurred")


    if (choose_model=='Logistic Regression'):
        file_upload=st.file_uploader("Upload Input csv file ",type=["csv"])
        try:
         if file_upload is not None:
            f1=pd.read_csv(file_upload)
            f1.isna().sum()
            f1=f1.dropna()
            X=f1[['age','previous_year_rating','length_of_service','KPI_Met','awards_won']]
            y=f1['is_promoted']
            d1=f1[['age','previous_year_rating','length_of_service','KPI_Met','awards_won','is_promoted']]
            from sklearn.model_selection import train_test_split
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
            if st.checkbox('Show Input Data'):
                st.write(d1)
            from sklearn.linear_model import LogisticRegression
            classifier = LogisticRegression()
            classifier.fit(X_train, y_train)
            y_pred=classifier.predict(X_test)
            from sklearn.metrics import accuracy_score
            score=accuracy_score(y_test,y_pred)
            from sklearn.metrics import confusion_matrix
            c1=confusion_matrix(y_test,y_pred)
            
            
            st.write("Confusion Matrix : ", c1)
            st.write("Model Accuracy : ", score)
            
            from sklearn.model_selection import cross_val_score
            cv=cross_val_score(classifier,X_train, y_train,cv=5,scoring='accuracy')
            st.write("Cross Calidation of Model : ", cv)
            
            import matplotlib.pyplot as plt
            from sklearn import metrics
            y_pred_proba = classifier.predict_proba(X_test)[::,1]
            fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
            auc = metrics.roc_auc_score(y_test, y_pred_proba)
            plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
            plt.legend(loc=4)
            st.pyplot(plt)
            
            file_upload=st.file_uploader("Upload csv file for Predictions",type=["csv"])
        #except:
            #st.header("An Error occured")
            
            
            
        #file_upload=st.file_uploader("Upload csv file for Predictions",type=["csv"])
        #try:
         if file_upload is not None:
            data=pd.read_csv(file_upload)
            data=data[['age','previous_year_rating','length_of_service','KPI_Met','awards_won']]
            
            data=data.dropna()
            predictions=classifier.predict(data)
            data['Prediction'] = predictions
            st.subheader("Find the Predicted Results below :")
            st.write(data)
            st.text("0 : Not Eligible for Promotion")
            st.text("1 : Eligible for Promotion")
            
        #except:
            #st.header("An Error occurred")
            
            
            csv = data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
            href = f'<a href="data:file/csv;base64,{b64}">Download The Prediction Results CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
            st.markdown(href, unsafe_allow_html=True)
            display_df = st.checkbox(label='Visualize the Predicted Value')
            
            if display_df:
                st.bar_chart(data['Prediction'].value_counts())
                st.text(data['Prediction'].value_counts())  
            
            
            
            hide_st_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
"""
            
            st.markdown(hide_st_style, unsafe_allow_html=True)
            
        except:
            st.header("An Error occurred")
    
        
         
            
                      
if __name__=='__main__':
    main()
    
