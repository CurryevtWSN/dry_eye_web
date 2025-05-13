
#%%load package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import shap
import sklearn
import joblib
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


#%%不提示warning信息
st.set_option('deprecation.showPyplotGlobalUse', False)

#%%set title
st.set_page_config(page_title='A Clinical Decision-Making Model for Dry Eye in Patients with Refractive Errors')
st.title('A Clinical Decision-Making Model for Dry Eye in Patients with Refractive Errors')

#%%set varixgbles selection
st.sidebar.markdown('## Varixgbles')


Age =  st.sidebar.slider("Age (year)", 0,100,value = 65, step = 1)
Sex = st.sidebar.selectbox("Sex", ('Male','Female'), index = 1)
Refractive_error_time = st.sidebar.slider("Refractive error time (year)",
                                          0.00,
                                          80.00
                                          ,value= 10.00, step=1.00)
Conjunctivitis = st.sidebar.selectbox("Conjunctivitis", ('No','Yes'), index = 1)
Cataract = st.sidebar.selectbox("Cataract", ('No','Yes'), index = 1)
Glaucoma = st.sidebar.selectbox("Glaucoma", ('No','Yes'), index = 1)
Strabismus = st.sidebar.selectbox("Strabismus", ('No','Yes'), index = 1)
History_of_ocular_surgery = st.sidebar.selectbox("History_of_ocular_surgery", ('No','Yes'), index = 1)
History_of_allergy = st.sidebar.selectbox("History_of_allergy", ('No','Yes'), index = 1)

#分割符号
st.sidebar.markdown('#  ')
st.sidebar.markdown('#  ')
st.sidebar.markdown('##### All rights reserved') 
st.sidebar.markdown('##### For communication and cooperation, please contact wshinana99@163.com, Shi-Nan Wu, Xiamen University')
#传入数据
map = {'Male':0,
       'Female':1,
       'No':0,
       'Yes':1
}

Sex =map[Sex]
Conjunctivitis =map[Conjunctivitis]
Cataract =map[Cataract]
Glaucoma =map[Glaucoma]
Strabismus =map[Strabismus]
History_of_ocular_surgery =map[History_of_ocular_surgery]
History_of_allergy =map[History_of_allergy]


# 数据读取，特征标注
#%%load model
xgb_model = joblib.load('xgb_model.pkl')

#%%load data
hp_train = pd.read_csv('balanced_data_5000.csv')

target = ["Dry_eye"]
y = np.array(hp_train[target])
sp = 0.5

is_t = (xgb_model.predict_proba(np.array([[Refractive_error_time,Age,Conjunctivitis,Sex,Cataract,
                                           Glaucoma,History_of_ocular_surgery,History_of_allergy,Strabismus]]))[0][1])> sp
prob = (xgb_model.predict_proba(np.array([[Refractive_error_time,Age,Conjunctivitis,Sex,Cataract,
                                           Glaucoma,History_of_ocular_surgery,History_of_allergy,Strabismus]]))[0][1])*1000//1/10
    

if is_t:
    result = 'High Risk Dry Eye Group'
else:
    result = 'Low Risk Dry Eye Group'
if st.button('Predict'):
    st.markdown('## Result:  '+str(result))
    if result == '  Low Risk Dry Eye Group':
        st.balloons()
    st.markdown('## Probability of High Risk Dry Eye Group:  '+str(prob)+'%')
    #%%cbind users data
    features = ["Refractive_error_time", "Age", "Conjunctivitis", "Sex",
                "Cataract", "Glaucoma", "History_of_ocular_surgery",
                "History_of_allergy", "Strabismus"]
    col_names = features
    X_last = pd.DataFrame(np.array([[Refractive_error_time,Age,Conjunctivitis,Sex,Cataract,
                                               Glaucoma,History_of_ocular_surgery,History_of_allergy,Strabismus]]))
    X_last.columns = col_names
    X_raw = hp_train[features]
    X = pd.concat([X_raw,X_last],ignore_index=True)
    if is_t:
        y_last = 1
    else:
        y_last = 0  
    y_raw = (np.array(hp_train[target]))
    y = np.append(y_raw,y_last)
    y = pd.DataFrame(y)
    model = xgb_model
    #%%calculate shap values
    sns.set()
    explainer = shap.Explainer(model, X)
    shap_values = explainer.shap_values(X)
    a = len(X)-1
    #%%SHAP Force logit plot
    st.subheader('SHAP Force logit plot of XGB model')
    fig, ax = plt.subplots(figsize=(12, 6))
    force_plot = shap.force_plot(explainer.expected_value,
                    shap_values[a, :], 
                    X.iloc[a, :], 
                    figsize=(25, 3),
                    # link = "logit",
                    matplotlib=True,
                    out_names = "Output value")
    st.pyplot(force_plot)
    #%%SHAP Water PLOT
    st.subheader('SHAP Water plot of XGB model')
    shap_values = explainer(X) # 传入特征矩阵X，计算SHAP值
    fig, ax = plt.subplots(figsize=(8, 8))
    waterfall_plot = shap.plots.waterfall(shap_values[a,:])
    st.pyplot(waterfall_plot)
    #%%ConfusionMatrix 
    st.subheader('Confusion Matrix of XGB model')
    xgb_prob = xgb_model.predict(X)
    cm = confusion_matrix(y, xgb_prob)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low risk', 'High risk'])
    sns.set_style("white")
    disp.plot(cmap='RdPu')
    plt.title("Confusion Matrix of XGB model")
    disp1 = plt.show()
    st.pyplot(disp1)

