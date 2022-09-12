import streamlit as st
import pickle
import numpy as np

# import the model
pipe = pickle.load(open('laptop_pipe.pkl','rb'))
a = pickle.load(open('laptop_dataframe.pkl','rb'))

st.title("Bishal Laptop Predictor Website")

# brand
company = st.selectbox('Brand',a['Company'].unique())

# type of laptop
type = st.selectbox('Type',a['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS',['No','Yes'])

# screen size
screen_size = st.number_input('Screen Size')

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('CPU',a['Processor'].unique())

hdd = st.selectbox('HDD(in GB)',a['HDD'])

ssd = st.selectbox('SSD(in GB)',a['SSD'])

gpu = st.selectbox('GPU',a['GPUs'].unique())

os = st.selectbox('OS',a['OpSys'].unique())

if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company,type,ram,os,weight,touchscreen,ips,ppi,cpu,gpu,hdd,ssd])

    query = query.reshape(1,12)
    st.title("The predicted price of this Laptop is " + str(int(np.exp(pipe.predict(query)[0]))))