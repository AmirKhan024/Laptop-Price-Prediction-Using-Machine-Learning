import streamlit as st
import pickle
import numpy as np

pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title('Laptop Price Prediction')

company = st.selectbox('Select Brand', df['Company'].unique())

type = st.selectbox('Select Type', df['TypeName'].unique())

ram = st.selectbox('Select RAM (in GB)', [2,4,6,8,12,16,24,32,64])

weight = st.number_input('Enter Weight (in kg)', min_value=0.0, max_value=10.0, step=0.1)

touchsreen = st.selectbox('Touchscreen', ['No', 'Yes'])

ips = st.selectbox('IPS', ['No', 'Yes'])

screen_size = st.number_input('Screen Size (in inches)', min_value=0.0, max_value=20.0, step=0.1)

resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800',
'2560x1600','2560x1440','2304x1440'])

cpu = st.selectbox('CPU', df['Cpu_Brand'].unique())

hdd = st.selectbox('HDD (in GB)', [0,128,256,512,1024,2048])

ssd = st.selectbox('SSD (in GB)', [0,128,256,512,1024])

gpu = st.selectbox('GPU', df['Gpu_Brand'].unique())

os = st.selectbox('OS', df['Os'].unique())

if st.button('Predict Price'):
    ppi = None

    if touchsreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = resolution.split('x')[0]
    Y_res = resolution.split('x')[1]
    ppi = (float(X_res) ** 2 + float(Y_res) ** 2) ** 0.5 / screen_size

    query = np.array([company,type,ram,os,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu])
    query = query.reshape(1, 12)
    st.title('Predicted Price : {}'.format(np.round(np.exp(pipe.predict(query)[0]), 2)))