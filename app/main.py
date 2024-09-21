import streamlit as st
import pickle
import pandas as pd
import torch
from torch import nn
import plotly.graph_objects as go
import numpy as np


def get_clean_data():
    data=pd.read_csv("data/data.csv")
    data["diagnosis"]=data['diagnosis'].map({'M':1,'B':0})
    data=data.drop(['id'],axis=1)
    return data

def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    data=get_clean_data()

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict={}
    for label,key in slider_labels:
        input_dict[key]=st.sidebar.slider(
            label=label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    return input_dict

def get_scaled_values(input_dict):
    data=get_clean_data()
    X=data.drop(["diagnosis"],axis=1)
    scaled_dict={}
    for key,val in input_dict.items():
        max_val=X[key].max()
        min_val=X[key].min()
        scaled_value=(val-min_val)/(max_val-min_val)
        scaled_dict[key]=scaled_value
    return scaled_dict
        
def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)
    categories = ['Radius', 'Texture', 'Perimeter', 'Area','Smoothness', 'Compactness', 'Concavity', 'Concave Points','Symmetry', 'Fractal Dimension']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
            r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
            ],
            theta=categories,
            fill='toself',
            name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
            r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
            ],
            theta=categories,
            fill='toself',
            name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
            r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
            ],
            theta=categories,
            fill='toself',
            name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
        showlegend=True
    )
    
    return fig
    
def add_predictions(input_data):
    scaler=pickle.load(open("model/scaler.pkl","rb"))

    class CancerModel(nn.Module):
        def __init__(self, input_features, output_features, hidden_units=8):
            super().__init__()
            self.linear_layer_stack = nn.Sequential(
                nn.Linear(in_features=input_features, out_features=hidden_units),
                nn.ReLU(), 
                nn.Linear(in_features=hidden_units, out_features=hidden_units),
                nn.ReLU(),
                nn.Linear(in_features=hidden_units, out_features=output_features),
            )
        def forward(self, x):
            return self.linear_layer_stack(x)
    
    model = CancerModel(input_features=30, output_features=1, hidden_units=8)
    model.load_state_dict(torch.load("model/saved_model.pt",weights_only=True))
    model.eval()
    
    # temp1=get_clean_data()
    # temp2=temp1.drop(["diagnosis"],axis=1).iloc[[21]]
    # st.write(temp1)
    # st.write(temp2)
    # input_array = temp2.to_numpy().reshape(1, -1)

    input_array=np.array(list(input_data.values())).reshape(1,-1)
    input_array_scaled=scaler.transform(input_array)
    input_tensor = torch.from_numpy(input_array_scaled).float()
    with torch.inference_mode():
        logits = model(input_tensor).squeeze()
        pred_prob = torch.sigmoid(logits)
        threshold=0.5
        prediction = (pred_prob>=threshold).float()
    prediction=int(prediction)

    # st.write("Scaled: ",input_array_scaled)
    # st.write("Unscaled: ",input_array)

    st.subheader("Cell Cluster Prediction")
    st.write("The Cell Cluster is:")
    if prediction==0:
        st.write("Benign")
    else:
        st.write("Malicious")
    st.write("Probability of being Benign: ",round(float(1-pred_prob),4))
    st.write("Probability of being Malicious: ",round(float(pred_prob),4))
    st.write("This application is designed to assist healthcare professionals and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for any medical concerns.")


def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    input_data=add_sidebar()
    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("This application leverages machine learning to predict whether breast cancer cells are malignant (cancerous) or benign (non-cancerous). Using a dataset of breast cancer cell features—such as cell size, shape, and texture—the model analyzes the input data and classifies it into one of two categories: malignant or benign. The goal is to assist healthcare professionals by providing a reliable, automated diagnostic tool that can support early detection of breast cancer, improving patient outcomes. The user-friendly interface allows for easy input of patient data and delivers accurate, real-time predictions.")
    col1,col2=st.columns([4,1])
    with col1:
        radar_chart=get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)





if __name__== "__main__":
    main()