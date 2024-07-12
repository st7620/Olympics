## Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

## Page Config
st.set_page_config(
    page_title="Paris Olympics 2024 Medal Predictions",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded"
    )

alt.themes.enable("dark")

## Load data
df_final = pd.read_csv('Datasets/final.csv')
df_results = pd.read_csv('Datasets/results.csv')
df_predictions = pd.read_csv('Datasets/predictions.csv')

## Sidebar
with st.sidebar:
    st.title('🥇 Paris Olympics 2024 Medal Predictions')

    country_list = list(df_final.Country.unique())

    selected_country = st.selectbox('Select a country', country_list)

## Graphs

# Time Series of Medals
def make_time_series(input_df, input_country):
    # Filter the data
    df_country = input_df.loc[input_df['Country'] == input_country]
    reduced_df = df_country[['Year', 'total']]

    # Create the chart
    chart = px.line(reduced_df, x='Year', y='total', title=f"{input_country}'s Total Medals Over Time")
    
    return chart

# MP Bar Chart

def bar_chart(input_df):
    labels = input_df["Models"]
    RMSE_Results = input_df["RMSE"]
    R2_Results = input_df["R2"]

    rg = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()

    ax.bar(rg, RMSE_Results, width, label="RMSE")
    ax.bar(rg + width, R2_Results, width, label='R2')
    ax.set_xticks(rg + width / 2)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Models")
    ax.set_ylabel("RMSE/R2")
    ax.set_ylim([0, 1])
    ax.set_title('Model Performance')
    ax.legend(loc='upper left', ncol=2)

    return fig   

# Bar Chart of Predictions:
def bar_chart_predictions(input_df):
    reduced_df = input_df[['Country', 'total']]
    reduced_df = reduced_df.sort_values(by="total", ascending=False)

    chart = px.bar(reduced_df, x='Country', y='total', title=f"Medal Predictions")
    
    return chart

# ACF and PACF Plots:
def acf_pacf_plot(input_df, input_country):
    # Filter the data
    df_country = input_df.loc[input_df['Country'] == input_country]
    reduced_df = df_country[['Year', 'total']]

    fig, ax = plt.subplots(2,1, figsize=(12,8))
    plot_acf(reduced_df)
    plot_pacf(reduced_df)
    ax[0].set_title('Autocorrelation Plot')
    ax[1].set_title('Partial Autocorrelation Plot')
    plt.tight_layout()
    return fig

## Dashboard Main Panel:
col = st.columns((2,3,2), gap='medium')

with col[0]:
    st.markdown('#### Total Medals Over Time')

    ts = make_time_series(df_final, selected_country)
    st.plotly_chart(ts, use_container_width=True)

    st.markdown('#### ACF and PACF Plots')
    acf = acf_pacf_plot(df_final, selected_country)
    st.pyplot(acf)

with col[1]:
    st.markdown('#### Medal Predictions')

    bc = bar_chart_predictions(df_predictions)
    st.plotly_chart(bc, use_container_width=True)

with col[2]:
    st.markdown('#### Model Performance')

    bc = bar_chart(df_results)
    st.pyplot(bc, use_container_width=True)

    with st.expander('About', expanded=True):
        st.write('''
            The first graph shows the total number of medals won by the selected country over time. You can select a country from the sidebar to see the data for that country. 
            ''')

