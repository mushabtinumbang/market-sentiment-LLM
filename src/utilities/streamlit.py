import streamlit as st
import time
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import torch

import src.utilities.utils as utils

from datetime import datetime
from src.utilities.config_ import combined_data_path
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
from src.features.run_scraper import run_scraper
from src.features.run_prediction import run_forecast
from src.features.postprocess_data import run_postprocess
from src.utilities.config_ import distilbert_model_path, config_path, log_path, model_path
from src.utilities.utils import format_dates, read_yaml
from streamlit_calendar import calendar

# Chart Variables
quadrant_colors = ["#2bad4e", "#eff229", "#f25829"]  
quadrant_text = ["<b>Negative</b>", "<b>Neutral</b>", "<b>Positive</b>"]
n_quadrants = len(quadrant_colors) 

def greet():
    st.toast('Hello!', icon='✅')
    time.sleep(1)
    st.toast('Welcome to the Market Dashboard', icon='✅')

def stspace(num):
    for j in range(num):
        st.write("")

def getdata():
    df = utils.load(os.path.join(combined_data_path, 'combined_data.feather'))
    return df

def get_distilbert_model():
    distilbert_model = DistilBertForSequenceClassification.from_pretrained(distilbert_model_path)
    return distilbert_model

def get_distilbert_tokenizer():
    distilbert_tokenizer = DistilBertTokenizer.from_pretrained(distilbert_model_path)
    return distilbert_tokenizer

def create_gauge_chart(current_value):
    # Configuration for the gauge chart
    plot_bgcolor = "#fff"  # Default white background
    quadrant_colors = [plot_bgcolor, "#2bad4e", "#eff229" , "#f25829"]  # Negative, Neutral, Positive
    quadrant_text = ["", "<b>Positive</b>", "<b>Neutral</b>", "<b>Negative</b>"]
    n_quadrants = len(quadrant_colors) - 1
    
    min_value = -1
    max_value = 1
    hand_length = np.sqrt(2) / 4
    hand_angle = np.pi * (1 - (max(min_value, min(max_value, current_value)) - min_value) / (max_value - min_value))

    fig = go.Figure(
        data=[
            go.Pie(
                values=[0.5] + [0.225, 0.05, 0.225], # 0.2375, 0.025, 0.2375 for .05 threshold
                rotation=90,
                hole=0.5,
                marker_colors=quadrant_colors,
                text=quadrant_text,
                textinfo="text",
                hoverinfo="skip",
                sort=False
            ),
        ],
        layout=go.Layout(
            showlegend=False,
            margin=dict(b=0, t=40, l=10, r=10),
            width=400,
            height=400  ,
            paper_bgcolor=plot_bgcolor,
            annotations=[
                go.layout.Annotation(
                    text=f"<b>Net Sentiment Score Value:</b><br>{current_value}",
                    x=0.5, xanchor="center", xref="paper",
                    y=0.25, yanchor="bottom", yref="paper",
                    showarrow=False,
                )
            ],
            shapes=[
                go.layout.Shape(
                    type="circle",
                    x0=0.48, x1=0.52,
                    y0=0.48, y1=0.52,
                    fillcolor="#333",
                    line_color="#333",
                ),
                go.layout.Shape(
                    type="line",
                    x0=0.5, x1=0.5 + hand_length * np.cos(hand_angle),
                    y0=0.5, y1=0.5 + hand_length * np.sin(hand_angle),
                    line=dict(color="#333", width=4)
                )
            ]
        )
    )
    
    return fig

def filter_df_by_date(df, start_date, end_date):
    # Filter the DataFrame based on the date range
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    filtered_df = filtered_df.sort_values('date', ascending=False).reset_index(drop=True)
    
    return filtered_df

def filter_df_by_stock(df, stock):
    return df[df["category"] == stock]

def calculate_sentiment_metrics(df):
    # count label
    label_counts = df['label'].value_counts()

    # get total for each label 
    total_neutral = label_counts.get('neutral', 0)
    total_positive = label_counts.get('positive', 0)
    total_negative = label_counts.get('negative', 0)
    
    # Weighted Sentiment Score (weights can be adjusted as needed)
    w_p = 1
    w_n = 1
    weighted_sentiment_score = (((w_p * total_positive) - (w_n * total_negative)) / (total_positive + total_negative + total_neutral))
    
    return total_negative, total_neutral, total_positive, weighted_sentiment_score

def get_total_unique_sources(df):

    # count label
    label_counts = df['source'].value_counts()

    # get total for each label 
    total_dailyfx = label_counts.get('dailyfx', 0)
    total_econtimes = label_counts.get('econtimes', 0)
    total_financialtimes = label_counts.get('financialtimes', 0)
    
    return total_dailyfx, total_econtimes, total_financialtimes

def predict_with_distilbert(
        text,
        loaded_model,
        loaded_tokenizer,
        batch_size=32
):
    # New text data to predict
    titles = [text]

    # Move model to the appropriate device (e.g., CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model.to(device)

    # Initialize an empty list to store predictions
    all_predictions = []

    # Process data in batches
    for i in range(0, len(titles), batch_size):
        batch_titles = titles[i:i + batch_size]
        
        # Tokenize the new text
        inputs = loaded_tokenizer(batch_titles, padding='max_length', truncation=True, max_length=64, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        # Make predictions
        loaded_model.eval()
        with torch.no_grad():
            outputs = loaded_model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
        
        # Convert predictions to labels
        int_to_label = {0: 'positive', 1: 'neutral', 2: 'negative'}
        predicted_labels = [int_to_label[pred.item()] for pred in predictions]
        all_predictions.extend(predicted_labels)

    return all_predictions

def display_news(filtered_df, start_idx, end_idx):
    for index, row in filtered_df[start_idx:end_idx].iterrows():
        title = row['title']
        date = row['date']
        url = row['url']
        category = row['category']
        label = row['label']
        summary = row['summary']

        try:
            source = row['source']

        except KeyError:
            source = "Investing.com"

        color = {
            "negative": "red",
            "positive": "limegreen",
            "neutral": "blue"
        }.get(label.lower(), "black")

        st.markdown(f'<span style="color:{color};">{label.capitalize()}</span>', unsafe_allow_html=True)

        # Assuming title and url are defined
        title_safe = title.replace('_', '&lowbar;')  # Replace underscores with HTML entity

        # Combine <h6> with <a> tag for the title with underline
        st.write(f"<h6><a href='{url}' style='color:black; font-size:20px; text-decoration:underline;'>{title_safe}</a></h6>", unsafe_allow_html=True)

        # Write summary
        st.write(f"<p>{summary if summary != 'None' else 'Premium Content'}<p>", unsafe_allow_html=True)

        # st.write(f"<h6>{count + 1}. {sentence}</h6>", unsafe_allow_html=True)
        st.write(f'###### Source: {source.capitalize()}. Category: {category.capitalize()}. Date: {date.strftime("%d-%m-%Y")}')
        stspace(2)

def get_min_max_date_by_source(df):
    # Initialize a dictionary to store the results
    min_max_dates = {}

    # List of sources to process
    sources = ["dailyfx", "econtimes", "financialtimes"]

    # Loop through each source to filter the DataFrame and get min and max dates
    for source in sources:
        source_df = df[df["source"] == source].reset_index(drop=True)
        if not source_df.empty:
            min_date = (source_df.date.min()).strftime('%Y-%m-%d')
            max_date = (source_df.date.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            min_date, max_date = None, None
        min_max_dates[source] = (min_date, max_date)

    # Extract the results for each source
    dailyfx_min, dailyfx_max = min_max_dates["dailyfx"]
    econtimes_min, econtimes_max = min_max_dates["econtimes"]
    financialtimes_min, financialtimes_max = min_max_dates["financialtimes"]

    return dailyfx_min, dailyfx_max, econtimes_min, econtimes_max, financialtimes_min, financialtimes_max

def create_calendar(
        DATE_VIEW,
        dailyfx_min, 
        dailyfx_max, 
        econtimes_min, 
        econtimes_max, 
        financialtimes_min, 
        financialtimes_max
):
    calendar_options = {
        "editable": "false",
        "navLinks": "false",
        "resources": "",
        "selectable": "true",
        "headerToolbar": {
            "left": "today prev,next",
            "center": "title",
            "right": "dayGridDay,dayGridWeek,dayGridMonth",
        },
        "initialDate": f"{DATE_VIEW}",
        "initialView": "dayGridMonth",
    }
    calendar_events = [
        {
            "title": "DailyFX",
            "start": f"{dailyfx_min}",
            "end": f"{dailyfx_max}",
            "resourceId": "a",
            "color": "#FF4B4B",
        }, 
        {
            "title": "Economic Times",
            "start": f"{econtimes_min}",
            "end": f"{econtimes_max}",
            "resourceId": "b",
            "color": "#3DD56D",
        },
        {
            "title": "Financial Times",
            "start": f"{financialtimes_min}",
            "end": f"{financialtimes_max}",
            "resourceId": "c",
        }
    ]
    custom_css = f"""
        .fc-event-past {{
            opacity: 0.8;
        }}
        .fc-event-time {{
            font-style: italic;
        }}
        .fc-event-title {{
            font-weight: 700;
        }}
        .fc-toolbar-title {{
            font-size: 2rem;
        }}
        .calendar-container {{
            width: 20%;
            height: 200px;
        }}
    """

    return calendar(events=calendar_events, options=calendar_options, custom_css=custom_css)

def run_scrape_streamlit(date, dailyfx, econtimes, financialtimes, suffix= "streamlit"):
    # load some config
    params = read_yaml(
        os.path.join(config_path, "main_config.yaml"), render=True, suffix=suffix
    )

    # define parameters
    scraper_params = params["run_scraper_pipeline_params"]
    scraper_params["date"] = format_dates(date)
    scraper_params["dailyfx"] = dailyfx
    scraper_params["econtimes"] = econtimes
    scraper_params["ftimes"] = financialtimes
    scraper_params["suffix"] = suffix

    # run scraper
    run_scraper(**scraper_params)

    return True

def run_predict_streamlit(date, dailyfx, econtimes, financialtimes, suffix= "streamlit"):
    # load some config
    params = read_yaml(
        os.path.join(config_path, "main_config.yaml"), render=True, suffix=suffix
    )

    # define parameters
    predict_params = params["run_prediction_pipeline_params"]
    predict_params["dailyfx"] = dailyfx
    predict_params["econtimes"] = econtimes
    predict_params["ftimes"] = financialtimes
    predict_params["suffix"] = suffix

    # run scraper
    run_forecast(**predict_params)

    return True

def run_postprocess_streamlit(date, dailyfx, econtimes, financialtimes, suffix= "streamlit"):
    # load some config
    params = read_yaml(
        os.path.join(config_path, "main_config.yaml"), render=True, suffix=suffix
    )

    # define parameters
    combine_params = params["postprocess_data_params"]
    combine_params["dailyfx"] = dailyfx
    combine_params["econtimes"] = econtimes
    combine_params["ftimes"] = financialtimes

    # run data postprocessings
    run_postprocess(**combine_params)

    return True

def get_bart():
    # Load the tokenizer
    tokenizer = BartTokenizer.from_pretrained(os.path.join(model_path, "bart-large-cnn"))
    # Load the model
    model = BartForConditionalGeneration.from_pretrained(os.path.join(model_path, "bart-large-cnn"))

    return tokenizer, model

def get_data_stock():
    df = utils.load(os.path.join(combined_data_path, 'combined_data_stock.feather'))
    return df

def summarize_with_bart(
        tokenizer,
        model,
        text
):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

    # Generate summary
    summary_ids = model.generate(inputs["input_ids"], max_length=100, min_length=25, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

def submit_bart():
    st.session_state.bart_input = st.session_state.bart_widget

def submit_distilbert():
    st.session_state.distilbert_input = st.session_state.distilbert_widget
    st.session_state.bart_input = ''