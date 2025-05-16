import streamlit as st
import pandas as pd
import re

from PIL import Image
from src.utilities.streamlit import *
from streamlit_option_menu import option_menu
from datetime import datetime, timedelta


# Set Layout Config
st.set_page_config(page_title="Market Dashboard", page_icon=":chart_with_upwards_trend:",layout="wide")
     
# Add Font
streamlit_style = """
			<style>
			@import url('https://fonts.cdnfonts.com/css/neue-haas-grotesk-display-pro');    

			html, body, [class*="css"]  {
			font-family: 'neue haas grotesk display pro';
			}
			</style>
			"""

# Apply Font
st.markdown(streamlit_style, unsafe_allow_html=True)

# Global Variables
MAX_NEWS_PER_COL = 10
DATE_VIEW = datetime.today().strftime("%Y-%m-%d") # set default date in the calendar as today.
NEUTRAL_THRESHOLD = 0.1

if True:
    # Greet
    if "greet" not in st.session_state:
        greet()
        st.session_state.greet = True

    # Cache df
    if "df" not in st.session_state:
        df = get_data_stock()
        df['date'] = pd.to_datetime(df['date'])
        st.session_state.df = df

    # Get From Cache
    else:
         df = st.session_state.df

    # Cache Model & Tokenizer
    if "distilbert_model" not in st.session_state:
        distilbert_model = get_distilbert_model()
        distilbert_tokenizer = get_distilbert_tokenizer()
        st.session_state.distilbert_model = distilbert_model
        st.session_state.distilbert_tokenizer = distilbert_tokenizer

    # Get From Cache
    else:
         distilbert_model = st.session_state.distilbert_model
         distilbert_tokenizer = st.session_state.distilbert_tokenizer

    # Cache BART Model & Tokenizer
    if "bart_model" not in st.session_state:
        bart_tokenizer, bart_model = get_bart()
        st.session_state.bart_model = bart_model
        st.session_state.bart_tokenizer = bart_tokenizer

    # Get from Cache
    else:
        bart_model = st.session_state.bart_model
        bart_tokenizer = st.session_state.bart_tokenizer

    # Summary caching
    if "summary_cache" not in st.session_state:
        st.session_state.summary_cache = {}

    col1, col2, col3 = st.columns((14, 6, 6))

    # Add Image
    with col3:
        st.image(Image.open("./img/logo-unpad.png"))

    # Add Header Text
    with col1:
        stspace(1)
        st.write("## Market Dashboard")

    # Add whitespace
    stspace(3)

    # Generate Tabs
    listTabs = [
    "Dashboard",
    "Predict & Summarize Sentiment"
    ]

    # Show tabs
    selected = option_menu(None, listTabs, 
        icons=['pie-chart', 'graph-up-arrow'], 
        menu_icon="cast", default_index=0, orientation="horizontal")
    
    # Dashboard
    if selected == "Dashboard":
        # create cols
        title1col, stock_col, datecol  = st.columns((9, 2, 2))

        # Overall Sentiment Part
        with title1col:
            stspace(2)
            st.write("### Overall Sentiment")

        # Stock option box column
        with stock_col:
            selected_stock = st.selectbox("Select Stock", ("NVDA", "INTC", "META"))

        # Date box column
        with datecol:
            # Find the minimum date
            min_date = df['date'].min().to_pydatetime()

            # Find the maximum date
            max_date = df['date'].max().to_pydatetime()
            
            # Date Input Widget
            date_chosen = st.date_input("Select Date", value=(max_date - timedelta(days=6), max_date), min_value=min_date, max_value=max_date, format="DD.MM.YYYY", disabled=False, label_visibility="visible")

            # Process start date
            start_date = date_chosen[0].strftime('%Y-%m-%d')

            # Get end date
            try:
                # Add one day to the end date
                end_date = (date_chosen[1] + timedelta(days=1)).strftime('%Y-%m-%d')

            # Handle errors
            except Exception as e:
                end_date = (date_chosen[0] + timedelta(days=1)).strftime('%Y-%m-%d')

        
        if selected_stock:
            # Filter df by stock
            filtered_df = filter_df_by_stock(df, selected_stock)

            # Compute Total Negatives, Neutrals, Positives, and NSS Score
            filtered_df = filter_df_by_date(filtered_df, start_date=start_date, end_date=end_date)

            try:
            # filtered_df.iloc[0].date <=
                neg, neut, pos, score = calculate_sentiment_metrics(filtered_df)
                news_found = True

            except ZeroDivisionError:
                news_found = False
                st.write("News not found on the specified date period. Change the stocks option or the date range.")

        if news_found:
            # Sample value
            sentiment_score = round(score, 3)
            sentiment_label = "Negative" if sentiment_score <= -NEUTRAL_THRESHOLD else "Neutral" if sentiment_score < NEUTRAL_THRESHOLD else "Positive"
            
            # create dashboard cols
            leftcol, midcol = st.columns(2)

            with leftcol:
                # Show Gauge Chart
                st.plotly_chart(create_gauge_chart(sentiment_score))
                
            with midcol:
                stspace(2)
                st.write(f"**Overall Net Sentiment Score (NSS) for a given period is** ***{sentiment_label}***.")
                mid_1, mid_2, mid_3 = st.columns((100, 100, 100))
                
                with mid_1:
                    # Give space :)
                    stspace(2)

                    # Negative Metrics
                    st.metric("Negative", f"{neg} News", "-2")

                with mid_2:
                    # Give spaazeeee
                    stspace(2)

                    # Neutral Metrics
                    st.metric("Neutral", f"{neut} News", "4")
                
                with mid_3:
                    # Give space
                    stspace(2)

                    # Positive Metrics
                    st.metric("Positive", f"{pos} News", "3")

                # Document news sources
                stspace(1)


            # Combine all the text from the 'title' column into one string
            current_key =f'{selected_stock}-{str(date_chosen)}'

            # Caching logic
            if current_key in st.session_state.summary_cache:
                combined_text = st.session_state.summary_cache[current_key]

            else:
                combined_text = summarize_with_bart(bart_tokenizer, bart_model, filtered_df['title'].str.cat(sep='. '))
                st.session_state.summary_cache[current_key] = combined_text

            # Use regex to split only at periods followed by a space
            sentences_list = [sentence.strip() for sentence in re.split(r'(?<!\w\.\w.)(?<=\w\.)\s+(?=[A-Z])', combined_text) if sentence]

            # Write and summarize news
            st.write("### News Summary")

            # Loop through news
            for count, sentence in enumerate(sentences_list):
                st.write(f"<h6>{count + 1}. {sentence}</h6>", unsafe_allow_html=True)
        

            # Setup size formatting
            len_filtered_df = filtered_df.shape[0]
            left_col = min(MAX_NEWS_PER_COL, ((len_filtered_df // 2) + 1) if len_filtered_df % 2 != 0 else (len_filtered_df // 2))
            right_col = min(MAX_NEWS_PER_COL, ((len_filtered_df // 2)))
            
            # 3rd Header
            newscol1, newscol2 = st.columns((2))
            with newscol1:
                # Write Header
                stspace(2)
                st.write("### News")
                
                # Display news in the first column
                display_news(filtered_df, 0, left_col)

            with newscol2:
                # Space
                stspace(5)
                
                # Display news in the second column
                display_news(filtered_df, left_col, left_col + right_col)

    if selected == "Predict & Summarize Sentiment":
        # Write Header
        st.write("### Predict News Headline Sentiment")

        # Write cache sessions for increased performance
        if 'distilbert_input' not in st.session_state:
            st.session_state.distilbert_input = ''

        if 'bart_input' not in st.session_state:
            st.session_state.bart_input = ''

        # Create input widget
        distilbert_input = st.text_input("Input News Headline to Predict Sentiment", key='distilbert_widget', on_change=submit_distilbert)

        if st.session_state.distilbert_input != '':

            # Predict the input
            distilbert_output = predict_with_distilbert(st.session_state.distilbert_input, distilbert_model, distilbert_tokenizer)
            
            # Show
            st.write(f"###### Predicted Sentiment for '{st.session_state.distilbert_input}': {distilbert_output[0]}")

        # Spaces    
        stspace(3)

        # Create text box for news content summarizer
        st.write("### Summarize News Content")
        summarizer_box_content = st.text_area(
            "Text to Summarize",
            value = "",
            key = 'bart_widget',
            on_change = submit_bart

        )

        # Summarizing with cache input
        if st.session_state.bart_input != '':
            summarized_text_content = summarize_with_bart(bart_tokenizer, bart_model, st.session_state.bart_input)
            st.write("#### Summary:")
            st.write(summarized_text_content)
