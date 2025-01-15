import streamlit as st
# import openai
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Load the dataset
@st.cache_data
def load_data():
    # Replace this with the actual path to your JSON file
    data = pd.read_json('stock-dashboard/stock_data_with_sentiment.json')
    # data['timestamp'] = pd.to_datetime(data['timestamp'])
    return data

data = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Stock Market Chatbot", "Dashboards", "Dataset"])

# 1. ChatGPT-like Prompt Generator Page
if page == "Stock Market Chatbot":
    st.title("Stock Market Chatbot")

    # User input for the prompt
    prompt = st.text_area(
        "**Enter your prompt:**",
        height=300,  # Default height for fallback
        key="prompt_input",
        placeholder="Type your prompt here..."
    )


    # Generate response button
    if st.button("Generate Response"):
        # Combine prompt and instructions
        final_prompt = prompt
        if instructions:
            final_prompt += f"\n\n{instructions}"

        if prompt:
            # Call OpenAI API
            with st.spinner("Generating response..."):
                try:
                    response = openai.Completion.create(
                        engine="text-davinci-003",
                        prompt=final_prompt,
                        max_tokens=150
                    )
                    st.success("Response generated!")
                    st.write(response.choices[0].text.strip())
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter a prompt.")

# 2. Dashboards Page
elif page == "Dashboards":
    st.title("Dashboards")

    # Sidebar filters
    st.sidebar.header("Filters")
    stocks = data['symbol'].unique()
    selected_stock = st.sidebar.selectbox("Select Stock:", ["All"] + list(stocks))
    
    # Date range slider for filtering the x-axis
    min_date = data['timestamp'].min().date()
    max_date = data['timestamp'].max().date()

    # Add a range slider
    start_date, end_date = st.sidebar.slider(
        "Select Date Range:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )
    

    # Apply filters
    filtered_data = data[
        (data['timestamp'] >= pd.Timestamp(start_date)) & 
        (data['timestamp'] <= pd.Timestamp(end_date))
    ]

    if selected_stock != "All":
        filtered_data = filtered_data[filtered_data['symbol'] == selected_stock]

    # Display filtered data
    st.subheader(f"Filtered data for stock: *{selected_stock}*")
    st.dataframe(filtered_data)
    
    

    # Line Chart: Closing Price Over Time
    st.subheader("Closing Price Over Time")
    if not filtered_data.empty:
        fig, ax = plt.subplots()
        ax.plot(filtered_data['timestamp'], filtered_data['close'], label="Closing Price", color="blue", marker='o')
        ax.set_xlabel("Date")
        ax.set_ylabel("Closing Price")
        ax.set_title("Closing Price Over Time")
        ax.legend()
        # Format x-axis to show day-level details
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        # ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))  # Show every day
        fig.autofmt_xdate()  # Rotate date labels for better visibility
        st.pyplot(fig)
    else:
        st.write("No data available for the selected filters.")

    # Bar Chart: Trade Volume Over Time
    st.subheader("Trade Volume Over Time")
    if not filtered_data.empty:
        fig, ax = plt.subplots()
        ax.bar(filtered_data['timestamp'], filtered_data['volume'], color="orange", label="Trade Volume")
        ax.set_xlabel("Date")
        ax.set_ylabel("Volume")
        ax.set_title("Trade Volume Over Time")
        ax.legend()
        st.pyplot(fig)
    else:
        st.write("No data available for the selected filters.")

    # Scatter Plot: Sentiment Comparison
    st.subheader("Sentiment Score vs Sentiment Score NLP")
    if not filtered_data.empty:
        fig, ax = plt.subplots()
        ax.scatter(
            filtered_data['sentiment_score'], 
            filtered_data['sentiment_score_nlp'], 
            color="green", label="Sentiment Comparison"
        )
        ax.set_xlabel("Sentiment Score")
        ax.set_ylabel("Sentiment Score NLP")
        ax.set_title("Sentiment Score Comparison")
        ax.legend()
        st.pyplot(fig)
    else:
        st.write("No data available for the selected filters.")

# 3. Dataset Page
elif page == "Dataset":
    st.title("Dataset Used for Fine-Tuning")

    # Load a sample dataset (replace with your actual dataset)
    st.subheader("Below is the dataset used for fine-tuning LLAMA 3.1:")

    # Example: Replace this with your real dataset
    dataset = pd.DataFrame(data)

    # Display the dataset
    st.dataframe(dataset)

    # Option to download dataset
    st.download_button(
        label="Download Dataset",
        data=dataset.to_csv(index=False),
        file_name="dataset.csv",
        mime="text/csv"
    )
