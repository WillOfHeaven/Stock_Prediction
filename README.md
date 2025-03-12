
# Stock Price Prediction and Analysis: From App to Advanced Analysis

This repository demonstrates the evolution of a stock price prediction
project, starting with a Streamlit application and progressing to an
in-depth analysis of Bidirectional LSTM models and moving averages.

## Part 1: Stock Price Prediction App

### Overview
A Streamlit-based web application for stock price prediction and
visualization.

### Key Features
- Historical stock data retrieval using Alpha Vantage API
- Interactive data visualization with Plotly
- Stock price prediction using a pre-trained LSTM model
- Moving average calculations and display (100-day and 200-day)

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/stock-price-prediction.git
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Alpha Vantage API key:
   - Sign up for a free API key at [Alpha Vantage](
https://www.alphavantage.co/)
   - Replace `api_key = 'XO2YT4XMHPUERBBP'` with your actual API key in the
script

### Usage

Run the Streamlit app:
```
streamlit run app.py
```

Enter a stock symbol (e.g., IBM) in the text input field to fetch and
analyze the stock data.

### Code Structure

The Streamlit app is structured as follows:

1. **Imports and Setup**
   - Import necessary libraries (numpy, pandas, matplotlib, plotly, etc.)
   - Set up Streamlit page configuration
   - Define API key and global variables

2. **Data Retrieval**
   - Function to fetch stock data from Alpha Vantage API
   - Error handling for API requests
   - Data parsing and initial processing

3. **Data Processing**
   - Convert API response to pandas DataFrame
   - Rename columns for clarity
   - Calculate moving averages (100-day and 200-day)

4. **Visualization Functions**
   - Create interactive Plotly charts for:
     - Closing price vs. time
     - Stock price with volume
     - Moving averages

5. **LSTM Model Integration**
   - Load pre-trained LSTM model
   - Prepare data for LSTM input (scaling, reshaping)
   - Make predictions using the model

6. **Streamlit UI Components**
   - Title and user input for stock symbol
   - Display raw data and descriptive statistics
   - Show interactive charts (stock price, volume, moving averages)
   - Display LSTM predictions

7. **Additional Features**
   - Download button for CSV data
   - Error handling and user feedback messages

8. **Main Execution Flow**
   - User inputs stock symbol
   - Fetch and process data
   - Display visualizations and predictions
   - Update UI based on user interactions

9. **Utility Functions**
   - Data conversion for CSV download
   - Date range calculations

### Dependencies

numpy, csv, json, matplotlib, pandas_datareader, pickle, requests, keras,
streamlit, datetime, sklearn, plotly

## Part 2: Advanced Analysis - Improving Stock Price Forecasting with
Bidirectional LSTM and Moving Averages

### Motivation
After developing the Streamlit app, I saw an opportunity to deepen my
understanding of stock price prediction techniques. This led to a
comprehensive analysis focusing on Bidirectional LSTM models and the use of
moving averages for forecasting.

### Analysis Report: "Improving Bidirectional LSTM for Stock Price
Forecasting with Moving Averages"

#### Author
Sambuddha Chatterjee

#### Institution
Kalinga Institute of Industrial Technology

#### Supervisor
Dr. Saurabh Bilgaiyan

#### Overview
This analysis explores the optimization of Bidirectional Long Short-Term
Memory (Bi-LSTM) models for stock price prediction in the Indian stock
market. It incorporates moving averages and focuses on comparing different
model configurations to improve prediction accuracy.

#### Key Components
1. Dataset creation from the Indian stock market
2. Implementation of Bidirectional LSTM model
3. Incorporation of moving averages (100-day and 200-day)
4. Comparison of stateful and stateless Bi-LSTM models
5. Analysis of different numbers of hidden layers

#### Methodology
1. Selection of companies from various sectors in NIFTY 50
2. Data preparation, including moving average calculations
3. Implementation of Bi-LSTM model with different configurations
4. Comparison of model performance with and without moving averages

#### Key Findings
- Impact of moving averages on prediction accuracy
- Performance differences between stateful and stateless Bi-LSTM models
- Optimal number of hidden layers for the Indian stock market context

## Continuous Improvement Journey

1. **Initial App Development**: Created a Streamlit app for basic stock
prediction and moving average visualization.
2. **Identification of Analysis Opportunities**: Recognized potential for
deeper analysis of prediction techniques.
3. **Expanded Data Analysis**: Incorporated a wider range of Indian stocks
and extended use of moving averages.
4. **Model Experimentation**: Implemented various Bi-LSTM configurations
and compared with moving average predictions.
5. **Comparative Analysis**: Performed detailed comparisons between
different model types and moving average strategies.
6. **Report Writing**: Synthesized findings into a comprehensive analysis
report.

## Future Directions

- Integration of analysis findings to enhance the Streamlit app
- Exploration of combining Bi-LSTM predictions with traditional technical
analysis indicators
- Investigation of sector-specific prediction models within the Indian
market

## Author

Sambuddha Chatterjee (Sammy)

## Acknowledgements

Special thanks to Dr. Saurabh Bilgaiyan for supervision and to Kalinga
Institute of Industrial Technology for supporting this analysis.

## License

This project is open-source and available under the [MIT License](LICENSE).
