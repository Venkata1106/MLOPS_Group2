import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import pandas as pd
import plotly.graph_objects as go

def get_stock_data(symbol):
    """Fetch stock data using yfinance"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        return df
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return None

def make_prediction(instance_dict):
    """Make prediction using Vertex AI endpoint"""
    project = "322603165747"
    endpoint_id = "4602032306335514624"
    location = "us-east1"
    api_endpoint = "us-east1-aiplatform.googleapis.com"

    try:
        client_options = {"api_endpoint": api_endpoint}
        client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

        instance = json_format.ParseDict(instance_dict, Value())
        instances = [instance]
        
        parameters_dict = {}
        parameters = json_format.ParseDict(parameters_dict, Value())

        endpoint = client.endpoint_path(
            project=project, location=location, endpoint=endpoint_id
        )

        response = client.predict(
            endpoint=endpoint, instances=instances, parameters=parameters
        )

        prediction = dict(response.predictions[0])
        return prediction
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def main():
    st.title("Stock Price Predictor")
    
    # User input
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, GOOGL)", "AAPL")
    
    if st.button("Predict"):
        # Get stock data
        df = get_stock_data(symbol)
        
        if df is not None and not df.empty:
            st.subheader("Recent Stock Data")
            st.dataframe(df)
            
            # Prepare features for prediction
            latest_data = df.iloc[-1]
            instance_dict = {
                "Date": latest_data.name.strftime("%m/%d/%y"),
                "Open": str(latest_data['Open']),
                "High": str(latest_data['High']),
                "Low": str(latest_data['Low']),
                "Volume": str(latest_data['Volume'])
            }
            
            # Make prediction
            prediction = make_prediction(instance_dict)
            
            if prediction:
                st.subheader("Prediction Results")
                
                # Create columns for displaying results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Predicted Price", f"${prediction['value']:.2f}")
                with col2:
                    st.metric("Lower Bound", f"${prediction['lower_bound']:.2f}")
                with col3:
                    st.metric("Upper Bound", f"${prediction['upper_bound']:.2f}")
                
                # Create a visualization
                fig = go.Figure()
                
                # Add historical prices
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    name='Historical Close Price',
                    line=dict(color='blue')
                ))
                
                # Add prediction point
                fig.add_trace(go.Scatter(
                    x=[df.index[-1] + timedelta(days=1)],
                    y=[prediction['value']],
                    name='Prediction',
                    mode='markers',
                    marker=dict(size=10, color='red'),
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=[prediction['upper_bound'] - prediction['value']],
                        arrayminus=[prediction['value'] - prediction['lower_bound']],
                    )
                ))
                
                fig.update_layout(
                    title=f'{symbol} Stock Price Prediction',
                    xaxis_title='Date',
                    yaxis_title='Price ($)',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig)

if __name__ == "__main__":
    main()