# Final Year Project

## Project Overview

This repository contains code for my final year project, focusing on time series analysis and stock price prediction. The project utilizes Python and implements two primary models: ARIMA and LSTM, to forecast future stock prices based on historical data.

## Key Features & Benefits

*   **ARIMA Model:** Implements the ARIMA (Autoregressive Integrated Moving Average) model for time series forecasting.
*   **LSTM Model:** Leverages Long Short-Term Memory (LSTM) networks, a type of recurrent neural network, for advanced time series prediction.
*   **Data Acquisition:** Integrates with the `yfinance` library to fetch real-time stock data from Yahoo Finance.
*   **Data Visualization:** Uses `matplotlib` and `seaborn` for visualizing data and model performance.
*   **Modular Design:** The project is structured into separate Python files for each model, promoting code reusability and maintainability.

## Prerequisites & Dependencies

Before running the code, ensure you have the following installed:

*   **Python:** Version 3.6 or higher.
*   **pip:** Python package installer.

Install the required Python packages using pip:

```bash
pip install pandas
pip install numpy
pip install yfinance
pip install matplotlib
pip install seaborn
pip install scipy
pip install statsmodels
pip install tensorflow
```

## Installation & Setup Instructions

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/divine-amakor/Final-Year-Project.git
    cd Final-Year-Project
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt # If you create one based on the dependencies above
    ```

3.  **Download the Keras Model:**

    Ensure the `keras_model.h5` file is in the project root directory.

## Usage Examples

### ARIMA Model

To run the ARIMA model, execute the `ARIMA.py` script:

```bash
python ARIMA.py
```

This script will:

1.  Fetch stock data (e.g., AAPL).
2.  Perform time series decomposition.
3.  Apply the Augmented Dickey-Fuller (ADF) test for stationarity.
4.  Fit an ARIMA model to the data.
5.  Generate forecasts.
6.  Display plots of the original data and forecasts.

### LSTM Model

To run the LSTM model, execute the `LSTM.py` script:

```bash
python LSTM.py
```

This script will:

1.  Fetch stock data (e.g., AAPL).
2.  Preprocess the data for LSTM.
3.  Build and train an LSTM model (using `keras_model.h5`, if available).
4.  Make predictions.
5.  Display a plot comparing the actual and predicted stock prices.

## Configuration Options

*   **Stock Ticker:** Modify the `ticker` variable in both `ARIMA.py` and `LSTM.py` to analyze different stocks. Example (inside `ARIMA.py` or `LSTM.py`):

    ```python
    ticker = 'MSFT' # change AAPL to MSFT or any other stock ticker
    ```

*   **Date Range:** Adjust the `START` and `END` dates in `LSTM.py` to specify the desired data range:

    ```python
    START = "2021-01-01"
    END = "2025-01-01"
    ```

## Contributing Guidelines

Contributions are welcome! To contribute:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes.
4.  Submit a pull request with a clear description of your changes.

## License Information

License not specified. All rights reserved.

## Acknowledgments

*   Yahoo Finance for providing stock market data through the `yfinance` library.
*   The developers of `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`, and `tensorflow` for their invaluable open-source libraries.
