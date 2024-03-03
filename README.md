# Bitcoin Price Prediction Using Neural Networks

This project focuses on predicting Bitcoin (BTC) prices using various neural network architectures, including Long Short-Term Memory (LSTM), Convolutional Neural Networks (CNN), and models incorporating regularization techniques like L1 and L2. The goal is to explore how different models and techniques can improve prediction accuracy and efficiency in forecasting Bitcoin's future prices based on historical data.

## Project Structure

The project is structured as follows:

- **Data Preprocessing**: Preparing the Bitcoin price data for modeling, including normalization and creation of time series sequences.
- **Model Building**: Developing different neural network models for time series prediction.
- **Training and Evaluation**: Training the models on historical data and evaluating their performance.
- **Optimization**: Applying regularization techniques to improve model performance.
- **Visualization**: Plotting training and validation loss to analyze model learning.

## Dataset

The dataset used in this project contains daily Bitcoin prices. Each record includes the following features:

- Date
- Open
- High
- Low
- Close
- Adjusted Close
- Volume

## Dependencies

- Python 3.8+
- NumPy
- Pandas
- Keras (TensorFlow backend)
- scikit-learn
- Matplotlib

## Setup and Installation

Ensure you have Python 3.8+ installed. Install the required Python packages using pip:

```sh
pip install numpy pandas tensorflow scikit-learn matplotlib
```

## Data Preprocessing

1. **Normalization**: Scale the features to a range between 0 and 1 for efficient training.
2. **Sequence Creation**: Transform the data into sequences to be used as inputs for the neural network models. Each sequence predicts the next day's closing price based on a window of previous days' prices.

## Models

### LSTM Model

A simple LSTM model to predict Bitcoin's closing price based on sequences of previous days' closing prices.

### CNN Model

A CNN model that utilizes convolutional layers to capture temporal dependencies in the sequence data for price prediction.

### Regularization Techniques

Models incorporating L1 and L2 regularization to prevent overfitting and improve model generalization.

## Training and Evaluation

Each model is trained using historical Bitcoin price data. The performance is evaluated based on the mean squared error (MSE) between the predicted and actual prices.

## Optimization

Experiment with different optimizers, learning rates, and regularization techniques to find the best model configuration.

## Visualization

Plot training and validation loss over epochs to monitor the learning process and identify overfitting.

## Usage

To train and evaluate a model, run the corresponding Python script. For example, to train the LSTM model:

```python
python lstm_model.py
```

Replace `lstm_model.py` with the script for the desired model.

## Results

Throughout this project, I explored several neural network architectures and regularization techniques to predict Bitcoin prices. Here are the key findings from the models tested:

### LSTM Model
- The LSTM model demonstrated a good baseline performance in capturing the temporal dependencies within the Bitcoin price series. However, it showed signs of overfitting with increasing epochs, as evidenced by the divergence between training and validation loss.

### CNN Model
- The CNN model, an unconventional choice for time series forecasting, performed surprisingly well in identifying patterns in the sequence data. It showed a competitive performance compared to the LSTM model, with a more stable convergence in loss, suggesting an efficient extraction of temporal features.

### Regularization Techniques (L1 and L2)
- Incorporating L1 regularization led to a sparse model with some weights pushed to zero, which helped in feature selection but did not significantly improve the prediction accuracy over the baseline LSTM model.
- L2 regularization, on the other hand, helped in smoothing the learning curve and slightly improved the model's generalization capability by penalizing large weights. This resulted in a more robust model against overfitting compared to the baseline LSTM model.

### Training and Validation Loss
- Visualization of training and validation loss over epochs for each model revealed important insights. Models with regularization (especially L2) showed a better balance between training and validation loss, indicating improved model generalization.

## Conclusion

This project's exploration into neural networks for Bitcoin price prediction has yielded several insights into the capabilities and limitations of different architectures and techniques. LSTM and CNN models both showed promise in forecasting time series data, each with unique strengths. LSTM models excel in capturing long-term dependencies, while CNNs can efficiently process temporal patterns in sequences.

The application of L1 and L2 regularization techniques provided a nuanced understanding of how to combat overfitting, with L2 regularization showing a slight edge in enhancing model performance through better generalization.

However, it's clear that there is no one-size-fits-all solution. The choice of model and technique depends heavily on the specific characteristics of the dataset and the forecasting goals. Future work could explore more sophisticated neural network architectures, such as hybrid models that combine LSTM and CNN layers, and the use of attention mechanisms to further improve prediction accuracy.

Additionally, incorporating external factors that influence Bitcoin prices, such as market sentiment, regulatory news, and macroeconomic indicators, could provide a more holistic approach to forecasting.

This project underscores the importance of continuous experimentation and optimization in the field of machine learning and financial forecasting, highlighting the potential for innovative approaches to enhance prediction models in the volatile and unpredictable cryptocurrency market.

## License

This project is open-sourced under the MIT License.
