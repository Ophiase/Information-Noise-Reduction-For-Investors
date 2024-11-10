# Information Noise Reduction For Investors 🏦

This research and development project proposes a methodology for investors to select relevant variables and mitigate informational noise in complex market models. The approach leverages neural networks as a flexible class of models, incorporating the assumption that the relationships between variables are non-linear. Neural networks are treated as probabilistic models that permit to estimate the likelihood of a variable contributing valuable information to the prediction task. 

## Table of Contents

- [Introduction](#introduction)
    - [Example of Model Reduction](#example-of-model-reduction)
- [Features](#features)
- [Setup](#setup)
- [Future Development](#future-development)

## Introduction

This project presents an approach to solve the variable selection problem for investors who want to reduce noise when analyzing market phenomena. By using neural networks, we aim to estimate the probability of each variable containing useful information, which can help investors prioritize their focus on the most relevant variables. 

Unlike traditional linear models, neural networks are used here solely to estimate the importance of variables rather than explaining the data itself. The goal is to help investors decide which variables to investigate further, reduce unnecessary complexity, and avoid information overload. 

For example, if a neural network produces similar results (same loss, using cross-validation) when considering a model with or without a specific variable, this indicates that the variable might not add significant information to the model. This insight helps investors avoid overcomplicating their analysis and focus on the truly impactful variables.

The focus of this project will be on cases where investors have up to 10 variables to work with, and they seek to understand the relationships between these variables in a non-linear way.

### Example of Model Reduction

Suppose a financial analyst is attempting to predict the likelihood of bankruptcy based on the following financial indicators: 
- ROA(C) before interest and depreciation before interest, Operating Gross Margin, Realized Sales Gross Margin, Operating Profit Rate, Pre-tax net Interest Rate, Cash Flow Per Share, Current Ratio, Quick Ratio, Total debt/Total net worth, Debt ratio %, among others.

After training a neural network on this dataset, the model may identify some variables, such as **Pre-tax net Interest Rate**, **Quick Ratio**, and **Debt ratio %**, as insignificant predictors of bankruptcy. In contrast, **ROA(C) before interest and depreciation before interest**, **Operating Gross Margin**, **Operating Profit Rate**, **Current Ratio**, and **Cash Flow Per Share** may emerge as the most significant predictors. By reducing the model to focus on these key variables, the analyst can streamline the analysis, minimize complexity, and reduce the noise from irrelevant features, making the bankruptcy prediction model more efficient and accurate for decision-making.



## Features

- **Variable Importance Estimation**: Uses neural networks to estimate the significance of each variable in relation to the target outcome.
- **Cross-Validation Model**: The project will leverage cross-validation to assess how the inclusion of each variable affects model performance.
- **Noise Reduction**: Identifies which variables are redundant or irrelevant to the model’s predictive power, reducing mental noise for investors.
- **Inter-variable Relationships**: Helps to detect dependencies between variables, showing which ones are functionally related to others.

## Setup

### Installation

- Docker installation

```sh
docker build -t information-noise-reduction-for-investors
```

- conda install of the dependencies :

```sh
conda env create -f environment.yml
```

### Usage

```python
# Imports
from information_noise_reduction.subset_generator import reverse_all_subsets_generator
from information_noise_reduction.evaluate_model import evaluate_subsets
from information_noise_reduction.interpretation import compute_variable_contributions
# define a model generator
def model_generator(input_dim: int) -> tf.keras.Sequential:
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='sigmoid', input_shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='sigmoid'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
# evaluate subsets
subset_gen = reverse_all_subsets_generator(feature_columns)
result = evaluate_subsets(
    df, target_col=prediction_column, 
    subset_gen=subset_gen, 
    model_generator=model_generator, 
    max_subsets=10, epochs=10)
# compute variables scores
variable_contributions = compute_variable_contributions(result)
```

### Demonstrations

Open the notebooks in `/examples`.

## Future Development

- Stochastic model generator
    - For a subset of variables, we want to evaluate different models, with different properties.
- Support larger datasets with more than about ten variables.
    - The number of subsets grow exponentialy with the number of variables. Thus, if we want to consider large set of variables. We need to only evaluate a finite number of subsets. 
    - How to choose them with good statistical properties ?
        - Proposition: Geometric Law to select the number $k$ of variable to remove each time, and then uniform law to select the subset of $N-k$ variables. 
        - If we pick twice a specific subset, we also pick another realization of the model generator and thus get more certainty on this subset loss.
- Improve the variable importance estimation technique by integrating other machine learning algorithms.
    - Eg: Use LSTM to analyse relevant variables in timeseries.
    - Eg: Use VAE to find the optimal latent space. \
    Then try to project the importance of each variables by using the loss of the subset they are part of.
- Implement an interactive dashboard to visualize variable importance and interrelationships.

