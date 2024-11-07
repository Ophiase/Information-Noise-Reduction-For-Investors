# Information Noise Reduction For Investors 🏦

This research and development project proposes a novel methodology for investors to select relevant variables and mitigate informational noise in complex market models. The approach leverages neural networks as a flexible class of models, incorporating the assumption that the relationships between variables are non-linear. Neural networks are treated as probabilistic models that estimate the likelihood of a variable contributing valuable information to the prediction task. 

## Table of Contents

- [Introduction](#introduction)
    - [Example of Model Reduction](#example-of-model-reduction)
- [Features](#features)
- Setup
    - [Installation](#installation)
    - [Run](#run)
- [Future Development](#future-development)

## Introduction

This project presents an approach to solve the variable selection problem for investors who want to reduce noise when analyzing market phenomena. By using neural networks, we aim to estimate the probability of each variable containing useful information, which can help investors prioritize their focus on the most relevant variables. 

Unlike traditional linear models, neural networks are used here solely to estimate the importance of variables rather than explaining the data itself. The goal is to help investors decide which variables to investigate further, reduce unnecessary complexity, and avoid information overload. 

For example, if a neural network produces similar results (same loss, using cross-validation) when considering a model with or without a specific variable, this indicates that the variable might not add significant information to the model. This insight helps investors avoid overcomplicating their analysis and focus on the truly impactful variables.

The focus of this project will be on cases where investors have up to 10 variables to work with, and they seek to understand the relationships between these variables in a non-linear way.

### Example of Model Reduction

Suppose an investor is attempting to predict the price of a stock based on the following 10 variables: 
- Interest Rate (IR), GDP Growth (GDP), Oil Price (OP), Stock Market Index (SMI), Inflation Rate (IRR), Unemployment Rate (UR), Currency Exchange Rate (CER), Government Debt (GD), Consumer Confidence Index (CCI), and the Company’s Revenue (CR). 

After training a neural network, the model might find that some of these variables, such as **GDP Growth**, **Unemployment Rate**, and **Government Debt**, do not contribute meaningfully to predicting the stock price. In contrast, **Interest Rate**, **Oil Price**, **Stock Market Index**, and **Company’s Revenue** would likely emerge as the most significant predictors. By reducing the model to focus on these variables, the investor can simplify their analysis, avoid unnecessary complexity, and reduce the noise introduced by irrelevant features, leading to a more efficient and effective model for decision-making.


## Features

- **Variable Importance Estimation**: Uses neural networks to estimate the significance of each variable in relation to the target outcome.
- **Cross-Validation Model**: The project will leverage cross-validation to assess how the inclusion of each variable affects model performance.
- **Noise Reduction**: Identifies which variables are redundant or irrelevant to the model’s predictive power, reducing mental noise for investors.
- **Inter-variable Relationships**: Helps to detect dependencies between variables, showing which ones are functionally related to others.

## Setup

### Installation

### Run

## Future Development

- Support larger datasets with more than about ten variables.
- Improve the variable importance estimation technique by integrating other machine learning algorithms.
    - Example: Use LSTM to analyse relevant variables in timeseries.
- Add more sophisticated noise reduction mechanisms to handle even more complex datasets.
- Implement an interactive dashboard to visualize variable importance and interrelationships.