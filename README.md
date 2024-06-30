
# Predicting Significant Precipitation Events in Northern Germany

## Description
The aim is to predict the occurrence and magnitude of precipitation events exceeding 10 mm per day in Northern Germany using daily precipitation data from 10 NOAA GHCND stations. The dataset is from January 1, 2014, to March 31, 2024.

## Table of Contents
 - [Data and Methods](#data-and-methods)
   - [Data Preprocessing](#data-preprocessing)
   - [Modeling Approach](#modeling-approach)
 - [Results](#results)
 - [Discussion and Conclusion](#discussion-and-conclusion)
 - [Possible Improvements](#possible-improvements)
 - [Time spent](#time-spent)

## Repository structure

## Data and Methods
### Data Preprocessing
The precipitation values in the dataset are skewed, with most values less than 10 mm, which could lead to class imbalance issues if directly used for classification. To address this, I analyzed the correlation between current precipitation and other variables, and found a slight positive correlation with temperature. To find more stonger features, I also calculated the correlation between present precipitation and past 30 days precipitation, and found a strong correlation up to 3 days, which then decreased.

![Distribution of Precipitation values](figures/lagcorr.png)

As precipitation at a point is also affected by the atmopsheric conditions in nearby locations, I also incorporated data from the three nearest stations for each station, including their lagged values for precipitation and temperature.

The final feature set comprised of station identifier, PRCP, SNWD, TMAX, TMIN, month (encoded using sine and cosine transformations), lagged values for PRCP, TMAX, TMIN, and neighboring station data.


### Modeling Approach
I evaluated four different models: Random Forest, XGBoost, Linear Regression, and LSTM. I used these models to predict the precipitation amount and then applied a rule-based classification to these predictions to classify events exceeding 10 mm.

## Results
I utilized two validation methods to assess the performance of my models:

1. Temporal split with the last year as the test set
2. Temporal cross-validation with 5 splits

### Regression Results (Temporal Split):
| Model           | MAE       | MSE        | RMSE      | R2         |
|-----------------|-----------|------------|-----------|------------|
| Random Forest   | 2.105349  | 12.146505  | 3.485184  | 0.175133   |
| XGBoost         | 2.119736  | 12.561753  | 3.544256  | 0.146934   |
| Linear Regression | 2.145356 | 12.700200  | 3.563734  | 0.137532   |

### Classification Results (Temporal Split):
| Model           | Accuracy  | Precision  | Recall    | F1 Score   |
|-----------------|-----------|------------|-----------|------------|
| Random Forest   | 0.661538  | 0.088957   | 0.725     | 0.158470   |
| XGBoost         | 0.682418  | 0.097087   | 0.750     | 0.171920   |
| Linear Regression | 0.684615 | 0.095082  | 0.725     | 0.168116   |

### Regression Results (Temporal Cross-Validation):
| Model           | MAE       | MSE        | RMSE      | R2         |
|-----------------|-----------|------------|-----------|------------|
| Random Forest   | 1.850467e+001 | 1.182214e+013 | 3.438333e+001 | 1.579848e-01 |
| XGBoost         | 1.855479e+001 | 1.262287e+013 | 3.552867e+001 | 1.009541e-01 |
| Linear Regression | 1.021862e+091 | 1.443506e+193 | 3.799350e+09 | -1.028117e+18 |

### Classification Results (Temporal Cross-Validation):
| Model           | Accuracy  | Precision  | Recall    | F1 Score   |
|-----------------|-----------|------------|-----------|------------|
| Random Forest   | 0.720749  | 0.092137   | 0.776406  | 0.164726   |
| XGBoost         | 0.762345  | 0.096191   | 0.679012  | 0.168511   |
| Linear Regression | 0.701630 | 0.070429  | 0.607682  | 0.126229   |

## Discussion and Conclusion
Predicting precipitation using only precipitation and temperature data is challenging. Incorporating additional variables such as atmospheric pressure, humidity, and solar radiation could help improve the performance of the models. 

## Possible Improvements
This approach could be improved by,
- Hyperparameter optimization to find the best possible values for the models.
- Incorporating other atmospheric variables from reanalysis datasets like ERA5 and MERRA-2.
- Evaluating the potential of pretrained time series models in this task, as discussed in a recent [paper](https://arxiv.org/pdf/2310.10688).
- XAI methods can be like SLISEMAP and SHAP used to understand which features influence specific heavy precipitation events.

## Time spent
* understanding the problem and conceptualising a solution: 1 hr
* obtaining and processing the data: 1 hr
* writing the analysis script(s): 6 hr
* visualising results: 1 hr
* making the notebook ready for presentation: 30 mins