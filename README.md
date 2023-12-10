# Major Power Outages Prediction 🔋

By Yi Xing (Ylesia) Wu (xw001@ucsd.edu) & Junyue Lin (junyuelin608@gmail.com)

## Framing the Problem

A power outage is defined as the loss of the electrical power network supply to an end user. This occurrence engenders a disruption in the provision of electricity, leading to an absence of power in residences, commercial establishments, and other facilities. Power outages can have different degrees of severity. According to the Department of Energy, major power outages refer to those that impacted at least 50,000 customers or caused an unplanned firm load loss of at least 300 MW.

Determining whether a power outage event is considered major is important for local authorities and organizations to take care of the ramifications of the outage. However, it is unlikely that information about the two criteria for a major outage, the number of people affected and the amount of unplanned loss, will be available right after an outage ends. Thus, determining whether an outage event is major is a crucial problem to be solved by a prediction model.

We will be using a binary classification model to predict whether an outage event is major. Since we are more concerned with correctly predicting those power outages that are indeed major, we will be using recall as our metric for examining model performance. 

### Data Cleaning

Just like what we have done in the previous analysis, which can be found [here](https://github.com/ylesia-wu/power-outage-analysis), we have converted the `xlsx` file into `csv` file, removed unnecessary rows and columns, converted the data type of each column as appropriate, and create new `pd.Timestamp` columns by combining existing columns. In addition, we created our response variable column `IS.MAJOR` by combining information from `CUSTOMERS.AFFECTED` and `DEMAND.LOSS.MW` columns based on the definition of major outages: outages that impacted at least 50,000 customers or caused an unplanned firm load loss of at least 300 MW. To handle missing values in relevant columns, we used imputation with appropriate methods such as probabilistic imputation. Last but not least, we split our dataset into a train set and a test set in the proportion of 3 to 1. 

Here are the first few rows of our train set:

|                             | YEAR   | MONTH | U.S._STATE | POSTAL.CODE | NERC.REGION | CLIMATE.REGION     | ANOMALY.LEVEL | CLIMATE.CATEGORY | CAUSE.CATEGORY     | CAUSE.CATEGORY.DETAIL | HURRICANE.NAMES | OUTAGE.DURATION | RES.PRICE | COM.PRICE | IND.PRICE | TOTAL.PRICE | RES.SALES   | COM.SALES   | IND.SALES   | TOTAL.SALES  | RES.PERCEN | COM.PERCEN | IND.PERCEN | RES.CUSTOMERS | COM.CUSTOMERS | IND.CUSTOMERS | TOTAL.CUSTOMERS | RES.CUST.PCT | COM.CUST.PCT | IND.CUST.PCT | PC.REALGSP.STATE | PC.REALGSP.USA | PC.REALGSP.REL | PC.REALGSP.CHANGE | UTIL.REALGSP | TOTAL.REALGSP | UTIL.CONTRI | PI.UTIL.OFUSA | POPULATION  | POPPCT_URBAN | POPPCT_UC | POPDEN_URBAN | POPDEN_UC | POPDEN_RURAL | AREAPCT_URBAN | AREAPCT_UC | PCT_LAND   | PCT_WATER_TOT | PCT_WATER_INLAND | OUTAGE.START           | OUTAGE.RESTORATION     | DAY.OF.WEEK | TIME.OF.DAY | DAY.OF.MONTH | SEASON | IS.MAJOR |
|-----------------------------|--------|-------|------------|-------------|-------------|-------------------|---------------|------------------|---------------------|-----------------------|-----------------|-----------------|-----------|-----------|-----------|------------|-------------|-------------|-------------|--------------|-------------|-------------|-------------|---------------|---------------|---------------|-----------------|--------------|--------------|--------------|------------------|----------------|----------------|---------------------|--------------|----------------|-------------|---------------|-------------|--------------|------------|--------------|----------------|---------------|--------------|------------|----------------|-----------------|----------------|---------------|-----------------------|-------------------------|-------------|-------------|--------------|--------|----------|
| 574                     | 2010.0 | 6.0   | Indiana    | IN          | RFC         | Central           | -0.4          | normal           | severe weather      | thunderstorm          | NaN             | 1980.0          | 9.47      | 8.18      | 5.77      | 7.59       | 3044801.0   | 2221596.0   | 3854047.0   | 9121925.0    | 33.37893     | 24.354465    | 42.250369    | 2742789.0      | 341727.0       | 18796.0        | 3103313.0       | 88.3826      | 11.0117      | 0.6057       | 43130.0          | 47287.0        | 0.91209        | 5.8                 | 5955.0       | 279927.0      | 2.12734      | 2.3           | 6490590.0   | 72.44        | 13.27      | 1860.0        | 1646.9      | 53.7         | 7.05           | 1.46         | 98.369028  | 1.628226       | 0.991214          | 2010-06-18 15:30:00  | 2010-06-20 00:30:00   | 4.0         | 15.0        | 18.0         | summer | True      |
| 58                      | 2003.0 | 1.0   | Ohio       | OH          | ECAR        | Central           | 0.9           | warm             | intentional attack  | vandalism             | NaN             | 1440.0          | 7.4       | 7.09      | 4.62      | 6.4        | 5609731.0   | 3846164.0   | 4666822.0   | 14123130.0   | 39.720168    | 27.233085    | 33.043822    | 4791889.0      | 583171.0       | 22247.0        | 5397308.0       | 88.7829      | 10.8048      | 0.4122       | 43223.0          | 45858.0        | 0.94254        | 1.4                 | 9304.0       | 494250.0      | 1.882448     | 3.5           | 11434788.0  | 77.92        | 12.61      | 2033.7        | 1740.1      | 69.9         | 10.82          | 2.05         | 91.154687  | 8.845313       | 1.057422          | 2003-01-25 14:00:00  | 2003-01-26 14:00:00   | 5.0         | 14.0        | 25.0         | winter | False      |
| 1489                    | 2016.0 | 3.0   | Washington | WA          | WECC        | Northwest         | 1.6           | warm             | intentional attack  | sabotage              | NaN             | 1919.0          | 9.22      | 8.48      | 4.4       | 7.74       | 3318889.0   | 2463677.0   | 2014125.0   | 7797125.0    | 42.565548    | 31.597249    | 25.831637    | 2985799.0      | 367847.0       | 29012.0        | 3382664.0       | 88.2677      | 10.8745      | 0.8577       | 57796.0          | 50660.0        | 1.140861       | 4.3                 | 3504.0       | 420809.0      | 0.832682    | 0.7           | 7280934.0   | 84.05        | 9.08       | 2380.0        | 1487.9      | 16.7         | 3.57           | 0.62         | 93.208786  | 6.791214       | 2.405397          | 2016-03-10 04:00:00  | 2016-03-11 11:59:00   | 3.0         | 4.0         | 10.0         | spring | False      |
| 900                     | 2011.0 | 11.0  | Wyoming    | WY          | WECC        | West North Central | -1.0          | cold             | intentional attack  | vandalism             | NaN             | 0.0             | 9.54      | 7.75      | 5.72      | 6.83       | 237488.0    | 387208.0    | 904115.0    | 1528811.0   | 15.534163    | 25.327395    | 59.138442    | 258528.0       | 59872.0        | 9067.0         | 327467.0        | 78.9478      | 18.2834      | 2.7688       | 64163.0          | 47586.0        | 1.348359       | -0.7                | 917.0        | 36421.0       | 2.517778    | 0.4           | 567768.0    | 64.76        | 40.25      | 1876.2        | 1757.6      | 2.0          | 0.2            | 0.13         | 99.263902  | 0.736098       | 0.736098          | 2011-11-04 10:46:00  | 2011-11-04 10:46:00   | 4.0         | 10.0        | 4.0          | fall   | False      |
| 239                     | 2006.0 | 2.0   | California | CA          | WECC        | West              | -0.6          | cold             | severe weather      | winter storm          | NaN             | 2645.0          | 13.45     | 11.47     | 9.57      | 11.72      | 6390806.0   | 8585658.0   | 3962595.0   | 19005521.0   | 33.62605     | 45.174547    | 20.849705    | 12689438.0     | 1751882.0      | 79036.0        | 14520869.0      | 87.3876      | 12.0646      | 0.5443       | 54508.0          | 48909.0        | 1.114478       | 2.7                 | 29047.0      | 1963442.0     | 1.479392    | 11.9          | 36021202.0  | 94.95        | 5.22       | 4303.7        | 2124.1      | 12.7         | 5.28           | 0.59         | 95.164177  | 4.835823       | 1.730658          | 2006-02-27 18:25:00  | 2006-03-01 14:30:00   | 0.0         | 18.0        | 27.0         | winter | True      |



Here are the first few rows of our test set:

|                  | **YEAR** | **MONTH** | **U.S._STATE** | **POSTAL.CODE** | **NERC.REGION** | **CLIMATE.REGION** | **ANOMALY.LEVEL** | **CLIMATE.CATEGORY** | **CAUSE.CATEGORY** | **CAUSE.CATEGORY.DETAIL** | **HURRICANE.NAMES** | **OUTAGE.DURATION** | **RES.PRICE** | **COM.PRICE** | **IND.PRICE** | **TOTAL.PRICE** | **RES.SALES** | **COM.SALES** | **IND.SALES** | **TOTAL.SALES** | **RES.PERCEN** | **COM.PERCEN** | **IND.PERCEN** | **RES.CUSTOMERS** | **COM.CUSTOMERS** | **IND.CUSTOMERS** | **TOTAL.CUSTOMERS** | **RES.CUST.PCT** | **COM.CUST.PCT** | **IND.CUST.PCT** | **PC.REALGSP.STATE** | **PC.REALGSP.USA** | **PC.REALGSP.REL** | **PC.REALGSP.CHANGE** | **UTIL.REALGSP** | **TOTAL.REALGSP** | **UTIL.CONTRI** | **PI.UTIL.OFUSA** | **POPULATION** | **POPPCT_URBAN** | **POPPCT_UC** | **POPDEN_URBAN** | **POPDEN_UC** | **POPDEN_RURAL** | **AREAPCT_URBAN** | **AREAPCT_UC** | **PCT_LAND** | **PCT_WATER_TOT** | **PCT_WATER_INLAND** | **OUTAGE.START** | **OUTAGE.RESTORATION** | **DAY.OF.WEEK** | **TIME.OF.DAY** | **DAY.OF.MONTH** | **SEASON** | **IS.MAJOR** |
|------------------|----------|-----------|----------------|-----------------|-----------------|---------------------|-------------------|------------------------|---------------------|---------------------------|---------------------|---------------------|--------------|--------------|--------------|-----------------|----------------|---------------|---------------|-----------------|------------------|------------------|------------------|-------------------|---------------------|----------------------|-------------------|-----------------------|-----------------|------------------|-----------------|------------------|---------------|----------------|---------------|----------------|---------------|------------------|-------------------|----------------|------------------|---------------------|--------------------|---------------------|------------------------|------------------|----------------------|------------------|------------------|-----------------|-------------------|---------------|----------------|-------------------|-------------|---------------|-------------|------------------|---------------|------------------|---------------------|----------------|--------------|------------------|-----------------------|---------------------|--------------------------|----------------|------------------|------------------|-----------|--------------|
| 60           | 2003.0   | 4.0       | Wisconsin      | WI              | MRO             | East North Central | 0.0               | normal                 | intentional attack  | vandalism                 | NaN                 | 1219.0              | 8.79         | 7.15         | 4.73         | 6.65            | 1491193.0      | 1504236.0      | 2060297.0      | 5055727.0       | 29.495125       | 29.75311         | 40.751745       | 2446109.0          | 301434.0            | 5704.0           | 2753247.0          | 88.8445          | 10.9483          | 0.2072           | 43553.0             | 45858.0            | 0.949736          | 2.4                 | 4616.0          | 238635.0         | 1.934335         | 1.9              | 5479203.0       | 70.15           | 14.35         | 2123.3          | 1671.5         | 32.5            | 3.47              | 0.9            | 82.689019       | 17.312508          | 3.049041          | 2003-04-28 15:41:00 | 2003-04-29 12:00:00 | 0.0            | 15.0           | 28.0           | spring    | False          |
| 1055         | 2012.0   | 10.0      | New Jersey     | NJ              | NPCC            | Northeast          | 0.3               | normal                 | severe weather      | hurricanes               | Sandy               | 11337.0             | 15.17        | 12.13        | 9.98         | 12.9            | 1846305.0      | 2995476.0      | 621545.0       | 5486658.0       | 33.650813       | 54.595639        | 11.328299       | 3455302.0          | 489943.0            | 12729.0          | 3957980.0          | 87.2996          | 12.3786          | 0.3216           | 55571.0             | 48156.0            | 1.153979          | 1.5                 | 9159.0          | 493246.0         | 1.856883         | 3.0              | 8874893.0       | 94.68           | 2.44          | 2851.2          | 1446.5         | 105.5          | 39.7              | 2.01           | 84.305858       | 15.682678          | 4.99828           | 2012-10-29 16:03:00 | 2012-11-06 12:00:00 | 0.0            | 16.0           | 29.0           | fall      | True          |
| 267          | 2006.0   | 7.0       | Connecticut    | CT              | NPCC            | Northeast          | 0.1               | normal                 | severe weather      | thunderstorm             | NaN                 | 145.0               | 16.41        | 14.1         | 11.86        | 14.82           | 1454521.0      | 1321827.0      | 450441.0       | 3246923.0       | 44.796905       | 40.710143        | 13.872857       | 1437836.0          | 152984.0            | 5361.0           | 1596183.0          | 90.0796          | 9.5844           | 0.3359           | 67400.0             | 48909.0            | 1.378069          | 3.0                 | 3797.0          | 237075.0         | 1.601603         | 1.5              | 3517460.0       | 87.99           | 3.16          | 1721.9          | 1272.4         | 142.3          | 37.72             | 1.83           | 87.353419       | 12.646581          | 3.084972          | 2006-07-18 20:07:00 | 2006-07-18 22:32:00 | 1.0            | 20.0           | 18.0           | summer    | False          |
| 111          | 2004.0   | 2.0       | New York       | NY              | NPCC            | Northeast          | 0.3               | normal                 | public appeal       | NaN                     | NaN                 | 2400.0              | 14.02        | 12.07        | 6.98         | 11.93           | 4171308.0      | 6174482.0      | 1729385.0      | 12287262.0      | 33.94823        | 50.251081        | 14.074616       | 6794431.0          | 981964.0            | 10132.0          | 7786682.0          | 87.2571          | 12.6108          | 0.1301           | 55866.0             | 47037.0            | 1.187703          | 3.1                 | 20000.0         | 1071033.0        | 1.867356         | 7.7              | 19171567.0      | 87.87           | 5.21          | 4161.4          | 1700.0         | 54.6           | 8.68              | 1.26          | 86.38255        | 13.61745           | 3.645862          | 2004-02-14 20:00:00 | 2004-02-16 12:00:00 | 5.0            | 20.0           | 14.0           | winter    | False          |
| 724          | 2011.0   | 5.0       | Michigan       | MI              | RFC             | East North Central | -0.4              | normal                 | intentional attack  | vandalism                 | NaN                 | 200.0               | 13.37        | 10.6         | 7.39         | 10.36           | 2378750.0      | 3136896.0      | 2664590.0      | 8180730.0       | 29.077478       | 38.34494         | 32.571543       | 4249136.0          | 521322.0            | 12961.0          | 4783420.0          | 88.8305          | 10.8985          | 0.271            | 39953.0             | 47586.0            | 0.839596          | 2.5                 | 8716.0          | 394564.0         | 2.209021         | 3.6              | 9876589.0       | 74.57           | 8.19          | 2034.1          | 1390.4         | 47.5           | 6.41              | 1.03          | 58.459995       | 41.540005          | 2.068987          | 2011-05-04 12:20:00 | 2011-05-04 15:40:00 | 2.0            | 12.0           | 4.0            | spring    | False          |


### Prediction Problem: Classification
We are doing a binary classification to classify whether a power outage is major, which means `CUSTOMERS.AFFECTED` is greater than or equal to 50,000 and `DEMAND.LOSS.MW` is greater than or equal to 300. We will be working with different models and comparing their performances on the prediction task.

### Response Variable
The response variable, `IS.MAJOR`, is a binary variable indicating whether a power outage is major or not. It has two possible values: True for being a major outage event and False for not being a major outage event.

### Justification for Response Variable
We choose to classify whether a power outage is major because understanding the severity of an outage in real-time is crucial for local authorities and organizations to make informed decisions and handle the ramifications of the events.

### Features
Using `CAUSE.CATEGORY`` as the only feature for prediction can achieve an accuracy of 85%-90% on the test set, but once it is used along with other features, it overshadows all other features. We will not be using `CAUSE.CATEGORY` as one of our features in both the baseline and the final models because we want to build a model that takes into account more factors, even if the other features will not have a performance that is as impressive as using only CAUSE.CATEGORY. We will be exploring other features that are available. Usually, we want to predict whether an outage was major right after it ended. At the time of prediction, we will not be able to immediately count the number of people affected or the amount of loss. Instead, we only have access to real-time information related to the outage, such as the aggregate data of local customers, and basic information about the specific outage, such as the time the outage happened and how long it lasted. 

### Metric for Evaluation
To evaluate the model's performance, we could have chosen metrics such as accuracy or precision. However, we are more interested in correctly identifying those events that are indeed major and not mistakenly dismissing them as non-major. Thus, we decided to use recall as our metric for evaluation. In our case, recall is the proportion of correctly identified major events out of all major events. On the other hand, accuracy measures the overall correctness of a model but lacks a specific focus, and precision cares about the proportion of actual major outages out of all outages that are classified as major, in which our model has slightly less interest. 

---

## Baseline Model

### Model Description
The model used in this prediction task is a logistic regression model. The selected features for the model are `OUTAGE.DURATION`, and `TIME.OF.DAY`. We standardized the `OUTAGE.DURATION` feature and binned the `TIME.OF.DAY` feature into intervals during pre-processing. 

### Features
* `OUTAGE.DURATION`: This is a quantitative feature representing the duration of the power outage in minutes. It is a numerical variable.
* `TIME.OF.DAY`: This is an ordinal feature indicating the hour of the day when the power outage event started. It is a categorical variable obtained from the `OUTAGE.START` feature. 

### Encoding
* During pre-processing, we used standardizer in sklearn to standardize the `OUTAGE.DURATION` feature. 
* We used KBinsDiscretizer to bin the ordinal feature `TIME.OF.DAY` into intervals. This encoding technique uses one-hot encoding to create binary columns for each unique bin, indicating which bin the `TIME.OF.DAY` value falls in.
* The 'remainder' parameter in the ColumnTransformer is set to `drop`, which means columns that are not passed in as arguments will be dropped from the model fitting process.

### Model Performance
For the testing set, the model achieved an accuracy of 68.23%, a precision of 68.26%, and a recall of 68.23%.

| Metric          | Score   |
|-----------------|---------|
| Accuracy        | 68.23%  |
| Precision       | 68.26%  |
| Recall          | 68.23%  |

In our dataset, 53% of the observations are major outages, whereas around 46% are not. 

| IS.MAJOR | Probability  |
|----------|--------------|
| True     | 0.532595     |
| False    | 0.467405     |

We think the accuracy score is not high enough because, if the model predicts all outages to be true, it will have an accuracy of around 53%. The accuracy we have right now is not very big of an improvement from 53%. 

### Summary
Upon analyzing the performance of the current model, there are a few observations to consider. The model achieves high accuracy, indicating that it is able to make correct predictions for the majority of instances. However, the precision and recall scores are relatively lower compared to the accuracy.

Precision represents the proportion of correctly predicted positive instances out of the total instances predicted as positive. In this context, it measures the ability of the model to correctly identify power outages occurring in the West Climate region. The precision score of 0.617 suggests that the model has some difficulty in precisely identifying these instances. There is a possibility of false positives, where the model incorrectly classifies a power outage as occurring in the West Climate region.

Recall, also known as sensitivity or true positive rate, represents the proportion of correctly predicted positive instances out of the actual positive instances. It measures the ability of the model to capture all the power outages occurring in the West Climate region. The recall score of 0.649 indicates that the model captures a substantial portion of these instances but still misses some, resulting in false negatives.

Considering the imbalanced nature of the classes (West Climate region vs. other regions), where the West Climate region may be a minority class, the lower precision and recall scores can be partly attributed to the class imbalance. It is important to note that optimizing for both precision and recall can be a trade-off, and the choice depends on the specific requirements of the problem. In some cases, precision might be more critical, while in others, recall might be the primary concern.

To further improve the model's performance, additional steps can be taken, such as:

- Feature engineering: Explore and include additional relevant features that might have a strong correlation with the West Climate region. These features could provide more discriminatory power and improve the model's predictive ability.

- Hyperparameter tuning: Experiment with different hyperparameter settings for the logistic regression model or try alternative classification algorithms to find a configuration that better balances precision and recall.

- Handling class imbalance: Implement techniques to address the class imbalance issue, such as oversampling the minority class (West Climate region) or using weighted loss functions to give more importance to the minority class during training.

Upon all these analysis, we will continue and use these strategies to train our final model using random forest classifier, which will be better and easier at hyperparam tuning process.

---

## Final Model

### Model Choosing and features:
After conducting several trials, we have decided to use the random forest classifier as our model for two main reasons. Firstly, although logistic regression performs well as a baseline model, it has a limited number of tunable hyperparameters compared to other models. This makes it challenging for us to fine-tune the final model effectively. Secondly, our dataset contains numerous categorical features, suggesting that a classifier may be a better choice. Here are the features we have chosen for our model:

- `CLIMATE.CATEGORY`: This feature is transformed using one-hot encoding. It provides valuable information about a region's climate, as different climate categories can have distinct patterns or effects on the predicted climate region.

- `CAUSE.CATEGORY.DETAIL`: Transformed using one-hot encoding, this feature captures specific details about the cause category, which could influence the climate region.

- `CAUSE.CATEGORY`: Also transformed using one-hot encoding, this feature represents the cause category of the outage and offers insights into the climate region.

- `PC.REALGSP`: Scaled using StandardScaler, this feature represents the real gross state product and is standardized to have a mean of 0 and unit variance. It may be relevant for predicting the climate region.

- `OUTAGE.DURATION(mins)`: Scaled using StandardScaler, this feature represents the duration of the outage in minutes. Scaling helps handle features with different scales and magnitudes. The duration of the outage may be related to the climate region, as regions with severe weather conditions may experience longer power outages.

- `PI.UTIL.OFUSA(%)`: Scaled using StandardScaler, this feature represents the utility of power infrastructure in the USA. Scaling is applied for consistency and to prevent dominance by features with larger values. The distribution of power infrastructure across different areas can provide information about the climate.

### Model Performance
Even before tuning any hyperparameters, our final model already outperformed the previous version. The test accuracy is approximately 93%. We utilized GridSearch to find the best hyperparameter values, including the number of estimators, maximum features, and minimum sample splits. This tuning resulted in a better-performing model, although the improvement was not significant. Ultimately, our accuracy reached around 93.5%. The recall is approximately 0.95, and the precision is around 0.67, showing significant improvement compared to our baseline model.

### Summary
The random forest classifier yielded promising results for our prediction task. It enhanced the overall accuracy and recall of our model. Moreover, this process highlighted the distinct advantages of each model. Logistic regression performed well even with a limited number of features, but the random forest classifier demonstrated superior performance when additional features were incorporated, along with optimized hyperparameters. Model selection is a crucial step in this process, and fine-tuning hyperparameters can further enhance model performance.

---

## Fairness Analysis

### Accuracy Analysis
For our fairness assessment, we have categorized the test dataset into two groups: power outages affecting more than 50,000 people and those affecting fewer than 50,000 individuals. An outage affecting more than 50,000 people is considered a severe outage. Our primary evaluation metric is accuracy. We propose a null hypothesis asserting that our model's accuracy for determining outage severity is roughly equivalent across all cases, with any observed differences attributable to random variability. Conversely, our alternative hypothesis suggests that the model demonstrates unfairness, with a higher accuracy for less severe power outages than for severe ones. We have selected the accuracy disparity between less severe and severe outages as our test statistic, with a significance level of 0.01. After running a permutation test 5,000 times, we obtained a p-value of 0.1286, which exceeds our significance level. This outcome leads us to retain the null hypothesis, indicating that our model, based on this accuracy metric, is fair. However, we cannot definitively assert its complete fairness as the permutation test results are also contingent on random chance. Hence, we recommend further testing with more data to verify if it is 'truly fair'.

<iframe src="asset/fairness1.html" width=800 height=600 frameBorder=0></iframe>


### R squared Analysis
Our secondary evaluation metric is the R-squared value. We present a similar null hypothesis suggesting that our model's R-squared value for determining outage severity is approximately equal in all scenarios, and any discrepancies are due to random chance. Alternatively, our model could be unfair if the R-squared value is higher for less severe outages compared to severe ones. We've chosen the R-squared difference between less severe and severe outages as our test statistic, with a significance level of 0.01. Upon executing a permutation test 5,000 times, we obtained a p-value of 0.0112. As this value is above our significance level, we cannot reject the null hypothesis, suggesting that our model is fair based on the R-squared analysis. However, it is worth noting that this p-value is very close to our set significance level. Given that permutation test results can vary with each iteration, we cannot confidently assert that our model is fair based on the R-squared fairness analysis alone.

<iframe src="asset/fairness2.html" width=800 height=600 frameBorder=0></iframe>

---
