<img src="http://imgur.com/1ZcRyrc.png" style="float: left; margin: 20px; height: 120px">

# A Fair Appraisal: Predicting Housing Prices in Ames, IA
### Fill out this cell as the project progresses, then move to README.md as technical report.

*Deval Mehta*

## Table of Contents
1) [Overview](#Overview) 
2) [Data](#Data-Dictionary)
3) [Requirements](#Requirements)
4) [Executive Summary](#Executive-Summary)
    1) [Purpose](#Purpose)
    2) [Methods](#Methods)
    3) [Findings](#Findings)
    4) [Next Steps](#Next-Steps)

## Overview
Tax appraisals for properties provide a wealth of information regarding various aspects of the home, some of which the homeowners and prospective buyers may never even think about! Professor of Statistics Dane De Cock took advantage of this to create the Ames Housing Dataset for his Statistical Regression students, intended to be the focus of a capstone project. The question at hand is simple: considering all of the information available regarding the attributes of various homes listed for sale in Ames, IA in 2008, can we reliably predict the fair price of an additional 2008 Ames, IA listing?

We employ linear regression to attempt to answer this question successfully, iteratively improving models as we go through data cleaning and exploration. Rather than cleaning all of our data at once, we consider the impact of various types of data on the model and aim to generate a model which predicts housing prices in the Ames of 2008 as reliably as possible. We set for ourselves a target margin of \\$20,000 which we were sadly unable to meet, but we believe it is possible with some additional time and work. In principle, this work may be extended to analogous data from any locale in any year. Ideally, we would provide a tool by which homeowners may independently appraise the value of their home in preparation to list it.

## Data Dictionary

### Original Features
The original dataset contains 79 non-index, non-price features, each introducing a different piece of information regarding a listing.

| Variable | Data Type | Description | Notes |
|---|---|---|---|
| MS SubClass | `int64` | The classification of the building | Codified to numbers; see the original data documentation for the cipher |
| MS Zoning | `string` | General zoning classification of the sale | Codified into strings; see the original data documentation for the cipher |
| Lot Frontage | `float64` | Linear feet of street connected to the property | |
| Lot Area | `int64` | Lot size in square feet | |
| Street | `string` | Type of road access to property | Gravel or Paved |
| Alley | `string` | Type of alley access to property | Gravel or Paved |
| Lot Shape | `string` | General shape of property | Degree of irregularity |
| Land Contour | `string` | Flatness of the property | Level, Banked, Hillside, or Low Depression |
| Utilities | `string` | Type of utilities available | Electric, Gas, Water, Sewer |
| Lot Config | `string` | Lot Configuration | Where on a block or in a neighborhood the lot lands |
| Land Slope | `string` | Slope of the property | Categorized from "gentle" to "severe" |
| Neighborhood | `string` | Physical locations within Ames city limits | |
| Condition 1 | `string` | Proximity to main road or railroad | |
| Condition 2 | `string` | Proximity to main road or railroad | if a second is present |
| Bldg Type | `string` | Type of dwelling | More modern classification than MS SubClass |
| House Style | `string` | Style of dwelling | Number of stories |
| Overall Qual | `int64` | Overal material and finish quality | Scale 1 - 10 |
| Overall Cond | `int64` | Overall condition rating | Scale 1 - 10 |
| Year Built | `int64` | Original construction date | |
| Year Remod/Add | `int64` | Remodel date | Same as construction date if no remodeling or additions |
| Roof Style | `string` | Type of roof | |
| Roof Matl | `string` | Roofing material | |
| Exterior 1st | `string` | Exterior covering on house | |
| Exterior 2nd | `string` | Exterior covering on house | if more than one material |
| Mas Vnr Type | `string` | Masonry veneer type | |
| Mas Vnr Area | `int64` | Masonry veneer area | |
| Exter Qual | `string` | Exterior material quality | Scale Poor to Excellent (six grades) |
| Exter Cond | `string` | Present condition of the material on the exterior | Same grading system as Exter Qual |
| Foundation | `string` | Type of foundation | |
| Bsmt Qual | `string` | Height of basement | |
| Bsmt Cond | `string` | General condition of basement | |
| Bsmt Exposure | `string` | Walkout or garden level basement walls | |
| BsmtFin Type 1 | `string` | Quality of basement finished area | |
| BsmtFin SF 1 | `int64` | Type 1 finished square footage | |
| BsmtFin Type 2 | `string` | Quality of second finished area (if present) | |
| BsmtFin SF 2 | `string` | Type 2 finished square feet | |
| Bsmt Unf SF | `int64` | Unfinished square feet of basement area | |
| Total Bsmt SF | `int64` | Total square feet of basement area | |
| Heating | `string` | Type of heating | |
| Heating QC | `string` | Heating quality and condition | |
| Central Air | `string` | Central air conditioning | |
| Electrical | `string` | Electrical system | |
| 1st Flr SF | `int64` | First floor square footage | |
| 2nd Flr SF | `int64` | Second floor square footage | if it exists |
| Low Qual Fin SF | `int64` | Low quality finished square footage | all floors |
| Gr Liv Area | `int64` | Above grade living area square footage | |
| Bsmt Full Bath | `int64` | Number of basement full bathrooms | |
| Bsmt Half Bath | `int64` | Number of basement half bathrooms | |
| Bedroom AbvGr | `int64` | Number of bedrooms above grade | |
| Kitchen AbvGr | `int64` | Number of kitchens above grade | |
| Kitchen Qual | `string` | Kitchen quality | |
| TotRms AbvGr | `int64` | Total rooms above grade | does not include bathrooms |
| Functional | `string` | Home functionality rating | |
| Fireplaces | `int64` | Number of fireplaces | |
| Fireplace Qu| `string` | Fireplace quality | |
| GarageType | `string` | Garage type and location | |
| Garage Yr Blt | `float64` | Year garage was built | |
| Garage Finish | `string` | Interior finish of the garage | |
| Garage Cars | `int64` | Car capacity of garage | |
| Garage Area | `int64` | Square footage of garage | |
| Garage Qual | `string` | Garage quality | |
| Garage Cond | `string` | Garage condition | |
| Paved Drive | `string` | Paved driveway | |
| Wood Deck SF | `int64` | Wood deck area in square feet | |
| Open Porch SF | `int64` | Open porch area in square feet | |
| 3Ssn Porch | `int64` | Three season porch area in square feet | |
| Screen Porch | `int64` | Screen porch area in square feet | |
| Pool Area | `int64` | Pool area in square feet | |
| Pool QC | `string` | Pool quality | |
| Fence | `string` | Fence quality | |
| Misc Feature | `string` | Miscellaneous feature not covered in other categories | |
| Misc Val | `int64` | Dollar value of miscellaneous feature | |
| Mo Sold | `int64` | Month sold | |
| Yr Sold | `int64` | Year sold | |
| Sale Type | `string` | Type of sale | |
| Sale Price | `int64` | The property's sale price in dollars | This is our response variable for our models |

### Engineered Features
| Variable | Data Type | Description | Notes |
|---|---|---|---|
| Kitchen Exter Qual | `float64` | Interaction between the exterior and kitchen quality features | |
| Fireplace Index | `float64` | Product of the number of fireplaces and the fireplace quality in a given home | |
| Garage Space | `float64` | Product of the number of cars that fit in a garage and the area of the garage in square feet | |

## Requirements
To replicate our analysis and predictive modeling, the following modules are necessary:


| Library | Module | Purpose |
|---|---|---|
| `numpy` | | Ease of basic aggregate operations on data |
| `pandas` | | Read our data into a DataFrame, clean it, engineer new features, and write it out to submission files |
| `matplotlib` | `pyplot` | Basic plotting functionality |
| `sklearn` | `compose` | Column transformation |
| `sklearn` | `impute` | Imputation methods |
| `sklearn` | `linear_model` | to write SLR and MLR models |
| `sklearn` | `metrics` | Evaluate our models |
| `sklearn` | `model_selection` | Perform train-test splitting |
| `sklearn` | `preprocessing` | Data preprocessing and feature engineering tasks |
| `seaborn` | | More control over plots |
| `warnings` | | Suppress many of the warnings `pandas` flags in response to things like using `inplace` arguments |

A prospective colleague or student interested in replicating our results or improving upon them would also require access to the [Ames Housing Dataset by Dean de Cock](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset). In our case, we have saved this data within the `datasets` directory, split into the [train](../datasets/'train.csv') and [test](../datasets/test.csv) files.

## Executive Summary
### The Data
Our goal is to predict the price of homes listed for sale in Ames, IA, given information about the properties in question. This is a much wider dataset than any with which we have previously worked. The data consists of 80 columns, ranging from zoning classification to the quality and condition of various parts of the home and exterior. As we progress with analysis and modeling, we may find that we have to engineer new features to build a more accurate predictive model. Our models will be built on linear regression, perhaps with limited polynomial features, since we are interested in predicting the sale price, which is a quantitative feature.

We identified some redundant information here, which allowed us to pare down the data a bit. The "overall" numbers are all aggregates or combinations of the individual values, so we can count them out in our analysis. We will also want to convert our "sliding scale" variables to a numeric data type, then convolve some of them. In particular, the area of something is likely to interact with its quality. There are many missing values in the data, as will be seen below. We want to reasonably impute as many of them as possible.

Regarding the missing values, it seems many of the categorical features have been encoded as "NA" to mean that no relevant property exists. This will be relatively easy to impute if we have a common way to refer to such an incident in mind that cannot be overwritten by a null-type.

### Establishing a Baseline
We create a baseline prediction before considering anything else about the data. We see that one of our features is `Overall Quality`, which is likely an aggregate of all of the other `quality` features in the dataset. We have already generated a baseline relying on this feature as the only predicate in breakfast hour, which we add here for completeness. To ensure that the baseline is the same across the cohort, we select a random seed here.

Our baseline does not perform remarkably well, with an $R^2$ of about 0.63 on the training data, suggesting that only 63\% of the variability in the data can be accounted for due to the features of the training set. That said, the model is not overfit, as the testing $R^2$ is comparable to the testing $R^2$.

### Native Numeric Variables - A Preliminary Model

Broadly speaking the features can be split into 12 groups, based on the information in the data dictionary, which provide insights into similar or related aspects of a given house:

* Identifying Information and Access (ID, PID, Lot Frontage, Street, Alley) - These will likely have very little impact on housing prices
* Building Classification (MS SubClass, Bldg Type, House Style, Year Built, Year Remodeled, Roof Style)
* Lot (Lot Area, Lot Shape, Land Contour, Lot Config, Land Slope)
* Location (Neighborhood, Condition 1, Condition 2)
* General (Overall Quality, Overall Condition)
* Exterior (Exterior 1st, Exterior 2nd, Exterior Quality, Exterior Condition, Paved Driveway, Miscellaneous Feature, Miscellaneous Value)
* Structure (Type of Foundation, Roof Material, Masonry Veneer Type, Masonry Veneer Area, Utilities, Heating, Heating Quality, Central Air, Electrical)
* Basement (Basement Quality, Basement Condition, Basement Exposure, Basement Finish Type 1, Basement Finish Type 2, Basement Square Footage Type 1, Basement Square Footage Type 2, Basement Unfinished Square Footage, Total Basement Square Footage)
* Rooms Above Grade (Total Rooms Above Grade, Bedrooms Above Grade, Kitchens Above Grade, Kitchen Quality)
* Fireplaces (Fireplaces, Fireplace Quality)
* Garages (Garage Type, Garage Year Built, Garage Finish, Garage Cars, Garage Area, Garage Quality, Garage Condition)
* Yard (Wood Deck Square Footage, Open Porch Square Footage, 3 Season Porch, Screen Porch, Pool Area, Pool Quality)

We will check how well many features correlate to our desired response (price), but before we do that, we can pare down the data a bit. The "quality" and "condition" features where they both exist are likely to provide the same information about an attribute. As such, we can maintain on the condition features and process them as numeric types, translating the "sliding scale" of strings to a 0-5 scale of ints. Where required, we can adjust the top end of the scale to be the number of available options.

Before we can do that, we will have to change all the "NA" entries. The consequences of simply dropping the data are catastrophic. The same is true of the testing data. We will have to selectively drop columns and impute otherwise. To minimize the number of columns we need to drop, let's first impute where we can. We'll replace the NaNs in categorical columns with the initialism "NV" for "No Value." That should get around the encoding issue present.

In order to effectively impute the data, we will have to treat categorical and numerical variables separately. There are only three different data types present in the data: `int64`, `float64`, and `object`. We can filter the data frame by `object` type columns to gather all the categorical data and impute it before transforming the "condition" columns as we previously mentioned. We should also impute the testing data whenever we impute the training data, to ensure that our predictions remain accurate. If the withheld "secret" data is not cleaned, we expect our score to be worse than would be accurate, as a validation algorithm would not necessarily clean data the way we are.

Since we're interested in performing a predictive analysis via linear regression, we ought to clean up the numerical variables as well. While we do that, we can do some preliminary EDA to see which variables we might pick for our first analysis, beyond a baseline. Let's have a look at the "state of the NaNs." 11 of our numeric columns contain null values. With this information, let's do some initial EDA to see how well each of the numeric variables (as they are) correlate to `SalePrice`. We can use this information to decide whether it might be worth imputing null-values in some of these columns. To check, we'll create a heatmap, much like we did in Lesson 305: Model Workflow.

<img src = ./images/preliminary_saleprice_heatmap.png>

There appears to be no easily apparent "cutoff" for what we might consider to be highly correlated. We can take a token from Principal Component Analysis (thanks to DSB Lead Instructor Matt Brems for mentioning this well in advance of our PCA lesson) to determine which features are worth considering for a regression analysis. The "elbow method" advocates plotting the features (sorted by correlation) on line plot or bar chart and identifying sharp drops between items. Where "the elbow bends," we would call our cutoff.

<img src = ./images/elbow_method.png>

Ignoring the initial "elbow" at `Overall Qual` which occurs for the obvious reason of nothing correlating as well to `SalePrice` as it does to itself, we have a very "early elbow" at the `Garage Area` feature, but we can imagine that a multiple linear regression model using only three features from a set of 79 would likely be terribly underfit. Instead, we look for our second "elbow," which we identify at the `BsmtFin SF 1` feature, denoting the area of the "1st" finished part of the basement. Note the steep descent from this feature to `Lot Frontage`. Choosing this to be the cutoff for our first multiple linear regression model provides us with 14 features of 79 to play with. Normally, we would seek another "elbow" to filter for strongly anti-correlated features, but seeing as the mostly highly anti-correlated feature has a correlation coefficient of -0.26 (compare that against the 0.42 for our "strong" cutoff), that seems unnecessary here. Incidentally, the only features in our restricted list with null values are now:
* `Garage Area`
* `Garage Cars`
* `Total Bsmt SF`
* `Garage Yr Blt`
* `Mas Vnr Area`
* `BsmtFin SF 1`

Thankfully, most of these have only 1 null value, so we will be able to identify the relevant rows and impute them as necessary.

The null values in our columns of interest are not isolated, as they have other corresponding information. For instance, listing 910201180 has a detached garage, with negligible information regarding it available. In a situation like this, our best option is to assume that the missing data falls within the expectation of our eventual model and simply delete it: we will *hope* that it is Missing Completely at Random (MCAR). In fact, in the interest of time and producing a preliminary model, we will simply remove all of the rows with a null value in any of our columns of interest. Note that we **cannot** do this for our testing data. Kaggle submissions require that there are precisely 878 rows of data, so we must impute the validation data. In the interest of time, we will impute each missing numerical values with 0, but ideally, we would have instantiated and used Simple Imputation to replace the missing values with the mean, assuming that the validation set is also MCAR.

We've already fit our baseline model to a simple linear regression. This preliminary multiple linear regression model aims to predict the price of a house for sale in Ames using only the native numeric columns in the dataset. We perform a train-test split to generate this prediction. Ideally we would have liked to use k-fold cross validation with the typical 5 folds and shuffling. We could have refined our model further by attempting to optimize over the number of folds.

According to our $R^2$ scores, roughly 80% of the variability in the data can be explained by our features and our model is relatively well-fit, since the testing and training $R^2$ scores are similar (the difference of the two is less than $10\%$ of the training $R^2$ score.

We would expect that this model will score around 35,000 on Kaggle, meaning that the prices predicted per house will be within $35,000 of the true value. Considering housing prices in 2008, this falls within 10% to 35% of the middle 50% of homes [citation needed]. Let's consider this visually as well.

<img src = ./images/preliminary_true_vs_predicted.png>

As we can see, most of our value-prediction pairs lie quite close to the line of equality. In principle, we could happily stop here, but we know we can do better. Let's make our Kaggle submission and consider a more robust model.

### Preparing a More Robust Model

Let us return to the *ordinal* categorical features. Rather than one-hot encoding all of the categorical features, we see that many of them are "sliding scales." In particular, we turn our attention to `Exter Qual`, `Exter Cond`, `Bsmt Qual`, `Bsmt Cond`, `Bsmt Exposure`, `BsmtFin Type 1`, `BsmtFin Type 2`, `Heating QC`, `Kitchen Qual`, `Fireplace Qu`, `Garage Qual`, `Garage Cond`, and `Pool QC`. Most of these features scale from "NV" (no value) to "Ex" (Excellent). We define a dictionary, as we did in breakfast hour on Friday, to convert all of these to numerical data types so that we might use them to build another model that (we hope) will perform better. We note that three of our ordinal features still have a different scale than the rest: `Bsmt Exposure`, `BsmtFin Type 1`, `BsmtFin Type 2`. As such, we will have to determine their possible values. Let us inspect the available values on these columns and create new ordinal mappings accordingly.

Each of these "quality" features likely contribute to the `Overall Qual` feature and we can imagine that the "condition" features is practically colinear to the quality feature to which it corresponds. To verify, let us create another heatmap against `Overall Qual`

<img src = ./images/quality_heatmap.png>

As we might suspect, the features that are present in most Ames homes (an exterior, a kitchen, and a basement) have quality ratings that contribute heavily to the overall quality, but the remaining features (anti)correlate weakly. Perhaps we may *not* have expected that the conditions deteriorate as the quality improves. We suspect something is going wrong here, so will neglect the condition features in our analysis for the sake of time. 

We can now construct a new model based on the native numeric features we isolated earlier, along with the ordinal features that strongly correlate to `Overall Qual`. To avoid colinearity issues, we will exclude `Overall Qual`. Further, we should consider this time whether any of of the features we have selected are correlated with each other. To check, we will consider both a heatmap of correlations and a pairplot.

<img src = ./images/num_and_ord_heatmap.png>

So there is some strong colinearity! The features that correlate highly to each other seem obvious:
* Fireplace quality and the number of fireplaces are related; homes with multiple fireplaces tend to maintain them better
* Garage area and garage cars are colinear; more cars require more space
* The garage and home tend to be built around the same year
* The basement and first floor have nearly identical areas
* Homes that have been more recently remodeled have higher quality across the board and many homes were never remodeled
* Homes with higher quality kitchens also tend to have higher quality exteriors
* A higher number of rooms above grade typically means that the living area above grade is also higher.

Let's plot the features that interact in pairs against each other and see what kind of relationship they have.

<img src = ./images/potentially_collinear_features.png>

A few takeaways here:
* The first floor and basement areas are roughly colinear, so we will opt to include only the first floor area
* The year the home and garage were built are also highly colinear, so we will opt to include only the year the home was built.
* The living area above grade and the total number of rooms above grade are highly colinear as well, so we will opt to include on the living area above grade.
* The remaining plots are all too sparse to call them colinear in any sense. Perhaps they will serve our model better as interactions. We will attempt to add interaction terms after running a model without interactions first.

With this information in tow, let us redefine the set of columns we will use for our next model.

<img src = ./images/num_and_ord_true_vs_predicted.png>

We have some mild improvement on our preliminary model. Rather than an RMSE around \\$35,000, we now expect our Kaggle submission to score around an RMSE of \\$33,500. Once again, we can attribute about 80\% of the variability in the model to the training features, but the model is not overfit, since it performs comparably on the testing data and training data. We can confirm this visually as well.

### Impact from Interaction Terms
Before scaling the data, we'll define the interaction terms we proposed above. Perhaps our model will be more accurate with interactions between features that appear to have some level of correlation, but are not colinear.

<img src = ./images/num_and_ord_with_interaction_true_vs_predicted.png>

Once again, we have closed the gap a bit. This time, we expect our RMSE to be around \\$32,250. Our $R^2$ throughout the process has remained around 80\% on the training set. Interestingly enough, we have a bit of a rise this time on the testing set. Model performance is consistently increasing on the testing data.

## Scaling - The Final Attempt
Now we come to the final pre-processing step and perhaps the one procedure that may lead to great improvement in our predictive ability: scaling. We have mentioned a few times now that scaling appears to be necessary due to the curving tail on the right end of our plots comparing the true and predicted price values. In reality, the curve appears to be something like a square root and we believe this to be the case because the features included in our regression models are on such different scales. If we scale all of our features of interest using the `StandardScaler()`, we can renormalize and recenter, allowing the parameters of the model to adjust in a way that provides a "fairer" contribution from each feature. We apply the same procedure we learned in Lesson 304 - Feature Engineering, with some modification. To ensure that our testing data is not affecting our training data, we will perform a train-test split first, then transform the data.

<img src = ./images/scaled_true_vs_predicted.png>

Scaling appears to have made a slight amount of a difference, though not as much as we would have liked. Across all of our models, we have accounted for 80\% of the variability in the model with the features in the training set and consistently done better on the testing set. Were more time available, the next step would be to regularize using `LassoCV`.

### Next Steps
Though we did attempt it, we were unable to successfully one-hot encode the non-ordinal catagorical features of our dataset. We would like to consider the impact of some of the salient categorical variables in a future model. In addition, time constraints did not permit the use of regularization techniques, which would likely be the final step, particularly employing LASSO regularlization, which would allow us to enter all of the features in the dataset and pare them down through regularization. When regularizing, we would likely also lean on `PolynomialFeatures` to study the impact of a greater number of interactions between features and feature self-interactions.