# Machine Learning Assignment 1
#
# Student number: C15423602
# Student name: Sylvia Siu Wei Wan
# Course: DT282/4
#

import numpy as np
import pandas as pd

# Read and loads the data set as a table
header = r'./data/feature_names.txt'
with open(header, 'r') as file: headerVal = file.read().split()
filename = r'./data/dataset.txt'

df = pd.read_csv(filename, names=headerVal)

# dataset table
print(df)

# Showing the different data types of the features
print(df.dtypes)


###### Continuous Data #######

# copying continuous feature into dataframe
contDF = df.select_dtypes(include=['int64']).copy()

# Refer to .describe() table for error checking
contDF.describe().round()

# Count
contCount = pd.DataFrame(contDF.count(), columns=['Count'])

# Percentage of missing values
contMiss = pd.DataFrame(contDF.isnull().sum()*100 / contDF.count(), columns=['Miss %']).round(2)

# Cardinality
contCard = pd.DataFrame(contDF.nunique(), columns=['Card.'])

# Minimum values
contMin = pd.DataFrame(contDF.min(), columns=['Min'])

# First Quartile
contFirstQrt = pd.DataFrame(columns=['1st Qrt'])
for i in list(contDF.columns.values): contFirstQrt.loc[i] = [contDF[i].quantile(0.25)]

# Mean
contMean = pd.DataFrame(contDF.mean().round(2), columns=['Mean'])

# Median
contMedian = pd.DataFrame(contDF.median().round(2), columns=['Median'])

# Third Quartile
contThirdQrt = pd.DataFrame(columns=['3rd Qrt'])
for i in list(contDF.columns.values) : contThirdQrt.loc[i] = [contDF[i].quantile(0.75)]

# Maximum values
contMax = pd.DataFrame(contDF.max(), columns=['Max'])

# Standard deviation
contStd = pd.DataFrame(contDF.std().round(), columns=['Std Dev'])

# Joining all of the frames
contFrames = [contCount, contMiss, contCard, contMin, contFirstQrt, contMean, contMedian, contThirdQrt, contMax, contStd]

contReport = pd.concat(contFrames, axis=1, sort=False)

# Attempt to set first data cell as "Featured names"
# contReport['Feature'] = list(contDF)
# contReport.set_index("Feature names", inplace=True)
# contReport = contDF.groupby(['Features'], as_index=False)
# contReport.head()

print(contReport)


###### Catagorical Data ######

# copying categorical features into datafram
catDF = df.select_dtypes(include=['object']).copy()

# Since 'id' is not considered in the categorical table, it'll get dropped
catDF.drop(['id'], axis=1, inplace=True)

print(catDF)

# Refer to the .describe() table for error checking
print(catDF.describe())

# Count
catCount = pd.DataFrame(catDF.count(), columns=['Count'])

# Cardinality
catCard = pd.DataFrame(catDF.nunique(), columns=['Card.'])

# Percentage of missing values
catMiss = pd.DataFrame(catDF.isnull().sum()*100 / catDF.count(), columns=['Miss %']).round(2)

# Mode
catMode = pd.DataFrame(columns=['Mode'])
for i in list(catDF.columns.values): catMode.loc[i] = [catDF[i].mode().to_string(index=False)]

# Mode Frequency
catFreq = pd.DataFrame(columns=['Mode Freq'])
for i in list(catDF.columns.values): catFreq.loc[i] = [catDF[i].value_counts().values[0]]

# Mode Percentage
catModePercent = pd.DataFrame(columns=['Mode %'])
for j in list(catDF.columns.values):
    catModePercent.loc[j] = [(catDF[j].value_counts().values[0]*100) / (catDF[j].count()-catDF[j].isnull().sum())]

# Second Mode
catSecMode = pd.DataFrame(columns=['2nd Mode'])
for j in list(catDF.columns.values): catSecMode.loc[j] = [catDF[j].value_counts().index[1]]

# Second mode frequency
catSecFreq = pd.DataFrame(columns=['2nd Mode Freq'])
for j in list(catDF.columns.values): catSecFreq.loc[j] = [catDF[j].value_counts().values[1]]

# Second mode percentage
catSecPercent = pd.DataFrame(columns=['2nd Mode %'])
for j in list(catDF.columns.values):
    catSecPercent.loc[j] = [(catDF[j].value_counts().values[1]*100) / (catDF[j].count()-catDF[j].isnull().sum())]

# Joining all the frames
catFrames = [catCount, catMiss, catCard, catMode, catFreq, catModePercent, catSecMode, catSecFreq, catSecPercent]

catReport = pd.concat(catFrames, axis=1, sort=False)

# Attempt to set first data cell as "Featured names"

# catReport['Feature'] = list(catDF)
# catReport.set_index("Feature names", inplace=True)
# catReport.head()
# catReport = df.groupby(['Features'], as_index=False)

print(catReport)

# Export data to csv files
contReport.to_csv('./data/c15423602CONT.csv')
catReport.to_csv('./data/c15423602CAT.csv')