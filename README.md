## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
#### Name : SHANMUGAVEL RM
#### Ref No : 212222230142
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![image](https://github.com/22008650/EXNO-3-DS/assets/122548204/605e13a3-5e7d-48d1-b849-4fc92477bc5e)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/22008650/EXNO-3-DS/assets/122548204/b89b1882-1a2e-4f3d-800f-d29026e91603)


```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/22008650/EXNO-3-DS/assets/122548204/bab8d528-dec6-4866-aac4-488ac89c96ba)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/22008650/EXNO-3-DS/assets/122548204/70cc74f8-4bf9-4a47-8104-663444df062d)

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```
![image](https://github.com/22008650/EXNO-3-DS/assets/122548204/f571ed58-56ad-4939-8a31-b8b00b89081f)


```
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/22008650/EXNO-3-DS/assets/122548204/1d15705d-75a7-4de6-b8d1-3eb577b694b1)

```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/22008650/EXNO-3-DS/assets/122548204/927828d3-f11f-44ea-87fc-a39fac4c2c09)


```
pip install --upgrade category_encoders
```
![image](https://github.com/22008650/EXNO-3-DS/assets/122548204/20ed386a-e4b1-4109-b042-00e62efc6286)

```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![image](https://github.com/22008650/EXNO-3-DS/assets/122548204/65d66e02-a658-482e-b796-056f862d6ddf)

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```
![image](https://github.com/22008650/EXNO-3-DS/assets/122548204/62b3adca-dbe5-4f01-8969-3218e8b0a45f)


```
dfb=pd.concat([df,nd],axis=1)
dfb
```
![image](https://github.com/22008650/EXNO-3-DS/assets/122548204/ff028c2f-d75e-438d-bc7c-b349b6359d2d)

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/22008650/EXNO-3-DS/assets/122548204/175ed370-1502-4ca4-b137-3f2215dfc2af)


```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/22008650/EXNO-3-DS/assets/122548204/425d6386-3cea-481d-bd64-e52376368df8)

```
df.skew()
```
![image](https://github.com/22008650/EXNO-3-DS/assets/122548204/02b25bdf-9bb7-4785-be78-fafdc47af83c)

```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/22008650/EXNO-3-DS/assets/122548204/240e95ed-5a07-4d9d-a379-a68fb25bb43a)


```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/22008650/EXNO-3-DS/assets/122548204/b9e141b3-c952-4075-baff-ae7694ed7e6c)


```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/22008650/EXNO-3-DS/assets/122548204/753f1c4b-8914-473a-b9f8-2350f99516e1)


```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/22008650/EXNO-3-DS/assets/122548204/e2c672f8-32cc-4ed8-9937-f2f87b647ceb)


```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/22008650/EXNO-3-DS/assets/122548204/6e5c3d01-3fad-49ef-a59e-4b321810d95e)


```
df.skew()
```
![image](https://github.com/22008650/EXNO-3-DS/assets/122548204/f4e7305f-5e0e-4c78-9759-49b83495ad21)


```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
```
![image](https://github.com/22008650/EXNO-3-DS/assets/122548204/7d43f3c8-3d8d-473b-ba49-798e774d515d)


```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/22008650/EXNO-3-DS/assets/122548204/35eeaf6a-7eba-4755-b0f6-16dfb76658c2)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/22008650/EXNO-3-DS/assets/122548204/e73d6fea-aa9d-47a5-b35b-eeed6646a0b1)


```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/22008650/EXNO-3-DS/assets/122548204/29cb0f5c-affc-4023-a489-a58d64c5263e)


```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/22008650/EXNO-3-DS/assets/122548204/e6ab84ac-52b2-4fcc-bf1e-9f8b658fc1af)


```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/22008650/EXNO-3-DS/assets/122548204/63c455e4-b80f-4481-818e-bd190c9b78ad)



       
# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
       
