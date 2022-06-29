import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from warnings import filterwarnings
filterwarnings('ignore')

data_path = 'C:\\Users\\mukes\\PycharmProjects\\ZomatoRest\\zomato.csv'
from pandas import read_csv

df= read_csv(data_path)

# Results
print(f'Dataset shape: {df.shape}')
df.head()

df.info()

df.isnull().sum()

feature_na=[feature for feature in df.columns if df[feature].isnull().sum()>0]
feature_na

#% of missing values
import numpy as np
for feature in feature_na:
    print('{} has {} % missing values'.format(feature,np.round(df[feature].isnull().sum()/len(df)*100,4)))

df['approx_cost(for two people)'].dtype

df['approx_cost(for two people)'].unique()

### right now it has some NAN Values so it will be of float data-type,dats why very first I have to convert it into string then
### I have to remove this comma
df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(str).apply(lambda x: x.replace(',', ''))

df['approx_cost(for two people)']=df['approx_cost(for two people)'].astype(float)
df['approx_cost(for two people)'].dtype


df['rate'].unique()
df['rate'][0].split('/')[0]

def split(x):
    return x.split('/')[0]

df['rate'].dtype

df['rate'].isnull().sum()

### right now it has some NAN Values so it will be of float data-type,dats why very first I have to convert it into string then
### I have to split it & access
df['rate']=df['rate'].astype(str).apply(split)
### ''' df['rate'] = df['rate'].astype(str).apply(lambda x: x.split('/')[0])'''

df['rate'].replace('NEW',0,inplace=True)
df['rate'].replace('-',0,inplace=True)

df['rate']=df['rate'].astype(str).astype(float)

df['rate'].dtype

import matplotlib.pyplot as plt
plt.figure(figsize=(20,12))
df['rest_type'].value_counts().nlargest(20).plot.bar(color='red')

### to provide styling to text on x-axis
plt.gcf().autofmt_xdate()

df.columns

df['rest_type'].value_counts()


def mark(x):
    if x in ('Quick Bites', 'Casual Dining'):
        return 'Quick Bites + Casual Dining'
    else:
        return 'other'

    ## Alternative using Lambda
### df['Top_types']=df['rest_type'].apply(lambda x: 'Quick Bites + Casual Dining' if x in ('Quick Bites', 'Casual Dining') else 'Other')

df['Top_types']=df['rest_type'].apply(mark)

df.head()

import plotly.express as px
values=df['Top_types'].value_counts()
labels=df['Top_types'].value_counts().index

fig = px.pie(df, values=values, names=labels,title='Restaurants Pie chart')
fig.show()

### Almost 60 % of restaurants are of Casual Dining & Quick Bites

df.head()
df.columns

df.dtypes

rest=df.groupby('name').agg({'votes': 'sum','url': 'count','approx_cost(for two people)': 'mean','rate': 'mean'}).reset_index()
rest

rest.columns = ['name', 'total_votes', 'total_unities', 'avg_approx_cost', 'mean_rating']
rest.head()

rest['votes_per_unity'] = rest['total_votes'] / rest['total_unities']
rest.head()

popular=rest.sort_values(by='total_unities', ascending=False)
popular

popular['name'].nunique()

popular.shape


import seaborn as sns
# Creating a figure for restaurants overview analysis
fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(20,30))

# Plot Pack 01 - Most popular restaurants (votes)

# Annotations
ax1.text(0.50, 0.30, int(popular['total_votes'].mean()), fontsize=45, ha='center')
ax1.text(0.50, 0.12, 'is the average of votes', fontsize=12, ha='center')
ax1.text(0.50, 0.00, 'received by restaurants', fontsize=12, ha='center')
ax1.axis('off')

sns.barplot(x='total_votes', y='name', data=popular.sort_values(by='total_votes', ascending=False)[0:5],ax=ax2, palette='plasma')
ax2.set_title('Top 5 Most Voted Restaurants', size=12)

sns.barplot(x='total_votes', y='name', data=popular.sort_values(by='total_votes', ascending=False).query('total_votes > 0').tail(),ax=ax3, palette='plasma_r')
ax3.set_title('Top 5 Less Voted Restaurants\n(with at least 1 vote)', size=12)


popular.columns

popular.head()

fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(20,30))
# Annotations
import numpy as np
ax1.text(0.50, 0.30, np.round(popular['avg_approx_cost'].mean(), 2), fontsize=45, ha='center')
ax1.text(0.50, 0.12, 'is mean approx cost', fontsize=12, ha='center')
ax1.text(0.50, 0.00, 'for Bengaluru restaurants', fontsize=12, ha='center')
ax1.axis('off')

sns.barplot(x='avg_approx_cost', y='name', data=popular.sort_values(by='avg_approx_cost', ascending=False)[0:5],ax=ax2, palette='plasma')
ax2.set_title('Top 5 Most Expensives Restaurants', size=12)

sns.barplot(x='avg_approx_cost', y='name', data=popular.sort_values(by='avg_approx_cost', ascending=False).query('avg_approx_cost > 0').tail(),ax=ax3, palette='plasma_r')
ax3.set_title('Top 5 Less Expensive Restaurants', size=12)


#How many restaurants offer Book Table service? And how about Online Order service?

import plotly.graph_objs as go
from plotly.offline import iplot
x=df['book_table'].value_counts()
labels=['not book','book']

trace=go.Pie(labels=labels, values=x,
               hoverinfo='label+percent', textinfo='percent',
               textfont=dict(size=25),
              pull=[0, 0, 0,0.2, 0]
               )
iplot([trace])


import plotly.express as px
x=df['online_order'].value_counts()
labels=['accepted','not accepted']

fig = px.pie(df, values=x, names=labels,title='Pie chart')
fig.show()

#Finding Best budget Restaurants in any location
#we will pass location and restaurant type as parameteres,function will return name of restaurants

def return_budget(location,restaurant):
    budget=df[(df['approx_cost(for two people)']<=400) & (df['location']==location) &
                     (df['rate']>4) & (df['rest_type']==restaurant)]
    return(budget['name'].unique())

return_budget('BTM',"Quick Bites")

#geographical analysis
#I need Latitudes & longitudes for each of the place for geaographical Data analysis,so to fetch lat,lon of each place,use Geopy
locations=pd.DataFrame({"Name":df['location'].unique()})
locations['new_Name']='Bangalore '+locations['Name']
locations.head()

pip install geopy
from geopy.geocoders import Nominatim
lat=[]
lon=[]
geolocator=Nominatim(user_agent="app")
for location in locations['Name']:
    location = geolocator.geocode(location)
    if location is None:
        lat.append(np.nan)
        lon.append(np.nan)
    else:
        lat.append(location.latitude)
        lon.append(location.longitude)
locations['latitude']=lat
locations['longitude']=lon
locations.to_csv('zomato_locations.csv',index=False)

#We have found out latitude and longitude of each location listed in the dataset using geopy.
Rest_locations=pd.DataFrame(df['location'].value_counts().reset_index())
Rest_locations.columns=['Name','count']
Rest_locations.head()

#Combining both the dataframe
Restaurant_locations=Rest_locations.merge(locations,on='Name',how="left").dropna()
Restaurant_locations.head()

def generateBaseMap(default_location=[12.97, 77.59], default_zoom_start=12):
    base_map = folium.Map(location=default_location, zoom_start=default_zoom_start)
    return base_map

import folium
from folium.plugins import HeatMap
basemap=generateBaseMap()

#Heatmap of Restaurant
HeatMap(Restaurant_locations[['lat','lon','count']].values.tolist(),zoom=20,radius=15).add_to(basemap)

#It is clear that restaurants tend to concentrate in central bangalore area.
"""The clutter of restaurants lowers are we move away from central.
So,potential restaurant entrepreneurs can refer this and find out good locations for their venture.
note heatmap is good when we have latitude,longitude or imporatnce of that particular place or count of that place
"""

#wordcloud of Customer Preference
data=df[df['rest_type']=='Quick Bites']
data['dish_liked']
stopwords=set(STOPWORDS)
dishes=''
for word in data['dish_liked']:
    words=word.split()
    # Converts each token into lowercase
    for i in range(len(words)):
        words[i] = words[i].lower()
    dishes=dishes+ " ".join(words)+" "
wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,stopwords = stopwords,width=1500, height=1500).generate(dishes)
plt.imshow(wordcloud)
plt.axis("off")

#analysing Reviews of Particular Restaurant
df['reviews_list'][0]
data=df['reviews_list'][0].lower()
data
import re
data2=re.sub('[^a-zA-Z]', ' ',data)
data2

data3=re.sub('rated', ' ',data2)
data3

data4=re.sub('x',' ',data3)
data4

re.sub(' +',' ',data4)

dataset=df[df['rest_type']=='Quick Bites']
type(dataset['reviews_list'][3])

total_review = ' '
for review in dataset['reviews_list']:
    review = review.lower()
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = re.sub('rated', ' ', review)
    review = re.sub('x', ' ', review)
    review = re.sub(' +', ' ', review)
    total_review = total_review + str(review)

wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(total_review)
# plot the WordCloud image
plt.figure(figsize = (8, 8))
plt.imshow(wordcloud)
plt.axis("off")

#Predicting the Success of a Restaurant

df.columns

df['rate'].unique()

# Splitting restaurants
### df['rated'] = df['rate'].apply(lambda x: 1 if x >= 0 else 0)
def assign(x):
    if x>0:
        return 1
    else:
        return 0
df['rated']=df['rate'].apply(assign)

df['rated'].unique()

new_restaurants = df[df['rated'] == 0]
train_val_restaurants = df.query('rated == 1')

#### By now we've already splitted our original data into new_restaurants and train_val_restaurants using pandas DataFrames. Let's  keep the first one aside for now and let's work only with the training and validation set. The next step is to create our target variable to be used in this classification task.

#### The main point here is to define a fair threshold for splitting the restaurants into good and bad ones. It would be a  really experimental decision and we must keep in mind that this approach is not the best one. Probably it would let margin for classification errors. Even so, let's try!

train_val_restaurants.head()

train_val_restaurants['rate'].unique()

### Defining a custom threshold for splitting restaurants into good and bad\

# Defining a custom threshold for splitting restaurants into good and bad
threshold = 3.75
train_val_restaurants['target'] = train_val_restaurants['rate'].apply(lambda x: 1 if x >= threshold else 0)

train_val_restaurants.head()

import matplotlib.pyplot as plt
x=train_val_restaurants['target'].value_counts()
labels=x.index
print(x)
plt.pie(x,explode=[0.0,0.1],autopct='%1.1f%%')

#### Ok, for our first trial it's fair. The meaning of all this is that we marked as good restaurants with a rate greater or equal to 3.75. Correct or not, let's continue to see what we can get from this.

#### The next step is to prepare some features for training our classification model.

### Feature Extraction


train_val_restaurants.columns
train_val_restaurants.head()
## train_val_restaurants['total_cuisines'] = train_val_restaurants['cuisines'].astype(str).apply(lambda x: len(x.split(',')))

def count(x):
    return len(x.split(','))

#### as it have some NAN value that why very first I have to convert into str  &  then apply a function
train_val_restaurants['total_cuisines']=train_val_restaurants['cuisines'].astype(str).apply(count)
train_val_restaurants['multiple_types']=train_val_restaurants['rest_type'].astype(str).apply(count)

train_val_restaurants.columns

imp_features=['online_order','book_table','location','rest_type','multiple_types','total_cuisines','listed_in(type)', 'listed_in(city)','approx_cost(for two people)','target']
data = train_val_restaurants[imp_features]

data.dropna(how='any',inplace=True)
data.isnull().sum()

# Splitting features by data type
cat_features= [col for col in data.columns if data[col].dtype == 'O']
num_features= [col for col in data.columns if data[col].dtype != 'O']

cat_features

for feature in cat_features:
    print('{} has total {} unique features'.format(feature, data[feature].nunique()))

#### But we will observe over here,we have many categories thus if we encode it using onne-hot encoding, it will consume more
#### memory in our system

data.shape

cols = ['location', 'rest_type', 'listed_in(city)']
for col in cols:
    print('Total feature in {} are {}'.format(col, data[col].nunique()))
    print(data[col].value_counts() / (len(data)) * 100)
    print('\n')

percent=data['location'].value_counts()/len(data)*100
values=percent.values

len(values[values>0.4])

#### lets set Threshold value 0.4 ,

values=data['location'].value_counts()/len(data)*100
values

threshold=0.4
imp=values[values>threshold]
imp

data['location']=np.where(data['location'].isin(imp.index),data['location'],'other')

##X_train['location']=X_train['location'].apply(lambda x:'other' if x not in imp.index else x)

data['location'].nunique()

values2=data['rest_type'].value_counts()/len(data)*100
values2

data['rest_type'].head(20)

len(values2[values2>0.3])

threshold=1.5
imp2=values2[values2>1.5]
imp2

imp2.index

data['rest_type'].isin(imp2.index)

data['rest_type']=np.where(data['rest_type'].isin(imp2.index),data['rest_type'],'other')
##data['rest_type'].apply(lambda x: 'other' if x not in imp2.index else x)


data['rest_type']

#### after apply feature reduction, we will observe less number of features

for feature in cat_features:
    print('{} has total {} unique features'.format(feature, data[feature].nunique()))

cat_features

import pandas as pd
data_cat = data[cat_features]
for col in cat_features:
    col_encoded = pd.get_dummies(data_cat[col],prefix=col,drop_first=True)
    data_cat=pd.concat([data_cat,col_encoded],axis=1)
    data_cat.drop(col, axis=1, inplace=True)

data_cat.shape
data_cat.head(10)

data_cat.shape

data.head()

data_final=pd.concat([data.loc[:,['multiple_types','total_cuisines','approx_cost(for two people)','target']],data_cat],axis=1)

data_final.shape

# Splitting the data
X = data_final.drop('target', axis=1)
y = data_final['target'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)

X_train.shape
# Import the random forest model.
from sklearn.ensemble import RandomForestClassifier

# Initialize the model with some parameters.
model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=1)

# Fit the model to the data.
model.fit(X_train, y_train)

# Make predictions.
predictions = model.predict(X_test)

# Compute the error.
from sklearn.metrics import confusion_matrix
confusion_matrix(predictions, y_test)

from sklearn.metrics import accuracy_score
accuracy_score(predictions,y_test)

#fit naive bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

### classifier models
models = []
models.append(('LogisticRegression', LogisticRegression()))
models.append(('Naive Bayes',GaussianNB()))
models.append(('RandomForest', RandomForestClassifier()))
models.append(('Decision Tree', DecisionTreeClassifier()))
models.append(('KNN', KNeighborsClassifier(n_neighbors = 5)))

for name,model in models:
    print(name)
    print(models)

# Make predictions on validation dataset

for name, model in models:
    print(name)
    model.fit(X_train, y_train)

    # Make predictions.
    predictions = model.predict(X_test)

    # Compute the error.
    from sklearn.metrics import confusion_matrix

    print(confusion_matrix(predictions, y_test))

    from sklearn.metrics import accuracy_score

    print(accuracy_score(predictions, y_test))
    print('\n')












