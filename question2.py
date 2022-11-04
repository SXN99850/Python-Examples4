import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

#For output table display
desired_width=320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',20)

#Reading the dataset
df=pd.read_csv("./K-Mean_Dataset.csv")
print(df.head())
X = df.iloc[:, 1:].values

#Using Imputer from sklearn to replace null values with mean:
# calculating mean using fit() and replacing or transforming null values using transform()
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X)
x = imputer.transform(X)

#wcss - Within cluster sum of square
# It is the sum of the squared distance between each member of the cluster and its centroid
wcss = []
#Using elbow method to find the optimal K value
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

#plotting the elbow graph using matplotlib
plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

#importing KMeans from sklearn
from sklearn.cluster import KMeans
nclusters = 2
km = KMeans(n_clusters=nclusters)
#train the model with data.
print(km.fit(x))

#apply a trained model to data.
y_cluster_kmeans = km.predict(x)
from sklearn import metrics, preprocessing
#importing metrics to find silhouette score
score = metrics.silhouette_score(x, y_cluster_kmeans)
print('Silhouette score:',score)

#Question3
#Standardize features by removing the mean and scaling to unit variance
scaler = preprocessing.StandardScaler()
scaler.fit(x)
X_scaled_array = scaler.transform(x)
X_scaled = pd.DataFrame(X_scaled_array)

#importing KMeans from sklearn
from sklearn.cluster import KMeans
nclusters = 2
km = KMeans(n_clusters=nclusters)
km.fit(X_scaled)
KMeans(n_clusters=2)

#finding the score for scaled features
y_scaled_cluster_kmeans = km.predict(X_scaled)
from sklearn import metrics
score = metrics.silhouette_score(X_scaled, y_scaled_cluster_kmeans)
print('Silhouette score on scaled features: ', score)