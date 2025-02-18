# Week4 Assignment
This repository utilizes machine learning to classify echoes into leads and sea ice.  
It computes the average echo shape and standard deviation for both categories and evaluates classification performance by comparing the results with ESA's official classification using a confusion matrix.  
##  Project Overview  
This project focuses on colocating Sentinel-3 (OLCI & SRAL) and Sentinel-2 optical data while leveraging unsupervised learning techniques to classify sea ice and leads.  
The goal is to develop an automated pipeline that enhances Earth Observation (EO) analysis by fusing different satellite datasets and applying machine learning models to classify environmental features.  
<!-- ABOUT THE PROJECT -->
### Built With
- NumPy – Efficient numerical computations and matrix operations  
- Pandas – Data manipulation and tabular processing  
- Matplotlib – Visualization of classification results  
- Rasterio – Handling Sentinel-2 geospatial raster data  
- netCDF4 – Processing Sentinel-3 altimetry data  
- Scikit-Learn – Machine learning models (K-Means, GMM)  
- Folium – Interactive geospatial data visualization  
- Shapely – Geometric operations for colocation analysis  
- Requests – API calls for Sentinel-3 metadata retrieval
  
<!-- Unsupervised Learning -->
## Unsupervised Learning
This section marks our journey into another significant domain of machine learning and AI: unsupervised learning. Rather than delving deep into theoretical intricacies, our focus here will be on offering a practical guide. We aim to equip you with a clear understanding and effective tools for employing unsupervised learning methods in real-world (EO) scenarios.

It's important to note that, while unsupervised learning encompasses a broad range of applications, our discussion will predominantly revolve around classification tasks. This is because unsupervised learning techniques are exceptionally adept at identifying patterns and categorising data when the classifications are not explicitly labeled. By exploring these techniques, you'll gain insights into how to discern structure and relationships within your datasets, even in the absence of predefined categories or labels.

The tasks in this notebook will be mainly two:

1.Discrimination of Sea ice and lead based on image classification based on Sentinel-2 optical data.

2.Discrimination of Sea ice and lead based on altimetry data classification based on Sentinel-3 altimetry data.

<!-- Introduction to Unsupervised Learning Methods [Bishop and Nasrabadi, 2006] -->
### Introduction to Unsupervised Learning Methods [Bishop and Nasrabadi, 2006]
**Introduction to K-means Clustering**

K-means clustering is a type of unsupervised learning algorithm used for partitioning a dataset into a set of k groups (or clusters), where k represents the number of groups pre-specified by the analyst. It classifies the data points based on the similarity of the features of the data {cite}macqueen1967some. The basic idea is to define k centroids, one for each cluster, and then assign each data point to the nearest centroid, while keeping the centroids as small as possible.

**Why K-means for Clustering?**

K-means clustering is particularly well-suited for applications where:

The structure of the data is not known beforehand: K-means doesn’t require any prior knowledge about the data distribution or structure, making it ideal for exploratory data analysis.
Simplicity and scalability: The algorithm is straightforward to implement and can scale to large datasets relatively easily.

**Key Components of K-means**

1.**Choosing K**: The number of clusters (k) is a parameter that needs to be specified before applying the algorithm.

2.**Centroids Initialization**: The initial placement of the centroids can affect the final results.

3.**Assignment Step**: Each data point is assigned to its nearest centroid, based on the squared Euclidean distance.

4.**Update Step**: The centroids are recomputed as the center of all the data points assigned to the respective cluster.

**The Iterative Process of K-means**

The assignment and update steps are repeated iteratively until the centroids no longer move significantly, meaning the within-cluster variation is minimised. This iterative process ensures that the algorithm converges to a result, which might be a local optimum.

**Advantages of K-means**

**Efficiency**: K-means is computationally efficient.

**Ease of interpretation**: The results of k-means clustering are easy to understand and interpret.

**Basic Code Implementation**

Below, you'll find a basic implementation of the K-means clustering algorithm. This serves as a foundational understanding and a starting point for applying the algorithm to your specific data analysis tasks.


```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
pip install rasterio
```

```python
pip install netCDF4
```

```python
# Python code for K-means clustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Sample data
X = np.random.rand(100, 2)

# K-means model
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()
```

<img width="563" alt="截屏2025-02-18 19 58 45" src="https://github.com/user-attachments/assets/a805964e-f299-4d0f-9b40-58cd63ae22e1" />

<!-- Gaussian Mixture Models (GMM) [Bishop and Nasrabadi, 2006] -->
### Gaussian Mixture Models (GMM) [Bishop and Nasrabadi, 2006]
#### Introduction to Gaussian Mixture Models

Gaussian Mixture Models (GMM) are a probabilistic model for representing normally distributed subpopulations within an overall population. The model assumes that the data is generated from a mixture of several Gaussian distributions, each with its own mean and variance {cite}reynolds2009gaussian, mclachlan2004finite. GMMs are widely used for clustering and density estimation, as they provide a method for representing complex distributions through the combination of simpler ones.

#### Why Gaussian Mixture Models for Clustering?
Gaussian Mixture Models are particularly powerful in scenarios where:

·**Soft clustering is needed**: Unlike K-means, GMM provides the probability of each data point belonging to each cluster, offering a soft classification and understanding of the uncertainties in our data.
·**Flexibility in cluster covariance**: GMM allows for clusters to have different sizes and different shapes, making it more flexible to capture the true variance in the data.

#### Key Components of GMM

Number of Components (Gaussians): Similar to K in K-means, the number of Gaussians (components) is a parameter that needs to be set.
Expectation-Maximization (EM) Algorithm: GMMs use the EM algorithm for fitting, iteratively improving the likelihood of the data given the model.
Covariance Type: The shape, size, and orientation of the clusters are determined by the covariance type of the Gaussians (e.g., spherical, diagonal, tied, or full covariance).
#### The EM Algorithm in GMM

The Expectation-Maximization (EM) algorithm is a two-step process:

·**Expectation Step (E-step)**: Calculate the probability that each data point belongs to each cluster.

·**Maximization Step (M-step)**: Update the parameters of the Gaussians (mean, covariance, and mixing coefficient) to maximize the likelihood of the data given these assignments.

This process is repeated until convergence, meaning the parameters do not significantly change from one iteration to the next.

#### Advantages of GMM

Soft Clustering: Provides a probabilistic framework for soft clustering, giving more information about the uncertainties in the data assignments.
Cluster Shape Flexibility: Can adapt to ellipsoidal cluster shapes, thanks to the flexible covariance structure.

#### Basic Code Implementation

Below, you'll find a basic implementation of the Gaussian Mixture Model. This should serve as an initial guide for understanding the model and applying it to your data analysis projects.

```python
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np

# Sample data
X = np.random.rand(100, 2)

# GMM model
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
y_gmm = gmm.predict(X)

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=y_gmm, cmap='viridis')
centers = gmm.means_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title('Gaussian Mixture Model')
plt.show()
```

<img width="571" alt="截屏2025-02-18 20 00 03" src="https://github.com/user-attachments/assets/2b398b88-57f4-41e7-b1fc-cc0d7b9ff3f2" />

#### Image Classification
Now, let's explore the application of these unsupervised methods to image classification tasks, focusing specifically on distinguishing between sea ice and leads in Sentinel-2 imagery.

## K-Means Implementation

```python
import rasterio
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

base_path = "/content/drive/MyDrive/Week 4/Unsupervised Learning/S2A_MSIL1C_20190301T235611_N0207_R116_T01WCU_20190302T014622.SAFE/GRANULE/L1C_T01WCU_A019275_20190301T235610/IMG_DATA/" # You need to specify the path
bands_paths = {
    'B4': base_path + 'T01WCU_20190301T235611_B04.jp2',
    'B3': base_path + 'T01WCU_20190301T235611_B03.jp2',
    'B2': base_path + 'T01WCU_20190301T235611_B02.jp2'
}

# Read and stack the band images
band_data = []
for band in ['B4']:
    with rasterio.open(bands_paths[band]) as src:
        band_data.append(src.read(1))

# Stack bands and create a mask for valid data (non-zero values in all bands)
band_stack = np.dstack(band_data)
valid_data_mask = np.all(band_stack > 0, axis=2)

# Reshape for K-means, only including valid data
X = band_stack[valid_data_mask].reshape((-1, 1))

# K-means clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
labels = kmeans.labels_

# Create an empty array for the result, filled with a no-data value (e.g., -1)
labels_image = np.full(band_stack.shape[:2], -1, dtype=int)

# Place cluster labels in the locations corresponding to valid data
labels_image[valid_data_mask] = labels

# Plotting the result
plt.imshow(labels_image, cmap='viridis')
plt.title('K-means clustering on Sentinel-2 Bands')
plt.colorbar(label='Cluster Label')
plt.show()

del kmeans, labels, band_data, band_stack, valid_data_mask, X, labels_image
```

<img width="593" alt="截屏2025-02-18 20 04 10" src="https://github.com/user-attachments/assets/3eddeb79-e1f8-4e81-abc2-08aabde12055" />

### GMM Implementation

```python
import rasterio
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Paths to the band images
base_path = "/content/drive/MyDrive/GEOL0069/2425/Week 4/Unsupervised Learning/S2A_MSIL1C_20190301T235611_N0207_R116_T01WCU_20190302T014622.SAFE/GRANULE/L1C_T01WCU_A019275_20190301T235610/IMG_DATA/" # You need to specify the path
bands_paths = {
    'B4': base_path + 'T01WCU_20190301T235611_B04.jp2',
    'B3': base_path + 'T01WCU_20190301T235611_B03.jp2',
    'B2': base_path + 'T01WCU_20190301T235611_B02.jp2'
}

# Read and stack the band images
band_data = []
for band in ['B4']:
    with rasterio.open(bands_paths[band]) as src:
        band_data.append(src.read(1))

# Stack bands and create a mask for valid data (non-zero values in all bands)
band_stack = np.dstack(band_data)
valid_data_mask = np.all(band_stack > 0, axis=2)

# Reshape for GMM, only including valid data
X = band_stack[valid_data_mask].reshape((-1, 1))

# GMM clustering
gmm = GaussianMixture(n_components=2, random_state=0).fit(X)
labels = gmm.predict(X)

# Create an empty array for the result, filled with a no-data value (e.g., -1)
labels_image = np.full(band_stack.shape[:2], -1, dtype=int)

# Place GMM labels in the locations corresponding to valid data
labels_image[valid_data_mask] = labels

# Plotting the result
plt.imshow(labels_image, cmap='viridis')
plt.title('GMM clustering on Sentinel-2 Bands')
plt.colorbar(label='Cluster Label')
plt.show()
```
<img width="541" alt="截屏2025-02-18 20 30 45" src="https://github.com/user-attachments/assets/d1d19e89-cbbc-4c43-8da8-54136d0027c1" />


### Altimetry Classification

Now, let's explore the application of these unsupervised methods to altimetry classification tasks, focusing specifically on distinguishing between sea ice and leads in Sentinel-3 altimetry dataset.

## Read in Functions Needed

Before delving into the modeling process, it's crucial to preprocess the data to ensure compatibility with our analytical models. This involves transforming the raw data into meaningful variables, such as peakniness and stack standard deviation (SSD), etc.

There are some NaN values in the dataset so one way to deal with this is to delete them.

Now, let's proceed with running the GMM model as usual. Remember, you have the flexibility to substitute this with K-Means or any other preferred model.

We can plot the mean waveform of each class.

```python
# mean and standard deviation for all echoes
mean_ice = np.mean(waves_cleaned[clusters_gmm==0],axis=0)
std_ice = np.std(waves_cleaned[clusters_gmm==0], axis=0)

plt.plot(mean_ice, label='ice')
plt.fill_between(range(len(mean_ice)), mean_ice - std_ice, mean_ice + std_ice, alpha=0.3)


mean_lead = np.mean(waves_cleaned[clusters_gmm==1],axis=0)
std_lead = np.std(waves_cleaned[clusters_gmm==1], axis=0)

plt.plot(mean_lead, label='lead')
plt.fill_between(range(len(mean_lead)), mean_lead - std_lead, mean_lead + std_lead, alpha=0.3)

plt.title('Plot of mean and standard deviation for each class')
plt.legend()
```

<img width="589" alt="截屏2025-02-18 20 36 07" src="https://github.com/user-attachments/assets/46e44e93-5a54-40b5-a84a-d87b2c67d893" />

```python
x = np.stack([np.arange(1,waves_cleaned.shape[1]+1)]*waves_cleaned.shape[0])
plt.plot(x,waves_cleaned)  # plot of all the echos
plt.show()
```
<img width="591" alt="截屏2025-02-18 20 38 30" src="https://github.com/user-attachments/assets/9d5fa1bd-1168-49da-8b4f-35192b659fcb" />
```python
# plot echos for the lead cluster
x = np.stack([np.arange(1,waves_cleaned[clusters_gmm==1].shape[1]+1)]*waves_cleaned[clusters_gmm==1].shape[0])
plt.plot(x,waves_cleaned[clusters_gmm==1])  # plot of all the echos
plt.show()
```
<img width="592" alt="截屏2025-02-18 20 43 04" src="https://github.com/user-attachments/assets/7115bd08-e625-47bf-b317-40b669a53768" />
```python
# plot echos for the sea ice cluster
x = np.stack([np.arange(1,waves_cleaned[clusters_gmm==0].shape[1]+1)]*waves_cleaned[clusters_gmm==0].shape[0])
plt.plot(x,waves_cleaned[clusters_gmm==0])  # plot of all the echos
plt.show()
```
<img width="583" alt="截屏2025-02-18 20 43 50" src="https://github.com/user-attachments/assets/3639054f-a4ad-4d9a-bf50-7d32be2da708" />

### Scatter Plots of Clustered Data

This code visualizes the clustering results using scatter plots, where different colors represent different clusters (clusters_gmm).

```python
plt.scatter(data_cleaned[:,0],data_cleaned[:,1],c=clusters_gmm)
plt.xlabel("sig_0")
plt.ylabel("PP")
plt.show()
plt.scatter(data_cleaned[:,0],data_cleaned[:,2],c=clusters_gmm)
plt.xlabel("sig_0")
plt.ylabel("SSD")
plt.show()
plt.scatter(data_cleaned[:,1],data_cleaned[:,2],c=clusters_gmm)
plt.xlabel("PP")
plt.ylabel("SSD")
```
<img width="617" alt="截屏2025-02-18 20 45 28" src="https://github.com/user-attachments/assets/6aa4850f-3206-4411-8cd3-f819af15cc25" />
<img width="611" alt="截屏2025-02-18 20 45 59" src="https://github.com/user-attachments/assets/ce1a686b-7a64-4c6c-b46b-5d1debdd7630" />

<img width="593" alt="截屏2025-02-18 20 46 32" src="https://github.com/user-attachments/assets/e422ea38-1c06-4a8d-89d7-32ed98ad5974" />

### Waveform Alignment Using Cross-Correlation

This code aligns waveforms in the cluster where clusters_gmm == 0 by using cross-correlation.

```python
from scipy.signal import correlate

# Find the reference point (e.g., the peak)
reference_point_index = np.argmax(np.mean(waves_cleaned[clusters_gmm==0], axis=0))

# Calculate cross-correlation with the reference point
aligned_waves = []
for wave in waves_cleaned[clusters_gmm==0][::len(waves_cleaned[clusters_gmm == 0]) // 10]:
    correlation = correlate(wave, waves_cleaned[clusters_gmm==0][0])
    shift = len(wave) - np.argmax(correlation)
    aligned_wave = np.roll(wave, shift)
    aligned_waves.append(aligned_wave)

# Plot aligned waves
for aligned_wave in aligned_waves:
    plt.plot(aligned_wave)

plt.title('Plot of 10 equally spaced functions where clusters_gmm = 0 (aligned)')
```
<img width="611" alt="截屏2025-02-18 20 47 59" src="https://github.com/user-attachments/assets/73eff198-4932-4e5f-aa8f-0a921a6f0b9c" />

### Compare with ESA data

In the ESA dataset, sea ice = 1 and lead = 2. Therefore, we need to subtract 1 from it so our predicted labels are comparable with the official product labels.
To evaluate the Gaussian Mixture Model (GMM) clustering, the true labels from `flag_cleaned_modified` are compared with the predicted labels from `clusters_gmm`. The confusion matrix (`confusion_matrix(true_labels, predicted_gmm)`) summarizes correct and misclassified instances, while the classification report (`classification_report(true_labels, predicted_gmm)`) provides key performance metrics, including precision, recall, and F1-score for each class.  

The results indicate high accuracy, with 8,856 sea ice and 3,293 lead instances correctly classified, and only 22 misclassified as sea ice and 24 misclassified as lead. With an overall accuracy of 100%, the GMM model effectively distinguishes between the two classes.

```python
Confusion Matrix:
[[8856   22]
 [  24 3293]]

Classification Report:
              precision    recall  f1-score   support

         0.0       1.00      1.00      1.00      8878
         1.0       0.99      0.99      0.99      3317

    accuracy                           1.00     12195
   macro avg       1.00      1.00      1.00     12195
weighted avg       1.00      1.00      1.00     12195
```






