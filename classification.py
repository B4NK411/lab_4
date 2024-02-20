import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


data = np.loadtxt("kinematic_features.txt")
ctrl_data = data[:41]
pd_data = data[41:]
#calculating mean of both classes
ctrl_mean = np.mean(ctrl_data, axis=0)
pd_mean = np.mean(pd_data, axis=0)
print(f"Mean for CTRL class: {ctrl_mean}")
print(f"Mean for PD class:{pd_mean}")


#calculating standard deviation for each class
ctrl_dev = np.std(ctrl_data,axis=0)
pd_dev = np.std(pd_data,axis=0)
print(f"standard deviation for PD class:{pd_dev}")
print(f"Standart deviation for CTRL class:{ctrl_dev}")

#calculating eucledian distance between the mean vectors
print(f"Eucledian distance between means is {np.linalg.norm(pd_mean - ctrl_mean)}")


# Plot histogram for the first feature(velocity
feature_data = data[:,0]
plt.figure(figsize=(8, 6))
plt.hist(feature_data, bins=20, color='skyblue', edgecolor='black')
plt.title(f'Histogram of Feature {1}')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Calculate mean and variance for the first feature
feature_mean = np.mean(feature_data)
feature_variance = np.var(feature_data)
print(f"Mean of Feature {1}: {feature_mean}")
print(f"Variance of Feature {1}: {feature_variance}")


# calculate the minkwoski distance between 2 feature vectors, one from CTRL and one from PD
feature_vector_1 = data[0]  # Selecting the first feature vector
feature_vector_2 = data[43]  # Selecting the second feature vector
minkwoski_distances = []
r_values = list(range(1, 11))
for r in r_values:
    distance = np.linalg.norm(feature_vector_1 - feature_vector_2, ord=r)
    minkwoski_distances.append(distance)
plt.figure(figsize=(10, 6))
plt.plot(r_values, minkwoski_distances, marker='o', linestyle='-')
plt.title('Minkowski Distance between Two Feature Vectors')
plt.xlabel('r')
plt.ylabel('Distance')
plt.grid(True)
plt.xticks(r_values)
plt.show()
# it is observed that as r increases, the distance decreases

