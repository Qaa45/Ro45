#1) Design and implement pattern recognition system to identify and extract unique species patterns from the Iris dataset.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['species'] = [data.target_names[i] for i in data.target]

print("\nUnique Patterns (Averages):")
print(df.groupby('species').mean())

sns.pairplot(df, hue='species')
plt.suptitle("Feature Relationships", y=1.02)
plt.show()

sns.barplot(x='species', y='sepal length (cm)', data=df)
plt.title("Mean Sepal Length")
plt.show()

sns.violinplot(x='species', y='petal width (cm)', data=df)
plt.title("Petal Width Distribution")
plt.show()

X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = RandomForestClassifier()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, pred))

pd.Series(model.feature_importances_, index=X.columns).plot(kind='barh', title="Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()









# 2)Develop a text classification model that can effectively identify, extract features, and 
# classify documents from the 20 Newsgroups dataset into one of the 20 predefined categories using  pattern recognition techniques.

import nltk
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

newsgroups = fetch_20newsgroups(subset='all')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

processed_data = [' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(doc.lower()) if word not in stop_words]) for doc in newsgroups.data]

X = TfidfVectorizer(max_features=5000).fit_transform(processed_data)
y = newsgroups.target 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = MultinomialNB().fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=newsgroups.target_names))







# 3) Design a statistical model to analyze wine quality using Gaussian distribution methods.
# Utilize synthetic data generated with NumPy or the Wine Quality Dataset


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(42)
features = ['alcohol', 'sulphates', 'citric_acid', 'residual_sugar', 'pH']
X = np.random.normal(5, 2, size=(1000, len(features)))
y = np.clip(np.round(np.random.normal(6, 1.5, 1000)), 3, 8).astype(int)
data = pd.DataFrame(X, columns=features)
data['quality'] = y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.hist(y_test, bins=range(3, 10), alpha=0.5, label='Actual')
plt.hist(y_pred, bins=range(3, 10), alpha=0.5, label='Predicted')
plt.xlabel('Wine Quality')
plt.ylabel('Count')
plt.title('Synthetic Wine Quality: Actual vs Predicted')
plt.legend()
plt.show()

# Feature distribution curves
plt.figure(figsize=(10, 6))
for col in features:
    mu, std = norm.fit(data[col])
    x = np.linspace(data[col].min(), data[col].max(), 100)
    plt.plot(x, norm.pdf(x, mu, std), label=col)
plt.title("Normal Distribution Fit of Features")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()






# 4)Develop a classification system for handwritten digit recognition using the MNIST dataset, 
# leveraging Bayes' Decision Theory to optimize decision-making and minimize classification error.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
mnist = fetch_openml("mnist_784")

X = mnist.data
y = mnist.target.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = GaussianNB()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
index = np.random.randint(0, len(X_test))
plt.imshow(X_test.iloc[index].values.reshape(28, 28), cmap="gray")
plt.title(f"Predicted Label: {y_pred[index]}")
plt.show()






# 5)Develop an anomaly detection system for high-dimensional network traffic data using the KDD Cup 1999 dataset.

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_kddcup99
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

data = fetch_kddcup99(percent10=True, as_frame=True)
df = data.frame
df.rename(columns={df.columns[-1]: 'label'}, inplace=True)

df = df.apply(LabelEncoder().fit_transform)
X = df.drop('label', axis=1)
y = (df['label'] != 11).astype(int)  # 11 = 'normal.' after label encoding

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X_train)

y_pred = (model.predict(X_test) == -1).astype(int)
print("Classification Report:\n", classification_report(y_test, y_pred))











# 6)Implement a Hidden Markov Model (HMM) to recognize the sequence of weather patterns (e.g., sunny, cloudy, rainy) based on temperature and humidity observations. 
# Use both discrete and continuous HMMs to compare their performance.

import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.preprocessing import KBinsDiscretizer
states = ["Sunny", "Cloudy", "Rainy"]
n_states = len(states)

means = np.array([[30, 40], [25, 50], [20, 80]])  
covars = np.array([[[5, 3], [3, 5]], [[4, 2], [2, 4]], [[3, 5], [5, 3]]])  

n_samples = 300
np.random.seed(42)
hidden_states = np.random.choice(n_states, n_samples, p=[0.5, 0.3, 0.2])
observations = np.array([np.random.multivariate_normal(means[s], covars[s]) for s in hidden_states])

plt.scatter(observations[:, 0], observations[:, 1], c=hidden_states, cmap="viridis")
plt.xlabel("Temperature")
plt.ylabel("Humidity")
plt.title("Weather Observations")
plt.show()

discretizer = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="uniform")
X_discrete = discretizer.fit_transform(observations).astype(int)
X_discrete = X_discrete.reshape(-1, 1)  # Reshape for HMM

hmm_discrete = hmm.MultinomialHMM(n_components=n_states, n_iter=100)
hmm_discrete.fit(X_discrete)
preds_discrete = hmm_discrete.predict(X_discrete)

accuracy_discrete = np.mean(preds_discrete[:n_samples] == hidden_states)
print(f"Discrete HMM Accuracy: {accuracy_discrete:.2f}")

hmm_continuous = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100)
hmm_continuous.fit(observations)
preds_continuous = hmm_continuous.predict(observations)
accuracy_continuous = np.mean(preds_continuous == hidden_states)
print(f"Continuous HMM Accuracy: {accuracy_continuous:.2f}")

plt.figure(figsize=(10, 4))
plt.plot(hidden_states[:50], "bo-", label="True States")
plt.plot(preds_discrete[:50], "r--", label="Discrete HMM")
plt.plot(preds_continuous[:50], "g.-", label="Continuous HMM")
plt.legend()
plt.xlabel("Time Step")
plt.ylabel("State")
plt.title("Comparison of True vs Predicted States")
plt.show()








# 7)Build a Discrete Hidden Markov Model (HMM) to analyze DNA sequences and predict gene regions. 
# Use Maximum Likelihood Estimation to train the model with a given dataset of labeled sequences


import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder

sequences = ["ATGCGT", "GCGTAA", "ATGCTG", "GATCCA", "CGTATT"]
encoder = LabelEncoder().fit(['A', 'T', 'G', 'C'])

# Prepare training data
X_train = np.concatenate([encoder.transform(list(seq)) for seq in sequences]).reshape(-1, 1)
lengths = [len(seq) for seq in sequences]

model = hmm.MultinomialHMM(n_components=2, n_iter=100)
model.fit(X_train, lengths)

test_sequence = "GTACGTA"
test_observed = np.array([encoder.transform([nuc])[0] for nuc in test_sequence]).reshape(-1, 1)

predicted_states = model.predict(test_observed)
predicted_labels = ''.join(['G' if s == 0 else 'N' for s in predicted_states])

print("Test DNA Sequence:      ", test_sequence)
print("Predicted Gene Regions:", predicted_labels)
print("Predicted States:      ", predicted_states)
print("Transition Probabilities:", model.transmat_)
print("Emission Probabilities:", model.emissionprob_)







# 8)Create a program that fits a mixture of Gaussians to a dataset of handwritten digit features and clusters them into distinct groups.
# Use the Expectation-Maximization method to estimate the parameters of the Gaussian mixture model.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml

print("Loading MNIST dataset...")
mnist = fetch_openml("mnist_784", version=1)
X = mnist.data.to_numpy().astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]
y = mnist.target.astype(int)

print("Applying PCA for dimensionality reduction...")
pca = PCA(n_components=50)  
X_reduced = pca.fit_transform(X)

# Fit Gaussian Mixture Model
print("Fitting Gaussian Mixture Model...")
gmm = GaussianMixture(n_components=10, covariance_type="full", random_state=42)
gmm.fit(X_reduced)
print("Predicting clusters...")
clusters = gmm.predict(X_reduced)

def plot_cluster_images(cluster_number, num_samples=10):
    indices = np.where(clusters == cluster_number)[0][:num_samples]
    fig, axes = plt.subplots(1, num_samples, figsize=(10, 2))
    for i, idx in enumerate(indices):
        axes[i].imshow(X[idx].reshape(28, 28), cmap="gray")  # Reshape and plot the image
        axes[i].axis("off")
    plt.suptitle(f"Cluster {cluster_number}")
    plt.show()

for i in range(5):
    plot_cluster_images(i)

print("Clustering completed successfully!")








# 9)Use non-parametric K-Nearest Neighbor (KNN) techniques to classify grayscale images of shapes (e.g., circles, squares, and triangles).
# Evaluate and compare the classification accuracy of both methods.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from skimage.draw import disk, rectangle, polygon

def generate_shapes_data():
    shapes, labels = [], []
    image_size = (28, 28) 

    # Circle
    for _ in range(100):
        img = np.zeros(image_size)
        rr, cc = disk((14, 14), 10)
        img[rr, cc] = 1
        shapes.append(img)
        labels.append('Circle')
    # Square
    for _ in range(100):
        img = np.zeros(image_size)
        rr, cc = rectangle((5, 5), extent=(18, 18))
        img[rr, cc] = 1
        shapes.append(img)
        labels.append('Square')
    # Triangle
    for _ in range(100):
        img = np.zeros(image_size)
        rr, cc = polygon([10, 18, 26], [6, 18, 6])
        img[rr, cc] = 1
        shapes.append(img)
        labels.append('Triangle')

    return np.array(shapes), np.array(labels)


X, y = generate_shapes_data()
X_flattened = X.reshape(X.shape[0], -1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X_flattened, y_encoded, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(f"Classification Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {label_encoder.inverse_transform([y_pred[i]])[0]}")
    plt.axis('off')
plt.show()









# 10)
# Build a Python application to classify iris flowers using the Nearest Neighbor Rule.
# Use a given dataset with features such as petal length and width.Experiment with different values of K and evaluate the model's accuracy.


import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X, y = datasets.load_iris(return_X_y=True)
X = X[:, 2:4]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
accuracies = [accuracy_score(y_test, KNeighborsClassifier(K).fit(X_train, y_train).predict(X_test)) for K in [1, 3, 5, 7, 9]]

for K, acc in zip([1, 3, 5, 7, 9], accuracies):
    print(f"K={K}, Accuracy={acc*100:.2f}%")

plt.plot([1, 3, 5, 7, 9], accuracies, marker='o')
plt.show()
