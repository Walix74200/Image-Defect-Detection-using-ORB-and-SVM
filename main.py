import cv2
import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import glob
from imgaug import augmenters as iaa

def extract_features(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return descriptors

def load_images_from_folder(folder):
    images = []
    for filename in glob.glob(folder):
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

correct_images_path = 'path to your reference images folder*.*'
error_images_path = 'path to your error images folder*.*'

# Charger les images de référence et les images avec erreurs
correct_images = load_images_from_folder(correct_images_path)
error_images = load_images_from_folder(error_images_path)

seq = iaa.Sequential([
    iaa.Fliplr(0.5),  
    iaa.Rotate((-10, 10)),  
    iaa.Affine(scale=(0.8, 1.2)),  
    iaa.Multiply((0.8, 1.2)),  
])

correct_images_augmented = seq(images=correct_images)
error_images_augmented = seq(images=error_images)

correct_images += correct_images_augmented
error_images += error_images_augmented

correct_features = [extract_features(img) for img in correct_images]
error_features = [extract_features(img) for img in error_images]

# Création des labels
correct_labels = [0] * len(correct_features)
error_labels = [1] * len(error_features)

features = correct_features + error_features
labels = correct_labels + error_labels

features, labels = zip(*[(f, l) for f, l in zip(features, labels) if f is not None])

all_descriptors = np.vstack(features)  

n_clusters = 50 
kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(all_descriptors)

def create_histogram(descriptors, kmeans):
    histogram = np.zeros(n_clusters)
    if descriptors is not None:
        clusters = kmeans.predict(descriptors)
        for c in clusters:
            histogram[c] += 1
    return histogram

features_histogram = [create_histogram(desc, kmeans) for desc in features]

X = np.array(features_histogram)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement de mon SVM
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Calcule de l'accuracy
print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')

# Fonction pour prédire si une nouvelle image a des erreurs d'impression
def predict_image(image):
    features = extract_features(image)
    if features is not None:
        histogram = create_histogram(features, kmeans)
        histogram = histogram.reshape(1, -1)
        prediction = clf.predict(histogram)
        return 'Correct' if prediction == 0 else 'Error'
    else:
        return 'No features detected'

new_image = cv2.imread('path to your test image.jpg', cv2.IMREAD_GRAYSCALE)
print(predict_image(new_image))
