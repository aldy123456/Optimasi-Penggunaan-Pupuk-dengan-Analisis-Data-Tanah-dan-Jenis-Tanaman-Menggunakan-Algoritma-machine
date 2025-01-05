from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Inisialisasi model KNN dengan 1 tetangga terdekat
knn_model = KNeighborsClassifier(n_neighbors=1)

# Melatih model dengan data latih yang telah di-resample
knn_model.fit(X_train_resampled, y_train_resampled)

# Memprediksi data uji
y_pred_knn = knn_model.predict(X_test)

# Menampilkan Classification Report
print("\nClassification Report KNN:")
print(classification_report(y_test, y_pred_knn))

# Menghitung Confusion Matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)

# Visualisasi Confusion Matrix menggunakan heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix KNN')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
