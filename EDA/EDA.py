import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Menampilkan deskripsi dan informasi data
data.describe()
data.info()

# Mengecek data duplikat
duplikat = data[data.duplicated()]
if not duplikat.empty:
    print("Data duplikat:")
    print(duplikat)
else:
    print("Tidak ada data duplikat.")

# Mengecek data kosong
kosong = data[data.isnull().any(axis=1)]
if not kosong.empty:
    print("\nData kosong:")
    print(kosong)
else:
    print("\nTidak ada data kosong.")

# Nilai unik pada kolom 'Crop'
unik_crop = data['Crop'].unique()
jumlah_unik_crop = data['Crop'].nunique()
print("Nilai unik pada kolom 'Crop' (Jumlah: {})".format(jumlah_unik_crop))
print(unik_crop)

# Nilai unik pada kolom 'Fertilizer'
unik_fertilizer = data['Fertilizer'].unique()
jumlah_unik_fertilizer = data['Fertilizer'].nunique()
print("\nNilai unik pada kolom 'Fertilizer' (Jumlah: {})".format(jumlah_unik_fertilizer))
print(unik_fertilizer)

# Nilai unik pada kolom 'Soil_color'
unik_soil_color = data['Soil_color'].unique()
jumlah_unik_soil_color = data['Soil_color'].nunique()
print("\nNilai unik pada kolom 'Soil_color' (Jumlah: {})".format(jumlah_unik_soil_color))
print(unik_soil_color)

# Visualisasi distribusi dan pola dalam data numerik
plt.figure(figsize=(12, 8))

# Histogram untuk fitur numerik
numeric_features = ['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature']
for i, feature in enumerate(numeric_features, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data[feature], kde=True)
    plt.title(f'Distribusi {feature}')

plt.tight_layout()
plt.show()

# Box plot untuk fitur numerik
plt.figure(figsize=(12, 8))
for i, feature in enumerate(numeric_features, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=data[feature])
    plt.title(f'Box Plot {feature}')

plt.tight_layout()
plt.show()

# Heatmap untuk matriks korelasi antar fitur
correlation_matrix = data[numeric_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriks Korelasi Antara Fitur')
plt.show()
