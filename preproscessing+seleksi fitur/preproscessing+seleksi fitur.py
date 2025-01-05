import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Mengencode kolom kategori
kolom_encode = ['Crop', 'Fertilizer', 'Soil_color', 'District_Name']
encoder_dict = {}

for kolom in kolom_encode:
    encoder = LabelEncoder()
    data[kolom + '_encoded'] = encoder.fit_transform(data[kolom])
    encoder_dict[kolom] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

for kolom in kolom_encode:
    data[kolom] = data[kolom + '_encoded']
    data.drop(columns=[kolom + '_encoded'], inplace=True)

print("DataFrame dengan data yang telah diencode:")
print(data)

print("\nDictionary korespondensi nilai asli dan nilai terkodena:")
print(encoder_dict)

# Memisahkan fitur dan target
X = data.drop(columns=['Fertilizer'])  # Memilih semua fitur kecuali target 'Fertilizer'
y = data['Fertilizer']

# Memilih fitur terbaik dengan metode Chi-Square
k = 7
chi2_selector = SelectKBest(chi2, k=k)
X_new = chi2_selector.fit_transform(X, y)

selected_features = data.columns[:-1][chi2_selector.get_support()]
print("\nSelected Features:")
print(selected_features)

# Mengonversi menjadi DataFrame untuk visualisasi yang lebih baik
X_new = X[selected_features]
y = data['Fertilizer']

# Menghitung varians per kelas pada kolom 'Fertilizer'
varians_per_kelas = data.groupby('Fertilizer').size().reset_index(name='Jumlah')

# Mengurutkan kelas berdasarkan jumlah dari yang terbesar ke yang terkecil
kelas_dengan_jumlah_terbanyak = varians_per_kelas.sort_values(by='Jumlah', ascending=False)

# Menghitung jumlah data yang akan dipilih dari kelas 17
target_count = 550

# Menghitung jumlah data yang tersedia pada kelas 17
available_count = kelas_dengan_jumlah_terbanyak.loc[kelas_dengan_jumlah_terbanyak['Fertilizer'] == 17, 'Jumlah'].values[0]

# Menentukan berapa banyak data yang akan dipilih dari kelas 17
ratio = target_count / available_count

# Mengambil data dari kelas 17 sesuai dengan rasio yang ditentukan
selected_data_17 = data[data['Fertilizer'] == 17].sample(frac=ratio, random_state=42)

# Mengambil data dari kelas lain
selected_data_others = data[data['Fertilizer'] != 17]

# Menggabungkan data dari kelas 17 dengan data dari kelas lain
selected_data = pd.concat([selected_data_17, selected_data_others])

# Pisahkan fitur dan target dari data yang terpilih
X = selected_data.drop(columns=['Fertilizer'])
y = selected_data['Fertilizer']

# Standarisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Hasil standarisasi fitur:")
print(pd.DataFrame(X_scaled, columns=X.columns).head())

# Membagi data menjadi train dan test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Oversampling menggunakan SMOTE
smote = SMOTE(sampling_strategy={11: 100, 3: 100, 8: 100, 0: 100, 12: 60, 18: 50, 6: 50, 16: 50, 4: 50}, k_neighbors=2)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Menampilkan distribusi kelas setelah oversampling
print("Distribusi kelas setelah SMOTE:")
print(y_train_resampled.value_counts())
