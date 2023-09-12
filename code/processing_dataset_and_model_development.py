# Mengimpor library pendukung
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import mplfinance as mpf

# Menyimpan file CSV dalam DataFrame
df = pd.read_csv('../dataset/monthly_dataset.csv')

# Memberikan label berdasarkan nilai fitur Close
df['Label'] = (df['Close'].diff() >= 0).astype(int)

# Membagi DataFrame menjadi data latih dan data uji menggunakan metode Train Test Split
train_data, test_data = train_test_split(df, train_size=0.75, test_size=0.25, shuffle=False)

# Memisahkan fitur dan label pada data latih dan data uji
train_feature = train_data.drop('Label', axis=1)
train_label = train_data['Label']
test_feature = test_data.drop('Label', axis=1)
test_label = test_data['Label']

...

# Membuat model algoritma Random Forest untuk menerapkan teknik klasifikasi
rf_model = RandomForestClassifier(n_estimators='nan', random_state=42)

# Menyiapkan hyperparameter model untuk jumlah decision tree
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
}

# Menyetel hyperparameter dan mengevaluasi model menggunakan metode Grid Search Cross Validation
grid_search = GridSearchCV(rf_model, param_grid, cv=5)

# Melatih model menggunakan fitur dan label pada data latih
grid_search.fit(train_feature, train_label)

# Mendapatkan hasil penyetelan hyperparameter model
grid_result = grid_search.cv_results_

# Mendapatkan indeks hasil penyetelan hyperparameter model terbaik
best_grid_result_index = grid_search.best_index_

# Mendapatkan hasil pelatihan dan evaluasi model untuk jumlah decision tree dan skor Rank Test
n_estimator = grid_result['param_n_estimators']
rank_test_score = grid_result['rank_test_score']

# Mendapatkan hasil pelatihan dan evaluasi model untuk jumlah decision tree dan skor Rank Test terbaik
best_n_estimator = n_estimator[best_grid_result_index]
best_rank_test_score = rank_test_score[best_grid_result_index]

# Memvisualisasikan hasil pelatihan dan evaluasi model untuk jumlah decision tree dan skor Rank Test terbaik
plt.plot(n_estimator, rank_test_score, color='C0')
plt.plot(best_n_estimator, best_rank_test_score, marker='v', color='C1')
plt.xlabel(f'\nBest Rank Test Score: {best_rank_test_score}\nBest N Estimator: {best_n_estimator}')
plt.ylabel('Score\n')
plt.show()

...

# Mendapatkan hasil pengujian model untuk skor Mean Test dan skor STD Test
mean_test_score = grid_result['mean_test_score']
std_test_score = grid_result['std_test_score']

# Mendapatkan hasil pengujian model untuk skor Mean Test dan skor STD Test terbaik
best_mean_test_score = mean_test_score[best_grid_result_index]
best_std_test_score = std_test_score[best_grid_result_index]

# Memvisualisasikan hasil pengujian model untuk skor Mean Test dan skor STD Test terbaik
plt.bar(['Mean Test', 'STD Test'], [best_mean_test_score, best_std_test_score], color=['C0', 'C1'])
plt.xlabel(f'\nMean Test Score: {best_mean_test_score:.2f}\nSTD Test Score: {best_std_test_score:.2f}')
plt.ylabel('Score\n')
plt.show()

...

# Mendapatkan hasil penyetelan hyperparameter model terbaik
best_grid_result = grid_search.best_estimator_

# Membuat model menggunakan hasil penyetelan hyperparameter terbaik
best_rf_model = best_grid_result

# Memprediksi label berdasarkan fitur pada data uji menggunakan model
predict_label = best_rf_model.predict(test_feature)

# Mendapatkan hasil evaluasi prediksi model untuk skor Accuracy Test
accuracy_test_score = accuracy_score(test_label, predict_label)

# Memvisualisasikan data uji
test_close_up = np.where(test_label == 1, test_feature['Close'], np.nan)
test_close_down = np.where(test_label == 0, test_feature['Close'], np.nan)
test_plot_up = mpf.make_addplot(test_close_up, type='step', color='C2', ylabel='\nTest Direction',
                                secondary_y=False, panel=1)
test_plot_down = mpf.make_addplot(test_close_down, type='step', color='C3', panel=1)

# Memvisualisasikan data hasil prediksi model
predict_close_up = np.where(predict_label == 1, test_feature['Close'], np.nan)
predict_close_down = np.where(predict_label == 0, test_feature['Close'], np.nan)
predict_plot_up = mpf.make_addplot(predict_close_up, type='step', color='C2', ylabel='\nPredict Direction',
                                   secondary_y=False, panel=2)
predict_plot_down = mpf.make_addplot(predict_close_down, type='step', color='C3', panel=2)

# Menggabungkan visualisasi data uji dan data hasil prediksi model
test_feature.set_index('Date', inplace=True)
test_feature.index = pd.to_datetime(test_feature.index, format='%Y%m%d')
mpf.plot(test_feature, type='candlestick', style='yahoo', xlabel=f'\nAccuracy Test Score: {accuracy_test_score:.2f}',
         ylabel='\nTest Price', addplot=[test_plot_up, test_plot_down, predict_plot_up, predict_plot_down])
mpf.show()
