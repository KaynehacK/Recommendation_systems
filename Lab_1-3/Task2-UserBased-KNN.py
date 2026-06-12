import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# ==========================================
# ГЛОБАЛЬНЫЕ ПАРАМЕТРЫ (Задание 2)
# ==========================================
K_USERS = 30  # Число k похожих пользователей
TOP_X = 10  # Сколько рекомендаций выводить
TEST_SIZE = 0.2  # Процент данных для контроля
MIN_RATINGS_PER_USER = 20  # Минимальное кол-во оценок у пользователя (в датасете и так >20)


# ==========================================
# ЗАГРУЗКА ДАННЫХ
# ==========================================
def load_data():
    ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv('ratings.dat', sep='::', names=ratings_cols, engine='python', encoding='latin-1')

    movies_cols = ['movie_id', 'title', 'genres']
    movies = pd.read_csv('movies.dat', sep='::', names=movies_cols, engine='python', encoding='latin-1')

    return ratings, movies


# ==========================================
# ПОДГОТОВКА И ОБУЧЕНИЕ
# ==========================================
def build_user_recommender(ratings):
    # Разделение на train и test
    train_df, test_df = train_test_split(ratings, test_size=TEST_SIZE, random_state=42)

    # Создаем сводную таблицу (User-Item Matrix)
    # Строки - ПОЛЬЗОВАТЕЛИ, Столбцы - ФИЛЬМЫ
    user_item_matrix = train_df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

    # Превращаем в разреженную матрицу
    sparse_matrix = csr_matrix(user_item_matrix.values)

    # Обучаем модель KNN для поиска похожих пользователей
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=K_USERS, n_jobs=-1)
    model_knn.fit(sparse_matrix)

    return model_knn, user_item_matrix, test_df


# ==========================================
# ФУНКЦИЯ РЕКОМЕНДАЦИИ ДЛЯ ПОЛЬЗОВАТЕЛЯ
# ==========================================
def get_user_recommendations(target_user_id, movies_df, model_knn, user_item_matrix):
    if target_user_id not in user_item_matrix.index:
        return "Пользователь не найден в обучающей выборке."

    # 1. Находим K похожих пользователей
    user_vector = user_item_matrix.loc[target_user_id].values.reshape(1, -1)
    distances, indices = model_knn.kneighbors(user_vector, n_neighbors=K_USERS + 1)

    # Индексы похожих пользователей (исключая самого себя)
    similar_user_indices = indices.flatten()[1:]
    similar_user_ids = user_item_matrix.index[similar_user_indices]
    sim_scores = 1 - distances.flatten()[1:]  # Сходство = 1 - дистанция

    # 2. Получаем фильмы, которые смотрели похожие пользователи
    # Выбираем только те фильмы, которые целевой пользователь еще НЕ смотрел (оценка 0)
    target_user_ratings = user_item_matrix.loc[target_user_id]
    unseen_movies = target_user_ratings[target_user_ratings == 0].index

    # Матрица оценок похожих пользователей для непросмотренных фильмов
    similar_users_ratings = user_item_matrix.loc[similar_user_ids, unseen_movies]

    # 3. Вычисляем прогнозный рейтинг для каждого фильма
    # Прогноз = (сумма оценок * сходство) / (сумма сходств)
    weighted_ratings = similar_users_ratings.T.dot(sim_scores)
    sum_of_similarities = np.array([sim_scores[similar_users_ratings[m] > 0].sum() for m in unseen_movies])

    # Избегаем деления на ноль
    sum_of_similarities[sum_of_similarities == 0] = 1e-9
    preds = weighted_ratings / sum_of_similarities

    # Формируем результат
    recommendations = pd.Series(preds, index=unseen_movies).sort_values(ascending=False)
    top_recs = recommendations.head(TOP_X)

    print(f"\nТоп-{TOP_X} рекомендаций для пользователя {target_user_id}:")
    results = []
    for i, (m_id, score) in enumerate(top_recs.items(), 1):
        title = movies_df[movies_df['movie_id'] == m_id]['title'].values[0]
        print(f"{i}. {title} (прогнозный рейтинг: {score:.2f})")
        results.append(m_id)

    return results


# ==========================================
# ОЦЕНКА КАЧЕСТВА (МЕТРИКИ)
# ==========================================
def evaluate_user_model(model_knn, user_item_matrix, test_df):
    """
    Метрики:
    1. RMSE (Root Mean Squared Error) - точность предсказания оценки.
    2. MAE (Mean Absolute Error) - средняя абсолютная ошибка.
    """
    print("\nРасчет метрик качества...")
    y_true = []
    y_pred = []

    # Берем случайную подвыборку из теста для скорости
    test_sample = test_df.sample(min(1000, len(test_df)), random_state=42)

    for _, row in test_sample.iterrows():
        uid = row['user_id']
        mid = row['movie_id']

        if uid in user_item_matrix.index and mid in user_item_matrix.columns:
            # Находим соседей пользователя
            user_vector = user_item_matrix.loc[uid].values.reshape(1, -1)
            distances, indices = model_knn.kneighbors(user_vector, n_neighbors=K_USERS + 1)

            sim_indices = indices.flatten()[1:]
            sim_ids = user_item_matrix.index[sim_indices]
            sim_values = 1 - distances.flatten()[1:]

            # Оценки соседей для этого фильма
            neighbor_ratings = user_item_matrix.loc[sim_ids, mid]

            # Считаем взвешенное среднее
            mask = neighbor_ratings > 0
            if mask.any():
                pred = np.sum(neighbor_ratings[mask] * sim_values[mask]) / np.sum(sim_values[mask])
                y_true.append(row['rating'])
                y_pred.append(pred)

    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    print(f"Метрика RMSE: {rmse:.4f}")
    print(f"Метрика MAE: {mae:.4f}")
    print(f"Оценка проведена на базе {len(y_true)} тестов.")
    print("--------------------------------------------------")


# ==========================================
# ЗАПУСК
# ==========================================
if __name__ == "__main__":
    # 1. Загрузка
    ratings, movies = load_data()

    # 2. Обучение
    print(f"User-based модель (K_USERS={K_USERS}, TOP_X={TOP_X})")
    model_knn, ui_matrix, test_df = build_user_recommender(ratings)

    # 3. Оценка
    evaluate_user_model(model_knn, ui_matrix, test_df)

    # 4. Демонстрация для пользователя (например, ID=10)
    target_user = 10
    get_user_recommendations(target_user, movies, model_knn, ui_matrix)