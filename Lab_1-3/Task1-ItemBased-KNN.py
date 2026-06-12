import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math

# ==========================================
# ГЛОБАЛЬНЫЕ ПАРАМЕТРЫ (Задание 1)
# ==========================================
K_NEIGHBORS = 20  # Число k соседей для алгоритма
TOP_X = 10  # Сколько рекомендаций выводить
TEST_SIZE = 0.2  # Процент данных для контроля (20%)
MIN_RATINGS_PER_MOVIE = 50  # Фильтрация малоизвестных фильмов для стабильности


# ==========================================
# ЗАГРУЗКА ДАННЫХ
# ==========================================
def load_data():
    # Загрузка рейтингов
    ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv('ratings.dat', sep='::', names=ratings_cols, engine='python', encoding='latin-1')

    # Загрузка названий фильмов
    movies_cols = ['movie_id', 'title', 'genres']
    movies = pd.read_csv('movies.dat', sep='::', names=movies_cols, engine='python', encoding='latin-1')

    return ratings, movies


# ==========================================
# ПОДГОТОВКА И ОБУЧЕНИЕ
# ==========================================
def build_recommender(ratings):
    # Фильтруем фильмы, чтобы оставить только те, у которых достаточно оценок
    movie_counts = ratings.groupby('movie_id').size()
    popular_movies = movie_counts[movie_counts >= MIN_RATINGS_PER_MOVIE].index
    filtered_ratings = ratings[ratings['movie_id'].isin(popular_movies)]

    # Разделение на train и test (по рейтингам)
    train_df, test_df = train_test_split(filtered_ratings, test_size=TEST_SIZE, random_state=42)

    # Создаем сводную таблицу (Pivot Table) для Item-based
    # Строки - фильмы, столбцы - пользователи
    user_item_matrix = train_df.pivot(index='movie_id', columns='user_id', values='rating').fillna(0)

    # Превращаем в разреженную матрицу для ускорения вычислений
    sparse_matrix = csr_matrix(user_item_matrix.values)

    # Обучаем модель KNN
    # Используем косинусное сходство (cosine similarity) - стандарт для РС
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=K_NEIGHBORS, n_jobs=-1)
    model_knn.fit(sparse_matrix)

    return model_knn, user_item_matrix, test_df


# ==========================================
# ФУНКЦИЯ РЕКОМЕНДАЦИИ
# ==========================================
def get_recommendations(movie_title, movies_df, model_knn, user_item_matrix):
    # Поиск ID фильма по названию
    try:
        movie_idx = movies_df[movies_df['title'].str.contains(movie_title, case=False, regex=False)].iloc[0]['movie_id']
    except IndexError:
        return "Фильм не найден в базе данных."

    if movie_idx not in user_item_matrix.index:
        return "Фильм был отфильтрован из-за малого количества оценок."

    # Получаем индекс строки в матрице
    query_index = user_item_matrix.index.get_loc(movie_idx)

    # Находим ближайших соседей
    distances, indices = model_knn.kneighbors(
        user_item_matrix.iloc[query_index, :].values.reshape(1, -1),
        n_neighbors=TOP_X + 1
    )

    print(f"\nРекомендации для фильма: {movie_title}")
    recs = []
    for i in range(1, len(distances.flatten())):
        target_movie_id = user_item_matrix.index[indices.flatten()[i]]
        title = movies_df[movies_df['movie_id'] == target_movie_id]['title'].values[0]
        dist = distances.flatten()[i]
        recs.append((title, dist))
        print(f"{i}: {title} (дистанция: {dist:.3f})")

    return recs


# ==========================================
# ОЦЕНКА КАЧЕСТВА (МЕТРИКИ)
# ==========================================
def evaluate_model(model_knn, user_item_matrix, test_df):
    """
    Расчет RMSE для предсказания рейтингов.
    Для Item-based KNN: рейтинг = взвешенное среднее оценок соседей.
    """
    print("\nРасчет метрик качества...")
    y_true = []
    y_pred = []

    # Берем случайную выборку из теста для ускорения оценки
    test_sample = test_df.sample(min(2000, len(test_df)), random_state=42)

    for _, row in test_sample.iterrows():
        uid = row['user_id']
        mid = row['movie_id']

        if mid in user_item_matrix.index:
            # Находим соседей фильма mid
            query_index = user_item_matrix.index.get_loc(mid)
            distances, indices = model_knn.kneighbors(
                user_item_matrix.iloc[query_index, :].values.reshape(1, -1),
                n_neighbors=K_NEIGHBORS
            )

            neighbor_ids = user_item_matrix.index[indices.flatten()[1:]]
            neighbor_distances = distances.flatten()[1:]

            # Пытаемся предсказать рейтинг как среднее соседей, которых этот пользователь уже оценивал
            weights = []
            ratings = []

            for i, n_id in enumerate(neighbor_ids):
                # Если у пользователя в обучающей выборке есть оценка соседа
                user_rating = user_item_matrix.loc[n_id, uid] if uid in user_item_matrix.columns else 0
                if user_rating > 0:
                    # Вес = 1 - дистанция (чем меньше дистанция, тем больше сходство)
                    sim = 1 - neighbor_distances[i]
                    weights.append(sim)
                    ratings.append(user_rating)

            if sum(weights) > 0:
                predicted_rating = sum(np.array(ratings) * np.array(weights)) / sum(weights)
                y_true.append(row['rating'])
                y_pred.append(predicted_rating)

    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    print(f"Метрика RMSE: {rmse:.4f} (на базе {len(y_true)} предсказаний)")
    print("--------------------------------------------------")


# ==========================================
# ОСНОВНОЙ ЦИКЛ
# ==========================================
if __name__ == "__main__":
    # 1. Загрузка
    print("Загрузка данных...")
    ratings, movies = load_data()

    # 2. Обучение
    print(f"Обучение модели (K={K_NEIGHBORS}, Test={TEST_SIZE * 100}%)...")
    model_knn, ui_matrix, test_df = build_recommender(ratings)

    # 3. Оценка
    evaluate_model(model_knn, ui_matrix, test_df)

    # 4. Демонстрация
    # Попробуем найти рекомендации для известного фильма
    example_movie = "Toy Story (1995)"
    get_recommendations(example_movie, movies, model_knn, ui_matrix)

    example_movie_2 = "Jurassic Park (1993)"
    get_recommendations(example_movie_2, movies, model_knn, ui_matrix)