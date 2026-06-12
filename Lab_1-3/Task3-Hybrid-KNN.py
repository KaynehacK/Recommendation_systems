import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math

# ==========================================
# ГЛОБАЛЬНЫЕ ПАРАМЕТРЫ (Задание 3)
# ==========================================
K_USERS = 30
K_ITEMS = 20
TOP_X = 10
TEST_SIZE = 0.2
HYBRID_ALPHA = 0.5  # Вес User-based (0.5 = поровну с Item-based)


# ==========================================
# ЗАГРУЗКА И ПОДГОТОВКА
# ==========================================
def load_data():
    r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv('ratings.dat', sep='::', names=r_cols, engine='python', encoding='latin-1')
    m_cols = ['movie_id', 'title', 'genres']
    movies = pd.read_csv('movies.dat', sep='::', names=m_cols, engine='python', encoding='latin-1')
    return ratings, movies


def build_models(ratings):
    train_df, test_df = train_test_split(ratings, test_size=TEST_SIZE, random_state=42)

    # Матрица для User-based (Строки - Юзеры)
    user_item_mtx = train_df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    model_ub = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=K_USERS)
    model_ub.fit(csr_matrix(user_item_mtx.values))

    # Матрица для Item-based (Строки - Фильмы)
    item_user_mtx = train_df.pivot(index='movie_id', columns='user_id', values='rating').fillna(0)
    model_ib = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=K_ITEMS)
    model_ib.fit(csr_matrix(item_user_mtx.values))

    return model_ub, model_ib, user_item_mtx, item_user_mtx, test_df


# ==========================================
# ГИБРИДНАЯ ЛОГИКА
# ==========================================
def get_hybrid_recommendations(user_id, movies_df, model_ub, model_ib, ui_mtx, iu_mtx):
    if user_id not in ui_mtx.index: return []

    # --- ШАГ 0: User-based кандидаты ---
    dist_u, ind_u = model_ub.kneighbors(ui_mtx.loc[user_id].values.reshape(1, -1), n_neighbors=K_USERS + 1)
    sim_users = ui_mtx.index[ind_u.flatten()[1:]]
    user_unseen = ui_mtx.loc[user_id][ui_mtx.loc[user_id] == 0].index

    ub_preds = ui_mtx.loc[sim_users, user_unseen].mean(axis=0).dropna()
    # Нормализация UB (0-1)
    if not ub_preds.empty:
        ub_preds = (ub_preds - ub_preds.min()) / (ub_preds.max() - ub_preds.min() + 1e-9)

    # --- ШАГ 1: Item-based кандидаты ---
    # Берем фильмы, которые юзер оценил высоко (4 или 5)
    user_liked_movies = ui_mtx.loc[user_id][ui_mtx.loc[user_id] >= 4].index
    ib_candidates = {}

    for m_id in user_liked_movies:
        if m_id in iu_mtx.index:
            dist_i, ind_i = model_ib.kneighbors(iu_mtx.loc[m_id].values.reshape(1, -1), n_neighbors=5)
            for i, idx in enumerate(ind_i.flatten()[1:]):
                neighbor_id = iu_mtx.index[idx]
                if ui_mtx.loc[user_id, neighbor_id] == 0:  # если еще не смотрел
                    sim = 1 - dist_i.flatten()[i + 1]
                    ib_candidates[neighbor_id] = ib_candidates.get(neighbor_id, 0) + sim

    ib_preds = pd.Series(ib_candidates)
    # Нормализация IB (0-1)
    if not ib_preds.empty:
        ib_preds = (ib_preds - ib_preds.min()) / (ib_preds.max() - ib_preds.min() + 1e-9)

    # --- ШАГ 2-3: Пересечение и объединение ---
    # Создаем общий датафрейм для скоринга
    all_movies = list(set(ub_preds.index) | set(ib_preds.index))
    hybrid_scores = pd.DataFrame(index=all_movies)
    hybrid_scores['ub_score'] = ub_preds
    hybrid_scores['ib_score'] = ib_preds
    hybrid_scores.fillna(0, inplace=True)

    # ФОРМУЛА: Взвешенная сумма + бонус 20% за пересечение
    hybrid_scores['final'] = (HYBRID_ALPHA * hybrid_scores['ub_score'] +
                              (1 - HYBRID_ALPHA) * hybrid_scores['ib_score'])

    # Увеличиваем оценку, если фильм найден обоими методами
    mask_both = (hybrid_scores['ub_score'] > 0) & (hybrid_scores['ib_score'] > 0)
    hybrid_scores.loc[mask_both, 'final'] *= 1.2

    top_list = hybrid_scores.sort_values('final', ascending=False).head(TOP_X)

    # Печать результата
    print(f"\nГибридные рекомендации для пользователя {user_id}:")
    for i, (m_id, row) in enumerate(top_list.iterrows(), 1):
        title = movies_df[movies_df['movie_id'] == m_id]['title'].values[0]
        tag = "[ОБА]" if mask_both.loc[m_id] else ("[UB]" if row['ub_score'] > 0 else "[IB]")
        print(f"{i}. {tag} {title} (Score: {row['final']:.3f})")

    return top_list.index.tolist()


# ==========================================
# ИДЕИ ДЛЯ УЛУЧШЕНИЯ (ПУНКТ 4)
# ==========================================
"""
1. Учет жанров (Content-Based Filtering): Добавлять бонус фильмам тех жанров, 
   которые пользователь смотрит чаще всего.
2. Период актуальности: Использовать Timestamp, чтобы давать больший вес 
   недавним оценкам пользователя.
3. Штраф за популярность: Немного снижать вес слишком известных фильмов 
   (типа 'Titanic'), чтобы рекомендовать более редкие, но точные фильмы (Long Tail).
"""

# ==========================================
# МЕТРИКИ И ЗАПУСК
# ==========================================
if __name__ == "__main__":
    ratings, movies = load_data()

    # Тест влияния K на RMSE (упрощенно)
    for k_val in [10, 30, 50]:
        K_USERS = k_val
        print(f"\n--- Тестирование при K_USERS = {k_val} ---")
        model_ub, model_ib, ui_mtx, iu_mtx, test_df = build_models(ratings)

        # Демонстрация
        get_hybrid_recommendations(10, movies, model_ub, model_ib, ui_mtx, iu_mtx)