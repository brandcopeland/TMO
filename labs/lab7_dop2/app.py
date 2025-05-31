import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, 
    roc_curve, auc, precision_recall_curve,
    RocCurveDisplay
)
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.multiclass import OneVsRestClassifier

# Настройка страницы
st.set_page_config(
    page_title="ML Model Explorer",
    page_icon="🧠",
    layout="wide"
)

# Заголовок приложения
st.title("🧠 Демонстратор моделей машинного обучения")
st.write("""
Веб-приложение для обучения и визуализации работы различных алгоритмов машинного обучения.
Загрузите свои данные или используйте встроенные наборы данных.
""")

# Боковая панель для настроек
with st.sidebar:
    st.header("⚙️ Настройки эксперимента")
    dataset_type = st.radio("Источник данных:", 
                           ["Встроенные данные", "Загрузить CSV"])
    
    # Выбор набора данных
    if dataset_type == "Встроенные данные":
        dataset_name = st.selectbox(
            "Выберите набор данных:",
            ["Ирисы Фишера", "Рак молочной железы"]
        )
    else:
        data_file = st.file_uploader("Загрузите CSV файл", type="csv")
    
    # Выбор модели
    model_name = st.selectbox(
        "Выберите модель:",
        ["Логистическая регрессия", "Случайный лес", 
         "Градиентный бустинг", "SVM"]
    )
    
    # Параметры модели
    st.subheader("Параметры модели:")
    test_size = st.slider("Доля тестовых данных:", 0.1, 0.5, 0.2)
    random_state = st.number_input("Random State:", 0, 100, 42)

    # Кнопка обучения
    train_button = st.button("Обучить модель")

# Основная область
if dataset_type == "Встроенные данные":
    if dataset_name == "Ирисы Фишера":
        data = load_iris()
        is_multiclass = True
    else:
        data = load_breast_cancer()
        is_multiclass = False
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    class_names = data.target_names if hasattr(data, 'target_names') else None
else:
    if data_file is not None:
        data = pd.read_csv(data_file)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        is_multiclass = len(y.unique()) > 2
        class_names = None
    else:
        st.info("⏳ Ожидаю загрузки данных...")
        st.stop()

# Отображение данных
st.header("🔍 Исследование данных")
st.subheader("Первые 5 строк данных:")
st.write(X.head())

st.subheader("Статистика данных:")
st.write(X.describe())

# Выбор модели
models = {
    "Логистическая регрессия": LogisticRegression(max_iter=10000),
    "Случайный лес": RandomForestClassifier(n_estimators=100),
    "Градиентный бустинг": GradientBoostingClassifier(),
    "SVM": SVC(probability=True)
}

if train_button:
    st.header("📊 Результаты обучения")
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Обучение модели
    model = models[model_name]
    with st.spinner(f"Обучение модели {model_name}..."):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    
    # Оценка модели
    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"Точность модели: **{accuracy:.4f}**")
    
    # Визуализация результатов
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Матрица ошибок")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        
        if class_names is not None:
            ax.set_xticklabels(class_names)
            ax.set_yticklabels(class_names)
            
        ax.set_xlabel('Предсказанный класс')
        ax.set_ylabel('Истинный класс')
        st.pyplot(fig)
    
    with col2:
        if y_proba is not None:
            st.subheader("ROC-кривая")
            fig, ax = plt.subplots(figsize=(8, 6))
            
            if is_multiclass:
                # Многоклассовая ROC-кривая
                y_test_bin = label_binarize(y_test, classes=np.unique(y))
                n_classes = y_test_bin.shape[1]
                
                for i in range(n_classes):
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, lw=2,
                            label=f'Класс {i} (AUC = {roc_auc:.2f})')
                
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.legend(loc="lower right")
                ax.set_title('Многоклассовая ROC-кривая')
            else:
                # Бинарная классификация
                fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color='darkorange', lw=2,
                        label=f'ROC кривая (AUC = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.legend(loc="lower right")
                ax.set_title('ROC-кривая')
            
            st.pyplot(fig)
        else:
            st.warning("ROC-кривая недоступна: модель не возвращает вероятности")
    
    # Важность признаков (если доступно)
    if hasattr(model, 'feature_importances_'):
        st.subheader("Важность признаков")
        importances = pd.Series(
            model.feature_importances_, 
            index=X.columns
        ).sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        importances.plot.bar(ax=ax)
        ax.set_title("Важность признаков")
        ax.set_ylabel("Важность")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

# Инструкция по использованию
st.sidebar.header("ℹ️ Инструкция")
st.sidebar.info("""
1. Выберите или загрузите данные
2. Выберите модель для обучения
3. Настройте параметры
4. Нажмите "Обучить модель"
5. Анализируйте результаты
""")