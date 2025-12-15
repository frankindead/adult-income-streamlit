import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

@st.cache_resource
def train_model():
    df = pd.read_csv('data.adult.csv')

    df = df.replace("?", np.nan).dropna() # заменяем ? на np.nan и удаляем пропуски

    y = df['>50K,<=50K'].replace({">50K": 1, "<=50K": 0}).astype(int) # целевая переменная
    X = df.drop(columns='>50K,<=50K')

    cat_features = X.select_dtypes(include=["object"]).columns.tolist() # типы признаков
    num_features = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )

    model = GradientBoostingClassifier( # выбрали лучшую модель для приложения
        n_estimators=88,     # определяем выявленные лучшие гиперпараметры модели
        max_features=None, 
        criterion="friedman_mse", 
        random_state=42
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ]) # определяем пайплайн

    pipe.fit(X, y) # фиттим под наши данные

    return pipe

st.set_page_config(
    page_title="Предсказатель дохода!!!",
    page_icon="💰",
    layout="centered"
)

st.title("Предсказатель дохода!!!") # заголовок и описание приложения
st.write(
    "Введите характеристики человека — модель предскажет, "
    "превышает ли доход порог 50K вечнозелёных."
)

model = train_model()

st.header("Ввод данных")

age = st.number_input("Возраст", min_value=0, max_value=120, value=35) # добавляем возможность прямого ввода числового значения для вещественных переменных
fnlwgt = st.number_input("Выборочный вес наблюдения", min_value=0, value=0)
education_num = st.number_input(
    "Уровень образования (числовой код)",
    min_value=1,
    max_value=20,
    value=10
)
capital_gain = st.number_input("Доход от капитала", min_value=0, value=0)
capital_loss = st.number_input("Убыток от капитала", min_value=0, value=0)
hours_per_week = st.number_input(
    "Часы работы в неделю",
    min_value=0,
    max_value=120,
    value=40
)

workclass_map = { # для категориальных признаков показываем пользователю русские варианты, а в модель передаём исходные английские значения
    "Частный сектор": "Private",
    "Работает на себя (без юрлица)": "Self-emp-not-inc",
    "Работает на себя (с юрлицом)": "Self-emp-inc",
    "Федеральное управление": "Federal-gov",
    "Местное управление": "Local-gov",
    "Региональное управление": "State-gov",
    "Без оплаты": "Without-pay",
    "Никогда не работал": "Never-worked",
}
workclass = workclass_map[
    st.selectbox("Тип занятости", list(workclass_map.keys()))
]

education_map = {
    "Бакалавр": "Bachelors",
    "Неполное высшее / колледж": "Some-college",
    "Средняя школа (выпускник)": "HS-grad",
    "Магистр": "Masters",
    "Доктор наук": "Doctorate",
    "Ассоциированная степень (акад.)": "Assoc-acdm",
    "Ассоциированная степень (проф.)": "Assoc-voc",
    "11 класс": "11th",
    "10 класс": "10th",
    "9 класс": "9th",
    "7–8 класс": "7th-8th",
}
education = education_map[
    st.selectbox("Образование", list(education_map.keys()))
]

marital_map = {
    "Женат / замужем": "Married-civ-spouse",
    "Никогда не состоял(а) в браке": "Never-married",
    "Разведен(а)": "Divorced",
    "В разлуке": "Separated",
    "Вдовец / вдова": "Widowed",
}
marital_status = marital_map[
    st.selectbox("Семейное положение", list(marital_map.keys()))
]

occupation_map = {
    "Техподдержка": "Tech-support",
    "Ремонт / ремесло": "Craft-repair",
    "Продажи": "Sales",
    "Руководитель / менеджер": "Exec-managerial",
    "Профессиональный специалист": "Prof-specialty",
    "Офисный сотрудник": "Adm-clerical",
    "Уборка / обслуживание": "Handlers-cleaners",
}
occupation = occupation_map[
    st.selectbox("Профессия", list(occupation_map.keys()))
]

relationship_map = {
    "Муж": "Husband",
    "Жена": "Wife",
    "Ребёнок": "Own-child",
    "Не в семье": "Not-in-family",
    "Не женат / не замужем": "Unmarried",
}
relationship = relationship_map[
    st.selectbox("Роль в домохозяйстве / Семейное положение", list(relationship_map.keys()))
]

race_map = {
    "Белый": "White",
    "Чёрный": "Black",
    "Азиат / Тихоокеанский регион": "Asian-Pac-Islander",
    "Коренной американец": "Amer-Indian-Eskimo",
    "Другое": "Other",
}
race = race_map[
    st.selectbox("Раса", list(race_map.keys()))
]

sex_map = {
    "Мужчина": "Male",
    "Женщина": "Female",
}
sex = sex_map[
    st.selectbox("Пол", list(sex_map.keys()))
]

native_country_map = {
    "США": "United-States",
    "Мексика": "Mexico",
    "Филиппины": "Philippines",
    "Германия": "Germany",
    "Канада": "Canada",
    "Индия": "India",
}
native_country = native_country_map[
    st.selectbox("Страна рождения", list(native_country_map.keys()))
]

input_df = pd.DataFrame([{ # собираем все введённые значения в датафрейм с теми же колонками, на которых обучалась модель
    "age": age,
    "workclass": workclass,
    "fnlwgt": fnlwgt,
    "education": education,
    "education-num": education_num,
    "marital-status": marital_status,
    "occupation": occupation,
    "relationship": relationship,
    "race": race,
    "sex": sex,
    "capital-gain": capital_gain,
    "capital-loss": capital_loss,
    "hours-per-week": hours_per_week,
    "native-country": native_country
}])

if st.button("Предсказать"): # при нажатии кнопки считаем вероятность и показываем результат пользователю
    proba = model.predict_proba(input_df)[0, 1] * 100

    st.subheader("Результат")
    st.metric("Вероятность дохода выше $50K", f"{proba:.3f}%")

    if proba >= 50:
        st.success("Доход, вероятно, превышает $50K")
    else:
        st.info("Доход, вероятно, не превышает $50K")