import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, RocCurveDisplay
from catboost import CatBoostClassifier
import os.path
from warnings import simplefilter
import time


# Функция для чтения файлов .tsv
def read_data(file_name):
    # Получение абсолютного пути к текущему каталогу, в котором читаем файл
    # train.tsv
    current_dir = os.path.dirname(os.path.abspath(__file__))

    print("Reading train.tsv...\n")
    data = pd.read_csv(os.path.join(current_dir, file_name), sep='\t')

    print('Data information:')
    data.info()

    print()

    return data


# Функция для очистки и преобразования библиотек
def clean_libs(libs):
    cleaned_libs = []
    for lib in libs:
        # У 223 файлов имеется библиотека msbvmb60 с указанием пути c:\\\/\/\/\/\//\\\\\\\//////\/\/\/windows\\\\...
        #  Удалим этот путь
        if 'c:\\' in lib:
            lib = 'msvbvm60'
        else:
            # Удаление расширений .dll и .sll
            lib = lib.replace('.dll', '')
            lib = lib.replace('.sll', '')

            # Удаление ненужных символов
            lib = lib.strip(' .//\\,![]><?|')

        cleaned_libs.append(lib)

    return cleaned_libs


# Функция для добавления уникальной библиотеки в список библиотек
def add_libs(unique_libs, libs):
    for lib in libs:
        unique_libs.add(lib)


# Создадие дамми переменных на основе списка уникальных библиотек:
def get_dummies(df, libs):
    for lib in libs:
        df[lib] = df['libs'].apply(lambda libs: 1 if lib in libs else 0).copy()

    print(
        f"Count of features:{len(df.columns[3:])}",
        f"Count of files: {len(df)}\n",
        sep='\n')


# Создание списка уникальных библиотек
def get_libs(df):
    unique_libs = set()
    df['libs'].apply(lambda libs: add_libs(unique_libs, libs))

    return unique_libs


# Предобработка данных
def feature_engineering(df):
    print('Feature Engineering...')

    # Т.к в тестовом наборе данных нет колонки filename, удалим ее
    if 'filename' in df.columns:
        df.drop('filename', axis=1, inplace=True)

    # преобразуем строки с библиотеками в список
    df['libs'] = (df['libs'].apply(lambda libs: libs.split(','))).copy()

    # Разведывательный анализ показал, что многие библиотеки, имеющие одно и то же название, например msvbvm60, заполнены
    # различными лишними подстроками: формат .dll, путь c:\\, пробелы и т.д. Чтобы отнсести их все к одной библиотеке,
    # уберем лишние символы
    df['libs'] = df['libs'].apply(lambda libs: clean_libs(libs)).copy()


# Обучение модели
def train_model(df):
    X_train = df.drop(['libs', 'is_virus'], axis=1)
    y_train = df['is_virus']

    # Гиперпараметры модели были подобраны на основе валидационной выборки
    params = {'verbose': 100,
              'random_seed': 42,
              'learning_rate': 0.125}

    print('Fit the model...')
    start = time.time()
    model = CatBoostClassifier(**params)
    model.fit(X=X_train, y=y_train)
    train_time = time.time() - start
    print()

    print('Done training!')
    print('Save the model as "anti_virus_detector.cbm"',
          f'Train time: {train_time}', sep='\n')
    model.save_model('anti_virus_detector.cmb', format="cbm")


if __name__ == "__main__":
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
    train_data = read_data('train.tsv')

    feature_engineering(train_data)
    unqiue_list = get_libs(train_data)
    get_dummies(train_data, unqiue_list)

    train_model(train_data)
