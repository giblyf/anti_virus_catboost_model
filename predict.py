from train import clean_libs, read_data, feature_engineering, get_dummies
from validate import load_model, predict
from warnings import simplefilter
import pandas as pd
from catboost import CatBoostClassifier, Pool


# Функция для записи результатов предсказания в файл
def write_test_results(y_pred):
    with open('prediction.txt', 'w') as f:
        f.write('prediction\n') 
        for pred in y_pred:
            f.write(f'{pred}\n')


# Функция, которая бъясняет предсказания модели CatBoostClassifier, 
# записывая в файл 'explain.txt' причины,по которым файлы считаются зловредными
def explain(model, test, y_pred):
    test = test.drop(['libs'], axis=1).copy()

    # Получение важности признаков с использованием модели
    feature_importances = model.get_feature_importance(
        Pool(data=test, label=None))

    # Создание словаря, сопоставляющего каждый признак его важности
    feature_importance_dict = dict(zip(test, feature_importances))

    # Открытие файла 'explain.txt' для записи результата
    with open('explain.txt', 'w') as file:
        # Итерация по предсказаниям и строкам тестового набора данных
        for i, (pred, row) in enumerate(zip(y_pred, test.iterrows())):
            if pred:  # Если модель считает файл зловредным
                reasons_yes = []  # Список причин для вирусных файлов
                reasons_no = []  # Список причин для не вирусных файлов

                # Итерация по признакам и их значениям в текущей строке
                for feature, value in row[1].items():
                    # Порог важности признака равен 3
                    if feature_importance_dict[feature] > 3 and value:
                        reasons_yes.append(f"{feature}")
                    elif feature_importance_dict[feature] > 3 and not value:
                        reasons_no.append(f"{feature}")

                # Формирование строки результата
                result = f"Есть библиотеки {', '.join(reasons_yes)} " * bool(
                    reasons_yes) + f"Нет библиотек {', '.join(reasons_no)}" * bool(reasons_no)
            else:
                result = f""

            # Запись результата в файл
            file.write(result + '\n')


if __name__ == "__main__":
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

    test_data = read_data('test.tsv')

    model = load_model('anti_virus_detector.cmb')

    feature_engineering(test_data)
    get_dummies(test_data, model.feature_names_)

    y_pred = predict(model, test_data)
    write_test_results(y_pred)
    explain(model, test_data, y_pred)
