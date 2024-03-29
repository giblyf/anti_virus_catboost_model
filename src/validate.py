from train import *
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# Загрузка модели Catboost

def load_model(model_name):
    # '..' переходит на уровень выше относительно src
    model_folder = os.path.join(CURRENT_DIR, '..', 'model')  

    # Путь к файлу модели
    model_file_path = os.path.join(model_folder, model_name)

    model = CatBoostClassifier()
    model.load_model(model_file_path)

    return model


# Функция для записи результатов предсказания
def write_val_results(y_test, y_pred):
   # Расчет матрицы ошибок (Confusion Matrix) и распаковка значений
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = accuracy_score(y_test, y_pred)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = f1_score(y_test, y_pred)

     # '..' переходит на уровень выше относительно src
    results_folder = os.path.join(CURRENT_DIR, '..', 'results')  

     # Проверка и создание папки, если она не существует
    os.makedirs(results_folder, exist_ok=True)


    # Путь к файлу модели

    print("The prediction result is saved as a file 'validation.txt'")
    with open(os.path.join(results_folder, 'validation.txt'), 'w') as f:
        f.write(f'True positive: {tp}\n')
        print(f"True positive: {tp}")

        f.write(f'False positive: {fp}\n')
        print(f"False positive: {fp}")

        f.write(f'False negative: {fn}\n')
        print(f"False negative: {fn}")

        f.write(f'True negative: {tn}\n')
        print(f"True negative: {tn}")

        f.write(f'Accuracy: {accuracy:.4f}\n')
        print(f"Accuracy: {accuracy}")

        f.write(f'Precision: {precision:.4f}\n')
        print(f"Precision: {precision}")

        f.write(f'Recall: {recall:.4f}\n')
        print(f"Recall: {recall}")

        f.write(f'F1: {f1:.4f}\n')
        print(f"F1: {f1}")


# Предсказание
def predict(model, test):
    print('Predicting...')

    X_test = test[model.feature_names_]
    y_pred = model.predict(X_test)

    return y_pred


if __name__ == "__main__":
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

    val_data = read_data('val.tsv')

    model = load_model('anti_virus_detector.cmb')

    feature_engineering(val_data)
    get_dummies(val_data, model.feature_names_)

    y_pred = predict(model, val_data)
    write_val_results(val_data['is_virus'], y_pred)
