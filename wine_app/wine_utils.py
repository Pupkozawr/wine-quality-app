import csv
import io

import numpy as np
import pandas as pd
from sklearn import metrics


FEATURE_LABELS = {
    'fixed_acidity': 'фиксированная кислотность',
    'volatile_acidity': 'летучая кислотность',
    'citric_acid': 'лимонная кислота',
    'residual_sugar': 'остаточный сахар',
    'chlorides': 'хлориды',
    'free_sulfur_dioxide': 'свободный диоксид серы',
    'total_sulfur_dioxide': 'общий диоксид серы',
    'density': 'плотность',
    'pH': 'кислотность ph',
    'sulphates': 'сульфаты',
    'alcohol': 'алкоголь',
    'is_red': 'красное вино'
}


def read_uploaded_csv(content, features):
    if not content.strip():
        raise ValueError('в файле нет строк с данными')
    try:
        frame = pd.read_csv(io.BytesIO(content), sep=None, engine='python', encoding='utf-8-sig')
    except (ValueError, UnicodeError, pd.errors.ParserError, csv.Error) as exc:
        raise ValueError('не удалось прочитать csv, проверь кодировку utf8 и разделители') from exc
    if frame.empty:
        raise ValueError('в файле нет строк с данными')
    frame.columns = frame.columns.str.strip()
    if frame.columns.duplicated().any():
        raise ValueError('названия столбцов не должны повторяться')
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise ValueError('в файле не хватает столбцов ' + ', '.join(missing))
    data = frame.loc[:, features].apply(pd.to_numeric, errors='coerce')
    invalid = data.columns[~np.isfinite(data.to_numpy(dtype=float)).all(axis=0)].tolist()
    if invalid:
        raise ValueError('нужны конечные числа без пропусков в столбцах ' + ', '.join(invalid))
    if 'is_red' in features and not data['is_red'].isin([0, 1]).all():
        raise ValueError('в столбце is_red допустимы только 0 и 1')
    return data


def classification_summary(y, probabilities, threshold):
    pred = (np.asarray(probabilities) >= threshold).astype(int)
    matrix = metrics.confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        'precision': metrics.precision_score(y, pred, zero_division=0),
        'recall': metrics.recall_score(y, pred, zero_division=0),
        'f1': metrics.f1_score(y, pred, zero_division=0),
        'accuracy': metrics.accuracy_score(y, pred),
        'fpr': fp / (fp + tn) if fp + tn else 0.0,
        'matrix': matrix,
        'selected': int(pred.sum()),
        'predictions': pred
    }


def prediction_table(data, probabilities, threshold):
    result = data.copy()
    result['вероятность хорошего вина'] = probabilities
    result['прогноз класса'] = (np.asarray(probabilities) >= threshold).astype(int)
    result['решение'] = np.where(result['прогноз класса'] == 1, 'рекомендовать', 'не рекомендовать')
    return result
