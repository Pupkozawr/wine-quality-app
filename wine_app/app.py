import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

from wine_utils import FEATURE_LABELS, classification_summary, prediction_table, read_uploaded_csv


APP_DIR = Path(__file__).resolve().parent
st.set_page_config(page_title='подборка вина', page_icon='🍷', layout='wide')


@st.cache_resource
def load_model(model_path, model_version):
    # один экземпляр модели для всех перезапусков приложения
    return joblib.load(model_path)


@st.cache_data
def load_test_data(csv_path, file_version):
    return pd.read_csv(csv_path)


@st.cache_data
def test_proba(_model, X, model_version):
    # изменение порога не запускает новый расчет вероятностей
    return _model.predict_proba(X)[:, 1]


@st.cache_data
def single_proba(_model, values, features, model_version):
    row = pd.DataFrame([values], columns=features)
    return float(_model.predict_proba(row)[0, 1])


@st.cache_data
def uploaded_proba(_model, X, model_version):
    return _model.predict_proba(X)[:, 1]


@st.cache_data
def ranking_curves(y, probabilities):
    fpr, tpr, _ = metrics.roc_curve(y, probabilities)
    precision, recall, _ = metrics.precision_recall_curve(y, probabilities)
    return fpr, tpr, precision, recall


def file_version(path):
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def reset_threshold(value):
    st.session_state['threshold'] = value


def main():
    st.title('подборка вина')
    st.caption('выбирай вина по вероятности высокой оценки и меняй строгость рекомендаций')
    try:
        model_path = APP_DIR / 'model.joblib'
        model_version = file_version(model_path)
        model = load_model(str(model_path), model_version)
        test_path = APP_DIR / 'test.csv'
        test = load_test_data(str(test_path), file_version(test_path))
        config = json.loads((APP_DIR / 'config.json').read_text(encoding='utf-8'))
        default_threshold = float(config['threshold'])
        if not 0 <= default_threshold <= 1:
            raise ValueError('порог в настройках должен быть от нуля до единицы')
        features = list(model.feature_names_in_)
        X_test = test.loc[:, features]
        y_test = test['target'].astype(int)
        probabilities = test_proba(model, X_test, model_version)
    except (OSError, ValueError, KeyError, AttributeError) as exc:
        st.error('не удалось загрузить модель или данные, проверь файлы в папке приложения')
        st.caption(str(exc))
        st.stop()

    with st.sidebar:
        st.header('строгость отбора')
        st.button('вернуть выбранный порог', on_click=reset_threshold, args=(default_threshold,))
        threshold = st.slider('порог хорошего вина', min_value=0.0, max_value=1.0,
                              value=default_threshold, step=0.001, format='%.3f', key='threshold')
        st.caption(f'порог из задания {default_threshold:.6f}')
        st.write('выше порог, меньше рекомендаций и обычно выше их точность')
        st.write('ниже порог, больше хороших вин попадет в подборку, но больше риск лишних рекомендаций')

    one_tab, metrics_tab, batch_tab, weights_tab = st.tabs([
        'одно вино', 'модельные метрики', 'пакетный прогноз', 'что выучила модель'
    ])

    with one_tab:
        st.subheader('характеристики вина')
        st.caption('диапазоны и начальные значения взяты из тестовых данных, единицы измерения совпадают с исходной таблицей')
        columns = st.columns(3)
        values = {}
        for position, feature in enumerate(name for name in features if name != 'is_red'):
            with columns[position % 3]:
                series = X_test[feature]
                values[feature] = st.number_input(
                    FEATURE_LABELS.get(feature, feature),
                    min_value=float(series.min()), max_value=float(series.max()),
                    value=float(series.median()),
                    step=0.0001 if feature == 'density' else 0.01,
                    format='%.5f' if feature == 'density' else '%.3f', key='feature_' + feature)
        values['is_red'] = int(st.toggle('красное вино', value=bool(X_test['is_red'].median()), key='is_red'))
        probability = single_proba(model, tuple(values[name] for name in features), tuple(features), model_version)
        st.metric('вероятность хорошего вина', f'{probability:.1%}')
        if probability >= threshold:
            st.success('рекомендовать вино')
        else:
            st.info('не рекомендовать при выбранном пороге')
        st.caption(f'текущий порог {threshold:.3f}')

    with metrics_tab:
        summary = classification_summary(y_test, probabilities, threshold)
        st.subheader('качество при выбранном пороге')
        metric_columns = st.columns(4)
        for column, label, value in zip(metric_columns,
                                        ['точность рекомендаций', 'полнота', 'f1', 'выбрано вин'],
                                        [f'{summary["precision"]:.1%}', f'{summary["recall"]:.1%}',
                                         f'{summary["f1"]:.3f}', str(summary['selected'])]):
            column.metric(label, value)
        left, right = st.columns([1, 2])
        with left:
            fig, ax = plt.subplots(figsize=(5, 4))
            metrics.ConfusionMatrixDisplay(summary['matrix'], display_labels=['остальные', 'хорошие']).plot(
                ax=ax, cmap='Blues', colorbar=False)
            ax.set_xlabel('прогноз модели')
            ax.set_ylabel('реальный класс')
            ax.set_title('матрица ошибок')
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        with right:
            fpr, tpr, precision_curve, recall_curve = ranking_curves(y_test.to_numpy(), probabilities)
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].plot(fpr, tpr, label='модель')
            axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray', label='случайное ранжирование')
            axes[0].scatter(summary['fpr'], summary['recall'], color='crimson', label='текущий порог', zorder=3)
            axes[0].set_xlabel('доля ложных срабатываний')
            axes[0].set_ylabel('полнота')
            axes[0].set_title('roc кривая')
            axes[1].step(recall_curve, precision_curve, where='post', label='модель')
            axes[1].axhline(y_test.mean(), linestyle='--', color='gray', label='доля хороших вин')
            axes[1].scatter(summary['recall'], summary['precision'], color='crimson', label='текущий порог', zorder=3)
            axes[1].set_xlabel('полнота')
            axes[1].set_ylabel('точность рекомендаций')
            axes[1].set_title('кривая точности и полноты')
            for ax in axes:
                ax.set_xlim(-0.02, 1.02)
                ax.set_ylim(-0.02, 1.02)
                ax.legend(fontsize=7)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        st.caption('точность показывает долю хороших вин среди рекомендованных, полнота показывает долю найденных хороших вин')
        st.caption('при изменении порога кривые остаются прежними, по ним перемещается рабочая точка')

    with batch_tab:
        st.subheader('прогноз для таблицы вин')
        st.write('загрузи csv с исходными названиями признаков, целевая переменная не нужна')
        st.download_button('скачать пример csv', data=X_test.head(5).to_csv(index=False).encode('utf-8-sig'),
                           file_name='example_wines.csv', mime='text/csv', key='example_download')
        with st.expander('нужные столбцы'):
            st.code(', '.join(features), language=None)
        upload = st.file_uploader('файл с характеристиками вин', type=['csv'], key='wine_csv')
        if upload is not None:
            try:
                uploaded = read_uploaded_csv(upload.getvalue(), features)
                batch_probabilities = uploaded_proba(model, uploaded, model_version)
                result = prediction_table(uploaded, batch_probabilities, threshold)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.caption(f'обработано вин {len(result)}, порог {threshold:.3f}')
                st.dataframe(result.rename(columns=FEATURE_LABELS), hide_index=True, width='stretch')
                st.download_button('скачать прогнозы', data=result.to_csv(index=False).encode('utf-8-sig'),
                                   file_name='wine_predictions.csv', mime='text/csv', key='predictions_download')

    with weights_tab:
        st.subheader('вклад признаков в прогноз')
        classifier = model.named_steps['logisticregression']
        coefficients = pd.Series(classifier.coef_.ravel(), index=features).sort_values()
        labels = [FEATURE_LABELS.get(name, name) for name in coefficients.index]
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#25734b' if value > 0 else '#ad3f4a' for value in coefficients]
        ax.barh(labels, coefficients.to_numpy(), color=colors)
        ax.axvline(0, color='gray', linewidth=0.8)
        ax.set_xlabel('коэффициент после стандартизации')
        ax.set_title('коэффициенты логистической регрессии')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        positive = coefficients[coefficients > 0].nlargest(3).index
        negative = coefficients[coefficients < 0].nsmallest(3).index
        st.write('сильнее поддерживают высокую оценку ' + ', '.join(FEATURE_LABELS.get(name, name) for name in positive))
        st.write('сильнее снижают оценку ' + ', '.join(FEATURE_LABELS.get(name, name) for name in negative))
        st.caption('положительный коэффициент повышает прогноз при росте признака, если остальные признаки фиксированы')
        st.caption('коэффициенты отражают связи в данных, причинное влияние из них не следует')


if __name__ == '__main__':
    main()
