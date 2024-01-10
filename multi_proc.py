import re
from string import punctuation

import pandas as pd
from joblib import Parallel
from joblib import delayed
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def clear_data(source_path: str, target_path: str, n_jobs: int):
    """Baseline process df

    Parameters
    ----------
    source_path : str
        Path to load dataframe from

    target_path : str
        Path to save dataframe to
    """
    data = pd.read_parquet(source_path)
    data = data.copy().dropna().reset_index(drop=True)

    # Разбиваем значение столбца на батчи
    # batch_size = 4
    # batches = [data['text'][i:i + batch_size] for i in range(0, len(data['text']), batch_size)]

    # Применяем функцию к каждому батчу параллельно
    # results = Parallel(n_jobs=n_jobs)(delayed(clean_batch)(text) for text in data['text'])
    results = Parallel(n_jobs=n_jobs)(delayed(map)(clean_batch, data['text']))

    # Объединяем результаты в один список
    # result = [item for item in results]

    # Создаем новый столбец с результатами
    data['cleaned_text'] = results

    # Выводим обновленный датафрейм
    print(data)
    data.to_parquet(target_path)


def clean_batch(text):
    lemmatizer = WordNetLemmatizer()

    text = str(text)
    text = re.sub(r"https?://[^,\s]+,?", "", text)
    text = re.sub(r"@[^,\s]+,?", "", text)

    stop_words = stopwords.words("english")
    transform_text = text.translate(str.maketrans("", "", punctuation))
    transform_text = re.sub(" +", " ", transform_text)

    text_tokens = word_tokenize(transform_text)

    lemma_text = [
        lemmatizer.lemmatize(word.lower()) for word in text_tokens
    ]
    cleaned_text = " ".join(
        [str(word) for word in lemma_text if word not in stop_words]
    )
    return cleaned_text

    # cleaned_text_list = []
    # for text in batch:
    #     text = str(text)
    #     text = re.sub(r"https?://[^,\s]+,?", "", text)
    #     text = re.sub(r"@[^,\s]+,?", "", text)
    #
    #     stop_words = stopwords.words("english")
    #     transform_text = text.translate(str.maketrans("", "", punctuation))
    #     transform_text = re.sub(" +", " ", transform_text)
    #
    #     text_tokens = word_tokenize(transform_text)
    #
    #     lemma_text = [
    #         lemmatizer.lemmatize(word.lower()) for word in text_tokens
    #     ]
    #     cleaned_text = " ".join(
    #         [str(word) for word in lemma_text if word not in stop_words]
    #     )
    #     cleaned_text_list.append(cleaned_text)
    #
    # return cleaned_text_list




