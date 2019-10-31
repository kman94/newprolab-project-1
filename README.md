### Репозиторий для первого проекта курса NewProLab-BigData по прогнозированию пола и возраста интернет-пользователей по логу посещения сайтов.

Структура файлов:
* `parse_url.ipynb` - скачивание содержимого доменов
* `read_url_data.ipynb` - создание словаря домен &rightarrow; список слов
* `domain_clusterisation.ipynb` - кластеризация доменов при помощи KMeans
* `model_pipeline.ipynb` - построение и обучение Pipeline (domain + category extraction &rightarrow; CountVectorizer &rightarrow; Multioutput LGBM &rightarrow; Top 50)
* `project01_gender-age.py` - запуск модели, предсказание
