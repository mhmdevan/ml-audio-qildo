# Эксперименты. Доли и первые доли такта

Проверка отвечает на три вопроса. Есть ли правильная сетка среди гипотез, даёт ли обучение на полных миксах выигрыш на новых композициях и сохраняется ли качество на живых исполнениях.

## Результаты финального кандидата

Все числа в таблице — средние по записям. F1 измеряется с допуском ±70 мс. Сравнение идёт с предыдущим гибридным трекером `mixed-residual.pt` на одинаковых файлах.

| Набор | Записей | Beat F1 | Downbeat F1 | CMLt | AMLt |
| --- | ---: | ---: | ---: | ---: | ---: |
| GuitarSet | 72 | 60,24 → **65,30%** | 34,75 → **49,04%** | 50,05 → **59,14%** | 57,85 → **64,13%** |
| Groove MIDI, исторический тест | 10 | 79,41 → **89,00%** | 60,86 → **78,61%** | 66,49 → **74,82%** | 89,11 → **94,39%** |
| Groove MIDI, дополнительные записи | 26 | 73,40 → **82,86%** | 55,22 → **67,27%** | 61,22 → **72,48%** | 84,95 → **99,09%** |
| AAM, полные синтетические миксы | 12 | 63,60 → **82,47%** | 30,50 → **77,32%** | 41,09 → **63,32%** | 57,70 → **80,25%** |
| ARTBeaT, задачи на пульс | 24 | 61,86 → **79,10%** | — | 38,04 → **58,60%** | 52,25 → **60,56%** |
| Winterreise, живые исполнения | 24 | — | 7,36 → **15,03%** | — | — |

- На GuitarSet число устойчивых ошибок ×2 и ÷2 уменьшилось с 58 до 17 из 984 тактов. На общих 311 тактах со всеми совпавшими долями число соседних сдвигов первой доли упало с 61 до 4. Пропуск или неточное время первой доли подросли с 4 до 6 треков из 72. Из первых восьми долей совпали 419 из 576 против 391 у baseline.
- На дополнительных записях Groove MIDI число октавных ошибок выросло со 126 до 164 из 1549 тактов. Ошибки первой доли выросли с 2 до 4 из 26. Среди 1009 общих подходящих тактов число соседних сдвигов уменьшилось со 149 до 97. На исторических 10 записях октавных ошибок стало 106 вместо 70 из 317. Разрыв между CMLt и AMLt показывает, что метрический уровень всё ещё часто выбран неверно.
- На AAM акустический режим дал 87,51% Beat F1 и 82,96% Downbeat F1. Финальная CNN без ранжировщика получила 84,69% и 81,82%, включая 80 октавных ошибок вместо 116 у ранжируемого режима. Единую конфигурацию выбрали по валидации трёх доменов. Отдельные варианты выигрывают на отдельных корпусах.
- На 11 фрагментах ARTBeaT с резкой сменой темпа Beat F1 вырос с 73,13% до 83,24%, а CMLt с 40,24% до 55,89%. На трёх фрагментах с постепенным изменением темпа Beat F1 вырос с 65,19% до 79,03%. Точные эталоны баров в этой проверке отсутствуют.
- На Winterreise акустический режим набрал 18,59% Downbeat F1 и правильно поставил первую размеченную сильную долю на 6 из 24 песен. У выбранной конфигурации совпали 2 из 24. Даже на 10 песнях в 3/4 или 4/4 её Downbeat F1 составляет 12,57%. Перенос на живой вокал с фортепиано остаётся слабым.

## Выбор конфигурации

| Модель и пул гипотез | Средняя целевая функция по доменам |
| --- | ---: |
| Прежняя CNN, ранжирование с учётом октавного сбоя | 0,60571 |
| Прежняя CNN, акустические кандидаты в общем пуле | 0,62719 |
| Иерархическая CNN, GuitarSet и Groove MIDI | 0,62555 |
| Иерархическая CNN с AAM, 4-я эпоха | 0,62278 |
| Иерархическая CNN с AAM, 7-я эпоха | 0,64836 |
| Иерархическая CNN с AAM, 12-я эпоха | **0,66651** |

Финальная CNN обучалась 14 эпох с seed 2026. Минимум validation loss, 0,89967, получен на 12-й эпохе. Ранжировщик калибровался на 112 записях. SHA-256 опубликованных весов `6f718846cc322381a67a65aa2a89cebb7adfeddd4ee6b4d386bdaf6da56e29d5`. Для `quality-ranker.json` SHA-256 `a6dc60d129d872a5858e37d9193dc43de9e9462cecb8bf43405f63f0614f609c`.

## Офлайн-инференс и готовность

На CPU Apple M4 в Docker Linux ARM64 с отключённой сетью проверены три четырёхминутных стерео-WAV. Худшее время полного вызова CLI составило **10,20 с**, пик RSS **320 МиБ**. В акустическом режиме худшее время составило 5,06 с при пике 279 МиБ. Во всех прогонах проверены границы времён, строгий порядок долей и поля JSON. Два WAV синтетические, третий — фрагмент *Inspired*, использованный для проверки времени и формата. [Kevin MacLeod, Inspired, CC BY 4.0](https://incompetech.com/music/royalty-free/index.html?isrc=USUAN1600022)

Инженерная проверка подтверждает запуск без сети, формат JSON и время до 120 секунд. Оценка качества подтверждает прирост средних F1 и CMLt на доступных сопоставимых выборках. Ошибки первой доли и метрического уровня требуют дальнейшей работы по указанным срезам. Закрытый набор Qildo и оценка слушателями остаются условиями продуктовой приёмки. GuitarSet имеет три тестовые музыкальные основы, Groove MIDI — одного тестового барабанщика, а Winterreise — одну запись цикла. Для вывода о переносе на каталог нужны полные записи разных исполнителей и жанров.

## Что измеряли до изменений

На 72 валидационных фрагментах GuitarSet эвристический выбор сетки давал 64,06% Beat F1 и 55,27% CMLt. Если выбрать из старых гипотез по эталону, достижимы около 73,18% и 70,12% соответственно. На 28 валидационных фрагментах Groove MIDI потолок достиг 84,4% Beat F1 и 79,03% CMLt. Такой выбор доступен только при диагностике, когда известна эталонная сетка.

Первый ранжировщик по F1 и CMLt улучшил Downbeat F1 на историческом тесте GuitarSet с 34,75% до 42,99%. Случаев ×2 и ÷2 стало больше, 58 → 122 из 984 тактов. При добавлении явного штрафа за октавную ошибку получились 40,36% Downbeat F1, 56,87% CMLt и 31 ошибочный такт. На новых 26 записях Groove MIDI у этой версии ошибки ×2 и ÷2 выросли с 126 до 147 тактов. Поэтому отслеживаем эти ошибки по каждому набору даже при росте всех F-мер.

## Протокол и данные

Модели и ранжировщики обучаются с нуля. Ранжировщик видит только `validation`. Для диагностики ранжировщика каждую группу по очереди исключали из подгонки его весов. Чекпойнт CNN отбирается по общему validation loss. Таким образом, контроль по группам оценивает перенос ранжировщика, а независимая оценка всего пайплайна проводится на `test`.

Конфигурации сравниваются по среднему из трёх доменов, GuitarSet, Groove MIDI и AAM. Целевая функция на шкале от 0 до 1 добавляет `0.40 × Beat F1 + 0.30 × Downbeat F1 + 0.25 × CMLt + 0.05 × точность первой доли` и вычитает `0.25 × долю тактов в сериях ошибок ×2 или ÷2`. Целевая функция нужна для отбора конфигураций. В отчёте о тестах метрики приведены по отдельности.

| Данные | Разбиение | Как интерпретировать |
| --- | --- | --- |
| GuitarSet v1.1.0 | 216 / 72 / 72, по музыкальной основе | Три музыкальные основы в тесте. Тот же тест применялся к предыдущим версиям. |
| Groove MIDI v1.0.0 | 128 / 28 / 10, по барабанщику | Десять записей исторической проверки. Дополнительные 26 записей того же отложенного барабанщика отсутствуют в ранних манифестах и подгонке. |
| AAM v1.1.0 | 36 / 12 / 12, по композиции | Тестовые миксы длиной 125–172 с. Аудио синтезировано из партитур, эталонная сетка взята из ARFF. |
| Schubert Winterreise v2.1, HU33 | 24 полных песни для проверки переноса | Историческая запись голоса и фортепиано. Оцениваются первые доли по ручным границам тактов. В цикле встречаются сложные и двудольные размеры. |
| ARTBeaT | 24 коротких примера для проверки пульса | Резкие и плавные смены темпа, синкопы, неоднозначные акценты, нечётные размеры. CSV содержит времена пульса без номера доли. |

- GuitarSet и Groove MIDI лицензированы по CC BY 4.0. [GuitarSet](https://zenodo.org/records/3371780), [Groove MIDI Dataset](https://magenta.withgoogle.com/datasets/groove)
- AAM лицензирован по CC BY 4.0. Зафиксированные списки композиций находятся в `benchmarks/aam-train.csv` и `benchmarks/aam-full.csv`. Более широкий список нужен для воспроизводимого отделения теста, в нём 24 дополнительных обучающих файла. [AAM](https://zenodo.org/records/5794629), [описание генерации](https://link.springer.com/article/10.1186/s13636-023-00278-7)
- Запись HU33 опубликована с Public Domain Mark 1.0, файл `03_ExtraMaterial/license_HU33.txt`. Разметка SWD лицензирована по CC BY 3.0. Использовано оригинальное исполнение первой песни из `HU33_originalShort_musopen`. В метрику входят только целочисленные номера тактов, а разметку внутритактовых долей для этого теста не реконструируем. [Schubert Winterreise Dataset](https://zenodo.org/records/10839767)
- ARTBeaT Рико Розенбуша лицензирован по CC BY 4.0. Использованы `Annotations/gt_csv`, первый калибровочный метроном исключён. [ARTBeaT](https://www.audiolabs-erlangen.com/resources/MIR/2024-ARTBeaT)

## Воспроизведение

Подготовьте GuitarSet по README. Для Groove MIDI скачайте официальный архив MIDI и распакуйте архив аудио. Для AAM нужны `0001-1000-annotations-v1.1.0.zip` и файлы `<id>_mix.flac` из `0001-1000-audio-mixes.zip`. Преобразуйте миксы из списка `benchmarks/aam-full.csv` в PCM WAV командой `ffmpeg -i <id>_mix.flac -ar 22000 -c:a pcm_s16le <id>_mix.wav`. Сложите WAV в `data/aam-source`.

```bash
python prepare_groove.py data/downloads/groove-v1.0.0-midionly.zip data/groove-source data/groove \
  --selection benchmarks/groove-166.csv
mv data/groove/manifest.json data/groove/snapshot-c.json
python prepare_groove.py data/downloads/groove-v1.0.0-midionly.zip data/groove-source data/groove-fresh \
  --selection benchmarks/groove-fresh.csv
python -m scripts.prepare_aam data/downloads/0001-1000-annotations-v1.1.0.zip data/aam-source \
  benchmarks/aam-train.csv data/aam
python -m scripts.prepare_aam data/downloads/0001-1000-annotations-v1.1.0.zip data/aam-source \
  benchmarks/aam-full.csv data/aam-full
```

```bash
python train.py data/guitarset/manifest.json runs/quality.pt \
  --additional-manifest data/groove/snapshot-c.json \
  --additional-manifest data/aam/manifest.json \
  --architecture hierarchical --augment --tempo-warp 0.12 \
  --channels 24 --batch-size 20 --epochs 14 --seed 2026
python -m scripts.fit_grid_ranker runs/quality.pt runs/quality-ranker.json \
  data/guitarset/manifest.json data/groove/snapshot-c.json data/aam/manifest.json \
  --include-acoustic
```

```bash
compare_run() {
  manifest="$1"
  label="$2"
  python evaluate.py manifest "$manifest" "runs/$label-baseline.json" \
    --split test --checkpoint checkpoints/mixed-residual.pt \
    --save-predictions "runs/$label-baseline" &&
  python evaluate.py manifest "$manifest" "runs/$label-quality.json" \
    --split test --checkpoint checkpoints/quality.pt --ranker checkpoints/quality-ranker.json \
    --compare "runs/$label-baseline" --save-predictions "runs/$label-quality"
}

compare_run data/guitarset/manifest.json guitar
compare_run data/groove/snapshot-c.json groove
compare_run data/groove-fresh/manifest.json groove-fresh
compare_run data/aam-full/manifest.json aam
```

Для проверки Schubert Winterreise соберите в `data/winterreise` по одному WAV и CSV `Schubert_D911-NN_HU33` из `01_RawData/audio_wav` и `02_Annotations/ann_audio_measure`. Для номера 01 положите пару из `03_ExtraMaterial/HU33_originalShort_musopen`. Туда же положите `ann_score_overview.csv` из `03_ExtraMaterial`.

```bash
python -m scripts.evaluate_winterreise data/winterreise runs/winter-acoustic.json
python -m scripts.evaluate_winterreise data/winterreise runs/winter-baseline.json \
  --checkpoint checkpoints/mixed-residual.pt
python -m scripts.evaluate_winterreise data/winterreise runs/winter-quality.json \
  --checkpoint checkpoints/quality.pt --ranker checkpoints/quality-ranker.json
python -m scripts.evaluate_artbeat data/downloads/ARTBeaT_Dataset.zip runs/artbeat-baseline.json \
  --checkpoint checkpoints/mixed-residual.pt
python -m scripts.evaluate_artbeat data/downloads/ARTBeaT_Dataset.zip runs/artbeat-quality.json \
  --checkpoint checkpoints/quality.pt --ranker checkpoints/quality-ranker.json
```
