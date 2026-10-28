#!/usr/bin/env python3
"""Текстовый отчёт presentation/REPORT.md по экспериментам skeleton_cv (числа из presentation/cv/stats.json)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CV = ROOT / "presentation" / "cv"
D = json.load(open(CV / "stats.json", encoding="utf-8"))
ST, META = D["stats"], D["meta"]
ERR = json.load(open(CV / "errors.json", encoding="utf-8")) if (CV / "errors.json").exists() else {}
JG = json.load(open(CV / "joint_groups.json", encoding="utf-8")) if (CV / "joint_groups.json").exists() else {}


def S(tag, model, metric="auc", p=3):
    try:
        return f"{ST[tag][model][metric]['mean']:.{p}f}"
    except KeyError:
        return "н/д"


def SD(tag, model, metric="auc", p=3):
    try:
        return f"{ST[tag][model][metric]['mean']:.{p}f}±{ST[tag][model][metric]['std']:.{p}f}"
    except KeyError:
        return "н/д"


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def nv(tag):
    return META[tag]["n_videos"] if tag in META else "н/д"


L = []
add = L.append

add("# Эксперименты по улучшению детекции дефекации по скелету собаки\n")
add(f"Всего проведено **{len(ST)} экспериментов** (каждый: 5 фолдов × 3-5 сидов). Источник данных: 166 видео из папок `data/0` "
    "(норма, 106) и `data/1` (дефекация, 60). Код: `scripts/extract_raw.py`, `scripts/skeleton_cv.py`, `scripts/run_queue.py`, "
    "`scripts/analyze_cv.py`; таблица всех запусков: `presentation/cv/table_all.md`; графики: `presentation/cv/*.png`; "
    "презентация: `presentation/cv_experiments.pptx`.\n")

add("## 1. Коротко\n")
add(f"- **Данные задают потолок.** Классификатор, которому доступны только разрешение, fps и длительность ролика, получает "
    f"ROC-AUC **{S('zoo_gbm', 'meta')}**. В датасете 64 клипа с камер парков есть только в классе «норма», а видео 1080p "
    f"составляют 65 из 106 нормальных и 1 из 60 положительных. Любой результат нужно читать на фоне этой планки.")
add(f"- **Модели на честном протоколе** (разбиение по источникам, оценка по видео): gbm {S('zoo_gbm_noasp', 'gbm')}, lstm "
    f"{S('zoo_nets_noasp', 'lstm')}, mamba {S('zoo_nets_noasp', 'mamba')}, tcn {S('zoo_nets_noasp', 'tcn')}, transformer "
    f"{S('zoo_nets_noasp', 'transformer')}, ST-GCN {S('zoo_nets_noasp', 'stgcn')} (только точки). Разница между lstm, tcn, "
    "transformer и Mamba в пределах шума; Mamba и внимание выигрыша не дали.")
add(f"- **Группировка точек не помогает.** Ни одна группа не превзошла полный набор из 20 точек: tcn {S('grp_all', 'tcn')} "
    f"у всех точек против {S('grp_hips_tail', 'tcn')} у таза+хвоста (в пределах шума), {S('grp_only_front', 'tcn')} у передних лап "
    f"(но с ложными срабатываниями на камерах {S('grp_only_front', 'tcn', 'cam_fpr_05', 2)}), {S('grp_only_tail', 'tcn')} у хвоста, "
    f"{S('grp_only_hips', 'tcn')} у таза.")
if "best_nets" in ST:
    add(f"- **Лучшая найденная конфигурация** (окно 30 шагов = 2 с при 15 кадр/с, порог доли кадров с позой 0,7, только "
        f"точки, без confidence, скоростей и аугментаций; {nv('best_nets')} видео): lstm **{SD('best_nets', 'lstm')}**, tcn "
        f"{S('best_nets', 'tcn')}, mamba {S('best_nets', 'mamba')}, ST-GCN {S('best_nets', 'stgcn')}, gbm {S('best_gbm', 'gbm')}; "
        f"контроль по метаданным на тех же видео {S('best_ctrl', 'meta')}.")
if "best_rep" in ST:
    add(f"- **Проверка на новых разбиениях** (другие сиды разбиения, без повторного подбора): lstm {SD('best_rep', 'lstm')}, "
        f"tcn {SD('best_rep', 'tcn')}, mamba {SD('best_rep', 'mamba')}, transformer {SD('best_rep', 'transformer')}.")
add("- **Рабочие метрики скромнее AUC.** При пороге 0,5 у базовых моделей precision около 0,66-0,70 и specificity 0,66-0,73; "
    "при 10% ложных тревог находится 50-65% дефекаций. Калибровка плохая (ECE 0,19-0,30), порог 0,8 из `video-scan/config.json` "
    "нужно подбирать заново.")
add("- **Сдвиг домена.** Если убрать камерные клипы из обучения, ложные срабатывания на камерах растут "
    f"(lstm {S('zoo_nets_noasp', 'lstm', 'cam_fpr_05', 2)} → {S('dom_excl_cam_noasp', 'lstm', 'cam_fpr_05', 2)}, tcn "
    f"{S('zoo_nets_noasp', 'tcn', 'cam_fpr_05', 2)} → {S('dom_excl_cam_noasp', 'tcn', 'cam_fpr_05', 2)}). Положительных камерных "
    "клипов в данных нет вообще, поэтому качество на реальных камерах оценить нельзя.\n")

add("## 2. Данные\n")
add("- 166 видео, около 21,8 минут: класс 0 - 106 видео (13,9 мин), класс 1 - 60 видео (7,9 мин). Медиана клипа 5 с (класс 0) и 6,9 с (класс 1).")
add("- Источники: камеры парков (`cinquez_east`, `cinquez_center`, `salem`, `china_park`) - 64 клипа, все в классе 0; сторонние видео "
    "(YouTube, istockphoto, getty и др.) - остальное. Точных дубликатов нет, но есть нарезки одного ролика (`Trim`…`Trim13` из одной "
    "подборки, `Download…`) и одноимённые ролики в обоих классах (Basset Hound, istockphoto-1498958799, istockphoto-464507358, «Как какает собака»).")
add("- Разрешение: короткая сторона 1080 у 65 из 106 нормальных видео и у 1 из 60 с дефекацией; fps от 15 до 60.")
add(f"- Из 166 видео окна даёт только 100-132 (зависит от параметров): в остальных собака найдена не в каждом кадре, либо ролик слишком короткий.\n")

add("## 3. Протокол\n")
add("1. Один проход YOLO (детектор собаки + pose) по всем видео сохраняет сырой скелет (`extract_raw.py`): рамка, 20 точек, confidence на каждый кадр.")
add("2. Видео приводятся к 15 кадр/с, из них нарезаются окна (по умолчанию 45 шагов = 3 с, шаг 5) с интерполяцией коротких пропусков.")
add("3. Признаки не зависят от разрешения: центр и масштаб тела по оси позвоночника, разворот по направлению собаки; длины звеньев, углы, скорости, confidence.")
add("4. Разбиение по группам источников (`StratifiedGroupKFold`), 5 фолдов, 3-5 сидов; внутри обучения 20% групп идут на валидацию для выбора эпохи.")
add("5. Оценка на уровне видео: среднее трёх лучших окон. Контроли: только метаданные (разрешение, fps, длительность), только форма рамки.")
add("6. Шум: одинаковая конфигурация tcn дала 0,842 и 0,854 в двух запусках; различия меньше ±0,02 AUC не считаем значимыми.\n")

add("## 4. Результаты\n")
add("### 4.1 Модели (T=45, порог 0,9, 100 видео)\n")
rows = []
for tag, model, name in (("zoo_gbm", "meta", "контроль: метаданные"), ("zoo_gbm", "aspect", "контроль: форма рамки"),
                         ("zoo_gbm", "gbm", "gbm (+рамка)"), ("zoo_gbm_noasp", "gbm", "gbm"),
                         ("zoo_nets_noasp", "lstm", "lstm"), ("zoo_nets_noasp", "tcn", "tcn"),
                         ("zoo_nets_noasp", "transformer", "transformer"), ("zoo_nets_noasp", "mamba", "Mamba (упрощённая)"),
                         ("zoo_nets_noasp", "stgcn", "ST-GCN")):
    rows.append([name, SD(tag, model), S(tag, model, "ap"), S(tag, model, "prec05", 2), S(tag, model, "rec05", 2),
                 S(tag, model, "spec05", 2), S(tag, model, "rec_fpr10", 2), S(tag, model, "cam_fpr_05", 2)])
add(table(rows, ["модель", "AUC", "AP", "precision@0,5", "recall@0,5", "specificity@0,5", "recall при FPR 10%", "ложные на камерах"]))
add("\nПримечание: «только точки» (кроме gbm с рамкой); ± - стандартное отклонение по сидам.\n")

add("### 4.2 Группы точек (только точки, без рамки; T=45)\n")
rows = [["все 20", SD("grp_all", "tcn"), S("grp_all", "tcn", "cam_fpr_05", 2), SD("grp_all", "stgcn")]]
for key, name in (("only_spine", "только позвоночник"), ("only_hips", "только таз"), ("only_tail", "только хвост"),
                  ("only_front", "только передние лапы"), ("only_rear", "только задние лапы"), ("spine_hips", "позвоночник+таз"),
                  ("hips_tail", "таз+хвост"), ("spine_tail", "позвоночник+хвост"), ("legs", "все лапы"),
                  ("tail_rear", "хвост+задние лапы"), ("hips_tail_rear", "таз+хвост+задние лапы"), ("drop_spine", "без позвоночника"),
                  ("drop_hips", "без таза"), ("drop_tail", "без хвоста"), ("drop_front", "без передних лап"), ("drop_rear", "без задних лап")):
    rows.append([name, SD(f"grp_{key}", "tcn"), S(f"grp_{key}", "tcn", "cam_fpr_05", 2), SD(f"grp_{key}", "stgcn")])
add(table(rows, ["набор точек", "AUC tcn", "ложные на камерах (tcn)", "AUC ST-GCN"]))
add("\nВывод: полный набор не уступает ни одной группе. В базовом режиме убрать таз или хвост стоит 0,02-0,03 AUC, позвоночник и лапы "
    "добавляют мало. «Только передние лапы» близки по AUC, но с сильно большим числом ложных срабатываний на камерах - "
    "признак того, что эта группа кодирует домен, а не поведение.\n")
if "bgrp_hips_tail" in ST:
    add("\n#### Те же группы в лучшей конфигурации\n")
    rows = [["все 20", SD("best_nets", "lstm"), SD("best_nets", "tcn")]]
    for key, name in (("only_spine", "только позвоночник"), ("only_hips", "только таз"), ("only_tail", "только хвост"),
                      ("only_front", "только передние лапы"), ("only_rear", "только задние лапы"), ("spine_hips", "позвоночник+таз"),
                      ("hips_tail", "таз+хвост"), ("drop_spine", "без позвоночника"), ("drop_hips", "без таза"),
                      ("drop_tail", "без хвоста"), ("drop_front", "без передних лап"), ("drop_rear", "без задних лап")):
        rows.append([name, SD(f"bgrp_{key}", "lstm"), SD(f"bgrp_{key}", "tcn")])
    add(table(rows, ["набор точек", "AUC lstm", "AUC tcn"]))
    add("\nВ лучшем режиме картина по группам иная: без хвоста и без задних лап потерь нет, без таза или позвоночника минус около 0,025, "
        "таз+хвост хуже полного набора примерно на 0,03. Значит, важность отдельных групп нестабильна между режимами, а сигнал распределён по "
        "всему телу. Устойчивый вывод один: ни одна группа не необходима и не достаточна, полный набор не хуже любой группы.\n")

if JG:
    add("\n### 4.3 Внимание по суставам (ST-GCN)\n")
    add(f"Доля внимания: позвоночник {JG['spine'] * 100:.0f}%, передние лапы {JG['front'] * 100:.0f}%, задние лапы {JG['rear'] * 100:.0f}%, "
        f"хвост {JG['tail'] * 100:.0f}%, таз {JG['hips'] * 100:.0f}%. Самые выделенные точки: конец передней лапы (индекс 15), "
        "задняя лапа (17), кончик хвоста (12). Осторожно: ST-GCN самая слабая модель, и внимание может отражать признаки домена.\n")

add("### 4.4 Окно и порог доли кадров с позой\n")
rows = []
for tag, name in (("win_T30", "T=30, порог 0,9"), ("win_T30_mv70", "T=30, порог 0,7"), ("win_T30_mv50", "T=30, порог 0,5"),
                  ("zoo_nets", "T=45, порог 0,9"), ("win_T45_mv70", "T=45, порог 0,7"), ("win_T45_mv50", "T=45, порог 0,5"),
                  ("win_T60", "T=60, порог 0,9"), ("win_T90", "T=90, порог 0,9"), ("win_T90_mv70", "T=90, порог 0,7"),
                  ("win_stride2", "T=45, шаг 2")):
    rows.append([name, nv(tag), S(tag, "lstm"), S(tag, "tcn"), S(tag, "stgcn")])
add(table(rows, ["окно", "видео", "AUC lstm", "AUC tcn", "AUC ST-GCN"]))
add("\nКороткие окна лучше и возвращают в оценку больше видео. У окон 60-90 шагов оценка идёт на 46-85 видео и без камерных клипов, "
    "поэтому сравнивать их с остальными нужно осторожно.\n")

add("### 4.5 Признаки и обучение (только точки, T=45)\n")
rows = []
for tag, name in (("grp_all", "базовый"), ("abl_no_conf", "без confidence"), ("abl_no_vel", "без скоростей"),
                  ("abl_no_aug", "без аугментаций"), ("abl_minimal", "минимум (без conf, vel, aug)"),
                  ("abl_ep20", "20 эпох"), ("abl_ep80", "80 эпох")):
    lstm = S("zoo_nets_noasp", "lstm") if tag == "grp_all" else S(tag, "lstm")
    rows.append([name, lstm, S(tag, "tcn"), S(tag, "stgcn")])
add(table(rows, ["вариант", "AUC lstm", "AUC tcn", "AUC ST-GCN"]))
add("\nАугментации скелета (поворот ±12°, масштаб ±15%, шум, выпадение 10% точек) и confidence на таком малом датасете скорее мешали; "
    "самый простой вариант оказался лучшим. Это стоит перепроверить на независимом тесте: подбор шёл по тем же фолдам.\n")

if "best_nets" in ST:
    add("### 4.6 Лучшая конфигурация\n")
    rows = []
    for tag, model, name in (("best_ctrl", "meta", "контроль: метаданные"), ("best_ctrl", "aspect", "контроль: форма рамки"),
                             ("best_gbm", "gbm", "gbm"), ("best_nets", "lstm", "lstm"), ("best_nets", "tcn", "tcn"),
                             ("best_nets", "mamba", "Mamba"), ("best_nets", "transformer", "transformer"),
                             ("best_nets", "stgcn", "ST-GCN"), ("best_rep", "lstm", "lstm, новые разбиения"),
                             ("best_rep", "tcn", "tcn, новые разбиения"), ("best_rep", "mamba", "Mamba, новые разбиения"),
                             ("best_rep", "transformer", "transformer, новые разбиения")):
        rows.append([name, SD(tag, model), S(tag, model, "ap"), S(tag, model, "prec05", 2), S(tag, model, "rec05", 2),
                     S(tag, model, "spec05", 2), S(tag, model, "rec_fpr10", 2), S(tag, model, "cam_fpr_05", 2)])
    add(table(rows, ["модель", "AUC", "AP", "precision@0,5", "recall@0,5", "specificity@0,5", "recall при FPR 10%", "ложные на камерах"]))
    add(f"\nПараметры: T=30, порог 0,7, без рамки, confidence, скоростей и аугментаций; {nv('best_nets')} видео.\n")

add("### 4.7 Сдвиг домена\n")
rows = []
for a, b, name in (("zoo_nets", "dom_excl_cam", "с рамкой"), ("zoo_nets_noasp", "dom_excl_cam_noasp", "только точки")):
    for m in ("lstm", "tcn", "stgcn"):
        rows.append([f"{m}, {name}", S(a, m), S(b, m), S(a, m, "cam_fpr_05", 2), S(b, m, "cam_fpr_05", 2)])
if "best_excl_cam" in ST:
    for m in ("lstm", "tcn"):
        rows.append([f"{m}, лучшая конфигурация", S("best_nets", m), S("best_excl_cam", m), S("best_nets", m, "cam_fpr_05", 2),
                     S("best_excl_cam", m, "cam_fpr_05", 2)])
add(table(rows, ["модель", "AUC (камеры в обучении)", "AUC (без камер)", "ложные на камерах (с камерами)", "ложные на камерах (без)"]))

add("\n### 4.8 Как собирать оценку видео из оценок окон\n")
rows = []
for m in ("lstm", "tcn", "transformer", "mamba", "stgcn"):
    rows.append([m, S("zoo_nets_noasp", m, "auc_k1"), S("zoo_nets_noasp", m, "auc"), S("zoo_nets_noasp", m, "auc_k5"),
                 S("zoo_nets_noasp", m, "auc_kmean")])
add(table(rows, ["модель", "макс. окна", "среднее 3 лучших", "среднее 5 лучших", "среднее всех"]))
add("\nРазличия небольшие; для lstm, Mamba и transformer среднее по всем окнам немного лучше, для ST-GCN хуже.\n")

add("## 5. Ошибки по роликам\n")
if ERR:
    add(f"При пороге 0,5 из {ERR['n_neg']} нормальных видео {ERR['fp_hi']} получили оценку ≥ 0,5, из {ERR['n_pos']} видео с дефекацией "
        f"{ERR['fn_lo']} получили < 0,5 (среднее по lstm/tcn/ST-GCN, только точки). Полный список: `presentation/cv/errors.md`.\n")
    add("Нормальные видео с самой высокой оценкой дефекации (первые 10):\n")
    add("\n".join(f"- {n}" for n in ERR["fp_names"]))
    add("\nВидео с дефекацией с самой низкой оценкой (первые 10):\n")
    add("\n".join(f"- {n}" for n in ERR["fn_names"]))
    add("\nРолик «Как какает собака» в нормальном классе и «Pooping» в одноимённых парах вызывают подозрение на ошибку разметки "
        "(или это отрезки без самого акта). Их стоит просмотреть глазами и при необходимости перенести.\n")

add("## 6. Сравнение с первым ноутбуком\n")
add("В Colab-прогоне первой версии (`new/PoopDetector.ipynb`, случайное разбиение окон, оценка на окнах) все 20 точек давали F1 0,857, "
    "accuracy 0,886 и ROC-AUC 0,964 на тесте из ~70 окон. На честном протоколе (разбиение по источникам, оценка по видео) "
    f"ближайший аналог даёт AUC {S('zoo_nets_noasp', 'lstm')}-{S('best_nets', 'lstm')}. Прямое сравнение некорректно (разные единицы оценки), но "
    "разрыв показывает, что прежние цифры были завышены утечкой между соседними окнами одного ролика и лёгкостью задачи «камера против стока».\n")

add("## 7. Выводы и рекомендации\n")
add("1. **Группировка точек не нужна как способ улучшить качество.** Полный набор не хуже ни одной группы; важность групп нестабильна между режимами (в базовом важнее таз и хвост, в лучшем - таз и позвоночник), ни одна группа не достаточна.")
add("2. **Архитектура вторична.** Mamba, трансформер и ST-GCN не дали выигрыша над LSTM/TCN/бустингом; на окнах в 2-3 секунды длинный контекст не нужен.")
add("3. **Простота помогает:** короткое окно (30 шагов при 15 кадр/с), без confidence, скоростей и аугментаций.")
add("4. **Главное узкое место - данные:** нужны положительные клипы с камер, разметка момента акта и проверка сомнительных меток.")
add("5. **Для video-scan:** перенести короткое окно и простые признаки, подобрать порог по recall при допустимых ложных тревогах "
    "(не 0,8 по умолчанию) и собирать срабатывания на камерах как кандидатов для ручной проверки.\n")

add("## 8. Ограничения\n")
add("- Мало данных: 100-132 видео с окнами, около 22 минут. Различия меньше ±0,02 AUC нельзя считать значимыми.")
add(f"- Подбор конфигурации шёл по тем же данным, что и оценка ({len(ST)} сравнений): цифры лучшей конфигурации оптимистичны, независимого теста нет.")
add("- Метка стоит на весь ролик, а не на момент акта; в «дефекации» нет камерных клипов.")
add("- Mamba реализована упрощённо на чистом PyTorch, без оптимизированных ядер.")
add("- Оценка идёт на уровне видео, а в production сервис видит поток: ложные тревоги на часах записи не измерены.")
add("- Метрика recall при 10% ложных тревог шумная: в оценке около 70 нормальных видео это 7 видео, поэтому она может заметно "
    "меняться между запусками (например, у Mamba 0,28 и 0,70 в двух прогонах лучшей конфигурации при близком AUC).\n")

add("## 9. Состояние video-scan (на момент отчёта)\n")
add("- Контейнер запущен с моделью `best_lstm_model.pth` (компактная LSTM из ноутбука): добавлены `compact_model.py` и ветка в `detector.py`; "
    "хелсчеки раз в 6 часов (`clip.health_check_interval_sec = 21600`).")
add("- Две из четырёх камер недоступны (`salem_dog_park`: видео недоступно; `china_park`: приватное видео); работают `cinquez_dog_park_east_cam` и `cinquez_dog_park_center_cam`.")
add("- Загрузка на Яндекс Диск не работает: токен отклоняется (401), клипы сохраняются локально в `video-scan/dataset_clips` на диске C: (свободно ~3 ГБ). Нужен новый токен.")

open(ROOT / "presentation" / "REPORT.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("report written", len(L), "blocks")
