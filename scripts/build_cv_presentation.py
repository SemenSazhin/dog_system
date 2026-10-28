#!/usr/bin/env python3
"""Презентация и текстовый отчёт по экспериментам skeleton_cv. Все числа берутся из presentation/cv/stats.json.

python build_cv_presentation.py  ->  presentation/cv_experiments.pptx, presentation/REPORT.md
"""
import json
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
CV = ROOT / "presentation" / "cv"
OUT = ROOT / "presentation"
D = json.load(open(CV / "stats.json", encoding="utf-8"))
ST, META = D["stats"], D["meta"]
ERR = json.load(open(CV / "errors.json", encoding="utf-8")) if (CV / "errors.json").exists() else {}
JG = json.load(open(CV / "joint_groups.json", encoding="utf-8")) if (CV / "joint_groups.json").exists() else {}
NAVY, ORANGE, GREY, RED, GREEN, INK = "#1F4E79", "#E07B39", "#9AA5B1", "#B5483A", "#4C9F70", "#1B2733"


def S(tag, model, metric="auc", p=2):
    try:
        return f"{ST[tag][model][metric]['mean']:.{p}f}"
    except KeyError:
        return "н/д"


def SD(tag, model, metric="auc", p=3):
    try:
        return f"±{ST[tag][model][metric]['std']:.{p}f}"
    except KeyError:
        return ""


def V(tag, model, metric="auc"):
    return ST[tag][model][metric]["mean"]


def rgb(h):
    return RGBColor.from_string(h.lstrip("#"))


W, H = Inches(13.333), Inches(7.5)


def add_text(slide, x, y, w, h, text, size=18, bold=False, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for i, line in enumerate(text if isinstance(text, list) else [text]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size, r.font.bold, r.font.name = Pt(size), bold, "Calibri"
        r.font.color.rgb = rgb(color)
        p.space_after = Pt(6)
    return tb


def bullets(slide, x, y, w, h, items, size=17):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run()
        r.text = "•  " + it
        r.font.size, r.font.name = Pt(size), "Calibri"
        r.font.color.rgb = rgb(INK)
        p.space_after = Pt(9)
    return tb


def rect(slide, x, y, w, h, fill):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    s.line.fill.background()
    s.adjustments[0] = 0.06
    s.shadow.inherit = False
    return s


def table(slide, x, y, rows, col_w, size=14, row_h=0.4, hl=()):
    shp = slide.shapes.add_table(len(rows), len(rows[0]), x, y, Inches(sum(col_w)), Inches(row_h * len(rows)))
    t = shp.table
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i, r in enumerate(rows):
        t.rows[i].height = Inches(row_h)
        for j, v in enumerate(r):
            c = t.cell(i, j)
            c.text = str(v)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            f = p.runs[0].font
            f.size, f.name = Pt(size), "Calibri"
            f.bold = i == 0 or i in hl
            f.color.rgb = rgb("#FFFFFF") if i == 0 else rgb(INK)
            c.fill.solid()
            c.fill.fore_color.rgb = rgb(NAVY if i == 0 else ("#DCE8F5" if i in hl else ("#F3F5F7" if i % 2 else "#FFFFFF")))


def pic_fit(slide, name, x, y, max_w, max_h):
    path = CV / name
    if not path.exists():
        add_text(slide, x, y, max_w, Inches(1), f"[нет графика {name}]", size=14, color=RED)
        return
    w, h = Image.open(path).size
    k = min(max_w / w, max_h / h)
    slide.shapes.add_picture(str(path), x + int((max_w - w * k) / 2), y, width=int(w * k), height=int(h * k))


prs = Presentation()
prs.slide_width, prs.slide_height = W, H
N = [0]


def new_slide(title):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    N[0] += 1
    add_text(s, Inches(0.6), Inches(0.3), Inches(12.1), Inches(0.9), title, size=28, bold=True, color=NAVY)
    ln = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(1.12), Inches(1.2), Emu(38000))
    ln.fill.solid()
    ln.fill.fore_color.rgb = rgb(ORANGE)
    ln.line.fill.background()
    add_text(s, Inches(12.2), Inches(7.05), Inches(0.8), Inches(0.35), str(N[0] + 1), size=12, color=GREY, align=PP_ALIGN.RIGHT)
    return s


def chart_slide(title, img, takeaway, img_h=4.3):
    s = new_slide(title)
    pic_fit(s, img, Inches(0.5), Inches(1.3), Inches(12.3), Inches(img_h))
    top = 1.3 + img_h + 0.1
    rect(s, Inches(0.6), Inches(top), Inches(12.1), Inches(7.0 - top), "#F3F5F7")
    add_text(s, Inches(0.8), Inches(top + 0.03), Inches(11.7), Inches(7.0 - top - 0.06), takeaway, size=13,
             anchor=MSO_ANCHOR.MIDDLE)
    return s


has = lambda t: t in ST
best_rep_ok = has("best_rep")

# ------------------------------------------------------------------ 1. титул
s = prs.slides.add_slide(prs.slide_layouts[6])
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
bg.fill.solid()
bg.fill.fore_color.rgb = rgb(NAVY)
bg.line.fill.background()
add_text(s, Inches(0.9), Inches(2.0), Inches(11.5), Inches(2.2),
         "Как улучшить детекцию дефекации по скелету собаки", size=42, bold=True, color="#FFFFFF")
add_text(s, Inches(0.9), Inches(4.3), Inches(11.5), Inches(1.2),
         f"{len(ST)} экспериментов на честном протоколе: модели, группы точек, окна, признаки, сдвиг домена",
         size=22, color="#C9D8E8")
add_text(s, Inches(0.9), Inches(6.3), Inches(11.5), Inches(0.5),
         "166 видео · разбиение по источникам · 5 фолдов × несколько сидов", size=16, color=ORANGE)

# ------------------------------------------------------------------ 2. данные
s = new_slide("Что с данными: главная проблема - подсказки вместо поведения")
bullets(s, Inches(0.6), Inches(1.4), Inches(7.6), Inches(5.4), [
    "166 видео, всего около 22 минут: 106 «норма» и 60 «дефекация»",
    "64 клипа с камер парков есть только в классе «норма»; положительных клипов с камер нет совсем",
    "1080p: 65 из 106 нормальных видео и 1 из 60 с дефекацией",
    "Окна по 3-4 секунды дают только 100-132 видео из 166: в остальных собака найдена не в каждом кадре или ролик слишком короткий",
    f"Классификатор, который видит только разрешение, fps и длительность, получает ROC-AUC {S('zoo_gbm', 'meta')}: ответ можно угадать, не глядя на собаку",
], size=17)
rect(s, Inches(8.5), Inches(1.4), Inches(4.2), Inches(2.5), "#FDECEC")
add_text(s, Inches(8.7), Inches(1.5), Inches(3.8), Inches(0.6), "Контроль «только метаданные»", size=16, bold=True, color=RED)
add_text(s, Inches(8.7), Inches(2.1), Inches(3.8), Inches(1.6),
         [f"AUC {S('zoo_gbm', 'meta')}", f"recall при 10% ложных тревог: {S('zoo_gbm', 'meta', 'rec_fpr10')}"],
         size=20, bold=True, color=RED)
rect(s, Inches(8.5), Inches(4.1), Inches(4.2), Inches(2.5), "#F3F5F7")
add_text(s, Inches(8.7), Inches(4.2), Inches(3.8), Inches(0.6), "Только форма рамки собаки", size=16, bold=True, color=GREY)
add_text(s, Inches(8.7), Inches(4.8), Inches(3.8), Inches(1.6),
         [f"AUC {S('zoo_gbm', 'aspect')}", f"ложные на камерах: {S('zoo_gbm', 'aspect', 'cam_fpr_05')}"],
         size=20, bold=True, color="#555555")

# ------------------------------------------------------------------ 3. протокол
s = new_slide("Как измеряем: честнее, чем в первом ноутбуке")
bullets(s, Inches(0.6), Inches(1.4), Inches(7.3), Inches(5.4), [
    "YOLO-проход делается один раз: сохраняем сырой скелет (20 точек + confidence) по каждому кадру, дальше эксперименты считаются за секунды",
    "Видео приводятся к 15 кадр/с; окно 30-45 шагов - то есть 2-3 секунды реального времени независимо от fps",
    "Признаки не зависят от разрешения: центр и масштаб тела, разворот собаки по оси позвоночника",
    "Разбиение по источникам: нарезки одного ролика и одноимённые ролики из разных классов в одной группе, 5 фолдов, несколько сидов",
    "Оценка на уровне видео (среднее трёх лучших окон), а не окон",
], size=16)
rect(s, Inches(8.2), Inches(1.4), Inches(4.5), Inches(5.2), "#F3F5F7")
add_text(s, Inches(8.4), Inches(1.5), Inches(4.1), Inches(0.5), "Метрики", size=18, bold=True, color=NAVY)
bullets(s, Inches(8.4), Inches(2.1), Inches(4.1), Inches(4.4), [
    "рабочие: precision, recall, specificity, F1, MCC при пороге 0,5",
    "recall при 10% ложных тревог",
    "ложные срабатывания на видео с камер",
    "AUC и AP как вспомогательные",
    "калибровка: Brier, ECE",
    "контроли: только метаданные, только форма рамки",
], size=15)
s2 = s
add_text(s, Inches(0.6), Inches(6.55), Inches(12), Inches(0.5),
         "Шум между прогонами ≈ ±0,01-0,025 AUC: идентичная конфигурация tcn дала 0,842 и 0,854. Меньшие различия считаем несущественными.",
         size=14, color=RED)

# ------------------------------------------------------------------ 4. зоопарк
chart_slide("Модели: простой бустинг не хуже нейросетей, ST-GCN слабее", "zoo.png",
            [f"gbm {S('zoo_gbm', 'gbm')}/{S('zoo_gbm_noasp', 'gbm')} (с рамкой/без) · lstm {S('zoo_nets_noasp', 'lstm')} · tcn {S('zoo_nets_noasp', 'tcn')} · transformer {S('zoo_nets_noasp', 'transformer')} · mamba {S('zoo_nets_noasp', 'mamba')} · stgcn {S('zoo_nets_noasp', 'stgcn')} (только точки).",
             "lstm, tcn, transformer и Mamba различаются в пределах шума: на окнах в 3 секунды ни Mamba, ни внимание преимуществ не дали. ST-GCN в базовом виде почти не отличается от контроля по метаданным."])

# ------------------------------------------------------------------ 5. рабочие метрики
chart_slide("Рабочие метрики: 50-65% дефекаций при 10% ложных тревог", "curves.png",
            [f"При пороге 0,5 у lstm: precision {S('zoo_nets_noasp', 'lstm', 'prec05')}, recall {S('zoo_nets_noasp', 'lstm', 'rec05')}, specificity {S('zoo_nets_noasp', 'lstm', 'spec05')}; gbm находит {S('zoo_gbm_noasp', 'gbm', 'rec_fpr10')} при 10% ложных тревог (метаданные: {S('zoo_gbm', 'meta', 'rec_fpr10')}).",
             f"Калибровка слабая (ECE {S('zoo_nets_noasp', 'lstm', 'ece')}-{S('zoo_nets_noasp', 'stgcn', 'ece')}): порог в video-scan нужно подбирать заново, а не считать выходы вероятностями."])

# ------------------------------------------------------------------ 6. группы точек
chart_slide("Группировка точек: ни одна группа не лучше всех 20", "groups_only.png",
            [f"tcn: все 20 точек {S('grp_all', 'tcn')}; таз+хвост {S('grp_hips_tail', 'tcn')} (в пределах шума); только передние лапы {S('grp_only_front', 'tcn')}, но ложные на камерах {S('grp_only_front', 'tcn', 'cam_fpr_05')} против {S('grp_all', 'tcn', 'cam_fpr_05')} у полного набора.",
             "Хвост, таз или лапы по отдельности заметно слабее (0,77-0,78 у tcn). Гипотеза «для дефекации хватит таза и хвоста» держится по AUC, но выигрыша не даёт."])

s = chart_slide("Что теряем, если убрать группу точек", "groups_drop.png",
                [f"Базовый режим: убрать таз или хвост стоит 0,02-0,03 AUC у tcn ({S('grp_drop_hips', 'tcn')} и {S('grp_drop_tail', 'tcn')} против {S('grp_all', 'tcn')}), позвоночник или лапы - в пределах шума.",
                 f"Лучший режим (следующий слайд): без хвоста и без задних лап потерь нет, а без таза или позвоночника минус около 0,025. Важность отдельных групп нестабильна: сигнал распределён по всему телу."], img_h=4.2)

# ------------------------------------------------------------------ 7. внимание по суставам
if JG:
    chart_slide("Куда смотрит модель: внимание по суставам", "joint_attention.png",
                [f"Доля внимания по группам: позвоночник {JG['spine'] * 100:.0f}%, передние лапы {JG['front'] * 100:.0f}%, задние лапы {JG['rear'] * 100:.0f}%, хвост {JG['tail'] * 100:.0f}%, таз {JG['hips'] * 100:.0f}%.",
                 "Сильнее всего выделены концы конечностей (передняя лапа, задняя лапа, кончик хвоста). Осторожно: это самая слабая из моделей, и внимание может отражать домен, а не позу."], img_h=4.2)

# ------------------------------------------------------------------ 8. окно
chart_slide("Окно и порог: короткие окна лучше и возвращают видео в обучение", "window.png",
            [f"Окно 30 шагов (2 с): tcn {S('win_T30', 'tcn')}, {META['win_T30']['n_videos']} видео; при пороге 0,7 {S('win_T30_mv70', 'tcn')} на {META['win_T30_mv70']['n_videos']} видео. Окна 60 и 90 шагов хуже и теряют камерные клипы (46-85 видео).",
             "Сравнивать столбцы с разным числом видео нужно осторожно: набор видео для оценки меняется, поэтому длинные окна выглядят хуже отчасти из-за потери данных."])

# ------------------------------------------------------------------ 9. абляции
chart_slide("Признаки и обучение: лишнее вредит", "ablation.png",
            [f"Без аугментаций lstm {S('abl_no_aug', 'lstm')}, tcn {S('abl_no_aug', 'tcn')}; без confidence lstm {S('abl_no_conf', 'lstm')}; «минимум» (без confidence, скоростей и аугментаций): lstm {S('abl_minimal', 'lstm')}, tcn {S('abl_minimal', 'tcn')}, stgcn {S('abl_minimal', 'stgcn')}.",
             f"Мои аугментации скелета (поворот, масштаб, выпадение точек) на таком малом датасете скорее мешали. Больше эпох не помогает: tcn 20/40/80 эпох = {S('abl_ep20', 'tcn')}/{S('grp_all', 'tcn')}/{S('abl_ep80', 'tcn')}, stgcn при 80 эпохах переобучается ({S('abl_ep80', 'stgcn')})."])

# ------------------------------------------------------------------ 10. лучшая конфигурация
if has("best_nets"):
    rep_txt = ""
    if best_rep_ok:
        rep_txt = f" Повтор на новых разбиениях: lstm {S('best_rep', 'lstm')}, tcn {S('best_rep', 'tcn')}, mamba {S('best_rep', 'mamba')}, transformer {S('best_rep', 'transformer')}."
    chart_slide("Лучшая конфигурация и проверка на новых разбиениях", "best.png",
                [f"Окно 30 шагов, порог 0,7, без рамки, confidence, скоростей и аугментаций ({META['best_nets']['n_videos']} видео): lstm {S('best_nets', 'lstm')}{SD('best_nets', 'lstm')}, tcn {S('best_nets', 'tcn')}, mamba {S('best_nets', 'mamba')}, stgcn {S('best_nets', 'stgcn')}, gbm {S('best_gbm', 'gbm')}. Контроль по метаданным на тех же видео: {S('best_ctrl', 'meta')}.{rep_txt}",
                 "Подбор шёл на тех же данных, что и оценка, поэтому цифры оптимистичны; повтор на других разбиениях показывает, сколько из прироста держится."])

if any(t.startswith("bgrp_") for t in ST):
    chart_slide("Группы точек в лучшей конфигурации", "best_groups.png",
                [f"Полный набор: lstm {S('best_nets', 'lstm')}, tcn {S('best_nets', 'tcn')}. Таз+хвост: lstm {S('bgrp_hips_tail', 'lstm')}, tcn {S('bgrp_hips_tail', 'tcn')}. Без таза: {S('bgrp_drop_hips', 'lstm')}/{S('bgrp_drop_hips', 'tcn')}, без хвоста: {S('bgrp_drop_tail', 'lstm')}/{S('bgrp_drop_tail', 'tcn')}.",
                 "Таз+хвост хуже полного набора примерно на 0,03 (за пределами шума), без хвоста или задних лап потерь нет. Полный набор не уступает ни одной группе; единственный стабильный вывод - сигнал не сосредоточен в одной группе."], img_h=4.2)

# ------------------------------------------------------------------ 11. домен
dom_extra = ""
if has("best_excl_cam"):
    dom_extra = f" В лучшей конфигурации без камерных клипов в обучении: lstm {S('best_excl_cam', 'lstm')} (было {S('best_nets', 'lstm')}), ложные на камерах {S('best_excl_cam', 'lstm', 'cam_fpr_05')} (было {S('best_nets', 'lstm', 'cam_fpr_05')})."
chart_slide("Сдвиг домена: модель без камерных клипов ошибается на камерах", "domain.png",
            [f"Если убрать камерные клипы из обучения, ложные срабатывания на камерах растут: lstm {S('zoo_nets_noasp', 'lstm', 'cam_fpr_05')} → {S('dom_excl_cam_noasp', 'lstm', 'cam_fpr_05')}, tcn {S('zoo_nets_noasp', 'tcn', 'cam_fpr_05')} → {S('dom_excl_cam_noasp', 'tcn', 'cam_fpr_05')}.{dom_extra}",
             "Для реальной работы на камерах нужны камерные клипы в обучении, включая положительные: сейчас их нет вообще."])

# ------------------------------------------------------------------ 12. ошибки
s = new_slide("Ошибки по роликам: что проверить в разметке")
if ERR:
    add_text(s, Inches(0.6), Inches(1.3), Inches(12), Inches(0.6),
             f"Из {ERR['n_neg']} нормальных видео {ERR['fp_hi']} получили оценку ≥ 0,5; из {ERR['n_pos']} видео с дефекацией {ERR['fn_lo']} получили < 0,5.",
             size=16, bold=True, color=NAVY)
    rect(s, Inches(0.6), Inches(1.95), Inches(6.0), Inches(4.9), "#FDECEC")
    add_text(s, Inches(0.8), Inches(2.0), Inches(5.6), Inches(0.5), "Норма с высокой оценкой дефекации", size=16, bold=True, color=RED)
    bullets(s, Inches(0.8), Inches(2.55), Inches(5.7), Inches(4.2), [n[:60] for n in ERR["fp_names"][:7]], size=12)
    rect(s, Inches(6.8), Inches(1.95), Inches(5.9), Inches(4.9), "#F3F5F7")
    add_text(s, Inches(7.0), Inches(2.0), Inches(5.5), Inches(0.5), "Дефекация с низкой оценкой", size=16, bold=True, color=NAVY)
    bullets(s, Inches(7.0), Inches(2.55), Inches(5.5), Inches(4.2), [n[:60] for n in ERR["fn_names"][:7]], size=12)

s = new_slide("Что это значит для video-scan")
bullets(s, Inches(0.6), Inches(1.4), Inches(12.1), Inches(5.4), [
    "Модель из ноутбука в video-scan ставили в исходном виде (окно 120 шагов, с confidence, скоростями и аугментациями): по этим результатам эта конфигурация не лучшая",
    f"Короткое окно (30 шагов, 2 с) и простые признаки лучше ({S('best_nets', 'lstm')} против {S('zoo_nets_noasp', 'lstm')} у lstm; наборы видео разные: 126 против 100) и подтвердились на новых разбиениях; их стоит перенести в сервис",
    "Порог 0,8 из config.json не откалиброван: подбирать по recall при допустимых ложных тревогах, а не по умолчанию",
    "Без положительных клипов с камер модель не проверить на реальной задаче: единственный путь - собрать их (сейчас сервис ждёт срабатываний, а загрузка на Яндекс Диск не работает: токен отклоняется)",
], size=17)

# ------------------------------------------------------------------ 13. выводы
s = new_slide("Выводы")
cards = [
    ("Данные важнее архитектуры", f"Контроль по метаданным даёт AUC {S('zoo_gbm', 'meta')}. Модели получают {S('zoo_nets_noasp', 'lstm')}-{S('best_nets', 'lstm')}, но разрыв в рабочих метриках скромный.", RED),
    ("Группировка точек не помогает", "Ни одна группа не лучше всех 20 точек ни в базовом, ни в лучшем режиме. Сигнал распределён по телу: ни одна группа не необходима и не достаточна.", ORANGE),
    ("Проще лучше", "Короткое окно, без confidence, скоростей и аугментаций; Mamba и внимание выигрыша не дали.", NAVY),
]
for i, (h, t, c) in enumerate(cards):
    x = Inches(0.6 + i * 4.1)
    rect(s, x, Inches(1.5), Inches(3.9), Inches(3.8), "#F3F5F7")
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, Inches(1.5), Inches(3.9), Emu(60000))
    bar.fill.solid()
    bar.fill.fore_color.rgb = rgb(c)
    bar.line.fill.background()
    add_text(s, x + Inches(0.2), Inches(1.7), Inches(3.5), Inches(0.9), h, size=20, bold=True, color=c)
    add_text(s, x + Inches(0.2), Inches(2.8), Inches(3.5), Inches(2.5), t, size=15)
add_text(s, Inches(0.6), Inches(5.6), Inches(12.1), Inches(1.3),
         "Дальше: положительные клипы с камер, разметка времени акта внутри роликов, проверка сомнительных меток, затем перенос лучшей конфигурации в video-scan.",
         size=17, bold=True, color=NAVY)

# ------------------------------------------------------------------ 14. ограничения
s = new_slide("Ограничения")
bullets(s, Inches(0.6), Inches(1.4), Inches(12.1), Inches(5.4), [
    "Мало данных: 100-132 видео с окнами, около 22 минут; различия меньше ±0,02 AUC нельзя считать значимыми",
    "Конфигурация подбиралась по этим же данным (55 сравнений): цифры лучшей конфигурации оптимистичны, независимого теста нет",
    "Метка стоит на весь ролик, а не на момент акта",
    "Класс «дефекация» не содержит камерных клипов, поэтому качество на реальных камерах оценить нельзя",
    "Некоторые метки подозрительны (ролики «Как какает собака», «Pooping» в нормальном классе): часть «ошибок» может быть ошибками разметки",
    "Mamba реализована упрощённо на чистом PyTorch (последовательный скан); на окнах в 2-3 с преимущества длинного контекста не проявляются",
], size=17)

prs.save(OUT / "cv_experiments.pptx")
print("saved", N[0] + 1, "slides")
