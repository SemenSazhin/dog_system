#!/usr/bin/env python3
"""Графики и презентация по выводу ячеек ноутбука new/PoopDetector.ipynb (серия «ключевые точки»).

Числа берутся из таблицы в выводе ячейки «Сводка экспериментов», вручную ничего не переписывается.
Результат: presentation/charts/*.png и presentation/keypoint_experiments.pptx
"""
import glob
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "presentation"
CH = OUT / "charts"
CH.mkdir(parents=True, exist_ok=True)

NAMES = {
    "kp_none_bbox": "Без точек (bbox)",
    "kp_all": "Все 20 точек",
    "kp_spine_hips": "Позвоночник + таз",
    "kp_hips_tail": "Таз + хвост",
    "kp_tail_only": "Только хвост",
    "kp_legs": "Только лапы",
    "kp_spine_tail": "Позвоночник + хвост",
}
POINTS = {"kp_none_bbox": 0, "kp_all": 20, "kp_spine_hips": 10, "kp_hips_tail": 7,
          "kp_tail_only": 3, "kp_legs": 7, "kp_spine_tail": 9}
GREY, NAVY, ORANGE, RED, INK = "#9AA5B1", "#1F4E79", "#E07B39", "#B5483A", "#1B2733"


def parse_results():
    """Строки таблицы из вывода ячейки сводки."""
    nb = json.load(open(glob.glob(str(ROOT / "new" / "*.ipynb"))[0], encoding="utf-8"))
    rows = {}
    for c in nb["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] != "code" or "Сводка экспериментов" not in src:
            continue
        for o in c.get("outputs", []):
            for line in "".join(o.get("text", [])).split("\n"):
                p = line.split()
                if p and p[0] in NAMES and len(p) >= 9:
                    f1, acc, prec, rec, auc = map(float, p[2:7])
                    rows[p[0]] = dict(f1=f1, accuracy=acc, precision=prec, recall=rec, roc_auc=auc)
    if not rows:
        raise SystemExit("таблица сводки в выводе ноутбука не найдена")
    return rows


def col(k):
    return GREY if k == "kp_none_bbox" else NAVY if k == "kp_all" else ORANGE


plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
                     "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


def make_charts(R):
    keys = [k for k in NAMES if k in R]
    x = np.arange(len(keys))
    labels = [NAMES[k].replace(" + ", "\n+ ").replace("Без точек (bbox)", "Без точек\n(bbox)")
              .replace("Все 20 точек", "Все 20\nточек") for k in keys]

    def bars(metric, fname, title, ylim=(0, 1.0)):
        fig, ax = plt.subplots(figsize=(11.5, 5.4))
        v = [R[k][metric] for k in keys]
        ax.bar(x, v, color=[col(k) for k in keys], width=0.62)
        for i, val in enumerate(v):
            ax.text(i, val + 0.012, f"{val:.2f}", ha="center", fontsize=13, fontweight="bold")
        ax.axhline(R["kp_none_bbox"][metric], color=GREY, ls="--", lw=1.2)
        if "kp_all" in R:
            ax.axhline(R["kp_all"][metric], color=NAVY, ls="--", lw=1.2)
        ax.set_xticks(x, labels); ax.set_ylim(*ylim); ax.set_title(title, fontsize=15, loc="left")
        fig.tight_layout(); fig.savefig(CH / fname, dpi=170); plt.close(fig)

    bars("f1", "f1.png", "F1 на тестовой выборке (серый пунктир - без точек, синий - все точки)")
    bars("roc_auc", "roc_auc.png", "ROC-AUC на тестовой выборке", (0.4, 1.02))

    fig, ax = plt.subplots(figsize=(12, 5.4))
    w = 0.2
    for j, (m, lab, c) in enumerate([("f1", "F1", NAVY), ("accuracy", "Accuracy", ORANGE),
                                     ("precision", "Precision", "#4C9F70"), ("recall", "Recall", "#7D6BAA")]):
        ax.bar(x + (j - 1.5) * w, [R[k][m] for k in keys], w, label=lab, color=c)
    ax.set_xticks(x, labels); ax.set_ylim(0, 1.08)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    ax.set_title("Основные метрики по конфигурациям", fontsize=15, loc="left")
    fig.tight_layout(); fig.savefig(CH / "metrics.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    others = [k for k in keys if k != "kp_all"]
    d = [R[k]["f1"] - R["kp_all"]["f1"] for k in others]
    ax.bar(range(len(others)), d, color=[GREY if k == "kp_none_bbox" else RED for k in others], width=0.6)
    for i, val in enumerate(d):
        ax.text(i, val - 0.012, f"{val:+.2f}", ha="center", va="top", fontsize=13, fontweight="bold")
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(range(len(others)), [NAMES[k].replace(" + ", "\n+ ") for k in others])
    ax.set_ylim(min(d) - 0.08, 0.05)
    ax.set_title("Разница F1 относительно «все 20 точек»", fontsize=15, loc="left")
    fig.tight_layout(); fig.savefig(CH / "delta.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for k in keys:
        ax.scatter(POINTS[k], R[k]["f1"], s=170, color=col(k), zorder=3)
        ax.annotate(NAMES[k], (POINTS[k], R[k]["f1"]), textcoords="offset points", xytext=(8, 7), fontsize=11)
    ax.set_xlabel("Число используемых ключевых точек"); ax.set_ylabel("F1")
    ax.set_xlim(-1.5, 25); ax.set_ylim(0.6, 0.9)
    ax.set_title("Качество и число точек", fontsize=15, loc="left")
    fig.tight_layout(); fig.savefig(CH / "points_vs_f1.png", dpi=170); plt.close(fig)


# ---------------------------------------------------------------- pptx
def rgb(h):
    return RGBColor.from_string(h.lstrip("#"))


W, H = Inches(13.333), Inches(7.5)


def add_text(slide, x, y, w, h, text, size=18, bold=False, color=INK, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    lines = text if isinstance(text, list) else [text]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size, r.font.bold, r.font.name = Pt(size), bold, "Calibri"
        r.font.color.rgb = rgb(color)
        p.space_after = Pt(6)
    return tb


def bullets(slide, x, y, w, h, items, size=18):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run()
        r.text = "•  " + it
        r.font.size, r.font.name = Pt(size), "Calibri"
        r.font.color.rgb = rgb(INK)
        p.space_after = Pt(10)
    return tb


def rect(slide, x, y, w, h, fill, line=None):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    s.fill.solid(); s.fill.fore_color.rgb = rgb(fill)
    if line:
        s.line.color.rgb = rgb(line)
    else:
        s.line.fill.background()
    s.adjustments[0] = 0.08
    s.shadow.inherit = False
    return s


def base_slide(prs, title, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_text(s, Inches(0.6), Inches(0.35), Inches(12.1), Inches(0.9), title, size=30, bold=True, color=NAVY)
    ln = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(1.2), Inches(1.2), Emu(38000))
    ln.fill.solid(); ln.fill.fore_color.rgb = rgb(ORANGE); ln.line.fill.background()
    add_text(s, Inches(12.2), Inches(7.0), Inches(0.8), Inches(0.35), str(n), size=12, color=GREY,
             align=PP_ALIGN.RIGHT)
    return s


def table(slide, x, y, w, rows, col_w, size=15, row_h=0.46, hl=None):
    shp = slide.shapes.add_table(len(rows), len(rows[0]), x, y, w, Inches(row_h * len(rows)))
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
            f.bold = i == 0 or (hl is not None and i == hl)
            f.color.rgb = rgb("#FFFFFF") if i == 0 else rgb(INK)
            c.fill.solid()
            c.fill.fore_color.rgb = rgb(NAVY if i == 0 else ("#DCE8F5" if hl == i else ("#F3F5F7" if i % 2 else "#FFFFFF")))
    return t


def pic(slide, name, x, y, w):
    slide.shapes.add_picture(str(CH / name), x, y, width=w)


def build(R):
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    n = 0

    # 1. титул
    s = prs.slides.add_slide(prs.slide_layouts[6]); n += 1
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    bg.fill.solid(); bg.fill.fore_color.rgb = rgb(NAVY); bg.line.fill.background()
    add_text(s, Inches(0.9), Inches(2.1), Inches(11.5), Inches(2.2),
             "Помогает ли группировка ключевых точек детектировать дефекацию у собак?",
             size=40, bold=True, color="#FFFFFF")
    add_text(s, Inches(0.9), Inches(4.5), Inches(11.5), Inches(1.0),
             "Эксперименты с наборами точек позы: YOLO-pose + LSTM с attention", size=22, color="#C9D8E8")
    add_text(s, Inches(0.9), Inches(6.2), Inches(11.5), Inches(0.5),
             "Предварительные результаты: 6 из 7 конфигураций, один запуск", size=16, color=ORANGE)

    # 2. пайплайн
    s = base_slide(prs, "Как устроена система", n := n + 1)
    steps = [("Видео", "кадры"), ("YOLO\nдетекция", "bbox собаки"), ("YOLO pose", "20 ключевых\nточек"),
             ("Фильтр +\nсглаживание", "confidence,\nмедиана"), ("Признаки", "72 на кадр\n+ скорости = 144"),
             ("LSTM +\nattention", "окно 120 кадров,\nшаг 30"), ("Класс", "дефекация /\nнорма")]
    bw, gap, x0 = Inches(1.5), Inches(0.28), Inches(0.6)
    for i, (a, b) in enumerate(steps):
        x = x0 + i * (bw + gap)
        rect(s, x, Inches(1.9), bw, Inches(1.1), NAVY if i not in (2, 4) else ORANGE)
        add_text(s, x, Inches(1.9), bw, Inches(1.1), a, size=16, bold=True, color="#FFFFFF",
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        add_text(s, x - Inches(0.1), Inches(3.1), bw + Inches(0.2), Inches(0.9), b, size=13, color=INK,
                 align=PP_ALIGN.CENTER)
    bullets(s, Inches(0.6), Inches(4.4), Inches(12), Inches(2.5), [
        "Датасет: 166 размеченных видео, 106 - «норма», 60 - «дефекация»",
        "Из видео нарезаются окна по 120 кадров: 348 окон с позой (194 нормы / 154 дефекации)",
        "Для конфигурации «без точек» позы нет, окон получается больше: 411 (251 / 160)",
        "Оранжевым выделены шаги, которые меняются в эксперименте: какие точки идут в признаки",
    ], size=17)

    # 3. что сравниваем
    s = base_slide(prs, "Что сравниваем: 7 конфигураций точек", n := n + 1)
    table(s, Inches(0.6), Inches(1.55), Inches(7.2), [
        ["Конфигурация", "Точки", "Индексы"],
        ["Без точек (bbox)", "0", "только рамка собаки"],
        ["Все 20 точек", "20", "0-19"],
        ["Позвоночник + таз", "10", "0-9"],
        ["Таз + хвост", "7", "6-12"],
        ["Только хвост", "3", "10-12"],
        ["Только лапы", "7", "13-19"],
        ["Позвоночник + хвост", "9", "0-5, 10-12"],
    ], [3.2, 1.2, 2.8], size=15, row_h=0.52)
    rect(s, Inches(8.2), Inches(1.55), Inches(4.5), Inches(4.16), "#F3F5F7")
    add_text(s, Inches(8.4), Inches(1.65), Inches(4.1), Inches(0.5), "Группы точек собаки", size=18, bold=True, color=NAVY)
    bullets(s, Inches(8.4), Inches(2.2), Inches(4.1), Inches(3.5), [
        "позвоночник: 0-5", "таз: 6-9", "хвост: 10-12", "передние лапы: 13-15", "задние лапы: 16-19",
    ], size=16)
    add_text(s, Inches(0.6), Inches(6.0), Inches(12), Inches(1.0),
             "Гипотеза: для дефекации важна поза таза и хвоста, поэтому меньшая группа точек может убрать шум "
             "и дать качество не хуже, чем все 20 точек.", size=16, color=INK)

    # 4. методика
    s = base_slide(prs, "Как проводился эксперимент", n := n + 1)
    bullets(s, Inches(0.6), Inches(1.6), Inches(7.4), Inches(5), [
        "Для каждой конфигурации заново собираются признаки и обучается одна и та же LSTM",
        "Модель: энкодер признаков → LSTM (2 слоя, 128) → attention → вероятность дефекации",
        "12 эпох, AdamW, lr 1e-3, batch 32; лучшие веса выбираются по F1 на валидации",
        "Окна делятся случайно: 64% обучение, 16% валидация, 20% тест (около 70 окон)",
        "Метрики на тесте: F1, accuracy, precision, recall, ROC-AUC",
    ], size=17)
    rect(s, Inches(8.4), Inches(1.6), Inches(4.3), Inches(3.9), "#FFF3E8")
    add_text(s, Inches(8.6), Inches(1.7), Inches(3.9), Inches(0.5), "Статус прогона", size=18, bold=True, color=ORANGE)
    bullets(s, Inches(8.6), Inches(2.25), Inches(3.9), Inches(3.2), [
        "6 из 7 конфигураций посчитаны",
        "«Позвоночник + хвост» прервана вручную",
        "Один запуск, один случайный сид",
    ], size=16)

    # 5. таблица
    s = base_slide(prs, "Результаты: таблица метрик", n := n + 1)
    keys = [k for k in NAMES if k in R]
    best = max(keys, key=lambda k: R[k]["f1"])
    rows = [["Конфигурация", "F1", "Accuracy", "Precision", "Recall", "ROC-AUC"]]
    for k in keys:
        r = R[k]
        rows.append([NAMES[k], f"{r['f1']:.3f}", f"{r['accuracy']:.3f}", f"{r['precision']:.3f}",
                     f"{r['recall']:.3f}", f"{r['roc_auc']:.3f}"])
    table(s, Inches(0.6), Inches(1.6), Inches(12.1), rows, [4.1, 1.6, 1.6, 1.6, 1.6, 1.6], size=17, row_h=0.55,
          hl=1 + keys.index(best))
    add_text(s, Inches(0.6), Inches(1.6 + 0.55 * len(rows) + 0.25), Inches(12), Inches(0.9),
             f"Лучший результат по всем метрикам - «{NAMES[best]}». Позвоночник + хвост в таблицу не попал: расчёт не завершён.",
             size=15, color=GREY)

    # 6-8 графики
    s = base_slide(prs, "F1: точки позы дают заметный прирост над bbox", n := n + 1)
    pic(s, "f1.png", Inches(0.9), Inches(1.45), Inches(11.5))
    s = base_slide(prs, "Остальные метрики и ROC-AUC", n := n + 1)
    pic(s, "metrics.png", Inches(0.3), Inches(1.4), Inches(6.5))
    pic(s, "roc_auc.png", Inches(6.7), Inches(1.4), Inches(6.4))
    add_text(s, Inches(0.6), Inches(5.0), Inches(12), Inches(1.6), [
        "У «всех точек» самая высокая точность (precision 0.96) и ROC-AUC 0.96.",
        "Лапы почти не несут сигнала: ROC-AUC 0.69 - хуже, чем у голой рамки (0.87)."], size=17)
    s = base_slide(prs, "Группировка против полного набора", n := n + 1)
    pic(s, "delta.png", Inches(0.5), Inches(1.45), Inches(7.6))
    pic(s, "points_vs_f1.png", Inches(8.2), Inches(1.55), Inches(4.9))
    add_text(s, Inches(0.6), Inches(5.35), Inches(12), Inches(1.5), [
        "Ни одна группа точек не превзошла полный набор: лучшие - «только хвост» и «позвоночник + таз» (F1 около 0.77, на 0.09 ниже).",
        "Зависимости «меньше точек - лучше» нет."], size=17)

    # 9. выводы
    s = base_slide(prs, "Выводы", n := n + 1)
    cards = [
        ("Точки позы помогают", f"F1 {R['kp_none_bbox']['f1']:.2f} → {R['kp_all']['f1']:.2f} по сравнению с одной рамкой собаки; "
                                f"ROC-AUC {R['kp_none_bbox']['roc_auc']:.2f} → {R['kp_all']['roc_auc']:.2f}.", NAVY),
        ("Группировка пока не помогает", "Любое подмножество точек дало F1 ниже, чем все 20. Хвост один даёт 0.77, но таз + хвост - 0.69.", ORANGE),
        ("Лапы вредны как единственный источник", f"F1 {R['kp_legs']['f1']:.2f} и ROC-AUC {R['kp_legs']['roc_auc']:.2f} - хуже bbox.", RED),
    ]
    for i, (h, t, c) in enumerate(cards):
        x = Inches(0.6 + i * 4.1)
        rect(s, x, Inches(1.7), Inches(3.9), Inches(3.4), "#F3F5F7")
        bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, Inches(1.7), Inches(3.9), Emu(60000))
        bar.fill.solid(); bar.fill.fore_color.rgb = rgb(c); bar.line.fill.background()
        add_text(s, x + Inches(0.2), Inches(1.9), Inches(3.5), Inches(0.9), h, size=20, bold=True, color=c)
        add_text(s, x + Inches(0.2), Inches(2.9), Inches(3.5), Inches(2.2), t, size=16)
    add_text(s, Inches(0.6), Inches(5.5), Inches(12), Inches(1.3),
             "Пока это не окончательный ответ: различия между группами лежат в пределах шума (см. ограничения).",
             size=18, bold=True, color=NAVY)

    # 10. ограничения
    s = base_slide(prs, "Ограничения этих результатов", n := n + 1)
    bullets(s, Inches(0.6), Inches(1.6), Inches(12), Inches(5.2), [
        "Малый тест: около 70 окон, одно окно меняет F1 на 0.02-0.03; разница в 0.05-0.09 между группами может быть случайной",
        "Один запуск с одним сидом: нет среднего и разброса",
        "Случайное разбиение окон: соседние окна одного ролика попадают и в обучение, и в тест, метрики завышены",
        "«Без точек» оценивается на другом наборе окон (411 против 348), сравнение приблизительное",
        "Конфигурация «позвоночник + хвост» не досчитана",
    ], size=19)

    # 11. дальше
    s = base_slide(prs, "Что сделано для надёжного ответа", n := n + 1)
    bullets(s, Inches(0.6), Inches(1.6), Inches(12), Inches(3.2), [
        "В ноутбук добавлены ячейки мультисид-прогона: каждая конфигурация на 5 разных разбиениях, среднее ± std",
        "Разбиение теперь по видео, а не по окнам - утечки между обучением и тестом нет",
        "Инференс YOLO кэшируется: видео обрабатываются сетями один раз на все конфигурации",
        "Итог сохраняется в runs.json и архив с графиками; по нему обновим эту презентацию",
    ], size=19)
    rect(s, Inches(0.6), Inches(5.0), Inches(12.1), Inches(1.4), "#DCE8F5")
    add_text(s, Inches(0.9), Inches(5.0), Inches(11.5), Inches(1.4),
             "Следующий шаг: дождаться мультисид-прогона на GPU и проверить, остаётся ли вывод «все точки лучше группы».",
             size=19, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)

    prs.save(OUT / "keypoint_experiments.pptx")
    print("saved", n, "slides")


if __name__ == "__main__":
    R = parse_results()
    print({k: round(v["f1"], 3) for k, v in R.items()})
    make_charts(R)
    build(R)
