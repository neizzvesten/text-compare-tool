from __future__ import annotations

import argparse
import html
import math
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Optional

# Optional dependencies. The application still starts without them.
try:
    from pymorphy3 import MorphAnalyzer
except Exception:
    MorphAnalyzer = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from docx import Document
except Exception:
    Document = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+(?:[-’'][A-Za-zА-Яа-яЁё0-9]+)*", re.UNICODE)
SENT_RE = re.compile(r"(?<=[.!?…])\s+|\n+")


@dataclass
class Match:
    category: str
    score: float
    text1: str
    text2: str
    details: str = ""


class Analyzer:
    def __init__(self, semantic_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.morph = MorphAnalyzer() if MorphAnalyzer is not None else None
        self.semantic_model_name = semantic_model_name
        self.semantic_model = None
        self.semantic_available = False
        self.semantic_note = ""

    @staticmethod
    def normalize_text(text: str) -> str:
        text = unicodedata.normalize("NFC", text.replace("\u00a0", " "))
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def words(text: str) -> list[str]:
        return [m.group(0).lower() for m in WORD_RE.finditer(text)]

    def lemmas_and_pos(self, text: str) -> tuple[list[str], list[str]]:
        tokens = self.words(text)
        if not self.morph:
            # Fallback: surface forms act as pseudo-lemmas; POS unavailable.
            return tokens, []
        lemmas: list[str] = []
        pos: list[str] = []
        for token in tokens:
            try:
                p = self.morph.parse(token)[0]
                lemmas.append(p.normal_form)
                coarse = (p.tag.POS or "X").split(",")[0]
                pos.append(coarse)
            except Exception:
                lemmas.append(token)
                pos.append("X")
        return lemmas, pos

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        parts = SENT_RE.split(text)
        out = []
        for part in parts:
            p = part.strip()
            if p and WORD_RE.search(p):
                out.append(p)
        return out

    @staticmethod
    def sequence_similarity(a: Iterable[str], b: Iterable[str]) -> float:
        return SequenceMatcher(None, list(a), list(b), autojunk=False).ratio()

    def exact_matches(self, a: str, b: str, min_words: int = 5) -> list[Match]:
        wa = self.words(a)
        wb = self.words(b)
        sm = SequenceMatcher(None, wa, wb, autojunk=False)
        matches: list[Match] = []
        seen: set[tuple[int, int, int]] = set()
        raw_a = self.split_sentences(a)
        raw_b = self.split_sentences(b)

        # Long exact word sequences.
        for block in sm.get_matching_blocks():
            i, j, n = block
            if n < min_words:
                continue
            key = (i, j, n)
            if key in seen:
                continue
            seen.add(key)
            frag = " ".join(wa[i:i+n])
            matches.append(Match(
                category="Полнотекстовое / дословное",
                score=1.0,
                text1=frag,
                text2=frag,
                details=f"Совпадает {n} слов подряд (пунктуация и пробелы не учитываются).",
            ))

        # Exact sentence matches are worth reporting even when they contain < min_words.
        normalized_b = {self.normalize_text(x).lower(): x for x in raw_b}
        already = {self.normalize_text(m.text1).lower() for m in matches}
        for sentence in raw_a:
            key = self.normalize_text(sentence).lower()
            if key in normalized_b and len(self.words(sentence)) >= 3 and key not in already:
                matches.append(Match(
                    category="Полнотекстовое / дословное",
                    score=1.0,
                    text1=sentence,
                    text2=normalized_b[key],
                    details="Полное совпадение предложения после нормализации пробелов.",
                ))
        return matches

    def sentence_pairs(self, a: str, b: str) -> list[tuple[str, str, float]]:
        sa = self.split_sentences(a)
        sb = self.split_sentences(b)
        pairs: list[tuple[str, str, float]] = []
        for x in sa:
            best = ("", 0.0)
            for y in sb:
                score = self.sequence_similarity(self.words(x), self.words(y))
                if score > best[1]:
                    best = (y, score)
            if best[0]:
                pairs.append((x, best[0], best[1]))
        return pairs

    def partial_matches(self, a: str, b: str, low: float = 0.65, high: float = 0.99) -> list[Match]:
        out = []
        for x, y, score in self.sentence_pairs(a, b):
            if low <= score < high and min(len(self.words(x)), len(self.words(y))) >= 4:
                out.append(Match(
                    category="Частичное / с минимальными изменениями",
                    score=score,
                    text1=x,
                    text2=y,
                    details="Высокое поверхностное сходство при наличии небольших изменений.",
                ))
        return out

    def lexical_matches(self, a: str, b: str, threshold: float = 0.5) -> list[Match]:
        sa = self.split_sentences(a)
        sb = self.split_sentences(b)
        out: list[Match] = []
        exact_keys = {(self.normalize_text(m.text1).lower(), self.normalize_text(m.text2).lower()) for m in self.exact_matches(a, b, 5)}
        for x in sa:
            lx, _ = self.lemmas_and_pos(x)
            set_x = set(lx)
            if len(set_x) < 3:
                continue
            best = None
            for y in sb:
                ly, _ = self.lemmas_and_pos(y)
                set_y = set(ly)
                if not set_y:
                    continue
                jac = len(set_x & set_y) / len(set_x | set_y)
                if best is None or jac > best[1]:
                    best = (y, jac, len(set_x & set_y))
            if best and best[1] >= threshold:
                y, score, common = best
                if (self.normalize_text(x).lower(), self.normalize_text(y).lower()) in exact_keys:
                    continue
                out.append(Match(
                    category="Лексическое",
                    score=score,
                    text1=x,
                    text2=y,
                    details=f"Совпадение по леммам: {common} общих лексических единиц; Jaccard={score:.2f}.",
                ))
        return out

    def grammatical_matches(self, a: str, b: str, threshold: float = 0.72) -> list[Match]:
        if not self.morph:
            return []
        sa = self.split_sentences(a)
        sb = self.split_sentences(b)
        out: list[Match] = []
        for x in sa:
            _, px = self.lemmas_and_pos(x)
            if len(px) < 4:
                continue
            best = None
            for y in sb:
                _, py = self.lemmas_and_pos(y)
                if len(py) < 4:
                    continue
                score = self.sequence_similarity(px, py)
                if best is None or score > best[1]:
                    best = (y, score)
            if best and best[1] >= threshold:
                out.append(Match(
                    category="Грамматическое / синтаксическое",
                    score=best[1],
                    text1=x,
                    text2=best[0],
                    details="Сходство последовательности частей речи; результат требует экспертной проверки.",
                ))
        return out

    @staticmethod
    def _tfidf_vectors(sentences: list[str]) -> list[dict[str, float]]:
        tokenized = [WORD_RE.findall(s.lower()) for s in sentences]
        df = Counter()
        for toks in tokenized:
            df.update(set(toks))
        n = max(1, len(tokenized))
        vectors: list[dict[str, float]] = []
        for toks in tokenized:
            counts = Counter(toks)
            total = max(1, len(toks))
            vec = {}
            for term, count in counts.items():
                idf = math.log((n + 1) / (df[term] + 1)) + 1
                vec[term] = (count / total) * idf
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            vectors.append({k: v / norm for k, v in vec.items()})
        return vectors

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        if len(a) > len(b):
            a, b = b, a
        return sum(v * b.get(k, 0.0) for k, v in a.items())

    def semantic_matches(self, a: str, b: str, threshold: float = 0.62) -> tuple[list[Match], str]:
        sa = self.split_sentences(a)
        sb = self.split_sentences(b)
        if not sa or not sb:
            return [], ""

        # Preferred method: multilingual sentence embeddings.
        if SentenceTransformer is not None:
            try:
                if self.semantic_model is None:
                    self.semantic_model = SentenceTransformer(self.semantic_model_name)
                va = self.semantic_model.encode(sa, normalize_embeddings=True)
                vb = self.semantic_model.encode(sb, normalize_embeddings=True)
                out: list[Match] = []
                for i, x in enumerate(sa):
                    scores = (vb @ va[i]) if hasattr(vb, "__matmul__") else []
                    if len(scores) == 0:
                        continue
                    j = int(scores.argmax())
                    score = float(scores[j])
                    if score >= threshold:
                        out.append(Match(
                            category="Семантическое (модель embeddings)",
                            score=score,
                            text1=x,
                            text2=sb[j],
                            details=f"Cosine similarity={score:.2f}; автоматическая оценка требует экспертной интерпретации.",
                        ))
                return out, f"Семантика: SentenceTransformer ({self.semantic_model_name})."
            except Exception as exc:
                self.semantic_note = f"Не удалось загрузить semantic-модель: {exc}. Использован резервный метод."

        # Dependency-free fallback: TF-IDF cosine similarity.
        all_sent = sa + sb
        vecs = self._tfidf_vectors(all_sent)
        va = vecs[:len(sa)]
        vb = vecs[len(sa):]
        out = []
        for i, x in enumerate(sa):
            best_j, best_score = -1, 0.0
            for j, y in enumerate(sb):
                score = self._cosine(va[i], vb[j])
                if score > best_score:
                    best_j, best_score = j, score
            if best_j >= 0 and best_score >= threshold:
                out.append(Match(
                    category="Лексико-семантическое (резервный метод)",
                    score=best_score,
                    text1=x,
                    text2=sb[best_j],
                    details=f"TF-IDF cosine={best_score:.2f}; это эвристическая оценка, а не полноценное семантическое понимание.",
                ))
        return out, "Семантика: резервный TF-IDF cosine (для демонстрационного прототипа)."

    def analyze(self, a: str, b: str) -> dict:
        a = self.normalize_text(a)
        b = self.normalize_text(b)
        exact = self.exact_matches(a, b)
        partial = self.partial_matches(a, b)
        lexical = self.lexical_matches(a, b)
        grammatical = self.grammatical_matches(a, b)
        semantic, semantic_method = self.semantic_matches(a, b)

        # Remove almost identical duplicate sentence-pair reports by text pair.
        groups = [exact, partial, lexical, grammatical, semantic]
        all_matches: list[Match] = []
        seen_pairs: set[tuple[str, str, str]] = set()
        for group in groups:
            for m in group:
                key = (m.category, self.normalize_text(m.text1).lower(), self.normalize_text(m.text2).lower())
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    all_matches.append(m)

        return {
            "counts": {
                "Полнотекстовые / дословные": len(exact),
                "Частичные": len(partial),
                "Лексические": len(lexical),
                "Грамматические / синтаксические": len(grammatical),
                "Семантические": len(semantic),
            },
            "semantic_method": semantic_method,
            "notes": [
                "Автоматические результаты являются поисковой подсказкой и не заменяют лингвистическую интерпретацию эксперта.",
                "Полнотекстовые совпадения в прототипе ищутся по последовательностям одинаковых слов; пунктуация и пробелы нормализуются.",
                "Пороговые значения предназначены для демонстрации и должны валидироваться на размеченном тестовом корпусе.",
            ],
            "matches": [asdict(m) for m in all_matches],
        }


# ---------------------------- file I/O ----------------------------

def load_file(path: str) -> str:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".txt", ".md", ".csv"}:
        return p.read_text(encoding="utf-8", errors="replace")
    if suffix == ".docx":
        if Document is None:
            raise RuntimeError("Для DOCX установите python-docx: pip install python-docx")
        doc = Document(str(p))
        return "\n".join(par.text for par in doc.paragraphs)
    if suffix == ".pdf":
        if PdfReader is None:
            raise RuntimeError("Для PDF установите pypdf: pip install pypdf")
        reader = PdfReader(str(p))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise RuntimeError(f"Формат {suffix or '(без расширения)'} пока не поддерживается.")


def save_html_report(path: str, result: dict) -> None:
    rows = []
    for i, m in enumerate(result["matches"], start=1):
        score = f"{m['score']:.2f}"
        rows.append(
            f"<section class='match'><h3>#{i} — {html.escape(m['category'])}</h3>"
            f"<div class='score'>Score: {score}</div>"
            f"<div class='grid'><div><b>Текст 1</b><p>{html.escape(m['text1'])}</p></div>"
            f"<div><b>Текст 2</b><p>{html.escape(m['text2'])}</p></div></div>"
            f"<p class='details'>{html.escape(m['details'])}</p></section>"
        )
    count_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in result["counts"].items()
    )
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in result["notes"])
    document = f"""<!doctype html>
<html lang='ru'><head><meta charset='utf-8'><title>Отчёт сравнения текстов</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1100px;margin:40px auto;padding:0 24px;color:#222;background:#fafafa}}
h1{{margin-bottom:6px}} .muted{{color:#666}} table{{border-collapse:collapse;width:100%;background:#fff;margin:20px 0}}
th,td{{border:1px solid #ddd;padding:10px;text-align:left}} th{{background:#f0f0f0}}
.match{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:16px;margin:16px 0}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}} .grid>div{{background:#f7f7f7;padding:12px;border-radius:6px}}
.score{{display:inline-block;padding:4px 8px;background:#eee;border-radius:4px;margin:6px 0 12px}}
.details{{color:#555}} li{{margin:6px 0}}
@media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>Отчёт сравнения двух текстов</h1>
<p class='muted'>Прототип для исследовательской/экспертной работы.</p>
<table><thead><tr><th>Тип совпадения</th><th>Количество найденных фрагментов</th></tr></thead><tbody>{count_rows}</tbody></table>
<h2>Метод семантического сравнения</h2><p>{html.escape(result['semantic_method'])}</p>
<h2>Обнаруженные фрагменты</h2>{''.join(rows) or '<p>Совпадения по заданным порогам не обнаружены.</p>'}
<h2>Ограничения прототипа</h2><ul>{notes}</ul>
</body></html>"""
    Path(path).write_text(document, encoding="utf-8")


# ---------------------------- CLI test ----------------------------

def run_self_test() -> int:
    a = (
        "В ходе исследования были получены следующие результаты. "
        "Компания понесла значительные финансовые потери. "
        "Учёные провели исследование влияния стресса."
    )
    b = (
        "В ходе исследования были получены следующие результаты. "
        "Компания понесла существенные финансовые потери. "
        "Исследование влияния стресса было проведено учёными."
    )
    analyzer = Analyzer()
    result = analyzer.analyze(a, b)
    print("SELF-TEST OK")
    print(result["counts"])
    print("semantic:", result["semantic_method"])
    assert result["counts"]["Полнотекстовые / дословные"] >= 1
    assert result["counts"]["Лексические"] >= 1
    assert result["counts"]["Грамматические / синтаксические"] >= 0
    assert result["matches"]
    return 0


# ---------------------------- GUI ----------------------------

def launch_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    analyzer = Analyzer()

    root = tk.Tk()
    root.title("Text Compare Tool — прототип для лингвистической экспертизы")
    root.geometry("1250x850")
    root.minsize(1000, 700)

    top = ttk.Frame(root, padding=10)
    top.pack(fill="x")
    ttk.Label(top, text="Сравнение текстов", font=("Segoe UI", 18, "bold")).pack(side="left")
    ttk.Label(top, text="Прототип: exact / partial / lexical / grammatical / semantic", foreground="#666").pack(side="left", padx=20)

    panes = ttk.Frame(root, padding=(10, 0, 10, 10))
    panes.pack(fill="both", expand=True)

    left_frame = ttk.LabelFrame(panes, text="Текст 1", padding=8)
    right_frame = ttk.LabelFrame(panes, text="Текст 2", padding=8)
    left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
    right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
    panes.columnconfigure(0, weight=1)
    panes.columnconfigure(1, weight=1)
    panes.rowconfigure(0, weight=1)

    left = tk.Text(left_frame, wrap="word", undo=True, font=("Segoe UI", 11))
    right = tk.Text(right_frame, wrap="word", undo=True, font=("Segoe UI", 11))
    left.pack(fill="both", expand=True)
    right.pack(fill="both", expand=True)

    def load_into(widget: tk.Text):
        path = filedialog.askopenfilename(filetypes=[
            ("Поддерживаемые файлы", "*.txt *.md *.docx *.pdf"),
            ("TXT/MD", "*.txt *.md"),
            ("DOCX", "*.docx"),
            ("PDF", "*.pdf"),
            ("Все файлы", "*.*"),
        ])
        if not path:
            return
        try:
            text = load_file(path)
            widget.delete("1.0", "end")
            widget.insert("1.0", text)
        except Exception as exc:
            messagebox.showerror("Ошибка загрузки", str(exc))

    controls = ttk.Frame(root, padding=10)
    controls.pack(fill="x")
    ttk.Button(controls, text="Загрузить файл 1", command=lambda: load_into(left)).pack(side="left")
    ttk.Button(controls, text="Загрузить файл 2", command=lambda: load_into(right)).pack(side="left", padx=6)
    ttk.Button(controls, text="Загрузить демо", command=lambda: load_demo(left, right)).pack(side="left", padx=6)
    compare_btn = ttk.Button(controls, text="СРАВНИТЬ", command=lambda: compare())
    compare_btn.pack(side="left", padx=18)
    export_btn = ttk.Button(controls, text="Сохранить HTML-отчёт", state="disabled")
    export_btn.pack(side="right")

    summary = ttk.LabelFrame(root, text="Результаты", padding=10)
    summary.pack(fill="both", expand=True, padx=10, pady=(0, 10))
    summary.columnconfigure(0, weight=1)
    summary.rowconfigure(1, weight=1)

    stats = ttk.Label(summary, text="Нажмите «СРАВНИТЬ», чтобы получить результаты.", font=("Segoe UI", 11, "bold"))
    stats.grid(row=0, column=0, sticky="w")

    tree = ttk.Treeview(summary, columns=("category", "score", "text1", "text2"), show="headings")
    tree.heading("category", text="Тип")
    tree.heading("score", text="Score")
    tree.heading("text1", text="Текст 1")
    tree.heading("text2", text="Текст 2")
    tree.column("category", width=220, anchor="w")
    tree.column("score", width=80, anchor="center")
    tree.column("text1", width=390, anchor="w")
    tree.column("text2", width=390, anchor="w")
    tree.grid(row=1, column=0, sticky="nsew")

    scroll = ttk.Scrollbar(summary, orient="vertical", command=tree.yview)
    scroll.grid(row=1, column=1, sticky="ns")
    tree.configure(yscrollcommand=scroll.set)

    status = ttk.Label(root, text="Готово", foreground="#555", padding=(10, 0, 10, 10))
    status.pack(fill="x")

    last_result = {"value": None}

    def compare():
        a = left.get("1.0", "end").strip()
        b = right.get("1.0", "end").strip()
        if not a or not b:
            messagebox.showwarning("Нет текста", "Заполните оба поля или загрузите два файла.")
            return
        compare_btn.config(state="disabled")
        status.config(text="Идёт сравнение…")
        root.update_idletasks()
        try:
            result = analyzer.analyze(a, b)
            last_result["value"] = result
            for item in tree.get_children():
                tree.delete(item)
            for m in result["matches"]:
                tree.insert("", "end", values=(
                    m["category"],
                    f"{m['score']:.2f}",
                    m["text1"][:180],
                    m["text2"][:180],
                ))
            c = result["counts"]
            stats.config(text=(
                f"Дословные: {c['Полнотекстовые / дословные']}   |   "
                f"Частичные: {c['Частичные']}   |   "
                f"Лексические: {c['Лексические']}   |   "
                f"Грамматические: {c['Грамматические / синтаксические']}   |   "
                f"Семантические: {c['Семантические']}"
            ))
            status.config(text=result["semantic_method"])
            export_btn.config(state="normal")
        except Exception as exc:
            messagebox.showerror("Ошибка сравнения", repr(exc))
            status.config(text="Ошибка")
        finally:
            compare_btn.config(state="normal")

    def export_report():
        result = last_result["value"]
        if not result:
            return
        path = filedialog.asksaveasfilename(defaultextension=".html", filetypes=[("HTML", "*.html")])
        if not path:
            return
        try:
            save_html_report(path, result)
            status.config(text=f"Отчёт сохранён: {path}")
        except Exception as exc:
            messagebox.showerror("Ошибка сохранения", str(exc))

    export_btn.config(command=export_report)

    def load_demo(left_widget, right_widget):
        a = (
            "В ходе исследования были получены следующие результаты. "
            "Компания понесла значительные финансовые потери. "
            "Учёные провели исследование влияния стресса."
        )
        b = (
            "В ходе исследования были получены следующие результаты. "
            "Компания понесла существенные финансовые потери. "
            "Исследование влияния стресса было проведено учёными."
        )
        left_widget.delete("1.0", "end")
        right_widget.delete("1.0", "end")
        left_widget.insert("1.0", a)
        right_widget.insert("1.0", b)

    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prototype text comparison tool for linguistic expertise")
    parser.add_argument("--test", action="store_true", help="Run a headless self-test")
    args = parser.parse_args()
    if args.test:
        raise SystemExit(run_self_test())
    launch_gui()
