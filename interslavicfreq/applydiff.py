#!/usr/bin/env python3
"""
applydiff_win.py — Надёжный патчер unified diff для Windows
v2.2 - исправлены синтаксические ошибки

Ключевые особенности:
• Частичное применение: если хунк не подходит — пропускается, остальные применяются
• Подробный отчёт о каждом хунке
• НИКОГДА не вылетает при двойном клике
• Сохраняет неприменённые хунки в .rej файл
• Умный поиск позиции с fuzz и offset
• Fuzzy matching для изменённого контекста
"""

import sys
import os
import re
import argparse
import traceback
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# СТРУКТУРЫ ДАННЫХ
# ═══════════════════════════════════════════════════════════════════════════════

HUNK_RE = re.compile(r'^@@\s*-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s*@@')

# Для извлечения diff-блоков из markdown сообщений чат-ботов
MARKDOWN_DIFF_BLOCK_RE = re.compile(r'```diff\s*\n(.*?)```', re.DOTALL)

SKIP_PREFIXES = (
    'diff --git', 'diff --combined', 'diff --cc', 'diff -',
    'index ', 'new file mode', 'deleted file mode',
    'old mode', 'new mode', 'similarity index',
    'rename from', 'rename to', 'copy from', 'copy to',
    'Binary files', 'GIT binary patch', 'Only in',
)


@dataclass
class HunkLine:
    """Строка внутри хунка"""
    op: str          # ' ', '+', '-'
    text: str        # содержимое без префикса
    has_newline: bool = True


@dataclass
class Hunk:
    """Один хунк (блок изменений)"""
    ostart: int          # начальная строка в оригинале (1-based)
    ocnt: int            # количество строк из оригинала
    nstart: int          # начальная строка в результате (1-based)
    ncnt: int            # количество новых строк
    lines: List[HunkLine] = field(default_factory=list)
    header: str = ""     # оригинальный заголовок @@ ... @@
    
    def get_old_lines(self) -> List[HunkLine]:
        """Строки, которые должны быть в оригинале (контекст + удаления)"""
        return [hl for hl in self.lines if hl.op in (' ', '-')]
    
    def get_new_lines(self) -> List[HunkLine]:
        """Строки, которые будут в результате (контекст + добавления)"""
        return [hl for hl in self.lines if hl.op in (' ', '+')]


@dataclass
class FilePatch:
    """Патч для одного файла"""
    src: str = ""
    dst: str = ""
    hunks: List[Hunk] = field(default_factory=list)
    
    @property
    def is_delete(self) -> bool:
        return '/dev/null' in self.dst
    
    @property
    def is_create(self) -> bool:
        return '/dev/null' in self.src
    
    @property
    def target_path(self) -> str:
        if self.is_delete:
            return self.src
        return self.dst


@dataclass
class HunkResult:
    """Результат применения одного хунка"""
    hunk: Hunk
    applied: bool = False
    matched_at: int = -1     # позиция где нашли (0-based)
    offset: int = 0          # смещение от ожидаемой позиции
    fuzz: int = 0            # использованный fuzz
    similarity: float = 1.0  # similarity ratio при fuzzy match
    error: str = ""


@dataclass
class FileResult:
    """Результат применения патча к файлу"""
    path: str
    operation: str = "unknown"  # 'create', 'modify', 'delete', 'skip', 'error'
    hunks_total: int = 0
    hunks_applied: int = 0
    hunks_failed: int = 0
    hunk_results: List[HunkResult] = field(default_factory=list)
    error: str = ""
    lines_before: int = 0
    lines_after: int = 0
    
    @property
    def success(self) -> bool:
        return self.hunks_failed == 0 and not self.error
    
    @property
    def partial(self) -> bool:
        return self.hunks_applied > 0 and self.hunks_failed > 0


# ═══════════════════════════════════════════════════════════════════════════════
# ПАРСЕР DIFF
# ═══════════════════════════════════════════════════════════════════════════════

class DiffParser:
    """Устойчивый парсер unified diff"""
    
    def __init__(self, text: str):
        self.lines = text.splitlines(keepends=False)
        self.pos = 0
        self.patches: List[FilePatch] = []
        self.warnings: List[str] = []
    
    def parse(self) -> List[FilePatch]:
        """Парсит весь diff, возвращает список патчей"""
        while self.pos < len(self.lines):
            try:
                if not self._try_parse_file():
                    self.pos += 1
            except Exception as e:
                self.warnings.append(f"Строка {self.pos + 1}: {e}")
                self.pos += 1
        return self.patches
    
    def _current(self) -> str:
        return self.lines[self.pos] if self.pos < len(self.lines) else ""
    
    def _peek(self, offset: int = 0) -> str:
        idx = self.pos + offset
        return self.lines[idx] if 0 <= idx < len(self.lines) else ""
    
    def _skip_junk(self):
        """Пропускаем служебные строки"""
        while self.pos < len(self.lines):
            line = self._current()
            if any(line.startswith(p) for p in SKIP_PREFIXES):
                self.pos += 1
            elif line.strip() == '':
                self.pos += 1
            else:
                break
    
    def _try_parse_file(self) -> bool:
        """Пытается распарсить патч файла, начиная с текущей позиции"""
        self._skip_junk()
        
        if self.pos >= len(self.lines):
            return False
        
        line = self._current()
        
        # Ищем --- заголовок
        if not line.startswith('--- '):
            return False
        
        src = self._clean_path(line[4:])
        self.pos += 1
        
        # Пропускаем мусор между --- и +++
        self._skip_junk()
        
        if self.pos >= len(self.lines):
            self.warnings.append(f"EOF после '--- {src}'")
            return False
        
        line = self._current()
        if not line.startswith('+++ '):
            self.warnings.append(f"Ожидался +++ после --- {src}")
            return False
        
        dst = self._clean_path(line[4:])
        self.pos += 1
        
        patch = FilePatch(src=src, dst=dst)
        
        # Парсим хунки
        while self.pos < len(self.lines):
            self._skip_junk()
            
            if self.pos >= len(self.lines):
                break
            
            line = self._current()
            
            if line.startswith('@@ '):
                hunk = self._parse_hunk()
                if hunk:
                    patch.hunks.append(hunk)
            elif line.startswith('--- '):
                # Следующий файл
                break
            else:
                break
        
        if patch.hunks or patch.is_create or patch.is_delete:
            self.patches.append(patch)
            return True
        
        return False
    
    def _parse_hunk(self) -> Optional[Hunk]:
        """Парсит один хунк"""
        header = self._current()
        m = HUNK_RE.match(header)
        
        if not m:
            self.warnings.append(f"Неверный хунк: {header[:50]}")
            self.pos += 1
            return None
        
        ostart = int(m.group(1))
        ocnt = int(m.group(2)) if m.group(2) else 1
        nstart = int(m.group(3))
        ncnt = int(m.group(4)) if m.group(4) else 1
        
        hunk = Hunk(ostart=ostart, ocnt=ocnt, nstart=nstart, ncnt=ncnt, header=header)
        self.pos += 1
        
        # Читаем строки хунка
        while self.pos < len(self.lines):
            line = self._current()
            
            # Конец хунка
            if line.startswith('@@ ') or line.startswith('--- ') or line.startswith('diff '):
                break
            
            # No newline маркер
            if line.startswith('\\ No newline') or line.startswith('\\'):
                if hunk.lines:
                    hunk.lines[-1].has_newline = False
                self.pos += 1
                continue
            
            # Обычная строка хунка
            if len(line) == 0:
                # Пустая строка = контекст
                hunk.lines.append(HunkLine(op=' ', text=''))
            elif line[0] in ' +-':
                hunk.lines.append(HunkLine(op=line[0], text=line[1:]))
            else:
                # Строка без префикса — скорее всего контекст
                hunk.lines.append(HunkLine(op=' ', text=line))
            
            self.pos += 1
        
        return hunk
    
    def _clean_path(self, raw: str) -> str:
        """Очищает путь от timestamp и пробелов"""
        raw = raw.strip()
        # Убираем timestamp
        if '\t' in raw:
            raw = raw.split('\t')[0].strip()
        return raw


# ═══════════════════════════════════════════════════════════════════════════════
# ПАТЧЕР
# ═══════════════════════════════════════════════════════════════════════════════

class SmartPatcher:
    """Умный патчер с частичным применением"""
    
    def __init__(self, root: str, strip: Optional[int] = None,
                 ignore_ws: bool = False, ignore_ws_change: bool = False,
                 fuzz: int = 3, max_offset: int = 1000,
                 min_similarity: float = 0.78, eol: str = 'auto'):
        self.root = root
        self.strip = strip
        self.ignore_ws = ignore_ws
        self.ignore_ws_change = ignore_ws_change
        self.fuzz = fuzz
        self.max_offset = max_offset
        self.min_similarity = min_similarity
        self.default_eol = '\r\n' if eol == 'crlf' else '\n' if eol == 'lf' else None
    
    def apply_patch(self, patch: FilePatch, dry_run: bool = False) -> FileResult:
        """Применяет патч к файлу"""
        target = self._normalize_path(patch.target_path)
        
        if not target:
            return FileResult(
                path="(unknown)", 
                operation="skip", 
                error="Не удалось определить путь"
            )
        
        full_path = os.path.join(self.root, target)
        
        # Инициализируем результат с операцией
        result = FileResult(
            path=target,
            operation="unknown",
            hunks_total=len(patch.hunks)
        )
        
        try:
            # Удаление
            if patch.is_delete:
                return self._do_delete(full_path, target, dry_run)
            
            # Чтение оригинала
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'r', encoding='utf-8', newline='') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    # Попробуем с другой кодировкой
                    with open(full_path, 'r', encoding='cp1251', newline='') as f:
                        content = f.read()
                
                orig_lines = content.splitlines(keepends=True)
                result.operation = 'modify'
            else:
                orig_lines = []
                result.operation = 'create'
            
            result.lines_before = len(orig_lines)
            
            # Применяем хунки
            new_lines, hunk_results = self._apply_hunks(orig_lines, patch.hunks)
            result.hunk_results = hunk_results
            result.hunks_applied = sum(1 for hr in hunk_results if hr.applied)
            result.hunks_failed = sum(1 for hr in hunk_results if not hr.applied)
            result.lines_after = len(new_lines)
            
            # Записываем результат
            if not dry_run:
                if result.hunks_applied > 0:
                    parent_dir = os.path.dirname(full_path)
                    if parent_dir and not os.path.exists(parent_dir):
                        os.makedirs(parent_dir, exist_ok=True)
                    
                    with open(full_path, 'w', encoding='utf-8', newline='') as f:
                        f.writelines(new_lines)
                
                # Сохраняем неприменённые хунки
                if result.hunks_failed > 0:
                    self._save_rejects(full_path, patch, hunk_results)
            
        except Exception as e:
            result.error = str(e)
            result.operation = 'error'
        
        return result
    
    def _do_delete(self, full_path: str, target: str, dry_run: bool) -> FileResult:
        """Удаление файла"""
        result = FileResult(path=target, operation='delete')
        
        if not os.path.exists(full_path):
            result.error = "Файл уже отсутствует"
            return result
        
        if not dry_run:
            try:
                os.remove(full_path)
            except Exception as e:
                result.error = str(e)
        
        return result
    
    def _apply_hunks(self, orig_lines: List[str], hunks: List[Hunk]) -> Tuple[List[str], List[HunkResult]]:
        """Применяет хунки последовательно"""
        lines = list(orig_lines)
        results: List[HunkResult] = []
        cumulative_offset = 0
        
        for hunk in hunks:
            hr = self._apply_single_hunk(lines, hunk, cumulative_offset)
            results.append(hr)
            
            if hr.applied:
                # Обновляем смещение
                added = sum(1 for hl in hunk.lines if hl.op == '+')
                removed = sum(1 for hl in hunk.lines if hl.op == '-')
                cumulative_offset += (added - removed)
        
        return lines, results
    
    def _apply_single_hunk(self, lines: List[str], hunk: Hunk, cum_offset: int) -> HunkResult:
        """Пытается применить один хунк"""
        old_lines = hunk.get_old_lines()
        expected_pos = max(0, hunk.ostart - 1 + cum_offset)
        
        # Пустой хунк (только добавления)
        if not old_lines:
            pos = min(expected_pos, len(lines))
            # Проверка на дублирование
            if self._would_create_duplicate(lines, hunk, pos):
                return HunkResult(
                    hunk=hunk, applied=False,
                    error="Хунк уже применён (обнаружено дублирование)"
                )
            self._do_apply(lines, hunk, pos)
            return HunkResult(hunk=hunk, applied=True, matched_at=pos)
        
        # Проверка на дублирование перед применением
        if self._would_create_duplicate(lines, hunk, expected_pos):
            return HunkResult(
                hunk=hunk, applied=False,
                error="Хунк уже применён (обнаружено дублирование)"
            )
        
        # Стратегия 1: точный поиск с fuzz
        for fuzz_level in range(self.fuzz + 1):
            trimmed = self._trim_context(old_lines, fuzz_level)
            if not trimmed:
                continue
            
            pos = self._search_exact(lines, trimmed, expected_pos)
            if pos is not None:
                # Дополнительная проверка на дублирование
                if self._would_create_duplicate(lines, hunk, pos):
                    return HunkResult(
                        hunk=hunk, applied=False,
                        error="Хунк уже применён (обнаружено дублирование)"
                    )
                self._do_apply(lines, hunk, pos, fuzz_level)
                return HunkResult(
                    hunk=hunk, applied=True, matched_at=pos,
                    offset=pos - expected_pos, fuzz=fuzz_level
                )
        
        # Стратегия 2: Fuzzy matching
        pos, similarity = self._search_fuzzy(lines, old_lines, expected_pos)
        if pos is not None and similarity >= self.min_similarity:
            # Дополнительная проверка на дублирование при fuzzy match
            if self._would_create_duplicate(lines, hunk, pos):
                return HunkResult(
                    hunk=hunk, applied=False,
                    error="Хунк уже применён (fuzzy обнаружил дублирование)"
                )
            self._do_apply(lines, hunk, pos, fuzz=0)
            return HunkResult(
                hunk=hunk, applied=True, matched_at=pos,
                offset=pos - expected_pos, similarity=similarity
            )
        
        # Не удалось применить
        return HunkResult(
            hunk=hunk, applied=False,
            error=f"Контекст не найден (строка {hunk.ostart}, fuzzy max={similarity:.0%})"
        )
    
    def _trim_context(self, old_lines: List[HunkLine], fuzz: int) -> List[HunkLine]:
        """Обрезает контекстные строки по краям"""
        if fuzz == 0:
            return old_lines
        
        # Находим индексы контекстных строк
        ctx = [(i, hl) for i, hl in enumerate(old_lines) if hl.op == ' ']
        if len(ctx) < fuzz * 2:
            return old_lines
        
        to_remove = set()
        # С начала
        for i in range(min(fuzz, len(ctx))):
            to_remove.add(ctx[i][0])
        # С конца
        for i in range(max(0, len(ctx) - fuzz), len(ctx)):
            to_remove.add(ctx[i][0])
        
        result = [hl for i, hl in enumerate(old_lines) if i not in to_remove]
        return result if result else old_lines
    
    def _search_exact(self, lines: List[str], pattern: List[HunkLine], expected: int) -> Optional[int]:
        """Точный поиск с расширяющимся радиусом"""
        max_pos = len(lines) - len(pattern) + 1
        
        # Сначала проверяем ожидаемую позицию
        if 0 <= expected <= max_pos:
            if self._matches_at(lines, pattern, expected):
                return expected
        
        # Расширяющийся поиск
        for delta in range(1, self.max_offset + 1):
            for candidate in [expected - delta, expected + delta]:
                if 0 <= candidate <= max_pos:
                    if self._matches_at(lines, pattern, candidate):
                        return candidate
        
        return None

    def _matches_at(self, lines: List[str], pattern: List[HunkLine], pos: int) -> bool:
        """Проверяет совпадение в позиции"""
        if pos < 0 or pos + len(pattern) > len(lines):
            return False
        
        for i, hl in enumerate(pattern):
            line_content = self._strip_eol(lines[pos + i])
            pattern_content = hl.text
            
            if not self._lines_equal(line_content, pattern_content):
                return False
        
        return True
    
    def _search_fuzzy(self, lines: List[str], pattern: List[HunkLine], expected: int) -> Tuple[Optional[int], float]:
        """Fuzzy поиск с SequenceMatcher"""
        if not pattern:
            return None, 0.0
        
        pattern_text = '\n'.join(hl.text for hl in pattern)
        best_pos = None
        best_ratio = 0.0
        
        # Ограниченное окно поиска
        window_size = len(pattern)
        start = max(0, expected - self.max_offset)
        end = min(len(lines) - window_size + 1, expected + self.max_offset)
        
        for pos in range(start, max(start, end)):
            window_text = '\n'.join(self._strip_eol(lines[pos + i]) for i in range(window_size))
            ratio = SequenceMatcher(None, pattern_text, window_text, autojunk=False).ratio()
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_pos = pos
        
        return best_pos, best_ratio
    
    def _would_create_duplicate(self, lines: List[str], hunk: Hunk, pos: int) -> bool:
        """Проверяет, не приведёт ли применение хунка к дублированию строк"""
        add_lines = [hl.text for hl in hunk.lines if hl.op == '+']
        if not add_lines:
            return False
        
        # Минимум 2 строки для проверки паттерна
        if len(add_lines) < 2:
            return False
        
        # Проверяем: не содержатся ли добавляемые строки уже рядом с позицией
        check_start = max(0, pos - 5)
        check_end = min(len(lines), pos + len(hunk.get_old_lines()) + len(add_lines) + 5)
        
        existing_text = [self._strip_eol(lines[i]) for i in range(check_start, check_end)]
        
        # Ищем последовательность добавляемых строк в существующем тексте
        for i in range(len(existing_text) - len(add_lines) + 1):
            match_count = 0
            for j, add_line in enumerate(add_lines):
                if i + j < len(existing_text) and self._lines_equal(existing_text[i + j], add_line):
                    match_count += 1
            # Если 80%+ добавляемых строк уже есть подряд — это дубликат
            if match_count >= len(add_lines) * 0.8:
                return True
        
        return False
    
    def _do_apply(self, lines: List[str], hunk: Hunk, pos: int, fuzz: int = 0):
        """Применяет хунк к lines (in-place)"""
        eol = self._detect_eol(lines)
        
        new_segment: List[str] = []
        src_idx = pos
        
        for hl in hunk.lines:
            if hl.op == ' ':
                # Контекст — берём из оригинала (сохраняем форматирование)
                if src_idx < len(lines):
                    new_segment.append(lines[src_idx])
                src_idx += 1
            elif hl.op == '-':
                # Удаление
                src_idx += 1
            elif hl.op == '+':
                # Добавление
                text = hl.text + (eol if hl.has_newline else '')
                new_segment.append(text)
        
        # Заменяем сегмент
        old_len = src_idx - pos
        lines[pos:pos + old_len] = new_segment
    
    def _lines_equal(self, a: str, b: str) -> bool:
        """Сравнение строк с учётом настроек"""
        if self.ignore_ws:
            return self._no_ws(a) == self._no_ws(b)
        if self.ignore_ws_change:
            return self._norm_ws(a) == self._norm_ws(b)
        return a == b
    
    def _no_ws(self, s: str) -> str:
        return ''.join(c for c in s if not c.isspace())
    
    def _norm_ws(self, s: str) -> str:
        return ' '.join(s.split())
    
    def _strip_eol(self, s: str) -> str:
        return s.rstrip('\r\n')
    
    def _detect_eol(self, lines: List[str]) -> str:
        for line in lines:
            if line.endswith('\r\n'):
                return '\r\n'
            if line.endswith('\n'):
                return '\n'
        return self.default_eol or '\r\n'
    
    def _normalize_path(self, raw: str) -> str:
        """Нормализует путь из diff"""
        path = raw.strip()
        if path == '/dev/null':
            return ''
        
        # Убираем timestamp
        if '\t' in path:
            path = path.split('\t')[0].strip()
        
        path = path.replace('\\', '/')
        
        if self.strip is not None:
            parts = path.split('/')
            path = '/'.join(parts[self.strip:])
        else:
            # Авто: убираем a/ или b/
            if path.startswith('a/') or path.startswith('b/'):
                path = path[2:]
        
        return path
    
    def _save_rejects(self, orig_path: str, patch: FilePatch, results: List[HunkResult]):
        """Сохраняет неприменённые хунки в .rej файл"""
        failed = [(hr.hunk, hr.error) for hr in results if not hr.applied]
        if not failed:
            return
        
        rej_path = orig_path + '.rej'
        try:
            with open(rej_path, 'w', encoding='utf-8') as f:
                f.write(f"# Отклонённые хунки для: {patch.target_path}\n")
                f.write(f"# Время: {datetime.now()}\n\n")
                f.write(f"--- {patch.src}\n")
                f.write(f"+++ {patch.dst}\n")
                
                for hunk, error in failed:
                    f.write(f"\n# ОШИБКА: {error}\n")
                    f.write(hunk.header + '\n')
                    for hl in hunk.lines:
                        f.write(f"{hl.op}{hl.text}\n")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# ВВОД/ВЫВОД
# ═══════════════════════════════════════════════════════════════════════════════

def read_diff_input(path: Optional[str]) -> str:
    """Читает diff из файла или интерактивно"""
    if path:
        # Пробуем разные кодировки
        for enc in ['utf-8', 'cp1251', 'latin-1']:
            try:
                with open(path, 'r', encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Не удалось прочитать файл {path}")
    
    # Интерактивный режим
    try:
        is_tty = sys.stdin.isatty()
    except Exception:
        is_tty = False
    
    if is_tty:
        print('╔════════════════════════════════════════════════════════════╗')
        print('║  ВСТАВЬТЕ UNIFIED DIFF НИЖЕ                                ║')
        print('║  Для завершения: точка (.) на отдельной строке или Ctrl+Z  ║')
        print('╚════════════════════════════════════════════════════════════╝')
        print()
        
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip() in ('.', 'END', 'EOF', '---END---'):
                break
            lines.append(line + '\n')
        return ''.join(lines)
    
    return sys.stdin.read()


def extract_diffs_from_message(text: str) -> str:
    """
    Извлекает unified diffs из сообщения чат-бота.
    
    Поддерживает:
    - Markdown блоки ```diff ... ```
    - Обычный текст с несколькими diff'ами
    - Смешанный формат (markdown + текст между блоками)
    """
    # Ищем markdown блоки с diff
    matches = MARKDOWN_DIFF_BLOCK_RE.findall(text)
    
    if matches:
        # Нашли markdown блоки — объединяем их
        extracted = '\n\n'.join(matches)
        return extracted
    
    # Проверяем, похоже ли это на обычный diff (есть --- и +++)
    if '--- ' in text and '+++ ' in text:
        return text
    
    # Может быть блоки без слова diff: ```\n--- a/file...```
    generic_block_re = re.compile(r'```\s*\n(---\s+\S.*?)```', re.DOTALL)
    generic_matches = generic_block_re.findall(text)
    
    if generic_matches:
        return '\n\n'.join(generic_matches)
    
    return text


def print_summary(results: List[FileResult], dry_run: bool):
    """Выводит итоговый отчёт"""
    print()
    print('═' * 70)
    print('                         ИТОГОВЫЙ ОТЧЁТ')
    print('═' * 70)
    
    total = len(results)
    success = sum(1 for r in results if r.success)
    partial = sum(1 for r in results if r.partial)
    failed = sum(1 for r in results if not r.success and not r.partial)
    
    total_hunks = sum(r.hunks_total for r in results)
    applied_hunks = sum(r.hunks_applied for r in results)
    failed_hunks = sum(r.hunks_failed for r in results)
    
    if dry_run:
        print('                      *** РЕЖИМ ПРОВЕРКИ ***')
        print()
    
    print(f'Файлов: {total}  |  OK: {success}  |  Частично: {partial}  |  Ошибка: {failed}')
    print(f'Хунков: {total_hunks}  |  Применено: {applied_hunks}  |  Отклонено: {failed_hunks}')
    print()
    
    # Детали по файлам
    for r in results:
        if r.success:
            icon = '[OK]'
        elif r.partial:
            icon = '[~~]'
        else:
            icon = '[XX]'
        
        stats = f'{r.hunks_applied}/{r.hunks_total}' if r.hunks_total else '-'
        op = r.operation.upper()[:6]
        print(f' {icon} [{op:6}] {r.path}  (хунки: {stats})')
        
        if r.error:
            print(f'           Ошибка: {r.error}')
        
        # Детали хунков (только если есть проблемы или интересные детали)
        for i, hr in enumerate(r.hunk_results, 1):
            if hr.applied:
                extras = []
                if hr.offset != 0:
                    extras.append(f'offset {hr.offset:+d}')
                if hr.fuzz > 0:
                    extras.append(f'fuzz {hr.fuzz}')
                if hr.similarity < 1.0:
                    extras.append(f'sim {hr.similarity:.0%}')
                if extras:
                    print(f'           хунк {i}: OK ({", ".join(extras)})')
            else:
                print(f'           хунк {i}: FAIL - {hr.error}')
    
    print()
    print('═' * 70)
    
    if failed_hunks > 0 and not dry_run:
        print('  Неприменённые хунки сохранены в .rej файлы')
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    
    ap = argparse.ArgumentParser(
        description='Умный патчер unified diff v2.2',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('diff', nargs='?', help='Файл с diff')
    ap.add_argument('-p', '--strip', type=int, default=None,
                    help='Удалить N компонентов пути (как patch -pN)')
    ap.add_argument('--dry-run', '-n', action='store_true',
                    help='Только проверка, без записи')
    ap.add_argument('--root', '-d', default=script_dir,
                    help=f'Корневая папка')
    ap.add_argument('--fuzz', '-F', type=int, default=3,
                    help='Уровень fuzz 0-5 (default: 3)')
    ap.add_argument('--max-offset', type=int, default=500,
                    help='Макс. смещение поиска (default: 500)')
    ap.add_argument('-w', '--ignore-whitespace', action='store_true',
                    help='Игнорировать пробелы')
    ap.add_argument('-b', '--ignore-space-change', action='store_true',
                    help='Игнорировать изменения пробелов')
    ap.add_argument('--eol', choices=['auto', 'lf', 'crlf'], default='auto',
                    help='Стиль переводов строк')
    
    args = ap.parse_args()
    
    # Читаем diff
    print()
    print('Чтение diff...')
    try:
        text = read_diff_input(args.diff)
    except FileNotFoundError:
        print(f'\n[ОШИБКА] Файл не найден: {args.diff}')
        return 1
    except Exception as e:
        print(f'\n[ОШИБКА] Не удалось прочитать: {e}')
        return 1
    
    if not text.strip():
        print('\n[ОШИБКА] Diff пуст')
        return 1
    
    # Извлекаем diff из markdown-сообщения (если это сообщение от чат-бота)
    text = extract_diffs_from_message(text)
    
    # Парсим
    print('Парсинг diff...')
    parser = DiffParser(text)
    patches = parser.parse()
    
    if parser.warnings:
        print(f'\nПредупреждения ({len(parser.warnings)}):')
        for w in parser.warnings[:5]:
            print(f'  ! {w}')
        if len(parser.warnings) > 5:
            print(f'  ... и ещё {len(parser.warnings) - 5}')
    
    if not patches:
        print('\n[ОШИБКА] Патчи не найдены!')
        print('Убедитесь, что это unified diff (--- / +++ / @@)')
        return 1
    
    print(f'\nНайдено: {len(patches)} файл(ов)')
    total_hunks = sum(len(p.hunks) for p in patches)
    print(f'Всего хунков: {total_hunks}')
    
    if args.dry_run:
        print('\n*** РЕЖИМ ПРОВЕРКИ — файлы НЕ изменяются ***')
    print()
    
    # Применяем
    patcher = SmartPatcher(
        root=args.root,
        strip=args.strip,
        ignore_ws=args.ignore_whitespace,
        ignore_ws_change=args.ignore_space_change,
        fuzz=args.fuzz,
        max_offset=args.max_offset,
        eol=args.eol
    )
    
    results: List[FileResult] = []
    
    for i, patch in enumerate(patches, 1):
        target = patcher._normalize_path(patch.target_path)
        print(f'[{i}/{len(patches)}] {target}... ', end='', flush=True)
        
        result = patcher.apply_patch(patch, dry_run=args.dry_run)
        results.append(result)
        
        if result.success:
            print(f'OK ({result.hunks_applied} хунков)')
        elif result.partial:
            print(f'ЧАСТИЧНО ({result.hunks_applied}/{result.hunks_total})')
        else:
            err_msg = result.error or "все хунки отклонены"
            print(f'ОШИБКА: {err_msg}')
    
    # Итоговый отчёт
    print_summary(results, args.dry_run)
    
    # Код возврата
    if all(r.success for r in results):
        return 0
    elif any(r.hunks_applied > 0 for r in results):
        return 1  # Частичный успех
    else:
        return 2  # Полный провал


if __name__ == '__main__':
    exit_code = 0
    
    try:
        exit_code = main()
    except KeyboardInterrupt:
        print('\n\n[Прервано]')
        exit_code = 130
    except Exception as e:
        print()
        print('╔══════════════════════════════════════════════════════════╗')
        print('║  ВНУТРЕННЯЯ ОШИБКА ПАТЧЕРА                               ║')
        print('╚══════════════════════════════════════════════════════════╝')
        print()
        print(f'Ошибка: {e}')
        print()
        traceback.print_exc()
        exit_code = 99
    
    # Пауза при интерактивном запуске (двойной клик в Windows)
    try:
        if sys.stdin and sys.stdin.isatty():
            print()
            input('>>> Нажмите Enter для выхода <<<')
    except Exception:
        pass
    
    sys.exit(exit_code)