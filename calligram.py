"""
Gerador de Caligramas Aleatórios
=================================
Seleciona uma imagem aleatória da pasta Images/ e um trecho aleatório de um
livro EPUB da pasta Livros/, extrai bordas e volumes da imagem via OpenCV e
renderiza o texto caractere a caractere para formar um caligrama.

Atualiza automaticamente a cada 5 segundos com novas combinações.
"""

import os
import re
import sys
import math
import time
import random
import threading
import textwrap
import unicodedata
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import tkinter as tk
from tkinter import ttk

# ─── Configurações ────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "Images"
LIVROS_DIR = BASE_DIR / "Livros"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

CANVAS_W   = 900          # largura do canvas de saída
CANVAS_H   = 900          # altura do canvas de saída
# INTERVAL   = 5000         # ms entre gerações
INTERVAL   = 500000         # ms entre gerações
MIN_TEXT   = 800          # mín. de caracteres para o trecho
MAX_TEXT   = 4000         # máx. de caracteres para o trecho

BASE_FONT_SIZE     =  24   # tamanho inicial da fonte (ajustável com +/-)
FONT_SIZE_MIN      = 6
FONT_SIZE_MAX      = 36

# Pesos das máscaras (bordas vs. volumes)
EDGE_WEIGHT   = 0.45
VOLUME_WEIGHT = 0.55

# Fontes (tenta usar uma monoespaçada do sistema; cai para default PIL)
FONT_CANDIDATES = [
    "C:/Windows/Fonts/consola.ttf",   # Consolas
    "C:/Windows/Fonts/cour.ttf",      # Courier New
    "C:/Windows/Fonts/lucon.ttf",     # Lucida Console
    "C:/Windows/Fonts/DejaVuSansMono.ttf",
]

# ─── Utilitários de Fonte ──────────────────────────────────────────────────────

def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


# ─── Extração de Texto dos EPUBs ──────────────────────────────────────────────

def _epub_text(epub_path: Path) -> str:
    """Retorna todo o texto de um EPUB como string limpa."""
    book = epub.read_epub(str(epub_path), options={"ignore_ncx": True})
    chunks = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text(separator=" ")
        # Normaliza espaços e hífens de quebra de linha
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 50:
            chunks.append(text)
    return " ".join(chunks)


def get_random_excerpt(min_len: int = MIN_TEXT, max_len: int = MAX_TEXT) -> tuple[str, str]:
    """Escolhe um livro aleatório e retorna (trecho, título_do_livro)."""
    epubs = list(LIVROS_DIR.glob("*.epub"))
    if not epubs:
        raise FileNotFoundError("Nenhum arquivo .epub encontrado em Livros/")
    chosen = random.choice(epubs)
    full_text = _epub_text(chosen)

    # Remove caracteres de controle
    full_text = "".join(
        c for c in full_text
        if unicodedata.category(c)[0] != "C" or c in "\n "
    )
    full_text = re.sub(r"\s+", " ", full_text).strip()

    if len(full_text) < min_len:
        excerpt = full_text
    else:
        max_start = max(0, len(full_text) - max_len)
        start = random.randint(0, max_start)
        length = random.randint(min_len, min(max_len, len(full_text) - start))
        excerpt = full_text[start : start + length]

    title = chosen.stem
    return excerpt, title


# ─── Processamento de Imagem ──────────────────────────────────────────────────

def build_masks(image_path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Retorna (gray, edge_mask, volume_mask) normalizados para [0,1].
    edge_mask   → contornos da imagem (Canny)
    volume_mask → regiões densas/escuras (threshold adaptativo + morfologia)
    """
    raw = np.frombuffer(Path(image_path).read_bytes(), dtype=np.uint8)
    img_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Não foi possível abrir a imagem: {image_path}")

    img_bgr = cv2.resize(img_bgr, (width, height), interpolation=cv2.INTER_AREA)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # --- Bordas (Canny) ---
    blurred    = cv2.GaussianBlur(gray, (3, 3), 0)
    edges      = cv2.Canny(blurred, 30, 100)
    kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_d    = cv2.dilate(edges, kernel, iterations=1)
    edge_mask  = edges_d.astype(np.float32) / 255.0

    # --- Volumes ---
    inv_gray   = cv2.bitwise_not(gray)
    blur2      = cv2.GaussianBlur(inv_gray, (7, 7), 0)
    _, thresh  = cv2.threshold(blur2, 60, 255, cv2.THRESH_BINARY)
    kernel2    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    filled     = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel2, iterations=2)
    vol_mask   = filled.astype(np.float32) / 255.0

    return gray, edge_mask, vol_mask


def combined_mask(edge_mask: np.ndarray, vol_mask: np.ndarray) -> np.ndarray:
    """Combina bordas e volumes numa única máscara [0,1]."""
    combined = np.clip(
        EDGE_WEIGHT * edge_mask + VOLUME_WEIGHT * vol_mask, 0, 1
    )
    return combined


# ─── Renderização do Caligrama ────────────────────────────────────────────────

_font_cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}

def _load_font_cached(size: int):
    if size not in _font_cache:
        _font_cache[size] = _load_font(size)
    return _font_cache[size]


class CharCell:
    """
    Representa uma letra do caligrama com sua posição original, cor,
    tamanho de fonte, fases aleatórias e estado físico para inércia.
    """
    __slots__ = ("x", "y", "char", "fsize", "color", "px", "py", "ox", "oy", "vx", "vy")

    def __init__(self, x, y, char, fsize, color):
        self.x     = x
        self.y     = y
        self.char  = char
        self.fsize = fsize
        self.color = color
        self.px = random.uniform(0, 2 * 3.14159)
        self.py = random.uniform(0, 2 * 3.14159)
        self.ox = 0.0
        self.oy = 0.0
        self.vx = 0.0
        self.vy = 0.0


def build_char_layout(
    image_path: Path,
    text: str,
    width: int = CANVAS_W,
    height: int = CANVAS_H,
    font_size: int = None,
) -> list:
    """
    Processa a imagem e o texto e retorna uma lista de CharCell.
    """
    if font_size is None:
        font_size = BASE_FONT_SIZE

    gray, edge_mask, vol_mask = build_masks(image_path, width, height)
    mask = combined_mask(edge_mask, vol_mask)

    img_pil = Image.open(image_path).convert("RGB").resize((width, height), Image.LANCZOS)
    img_arr = np.array(img_pil)

    base_font_size = font_size
    line_height    = base_font_size + 2
    char_width     = max(1, int(base_font_size * 0.62))

    cells    = []
    text_idx = 0
    text_len = len(text)

    for row_px in range(0, height - line_height, line_height):
        for col_px in range(0, width - char_width, char_width):
            if text_idx >= text_len:
                text_idx = 0

            intensity = float(mask[row_px + line_height // 2, col_px + char_width // 2])

            if intensity < 0.15:
                continue

            char = text[text_idx]
            text_idx += 1

            if char == " " and intensity < 0.3:
                continue

            if intensity > 0.7:
                fsize = base_font_size - 1
            elif intensity > 0.4:
                fsize = base_font_size
            else:
                fsize = base_font_size + 1

            r, g, b = img_arr[row_px + line_height // 2, col_px + char_width // 2]
            factor  = max(0.1, 1.0 - intensity * 0.7)
            color   = (int(r * factor), int(g * factor), int(b * factor))

            cells.append(CharCell(col_px, row_px, char, fsize, color))

    return cells


def draw_chars(
    cells: list,
    title: str,
    width: int = CANVAS_W,
    height: int = CANVAS_H,
    motion_level: float = 0.0,
) -> Image.Image:
    """
    Desenha todas as CharCell num canvas branco com inércia física e
    um tremulado mínimo constante, impedindo que o movimento pare totalmente.
    """
    canvas = Image.new("RGB", (width, height), "white")
    draw   = ImageDraw.Draw(canvas)

    t = time.time()

    # Define um nível mínimo de movimento constante (ex: 0.03 = 3% de agitação contínua)
    # Se a webcam estiver parada (0.0), o sistema usará 0.03 como base.
    min_motion = 0.022
    continuous_level = max(motion_level, min_motion)

    target_amplitude = continuous_level * 12.0

    for cell in cells:
        # Como o nível mínimo nunca é zero, o alvo da onda sempre existirá
        target_ox = target_amplitude * math.sin(t * 4.0 + cell.px)
        target_oy = target_amplitude * math.cos(t * 3.0 + cell.py)

        spring = 0.12   # Rigidez da mola
        damping = 0.82  # Atrito

        ax = (target_ox - cell.ox) * spring
        ay = (target_oy - cell.oy) * spring

        cell.vx = (cell.vx + ax) * damping
        cell.vy = (cell.vy + ay) * damping

        cell.ox += cell.vx
        cell.oy += cell.vy

        font = _load_font_cached(cell.fsize)
        draw.text(
            (cell.x + cell.ox, cell.y + cell.oy),
            cell.char,
            font=font,
            fill=cell.color,
        )

    # Legenda no rodapé
    legend_font = _load_font_cached(11)
    draw.rectangle([(0, height - 26), (width, height)], fill=(240, 240, 240))
    draw.text((10, height - 22), f"📖 {title}", font=legend_font, fill=(60, 60, 60))

    return canvas


def render_calligram(
    image_path: Path,
    text: str,
    title: str,
    width: int = CANVAS_W,
    height: int = CANVAS_H,
    font_size: int = None,
) -> tuple:
    cells = build_char_layout(image_path, text, width, height, font_size)
    img   = draw_chars(cells, title, width, height, motion_level=0.0)
    return img, cells


# ─── Detecção de Movimento (Webcam) ───────────────────────────────────────────

class WebcamMotionDetector:
    _AMPLIFY = 18.0

    def __init__(self):
        self._cap         = None
        self._thread      = None
        self._running     = False
        self._level       = 0.0
        self._lock        = threading.Lock()
        self._bg_sub      = cv2.createBackgroundSubtractorMOG2(
            history=30, varThreshold=40, detectShadows=False
        )

    def start(self, camera_index: int = 0) -> bool:
        self._cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            return False
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True, name="webcam-motion")
        self._thread.start()
        return True

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None

    @property
    def level(self) -> float:
        with self._lock:
            return min(1.0, self._level * self._AMPLIFY)

    @property
    def raw_level(self) -> float:
        with self._lock:
            return self._level

    def _loop(self):
        while self._running and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            small   = cv2.resize(frame, (320, 240))
            fg_mask = self._bg_sub.apply(small)

            raw = float(np.count_nonzero(fg_mask)) / fg_mask.size

            with self._lock:
                # Suavização exponencial com slow out gradual ao parar
                self._level = self._level * 0.85 + raw * 0.15


# ─── Efeito de Distúrbio ──────────────────────────────────────────────────────

def apply_disturbance(base_img: Image.Image, level: float) -> Image.Image:
    if level < 0.02:
        return base_img

    arr = np.array(base_img, dtype=np.uint8)
    h, w = arr.shape[:2]
    t    = time.time()

    amplitude = level * 35.0
    freq_y    = 3.5 + level * 2.0
    freq_x    = 2.0 + level * 1.5

    ys = np.arange(h, dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)
    xmap, ymap = np.meshgrid(xs, ys)

    xmap = xmap + amplitude * np.sin(
        2.0 * np.pi * (ymap / h * freq_y + t * 0.7)
    )
    ymap = ymap + (amplitude * 0.4) * np.cos(
        2.0 * np.pi * (xmap / w * freq_x + t * 0.5)
    )

    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    warped  = cv2.remap(
        arr_bgr,
        xmap.astype(np.float32),
        ymap.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    shift = max(1, int(level * 10))
    b, g, r = cv2.split(warped)
    b = np.roll(b,  shift, axis=1)
    r = np.roll(r, -shift, axis=1)
    warped = cv2.merge([b, g, r])

    if level > 0.6:
        k = 3
        warped = cv2.blur(warped, (k, 1))

    result = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result)


# ─── Interface Tkinter ────────────────────────────────────────────────────────

class CalligramApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🖋 Gerador de Caligramas — Borges")
        self.resizable(False, False)
        self.configure(bg="#1a1a2e")

        self._status_var = tk.StringVar(value="Iniciando…")
        status_bar = tk.Label(
            self,
            textvariable=self._status_var,
            bg="#16213e",
            fg="#e2e2e2",
            font=("Consolas", 10),
            anchor="w",
            padx=10,
            pady=4,
        )
        status_bar.pack(fill=tk.X, side=tk.TOP)

        self._label = tk.Label(self, bg="#1a1a2e", cursor="watch")
        self._label.pack(padx=12, pady=8)

        ctrl_frame = tk.Frame(self, bg="#16213e", pady=6)
        ctrl_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self._paused = False
        self._pause_btn = tk.Button(
            ctrl_frame,
            text="⏸  Pausar",
            command=self._toggle_pause,
            bg="#0f3460",
            fg="white",
            relief=tk.FLAT,
            font=("Consolas", 10, "bold"),
            padx=14,
            pady=4,
        )
        self._pause_btn.pack(side=tk.LEFT, padx=12)

        skip_btn = tk.Button(
            ctrl_frame,
            text="⏭  Próximo",
            command=self._skip,
            bg="#533483",
            fg="white",
            relief=tk.FLAT,
            font=("Consolas", 10, "bold"),
            padx=14,
            pady=4,
        )
        skip_btn.pack(side=tk.LEFT, padx=4)

        self._webcam_btn = tk.Button(
            ctrl_frame,
            text="📷  Webcam",
            command=self._toggle_webcam,
            bg="#1a3a1a",
            fg="#88cc88",
            relief=tk.FLAT,
            font=("Consolas", 10, "bold"),
            padx=14,
            pady=4,
        )
        self._webcam_btn.pack(side=tk.LEFT, padx=4)

        self._motion_canvas = tk.Canvas(
            ctrl_frame, width=60, height=16,
            bg="#16213e", highlightthickness=0,
        )
        self._motion_canvas.pack(side=tk.LEFT, padx=4)
        self._motion_bar = self._motion_canvas.create_rectangle(
            0, 2, 0, 14, fill="#44ff88", outline=""
        )

        self._font_label = tk.Label(
            ctrl_frame,
            text=f"A  {BASE_FONT_SIZE}pt",
            bg="#16213e",
            fg="#aaaacc",
            font=("Consolas", 10),
            padx=10,
        )
        self._font_label.pack(side=tk.RIGHT, padx=12)

        tk.Label(
            ctrl_frame,
            text="+ / −  tamanho",
            bg="#16213e",
            fg="#555577",
            font=("Consolas", 9),
        ).pack(side=tk.RIGHT, padx=4)

        self._timer_id       = None
        self._tk_image       = None
        self._counter        = 0
        self._font_size      = BASE_FONT_SIZE

        self._webcam_active  = False
        self._detector       = WebcamMotionDetector()
        self._base_calligram = None
        self._char_cells     = []
        self._char_title     = ""
        self._disturb_id     = None

        self.bind("<plus>",        lambda e: self._change_font(+1))
        self.bind("<equal>",       lambda e: self._change_font(+1))
        self.bind("<KP_Add>",      lambda e: self._change_font(+1))
        self.bind("<minus>",       lambda e: self._change_font(-1))
        self.bind("<KP_Subtract>", lambda e: self._change_font(-1))

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._generate)

    def _generate(self):
        images = (
            list(IMAGES_DIR.glob("*.png"))
            + list(IMAGES_DIR.glob("*.jpg"))
            + list(IMAGES_DIR.glob("*.jpeg"))
        )
        if not images:
            self._status_var.set("❌ Nenhuma imagem encontrada em Images/")
            return

        image_path = random.choice(images)
        self._status_var.set(f"⏳ Gerando… [{image_path.name}]")
        self.update_idletasks()

        try:
            excerpt, title = get_random_excerpt()
            calligram, cells = render_calligram(
                image_path, excerpt, title, font_size=self._font_size
            )

            self._char_cells  = cells
            self._char_title  = title
            self._base_calligram = calligram

            self._counter += 1
            out_name = OUTPUT_DIR / f"caligrama_{self._counter:04d}.png"
            calligram.save(str(out_name))

            from PIL import ImageTk
            self._tk_image = ImageTk.PhotoImage(calligram)
            self._label.configure(image=self._tk_image)
            self._status_var.set(
                f"✅  [{self._counter}]  {image_path.name}  |  📖 {title}  |  "
                f"{len(excerpt)} chars  |  próximo em {INTERVAL // 1000}s"
            )
        except Exception as exc:
            self._status_var.set(f"❌ Erro: {exc}")
            import traceback
            traceback.print_exc()

        if not self._paused:
            self._timer_id = self.after(INTERVAL, self._generate)

    def _toggle_pause(self):
        self._paused = not self._paused
        if self._paused:
            if self._timer_id:
                self.after_cancel(self._timer_id)
                self._timer_id = None
            self._pause_btn.configure(text="▶  Retomar", bg="#2c6e49")
            self._status_var.set("⏸  Pausado")
        else:
            self._pause_btn.configure(text="⏸  Pausar", bg="#0f3460")
            self._generate()

    def _change_font(self, delta: int):
        new_size = self._font_size + delta
        new_size = max(FONT_SIZE_MIN, min(FONT_SIZE_MAX, new_size))
        if new_size == self._font_size:
            return
        self._font_size = new_size
        self._font_label.configure(text=f"A  {new_size}pt")
        self._skip()

    def _skip(self):
        if self._timer_id:
            self.after_cancel(self._timer_id)
            self._timer_id = None
        self._paused = False
        self._pause_btn.configure(text="⏸  Pausar", bg="#0f3460")
        self._generate()

    def _toggle_webcam(self):
        if not self._webcam_active:
            self._status_var.set("⏳ Abrindo câmera…")
            self.update_idletasks()
            ok = self._detector.start()
            if not ok:
                self._status_var.set("❌ Câmera não encontrada (verifique a webcam)")
                return
            self._webcam_active = True
            self._webcam_btn.configure(text="🔴  Webcam ON", bg="#5a1a1a", fg="#ff8888")
            self._disturbance_loop()
        else:
            self._webcam_active = False
            self._detector.stop()
            if self._disturb_id:
                self.after_cancel(self._disturb_id)
                self._disturb_id = None
            self._webcam_btn.configure(text="📷  Webcam", bg="#1a3a1a", fg="#88cc88")
            if self._base_calligram:
                from PIL import ImageTk
                self._tk_image = ImageTk.PhotoImage(self._base_calligram)
                self._label.configure(image=self._tk_image)
            self._motion_canvas.coords(self._motion_bar, 0, 2, 0, 14)

    def _disturbance_loop(self):
        if not self._webcam_active:
            return

        level = self._detector.level

        bar_w = int(60 * level)
        r = min(255, int(level * 510))
        g = max(0,   255 - int(level * 300))
        color = f"#{r:02x}{g:02x}44"
        self._motion_canvas.coords(self._motion_bar, 0, 2, bar_w, 14)
        self._motion_canvas.itemconfig(self._motion_bar, fill=color)

        if self._char_cells:
            try:
                from PIL import ImageTk
                frame = draw_chars(
                    self._char_cells,
                    self._char_title,
                    motion_level=level,
                )
                self._tk_image = ImageTk.PhotoImage(frame)
                self._label.configure(image=self._tk_image)
            except Exception:
                pass

        self._disturb_id = self.after(50, self._disturbance_loop)

    def _on_close(self):
        self._webcam_active = False
        self._detector.stop()
        self.destroy()


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = CalligramApp()
    app.mainloop()
