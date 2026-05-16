import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFilter
from sklearn import svm
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import threading


# ══════════════════════════════════════════════════════════════════
#  BƯỚC 1: Train SVM với MNIST 28×28 (5000 mẫu, ~15-20 giây)
# ══════════════════════════════════════════════════════════════════

def train_model():
    print("Đang tải MNIST dataset...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X, y = mnist.data / 255.0, mnist.target.astype(int)

    # Dùng 5000 mẫu train, 1000 mẫu test → đủ nhanh + đủ chính xác
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, y_train = X_train[:5000], y_train[:5000]
    X_test,  y_test  = X_test[:1000],  y_test[:1000]

    print(f"Train: {len(X_train)}  Test: {len(X_test)}")
    print("Đang train SVM (RBF kernel)...")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    clf = svm.SVC(kernel="rbf", C=5, gamma="scale", cache_size=500)
    clf.fit(X_train_sc, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test_sc))
    print(f"Accuracy: {acc*100:.2f}%")
    return clf, scaler, acc


# ══════════════════════════════════════════════════════════════════
#  BƯỚC 2: Tiền xử lý ảnh vẽ tay → vector 784 chiều (28×28)
# ══════════════════════════════════════════════════════════════════

def preprocess_canvas_image(pil_image, scaler):
    """
    Chuẩn hóa ảnh canvas về đúng định dạng MNIST 28×28:
    - Nền đen, chữ trắng (MNIST convention)
    - Resize về 20×20, đặt vào giữa khung 28×28
    - Normalize rồi scale với scaler đã train
    """
    # Chuyển sang grayscale
    img = pil_image.convert("L")

    # Invert: canvas trắng+nét đen → MNIST đen+nét trắng
    img = ImageOps.invert(img)

    # Kiểm tra có nét vẽ không
    arr_check = np.array(img)
    if arr_check.max() < 10:
        return None  # Canvas gần như trống

    # Làm mờ nhẹ để giảm nhiễu (giống MNIST gốc)
    img = img.filter(ImageFilter.GaussianBlur(radius=1))

    # Crop vùng có nét vẽ
    bbox = img.getbbox()
    if bbox is None:
        return None
    img = img.crop(bbox)

    # Scale vừa với 20×20 (giữ tỉ lệ)
    img.thumbnail((20, 20), Image.LANCZOS)

    # Đặt vào giữa khung 28×28 (như MNIST chuẩn)
    canvas28 = Image.new("L", (28, 28), 0)
    x_off = (28 - img.width)  // 2
    y_off = (28 - img.height) // 2
    canvas28.paste(img, (x_off, y_off))

    # Normalize [0, 1] và flatten
    vec = np.array(canvas28, dtype=np.float64).flatten() / 255.0
    vec = vec.reshape(1, -1)

    # Áp dụng scaler đã train
    vec_scaled = scaler.transform(vec)
    return vec_scaled


# ══════════════════════════════════════════════════════════════════
#  BƯỚC 3: Giao diện Tkinter
# ══════════════════════════════════════════════════════════════════

class HandwritingApp:
    CANVAS_SIZE = 280
    PEN_WIDTH   = 20     # Nét dày hơn một chút
    BG_COLOR    = "#FFFFFF"
    PEN_COLOR   = "#000000"

    def __init__(self, root):
        self.root   = root
        self.model  = None
        self.scaler = None
        self.acc    = 0.0
        self._setup_ui()
        self._start_training()

    # ── Giao diện ────────────────────────────────────────────────

    def _setup_ui(self):
        self.root.title("SVM Nhận diện Chữ viết Tay")
        self.root.resizable(False, False)
        self.root.configure(bg="#1E1E2E")

        # Header
        header = tk.Frame(self.root, bg="#1E1E2E")
        header.pack(pady=(18, 4))

        tk.Label(header,
                 text="✍️  Nhận diện Chữ số",
                 font=("Helvetica", 20, "bold"),
                 bg="#1E1E2E", fg="#CDD6F4").pack()

        self.status_lbl = tk.Label(
            header,
            text="⏳ Đang train SVM với MNIST (15-20 giây)...",
            font=("Helvetica", 10),
            bg="#1E1E2E", fg="#F9E2AF"   # vàng = đang chờ
        )
        self.status_lbl.pack(pady=2)

        # Progress bar
        self.progress = ttk.Progressbar(
            header, mode="indeterminate", length=300
        )
        self.progress.pack(pady=4)
        self.progress.start(12)

        # Canvas vẽ
        canvas_frame = tk.Frame(
            self.root, bg="#313244",
            highlightthickness=3,
            highlightbackground="#89B4FA"
        )
        canvas_frame.pack(padx=30, pady=10)

        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.CANVAS_SIZE,
            height=self.CANVAS_SIZE,
            bg=self.BG_COLOR,
            cursor="crosshair",
            highlightthickness=0,
        )
        self.canvas.pack()
        self._draw_placeholder()

        self.canvas.bind("<Button-1>",        self._start_draw)
        self.canvas.bind("<B1-Motion>",       self._draw)
        self.canvas.bind("<ButtonRelease-1>", self._stop_draw)

        # PIL image backend
        self.pil_image    = Image.new("RGB", (self.CANVAS_SIZE, self.CANVAS_SIZE), "white")
        self.pil_draw     = ImageDraw.Draw(self.pil_image)
        self._prev_xy     = None
        self._has_drawing = False

        # Kết quả
        result_frame = tk.Frame(self.root, bg="#1E1E2E")
        result_frame.pack(pady=6)

        tk.Label(result_frame, text="Dự đoán:",
                 font=("Helvetica", 13),
                 bg="#1E1E2E", fg="#A6ADC8").pack()

        self.result_lbl = tk.Label(
            result_frame, text="—",
            font=("Helvetica", 72, "bold"),
            bg="#1E1E2E", fg="#89B4FA", width=3
        )
        self.result_lbl.pack()

        self.confidence_lbl = tk.Label(
            result_frame, text="",
            font=("Helvetica", 10),
            bg="#1E1E2E", fg="#6C7086"
        )
        self.confidence_lbl.pack()

        # Nút
        btn_frame = tk.Frame(self.root, bg="#1E1E2E")
        btn_frame.pack(pady=(6, 18))

        self.predict_btn = tk.Button(
            btn_frame, text="🔍  Predict",
            font=("Helvetica", 13, "bold"),
            bg="#89B4FA", fg="#1E1E2E",
            activebackground="#74C7EC",
            relief="flat", bd=0,
            padx=24, pady=10,
            cursor="hand2",
            command=self._predict,
            state="disabled"
        )
        self.predict_btn.pack(side="left", padx=10)

        self.clear_btn = tk.Button(
            btn_frame, text="🗑️  Xóa",
            font=("Helvetica", 13),
            bg="#313244", fg="#CDD6F4",
            activebackground="#45475A",
            relief="flat", bd=0,
            padx=20, pady=10,
            cursor="hand2",
            command=self._clear,
        )
        self.clear_btn.pack(side="left", padx=10)

        # Import ttk ở đây để dùng progressbar
        from tkinter import ttk as _ttk
        self.__class__._ttk = _ttk

    def _draw_placeholder(self):
        cx = cy = self.CANVAS_SIZE // 2
        self.canvas.create_text(cx, cy - 12,
                                text="Vẽ chữ số (0–9) ở đây",
                                font=("Helvetica", 13), fill="#BBBBBB",
                                tags="placeholder")
        self.canvas.create_text(cx, cy + 16,
                                text="Dùng chuột để vẽ",
                                font=("Helvetica", 10), fill="#CCCCCC",
                                tags="placeholder")

    # ── Train model ───────────────────────────────────────────────

    def _start_training(self):
        def _train():
            clf, scaler, acc = train_model()
            self.model  = clf
            self.scaler = scaler
            self.acc    = acc
            self.root.after(0, self._on_trained)

        threading.Thread(target=_train, daemon=True).start()

    def _on_trained(self):
        self.progress.stop()
        self.progress.pack_forget()
        self.status_lbl.config(
            text=f"✅ Model sẵn sàng  |  Accuracy: {self.acc*100:.1f}%  "
                 f"|  Vẽ chữ số rồi bấm Predict",
            fg="#A6E3A1"
        )
        self.predict_btn.config(state="normal")

    # ── Vẽ ───────────────────────────────────────────────────────

    def _start_draw(self, event):
        if not self._has_drawing:
            self.canvas.delete("placeholder")
            self._has_drawing = True
        self._prev_xy = (event.x, event.y)

    def _draw(self, event):
        if self._prev_xy is None:
            return
        x0, y0 = self._prev_xy
        x1, y1 = event.x, event.y
        r = self.PEN_WIDTH // 2

        self.canvas.create_oval(x1-r, y1-r, x1+r, y1+r,
                                fill=self.PEN_COLOR, outline="")
        self.canvas.create_line(x0, y0, x1, y1,
                                fill=self.PEN_COLOR,
                                width=self.PEN_WIDTH,
                                capstyle=tk.ROUND,
                                joinstyle=tk.ROUND)

        self.pil_draw.ellipse([x1-r, y1-r, x1+r, y1+r], fill=self.PEN_COLOR)
        self.pil_draw.line([x0, y0, x1, y1],
                           fill=self.PEN_COLOR, width=self.PEN_WIDTH)
        self._prev_xy = (x1, y1)

    def _stop_draw(self, event):
        self._prev_xy = None

    # ── Predict ──────────────────────────────────────────────────

    def _predict(self):
        if self.model is None:
            messagebox.showinfo("Thông báo", "Model chưa sẵn sàng!")
            return
        if not self._has_drawing:
            messagebox.showwarning("Chú ý", "Bạn chưa vẽ gì!")
            return

        vec = preprocess_canvas_image(self.pil_image, self.scaler)
        if vec is None:
            messagebox.showwarning("Chú ý", "Không nhận diện được nét vẽ!")
            return

        pred = self.model.predict(vec)[0]

        # Tính confidence dựa trên khoảng cách SVM
        dist     = self.model.decision_function(vec)[0]
        exp_dist = np.exp(dist - dist.max())
        conf_pct = (exp_dist[pred] / exp_dist.sum()) * 100
        conf_pct = min(conf_pct, 99.9)

        self.result_lbl.config(text=str(pred), fg="#89B4FA")
        self.confidence_lbl.config(
            text=f"Confidence ≈ {conf_pct:.1f}%",
            fg="#A6ADC8"
        )

    # ── Xóa ──────────────────────────────────────────────────────

    def _clear(self):
        self.canvas.delete("all")
        self.pil_image    = Image.new("RGB", (self.CANVAS_SIZE, self.CANVAS_SIZE), "white")
        self.pil_draw     = ImageDraw.Draw(self.pil_image)
        self._has_drawing = False
        self._draw_placeholder()
        self.result_lbl.config(text="—")
        self.confidence_lbl.config(text="")


# ══════════════════════════════════════════════════════════════════
#  CHẠY APP
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from tkinter import ttk   # import để progress bar hoạt động
    root = tk.Tk()
    app  = HandwritingApp(root)
    root.mainloop()
