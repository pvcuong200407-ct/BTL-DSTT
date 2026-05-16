"""
SVM & Ma Trận Bán Dương (PSD) - Python Implementation
======================================================
 Biểu đồ phân tách dữ liệu với 3 kernel (Linear, RBF, Polynomial)


Yêu cầu:
    pip install scikit-learn matplotlib numpy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from sklearn import svm, datasets
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)


# ══════════════════════════════════════════════════════════════
#  PHẦN 1 — Biểu đồ phân tách với 3 kernel SVM
# ══════════════════════════════════════════════════════════════

def plot_decision_boundary(ax, clf, X, y, title, kernel_name, color):
    """Vẽ vùng quyết định, margin, support vectors."""
    # Tạo lưới điểm
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Hàm quyết định
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Vùng màu quyết định
    cmap_bg = ListedColormap(["#FFDDC1", "#C1D8FF"])
    ax.contourf(xx, yy, Z, levels=[-3, 0, 3], alpha=0.3,
                colors=["#fb923c", "#378ADD"])

    # Đường ranh giới (Z=0) và margin (Z=±1)
    ax.contour(xx, yy, Z, levels=[-1, 0, 1],
               linestyles=["--", "-", "--"],
               linewidths=[1.2, 2.0, 1.2],
               colors=["gray", "black", "gray"])

    # Dữ liệu
    colors_pts = ["#fb923c" if yi == -1 else "#378ADD" for yi in y]
    ax.scatter(X[:, 0], X[:, 1], c=colors_pts, s=30,
               edgecolors="white", linewidths=0.5, zorder=3)

    # Support vectors
    sv = clf.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=120, facecolors="none",
               edgecolors="black", linewidths=1.8, zorder=4,
               label=f"Support Vectors ({len(sv)})")

    # Tính accuracy
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred) * 100

    ax.set_title(f"{title}\nAcc: {acc:.1f}%  |  SVs: {len(sv)}",
                 fontsize=10, fontweight="bold", color=color, pad=6)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "bottom", "left"]].set_color(color)
    ax.spines[["top", "right", "bottom", "left"]].set_linewidth(1.5)
    ax.legend(fontsize=7, loc="upper right")

    # Chú thích kernel formula
    formulas = {
        "linear":  r"$K(x,y) = x^Ty$",
        "rbf":     r"$K(x,y) = e^{-\gamma \|x-y\|^2}$",
        "poly":    r"$K(x,y) = (x^Ty + 1)^3$",
    }
    ax.text(0.02, 0.04, formulas[kernel_name], transform=ax.transAxes,
            fontsize=9, color=color, fontstyle="italic",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))


def plot_kernel_comparison():
    """So sánh 3 kernel trên 3 loại dataset."""

    # ── 3 datasets ──────────────────────────────────────────────
    datasets_info = [
        ("Moons",   *make_moons(n_samples=120, noise=0.15, random_state=42)),
        ("Circles", *make_circles(n_samples=120, noise=0.1, random_state=7)),
        ("Linear",  *make_classification(
            n_samples=120, n_features=2, n_redundant=0,
            n_informative=2, random_state=13, n_clusters_per_class=1)),
    ]
    # Đổi nhãn 0→-1
    for i in range(len(datasets_info)):
        name, X, y = datasets_info[i]
        y = np.where(y == 0, -1, 1)
        datasets_info[i] = (name, X, y)

    # ── 3 kernels ───────────────────────────────────────────────
    kernels_info = [
        ("Linear Kernel",     "linear", "#378ADD",
         svm.SVC(kernel="linear", C=1.0)),
        ("RBF / Gaussian",    "rbf",    "#1D9E75",
         svm.SVC(kernel="rbf", gamma=0.8, C=1.0)),
        ("Polynomial (d=3)",  "poly",   "#D4537E",
         svm.SVC(kernel="poly", degree=3, coef0=1, C=1.0)),
    ]

    fig = plt.figure(figsize=(15, 12))
    fig.patch.set_facecolor("#F8F9FA")

    # Tiêu đề chính
    fig.suptitle(
        "SVM Kernels & Ma Trận Bán Dương (PSD)\n"
        "Mỗi kernel hợp lệ vì ma trận Gram K thỏa: $z^TKz \\geq 0$ $\\forall z \\in \\mathbb{R}^n$",
        fontsize=13, fontweight="bold", y=0.98
    )

    # 3 hàng (dataset) × 3 cột (kernel)
    gs = gridspec.GridSpec(3, 3, hspace=0.45, wspace=0.2,
                           top=0.92, bottom=0.05, left=0.04, right=0.98)

    scaler = StandardScaler()

    for row, (ds_name, X, y) in enumerate(datasets_info):
        X_scaled = scaler.fit_transform(X)

        for col, (k_name, k_id, k_color, clf) in enumerate(kernels_info):
            # Clone clf để tránh reuse
            clf_copy = type(clf)(**clf.get_params())
            clf_copy.fit(X_scaled, y)

            ax = fig.add_subplot(gs[row, col])
            plot_decision_boundary(ax, clf_copy, X_scaled, y,
                                   f"{k_name}\n[{ds_name}]", k_id, k_color)

            # Nhãn dataset bên trái (chỉ cột đầu)
            if col == 0:
                ax.set_ylabel(ds_name, fontsize=11, fontweight="bold",
                              rotation=0, labelpad=60, va="center")

    # Chú thích chung
    note = (
        "━  Decision Boundary (Z=0)   "
        "- -  Margin (Z=±1)   "
        "⊙  Support Vectors   "
        "●  Class +1   "
        "●  Class −1"
    )
    fig.text(0.5, 0.01, note, ha="center", fontsize=8.5, color="#555")

    plt.savefig("svm_kernel_comparison.png", dpi=150, bbox_inches="tight")
    print("✓ Đã lưu: svm_kernel_comparison.png")
    plt.show()



# ══════════════════════════════════════════════════════════════
#  BONUS — Minh họa tính PSD của Gram matrix
# ══════════════════════════════════════════════════════════════

def visualize_gram_matrix():
    """Minh họa ma trận Gram K là PSD cho mỗi kernel."""
    from sklearn.metrics.pairwise import (
        linear_kernel, rbf_kernel, polynomial_kernel
    )

    rng = np.random.RandomState(42)
    X_sample = rng.randn(15, 2)          # 15 điểm 2D

    kernel_fns = {
        "Linear $K=X X^T$": linear_kernel(X_sample),
        "RBF $K=e^{-\\gamma ||x-y||^2}$": rbf_kernel(X_sample, gamma=0.5),
        "Polynomial $(X^TY+1)^3$": polynomial_kernel(X_sample, degree=3, coef0=1),
    }
    colors = ["#378ADD", "#1D9E75", "#D4537E"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(
        "Ma trận Gram K — Tính Bán Dương (PSD)\n"
        r"$z^T K z \geq 0 \;\forall z$ $\Leftrightarrow$ tất cả eigenvalue $\lambda_i \geq 0$",
        fontsize=12, fontweight="bold"
    )

    for ax, (name, K), color in zip(axes, kernel_fns.items(), colors):
        # Heatmap
        im = ax.imshow(K, cmap="Blues", aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(name, fontsize=10, color=color, fontweight="bold")
        ax.set_xlabel("j", fontsize=9)
        ax.set_ylabel("i", fontsize=9)

        # Tính eigenvalues
        eigvals = np.linalg.eigvalsh(K)
        min_eig = eigvals.min()
        psd_check = "✓ PSD" if min_eig >= -1e-10 else "✗ Not PSD"
        psd_color = "#1D9E75" if min_eig >= -1e-10 else "#E24B4A"

        ax.text(0.03, 0.97,
                f"{psd_check}\nmin λ = {min_eig:.2e}",
                transform=ax.transAxes, fontsize=9,
                va="top", color=psd_color, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=3))

    plt.tight_layout()
    plt.savefig("svm_gram_matrix.png", dpi=150, bbox_inches="tight")
    print("✓ Đã lưu: svm_gram_matrix.png")
    plt.show()


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   SVM & Ma Trận Bán Dương — Python Demo                 ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # ── So sánh 3 kernel ─────────────────────────────────
    print("► Vẽ biểu đồ phân tách với 3 kernel...")
    plot_kernel_comparison()

    # ── Bonus: Gram matrix ───────────────────────────────────────
    print("\n► Bonus: Minh họa Gram matrix PSD...")
    visualize_gram_matrix()

 

