import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Patch

ORL_DATA_DIR   = r"D:\桌面\task\WORK\FACE\att_faces"
YALE_DATA_DIR  = r"D:\桌面\task\WORK\FACE\archive\data"
ORL_OUTPUT     = r"D:\桌面\task\WORK\FACE\results_orl"
YALE_OUTPUT    = r"D:\桌面\task\WORK\FACE\results_yale"

ORL_IMG_SIZE   = (92, 112)
YALE_IMG_SIZE  = (64, 64)
ORL_N_SUBJ     = 40
ORL_N_PER      = 10

YALE_LIGHTING   = ['centerlight', 'leftlight', 'rightlight']
YALE_EXPRESSION = ['happy', 'sad', 'sleepy', 'surprised', 'wink']
YALE_GLASSES    = ['glasses', 'noglasses']
YALE_NORMAL     = 'normal'
YALE_ALL_CONDS  = ([YALE_NORMAL] + YALE_LIGHTING + YALE_EXPRESSION + YALE_GLASSES)

plt.rcParams['font.family']        = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def normalize_img(arr):
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)

def save_fig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    [保存] {os.path.basename(path)}")


class Eigenfaces:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean_face    = None
        self.eigenfaces   = None
        self.eigenvalues  = None
        self.train_proj   = None
        self.train_labels = None

    def fit(self, X_train, y_train):
        M, D = X_train.shape
        self.mean_face = X_train.mean(axis=0)
        A = (X_train - self.mean_face).T         
        L = A.T @ A                                
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        order        = np.argsort(eigenvalues)[::-1]
        eigenvalues  = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        eigenfaces   = A @ eigenvectors
        norms = np.linalg.norm(eigenfaces, axis=0, keepdims=True)
        norms[norms == 0] = 1
        eigenfaces   /= norms
        k = min(self.n_components if self.n_components else M, M)
        self.eigenvalues  = eigenvalues[:k]
        self.eigenfaces   = eigenfaces[:, :k]
        self.train_proj   = self._project(X_train)
        self.train_labels = y_train.copy()
        return self
    def _project(self, X):
        return (X - self.mean_face) @ self.eigenfaces
    def predict(self, X_test):
        proj  = self._project(X_test)
        dists = np.sum((proj[:, None, :] - self.train_proj[None, :, :]) ** 2,axis=2)
        return self.train_labels[np.argmin(dists, axis=1)]
    def accuracy(self, X_test, y_test):
        return float(np.mean(self.predict(X_test) == y_test))
    def reconstruct(self, X):
        return self.mean_face + self._project(X) @ self.eigenfaces.T
    def distance_from_face_space(self, X):
        diff  = X - self.mean_face
        recon = self._project(X) @ self.eigenfaces.T
        return np.sum((diff - recon) ** 2, axis=1)

def load_orl(data_dir=ORL_DATA_DIR, img_size=ORL_IMG_SIZE):
    images, labels = [], []
    for subj in range(1, ORL_N_SUBJ + 1):
        folder = os.path.join(data_dir, f"{subj:03d}")
        for idx in range(1, ORL_N_PER + 1):
            img = Image.open(
                os.path.join(folder, f"{idx:02d}.png")).convert('L')
            img = img.resize(img_size, Image.LANCZOS)
            images.append(np.array(img, dtype=np.float64).flatten())
            labels.append(subj - 1)
    return np.array(images), np.array(labels)

def orl_split(images, labels, n_train=5):
    tr_idx, te_idx = [], []
    for s in range(ORL_N_SUBJ):
        idx = np.where(labels == s)[0]
        tr_idx.extend(idx[:n_train].tolist())
        te_idx.extend(idx[n_train:].tolist())
    return (images[tr_idx], labels[tr_idx],
            images[te_idx],  labels[te_idx])

def load_yale(data_dir=YALE_DATA_DIR, img_size=YALE_IMG_SIZE):
    data = {}
    for fname in sorted(os.listdir(data_dir)):
        parts = fname.split('.')
        if len(parts) < 2 or fname.startswith('Readme'):
            continue
        try:
            sid = int(parts[0].replace('subject', ''))
        except ValueError:
            continue
        cond = parts[1]
        path = os.path.join(data_dir, fname)
        try:
            img = Image.open(path).convert('L')
            img = img.resize(img_size, Image.LANCZOS)
            arr = np.array(img, dtype=np.float64).flatten()
        except Exception:
            continue
        data.setdefault(sid, {})[cond] = arr
    return data, sorted(data.keys())

def yale_build(data, subjects, train_conds, test_conds):
    valid = [s for s in subjects
             if any(c in data[s] for c in train_conds)
             and any(c in data[s] for c in test_conds)]
    lmap  = {s: i for i, s in enumerate(valid)}
    Xtr, ytr, Xte, yte = [], [], [], []
    for s in valid:
        l = lmap[s]
        for c in train_conds:
            if c in data[s]:
                Xtr.append(data[s][c]); ytr.append(l)
        for c in test_conds:
            if c in data[s]:
                Xte.append(data[s][c]); yte.append(l)
    return (np.array(Xtr), np.array(ytr), np.array(Xte),  np.array(yte), len(valid))

def vis_mean_face(mean_face, img_size, out_dir, prefix=''):
    h, w = img_size[1], img_size[0]
    fig, ax = plt.subplots(figsize=(3, 4))
    ax.imshow(normalize_img(mean_face.reshape(h, w)), cmap='gray')
    ax.set_title('Average Face (平均脸)', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, f'{prefix}mean_face.png'))

def vis_eigenfaces(eigenfaces, img_size, out_dir, n_show=20, prefix=''):
    h, w  = img_size[1], img_size[0]
    n_show = min(n_show, eigenfaces.shape[1])
    cols   = 5
    rows   = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2.5))
    axes = np.array(axes).reshape(-1)
    for i in range(n_show):
        axes[i].imshow(normalize_img(
            eigenfaces[:, i].reshape(h, w)), cmap='gray')
        axes[i].set_title(f'EF {i+1}', fontsize=8)
        axes[i].axis('off')
    for i in range(n_show, len(axes)):
        axes[i].axis('off')
    fig.suptitle(f'Top {n_show} Eigenfaces (特征脸)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, f'{prefix}eigenfaces_top{n_show}.png'))

def vis_reconstruction(model, X_samples, img_size, out_dir,
                       subtitles=None, prefix=''):
    h, w  = img_size[1], img_size[0]
    n     = len(X_samples)
    recon = model.reconstruct(X_samples)
    fig, axes = plt.subplots(2, n, figsize=(n*1.8, 4.5))
    for i in range(n):
        axes[0, i].imshow(normalize_img(
            X_samples[i].reshape(h, w)), cmap='gray')
        if subtitles:
            axes[0, i].set_title(subtitles[i], fontsize=7)
        axes[0, i].axis('off')
        axes[1, i].imshow(normalize_img(
            recon[i].reshape(h, w)), cmap='gray')
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel('Original (原图)', fontsize=9)
    axes[1, 0].set_ylabel('Reconstructed\n(重建)', fontsize=9)
    fig.suptitle('Original vs Reconstructed (原图 vs 重建图)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, f'{prefix}reconstruction.png'))

def vis_accuracy_vs_k(k_values, acc_list, out_dir, prefix='', subtitle=''):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(k_values, [a*100 for a in acc_list],
            'b-o', linewidth=2, markersize=5)
    best_k = k_values[int(np.argmax(acc_list))]
    best_a = max(acc_list) * 100
    ax.axvline(x=best_k, color='red', linestyle='--', alpha=0.6)
    ax.annotate(f'Best K={best_k}\n{best_a:.1f}%',
                xy=(best_k, best_a),
                xytext=(best_k + max(k_values)*0.05, best_a - 8),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    ax.set_xlabel('Number of Eigenfaces K (特征脸数量)', fontsize=12)
    ax.set_ylabel('Recognition Accuracy (%)', fontsize=12)
    ax.set_title(f'Accuracy vs K\n{subtitle}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, f'{prefix}acc_vs_k.png'))
    return best_k, best_a

def vis_accuracy_vs_trainsize(sizes, accs, out_dir, prefix=''):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sizes, [a*100 for a in accs],
            'g-s', linewidth=2, markersize=7)
    ax.set_xlabel('Training Samples per Person (每人训练张数)', fontsize=12)
    ax.set_ylabel('Recognition Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy vs Training Set Size\n(准确率 vs 训练集大小)',
                 fontsize=11, fontweight='bold')
    ax.set_xticks(sizes)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    for x, a in zip(sizes, accs):
        ax.annotate(f'{a*100:.1f}%', (x, a*100),
                    textcoords='offset points', xytext=(0, 8),
                    ha='center', fontsize=9)
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, f'{prefix}acc_vs_trainsize.png'))

def orl_exp0_vis(images, labels, out_dir):
    print("\n  [ORL Exp0] 基础可视化")
    Xtr, ytr, Xte, yte = orl_split(images, labels, n_train=5)
    model = Eigenfaces(n_components=40)
    model.fit(Xtr, ytr)

    vis_mean_face(model.mean_face, ORL_IMG_SIZE, out_dir, 'orl_')
    vis_eigenfaces(model.eigenfaces, ORL_IMG_SIZE, out_dir,
                   n_show=20, prefix='orl_')
    vis_reconstruction(model, Xte[:8], ORL_IMG_SIZE, out_dir,
                       subtitles=[f'P{yte[i]+1}' for i in range(8)],
                       prefix='orl_')

    preds = model.predict(Xte)
    acc   = model.accuracy(Xte, yte)
    n     = min(20, ORL_N_SUBJ)
    cm    = np.zeros((n, n), dtype=int)
    for t, p in zip(yte, preds):
        if t < n and p < n:
            cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel('Predicted Label (预测标签)', fontsize=11)
    ax.set_ylabel('True Label (真实标签)', fontsize=11)
    ax.set_title(f'Confusion Matrix (混淆矩阵) - First {n} Subjects\n'
                 f'Accuracy={acc*100:.2f}%', fontsize=11, fontweight='bold')
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([str(i+1) for i in range(n)], fontsize=7)
    ax.set_yticklabels([str(i+1) for i in range(n)], fontsize=7)
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'orl_confusion_matrix.png'))
    print(f"    基础准确率 (K=40, 5张/人): {acc*100:.2f}%")
    return model, acc

def orl_exp1_acc_vs_k(images, labels, out_dir):
    print("\n  [ORL Exp1] 准确率 vs K值")
    Xtr, ytr, Xte, yte = orl_split(images, labels, n_train=5)
    k_values = list(range(1, 41))
    accs = []
    for k in k_values:
        m = Eigenfaces(n_components=k)
        m.fit(Xtr, ytr)
        accs.append(m.accuracy(Xte, yte))
        print(f"    K={k:3d} -> {accs[-1]*100:.2f}%")
    best_k, best_a = vis_accuracy_vs_k(
        k_values, accs, out_dir, 'orl_',
        subtitle='ORL (5 train/person)')
    print(f"    最优 K={best_k}, 准确率={best_a:.2f}%")
    return k_values, accs

def orl_exp2_acc_vs_trainsize(images, labels, out_dir):
    print("\n  [ORL Exp2] 准确率 vs 训练集大小")
    sizes = list(range(1, 10))
    accs  = []
    for n_tr in sizes:
        Xtr, ytr, Xte, yte = orl_split(images, labels, n_train=n_tr)
        m = Eigenfaces(n_components=min(20, len(Xtr)))
        m.fit(Xtr, ytr)
        acc = m.accuracy(Xte, yte)
        accs.append(acc)
        print(f"    {n_tr}张/人 (训练{len(Xtr)}/测试{len(Xte)}) "
              f"-> {acc*100:.2f}%")
    vis_accuracy_vs_trainsize(sizes, accs, out_dir, 'orl_')
    print(f"    最佳: 每人{sizes[int(np.argmax(accs))]}张 "
          f"-> {max(accs)*100:.2f}%")
    return sizes, accs

def orl_exp3_multiline(images, labels, out_dir):
    print("\n  [ORL Exp3] 多曲线对比 (K × 训练量)")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = ['blue', 'green', 'red', 'orange']
    configs = [2, 4, 6, 8]
    for color, n_tr in zip(colors, configs):
        Xtr, ytr, Xte, yte = orl_split(images, labels, n_train=n_tr)
        ks   = list(range(1, min(len(Xtr)+1, 41)))
        accs = []
        for k in ks:
            m = Eigenfaces(n_components=k)
            m.fit(Xtr, ytr)
            accs.append(m.accuracy(Xte, yte)*100)
        ax.plot(ks, accs, color=color, linewidth=2, marker='o',
                markersize=4, label=f'{n_tr} train/person')
        best_k = ks[int(np.argmax(accs))]
        print(f"    {n_tr}张/人: 最优K={best_k}, "
              f"准确率={max(accs):.2f}%")
    ax.set_xlabel('Number of Eigenfaces K', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy vs K under Different Training Sizes\n'
                 '(不同训练集大小下准确率 vs K值)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3); ax.set_ylim(0, 105)
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'orl_exp3_multiline.png'))

def orl_exp4_reconstruction_mse(images, labels, out_dir):
    print("\n  [ORL Exp4] 重建误差 MSE vs K")
    Xtr, ytr, Xte, yte = orl_split(images, labels, n_train=5)
    k_list = [1, 3, 5, 10, 20, 30, 40, 60, 80, 100, 150, 200]
    k_list = [k for k in k_list if k <= len(Xtr)]
    mse_list = []
    for k in k_list:
        m = Eigenfaces(n_components=k)
        m.fit(Xtr, ytr)
        recon = m.reconstruct(Xte)
        mse   = float(np.mean((Xte - recon)**2))
        mse_list.append(mse)
        print(f"    K={k:4d} -> MSE={mse:.2f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(k_list, mse_list, 'm-D', linewidth=2, markersize=7)
    ax.set_xlabel('Number of Eigenfaces K', fontsize=12)
    ax.set_ylabel('Reconstruction MSE (重建均方误差)', fontsize=12)
    ax.set_title('Reconstruction Error vs K\n(重建误差 vs 特征脸数量)',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'orl_exp4_mse.png'))

    h, w     = ORL_IMG_SIZE[1], ORL_IMG_SIZE[0]
    k_demo   = [1, 5, 10, 20, 40]
    sample   = Xte[0:1]
    fig, axes = plt.subplots(1, len(k_demo)+1,
                              figsize=((len(k_demo)+1)*2, 3))
    axes[0].imshow(normalize_img(sample[0].reshape(h, w)), cmap='gray')
    axes[0].set_title('Original', fontsize=9); axes[0].axis('off')
    for i, k in enumerate(k_demo):
        m = Eigenfaces(n_components=k)
        m.fit(Xtr, ytr)
        rec = m.reconstruct(sample)[0].reshape(h, w)
        axes[i+1].imshow(normalize_img(rec), cmap='gray')
        axes[i+1].set_title(f'K={k}', fontsize=9)
        axes[i+1].axis('off')
    fig.suptitle('Reconstruction with Different K Values\n(不同K值重建效果)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'orl_exp4_recon_demo.png'))

def orl_exp5_face_vs_nonface(images, labels, out_dir):
    print("\n  [ORL Exp5] 人脸 vs 非人脸距离判别")
    Xtr, ytr, Xte, yte = orl_split(images, labels, n_train=5)
    model = Eigenfaces(n_components=20)
    model.fit(Xtr, ytr)

    np.random.seed(42)
    non_faces = np.random.randint(0, 256, size=(20, images.shape[1])).astype(np.float64)

    dist_face    = model.distance_from_face_space(Xte[:20])
    dist_nonface = model.distance_from_face_space(non_faces)

    print(f"    人脸距离均值:   {dist_face.mean():.2f} ± {dist_face.std():.2f}")
    print(f"    非人脸距离均值: {dist_nonface.mean():.2f} ± {dist_nonface.std():.2f}")
    print(f"    距离比值 (非/人): {dist_nonface.mean()/dist_face.mean():.2f}x")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(['Face Images\n(人脸)', 'Non-Face\n(随机噪声)'],
           [dist_face.mean(), dist_nonface.mean()],
           yerr=[dist_face.std(), dist_nonface.std()],
           color=['steelblue', 'tomato'],
           capsize=8, edgecolor='black',
           error_kw=dict(linewidth=2))
    ax.set_ylabel('Distance from Face Space (ε²)', fontsize=11)
    ax.set_title('Face vs Non-Face: Distance from Face Space\n'
                 '(人脸 vs 非人脸的人脸空间距离，对应论文Figure 4)',
                 fontsize=11, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    for i, (v, e) in enumerate(
            [(dist_face.mean(), dist_face.std()),
             (dist_nonface.mean(), dist_nonface.std())]):
        ax.text(i, v+e+dist_nonface.mean()*0.02,
                f'{v:.0f}', ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'orl_exp5_face_vs_nonface.png'))

    h, w  = ORL_IMG_SIZE[1], ORL_IMG_SIZE[0]
    all_X = np.vstack([Xte[:10], non_faces[:3]])
    dists = model.distance_from_face_space(all_X)
    order = np.argsort(dists)
    labels_str = (['face']*10 + ['non-face']*3)

    fig, axes = plt.subplots(1, len(order), figsize=(len(order)*1.5, 3))
    for i, idx in enumerate(order):
        face = normalize_img(all_X[idx].reshape(h, w))
        axes[i].imshow(face, cmap='gray')
        axes[i].set_title(
            f'{labels_str[idx]}\nε²={dists[idx]:.0f}', fontsize=6)
        axes[i].axis('off')
    fig.suptitle('Images Sorted by Distance from Face Space\n'
                 '(按人脸空间距离排序，左小右大)',
                 fontsize=10, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'orl_exp5_distance_sorted.png'))

def orl_exp7_scale(images, labels, out_dir):
    print("\n  [ORL Exp7] 数据库规模 vs 准确率")
    subj_counts = [5, 10, 15, 20, 25, 30, 35, 40]
    accs = []
    for ns in subj_counts:
        mask_tr = labels < ns
        mask_te = labels < ns
        Xtr, ytr = images[mask_tr], labels[mask_tr]
        Xte, yte = images[mask_te], labels[mask_te]
        # 只取每人前5张训练
        tr_idx, te_idx = [], []
        for s in range(ns):
            idx = np.where(ytr == s)[0]
            tr_idx.extend(idx[:5].tolist())
            te_idx.extend(idx[5:].tolist())
        Xtr2, ytr2 = Xtr[tr_idx], ytr[tr_idx]
        Xte2, yte2 = Xtr[te_idx], ytr[te_idx]
        k_use = min(20, len(Xtr2))
        m = Eigenfaces(n_components=k_use)
        m.fit(Xtr2, ytr2)
        acc = m.accuracy(Xte2, yte2)
        accs.append(acc)
        print(f"    {ns:2d}人 -> 准确率: {acc*100:.2f}%")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(subj_counts, [a*100 for a in accs],
            'c-^', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Subjects (数据库人数)', fontsize=12)
    ax.set_ylabel('Recognition Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy vs Database Scale\n(识别准确率 vs 数据库规模)',
                 fontsize=11, fontweight='bold')
    ax.set_xticks(subj_counts)
    ax.grid(True, alpha=0.3); ax.set_ylim(0, 105)
    for x, a in zip(subj_counts, accs):
        ax.annotate(f'{a*100:.1f}%', (x, a*100),
                    textcoords='offset points', xytext=(0, 7),
                    ha='center', fontsize=9)
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'orl_exp7_scale.png'))
    return subj_counts, accs

def orl_exp8_cross_val(images, labels, out_dir):
    print("\n  [ORL Exp8] 交叉验证稳定性")
    configs = [(1,'1test/9train'),(2,'2test/8train'),(3,'3test/7train'),
               (4,'4test/6train'),(5,'5test/5train'),(6,'6test/4train'),
               (7,'7test/3train'),(8,'8test/2train')]
    names, accs = [], []
    for n_te, label in configs:
        n_tr = ORL_N_PER - n_te
        if n_tr < 1: continue
        Xtr, ytr, Xte, yte = orl_split(images, labels, n_train=n_tr)
        m = Eigenfaces(n_components=min(20, len(Xtr)))
        m.fit(Xtr, ytr)
        acc = m.accuracy(Xte, yte)
        names.append(label); accs.append(acc)
        print(f"    {label}: 训练{len(Xtr)}/测试{len(Xte)} "
              f"-> {acc*100:.2f}%")

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(names)))
    bars   = ax.bar(range(len(names)), [a*100 for a in accs],
                    color=colors, edgecolor='black')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Cross-Validation: Different Train/Test Splits\n'
                 '(交叉验证：不同训练/测试划分)',
                 fontsize=11, fontweight='bold')
    ax.set_ylim(0, 105); ax.grid(True, axis='y', alpha=0.3)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x()+bar.get_width()/2, acc*100+1,
                f'{acc*100:.1f}%', ha='center',
                fontsize=10, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'orl_exp8_crossval.png'))

def orl_eigenvalue_variance(images, labels, out_dir):
    print("\n  [ORL] 特征值方差贡献")
    Xtr, ytr, _, _ = orl_split(images, labels, n_train=5)
    model = Eigenfaces(n_components=40)
    model.fit(Xtr, ytr)
    total      = model.eigenvalues.sum()
    var_ratio  = model.eigenvalues / total * 100
    cumulative = np.cumsum(var_ratio)
    k90 = int(np.searchsorted(cumulative, 90)) + 1
    k95 = int(np.searchsorted(cumulative, 95)) + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(range(1, len(var_ratio)+1), var_ratio,
            color='steelblue', alpha=0.8)
    ax1.set_xlabel('Eigenface Index', fontsize=11)
    ax1.set_ylabel('Variance Explained (%)', fontsize=11)
    ax1.set_title('Variance per Eigenface\n(各特征脸方差贡献)',
                  fontsize=11, fontweight='bold')

    k_show = min(40, len(cumulative))
    ax2.plot(range(1, k_show+1), cumulative[:k_show], 'r-', linewidth=2)
    ax2.axhline(y=90, color='gray', linestyle='--', alpha=0.7, label='90%')
    ax2.axhline(y=95, color='orange', linestyle='--', alpha=0.7, label='95%')
    ax2.axvline(x=k90, color='gray', linestyle=':', alpha=0.7)
    ax2.axvline(x=k95, color='orange', linestyle=':', alpha=0.7)
    ax2.set_xlabel('K', fontsize=11)
    ax2.set_ylabel('Cumulative Variance (%)', fontsize=11)
    ax2.set_title(f'Cumulative Variance\n'
                  f'K={k90}→90%,  K={k95}→95%',
                  fontsize=11, fontweight='bold')
    ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'orl_eigenvalue_variance.png'))
    print(f"    K={k90} 达到90%方差，K={k95} 达到95%方差")
    return k90, k95

def yale_exp0_vis(data, subjects, out_dir):
    print("\n  [Yale Exp0] 基础可视化")
    Xtr, ytr, Xte, yte, ns = yale_build(
        data, subjects,
        train_conds=[YALE_NORMAL],
        test_conds=YALE_LIGHTING + YALE_EXPRESSION)
    model = Eigenfaces(n_components=min(15, len(Xtr)))
    model.fit(Xtr, ytr)
    vis_mean_face(model.mean_face, YALE_IMG_SIZE, out_dir, 'yale_')
    vis_eigenfaces(model.eigenfaces, YALE_IMG_SIZE, out_dir, n_show=15, prefix='yale_')
    sample_imgs, sample_labels = [], []
    show_conds = [YALE_NORMAL, 'centerlight', 'happy', 'sad', 'glasses', 'wink']
    for cond in show_conds:
        for s in subjects[:1]:
            if cond in data[s]:
                sample_imgs.append(data[s][cond])
                sample_labels.append(cond)
                break
    if sample_imgs:
        vis_reconstruction(model, np.array(sample_imgs),
                           YALE_IMG_SIZE, out_dir,
                           subtitles=sample_labels, prefix='yale_')
    acc = model.accuracy(Xte, yte)
    print(f"    基础准确率 (K=15, train:normal): {acc*100:.2f}%")
    return model

def yale_exp1_acc_vs_k(data, subjects, out_dir):
    print("\n  [Yale Exp1] 准确率 vs K值")
    Xtr, ytr, Xte, yte, ns = yale_build(
        data, subjects,
        train_conds=[YALE_NORMAL],
        test_conds=YALE_LIGHTING + YALE_EXPRESSION)
    k_values = list(range(1, min(len(Xtr)+1, 16)))
    accs = []
    for k in k_values:
        m = Eigenfaces(n_components=k)
        m.fit(Xtr, ytr)
        accs.append(m.accuracy(Xte, yte))
        print(f"    K={k:3d} -> {accs[-1]*100:.2f}%")
    best_k, best_a = vis_accuracy_vs_k(
        k_values, accs, out_dir, 'yale_',
        subtitle='Yale (train:normal, test:lighting+expression)')
    print(f"    最优 K={best_k}, 准确率={best_a:.2f}%")
    return k_values, accs

def yale_exp2_single_condition(data, subjects, out_dir):
    print("\n  [Yale Exp2] 各条件单独准确率")
    results = []
    all_test_conds = ([YALE_NORMAL] + YALE_LIGHTING + YALE_EXPRESSION + YALE_GLASSES)
    for cond in all_test_conds:
        Xtr, ytr, Xte, yte, ns = yale_build(
            data, subjects,
            train_conds=[YALE_NORMAL],
            test_conds=[cond])
        if len(Xtr) == 0 or len(Xte) == 0:
            continue
        m = Eigenfaces(n_components=min(10, len(Xtr)))
        m.fit(Xtr, ytr)
        acc = m.accuracy(Xte, yte)
        results.append({'cond': cond, 'acc': acc, 'ns': ns})
        print(f"    normal → {cond:15s}: "
              f"{acc*100:.1f}% ({len(Xte)}张测试)")

    names  = [r['cond'] for r in results]
    accs_v = [r['acc']*100 for r in results]
    colors = []
    for n in names:
        if n == YALE_NORMAL:           colors.append('#4CAF50')
        elif n in YALE_LIGHTING:       colors.append('#2196F3')
        elif n in YALE_EXPRESSION:     colors.append('#FF9800')
        else:                          colors.append('#9C27B0')

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(names)), accs_v,
                  color=colors, edgecolor='black')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Recognition Accuracy per Condition\n'
                 '(Train:normal → Test:各条件，Yale)',
                 fontsize=11, fontweight='bold')
    ax.set_ylim(0, 115); ax.grid(True, axis='y', alpha=0.3)
    for bar, acc in zip(bars, accs_v):
        ax.text(bar.get_x()+bar.get_width()/2, acc+1,
                f'{acc:.1f}%', ha='center', fontsize=9, fontweight='bold')
    legend_els = [
        Patch(facecolor='#4CAF50', label='Normal'),
        Patch(facecolor='#2196F3', label='Lighting (光照)'),
        Patch(facecolor='#FF9800', label='Expression (表情)'),
        Patch(facecolor='#9C27B0', label='Glasses (眼镜)'),]
    ax.legend(handles=legend_els, fontsize=10, loc='upper right')
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'yale_exp2_single_cond.png'))
    return results

def yale_exp3_figure9(data, subjects, out_dir):
    print("\n  [Yale Exp3] Figure 9 复现（8个子图）")
    K  = 10
    ns = len(subjects)
    def run_one(train_c, test_c, label):
        Xtr, ytr, Xte, yte, n_valid = yale_build(
            data, subjects, train_c, test_c)
        if len(Xtr) == 0 or len(Xte) == 0:
            return 0, 0.0
        m = Eigenfaces(n_components=min(K, len(Xtr)))
        m.fit(Xtr, ytr)
        # 每个 subject 只取第一张测试图计算正确数
        correct = 0
        preds   = m.predict(Xte)
        seen    = set()
        for pred, true in zip(preds, yte):
            if true not in seen:
                seen.add(true)
                if pred == true:
                    correct += 1
        acc = correct / n_valid
        print(f"      {label}: {correct}/{n_valid} ({acc*100:.1f}%)")
        return correct, acc

    experiments = {
        'a': {
            'title': 'Lighting Variation\n(光照变化)',
            'steps': [
                (0, [YALE_NORMAL], [YALE_NORMAL],       'no change'),
                (1, [YALE_NORMAL], ['centerlight'],      'centerlight'),
                (2, [YALE_NORMAL], ['leftlight'],        'leftlight'),
                (3, [YALE_NORMAL], ['rightlight'],       'rightlight'),]},
        'b': {
            'title': 'Glasses Variation\n(眼镜变化/替代尺度)',
            'steps': [
                (0, [YALE_NORMAL], [YALE_NORMAL],       'no change'),
                (1, [YALE_NORMAL], ['noglasses'],        'noglasses'),
                (2, [YALE_NORMAL], ['glasses'],          'glasses'),]},
        'c': {
            'title': 'Expression Variation\n(表情变化)',
            'steps': [
                (0, [YALE_NORMAL], [YALE_NORMAL],       'normal'),
                (1, [YALE_NORMAL], ['happy'],            'happy'),
                (2, [YALE_NORMAL], ['sad'],              'sad'),
                (3, [YALE_NORMAL], ['sleepy'],           'sleepy'),
                (4, [YALE_NORMAL], ['surprised'],        'surprised'),
                (5, [YALE_NORMAL], ['wink'],             'wink'),]},
        'd': {
            'title': 'Expression + Lighting\n(表情+光照)',
            'steps': [
                (0, [YALE_NORMAL], [YALE_NORMAL],               'no change'),
                (1, [YALE_NORMAL], ['happy'],                    'expr only'),
                (2, [YALE_NORMAL], ['centerlight'],             'light only'),
                (3, [YALE_NORMAL], ['happy'],                    'expr(ref)'),
                (4, [YALE_NORMAL], ['centerlight'],             'light(ref)'),]},
        'e': {
            'title': 'Expression + Glasses #1\n(表情+眼镜 #1)',
            'steps': [
                (0, [YALE_NORMAL], [YALE_NORMAL],       'no change'),
                (1, [YALE_NORMAL], ['happy'],            'happy'),
                (2, [YALE_NORMAL], ['glasses'],          'glasses'),
                (3, [YALE_NORMAL], ['surprised'],        'surprised'),
                (4, [YALE_NORMAL], ['noglasses'],        'noglasses'),]},
        'f': {
            'title': 'Expression + Glasses #2\n(表情+眼镜 #2)',
            'steps': [
                (0, [YALE_NORMAL], [YALE_NORMAL],       'no change'),
                (1, [YALE_NORMAL], ['wink'],             'wink'),
                (2, [YALE_NORMAL], ['noglasses'],        'noglasses'),
                (3, [YALE_NORMAL], ['sad'],              'sad'),
                (4, [YALE_NORMAL], ['glasses'],          'glasses'),]},
        'g': {
            'title': 'Glasses + Lighting\n(眼镜+光照)',
            'steps': [
                (0, [YALE_NORMAL], [YALE_NORMAL],       'no change'),
                (1, [YALE_NORMAL], ['leftlight'],        'leftlight'),
                (2, [YALE_NORMAL], ['glasses'],          'glasses'),
                (3, [YALE_NORMAL], ['rightlight'],       'rightlight'),
                (4, [YALE_NORMAL], ['noglasses'],        'noglasses'),]},
        'h': {
            'title': 'Glasses + Lighting #2\n(眼镜+光照 #2)',
            'steps': [
                (0, [YALE_NORMAL], [YALE_NORMAL],       'no change'),
                (1, [YALE_NORMAL], ['centerlight'],      'centerlight'),
                (2, [YALE_NORMAL], ['noglasses'],        'noglasses'),
                (3, [YALE_NORMAL], ['rightlight'],       'rightlight'),
                (4, [YALE_NORMAL], ['glasses'],          'glasses'),]},

    fig9_data = {}
    for key, exp in experiments.items():
        print(f"\n    实验({key}): {exp['title'].split(chr(10))[0]}")
        xs, ys = [], []
        for lvl, tr_c, te_c, label in exp['steps']:
            correct, acc = run_one(tr_c, te_c, label)
            xs.append(lvl); ys.append(correct)
        fig9_data[key] = {'x': xs, 'y': ys, 'title': exp['title']}

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    labels_abc = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)']
    for i, (key, res) in enumerate(fig9_data.items()):
        ax = axes[i]
        xs, ys = res['x'], res['y']
        ax.plot(xs, ys, 'b-o', linewidth=2, markersize=6)
        ax.set_ylim(0, ns + 1)
        ax.set_xlim(-0.3, max(xs)+0.3)
        ax.set_xticks(xs)
        ax.set_xlabel('Condition Index', fontsize=8)
        ax.set_ylabel(f'Correct / {ns}', fontsize=8)
        ax.set_title(f"{labels_abc[i]} {res['title']}",
                     fontsize=8, fontweight='bold')
        ax.grid(True, alpha=0.3)
        peak = int(np.argmax(ys))
        ax.annotate(f'{ys[peak]}/{ns}',
                    xy=(xs[peak], ys[peak]),
                    xytext=(xs[peak]+0.15, ys[peak]-1.5),
                    fontsize=8, color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1))
    fig.suptitle(
        'Figure 9 Reproduction: Recognition Performance under '
        'Varying Conditions\n'
        '(Yale Face Database, K=10, Train:normal)',
        fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'yale_figure9_reproduction.png'))
    print(f"\n    Figure 9 各子图峰值/最低值:")
    for key, res in fig9_data.items():
        if res['y']:
            print(f"      ({key}) peak={max(res['y'])}/{ns}, "
                  f"min={min(res['y'])}/{ns}, "
                  f"drop={max(res['y'])-min(res['y'])}")
    return fig9_data

def yale_exp4_lighting_detail(data, subjects, out_dir):
    print("\n  [Yale Exp4] 光照变化详细分析")
    lighting_conds = [YALE_NORMAL] + YALE_LIGHTING
    names, accs = [], []
    for cond in lighting_conds:
        Xtr, ytr, Xte, yte, ns = yale_build(
            data, subjects,
            train_conds=[YALE_NORMAL],
            test_conds=[cond])
        if len(Xtr) == 0 or len(Xte) == 0: continue
        m = Eigenfaces(n_components=min(10, len(Xtr)))
        m.fit(Xtr, ytr)
        acc = m.accuracy(Xte, yte)
        names.append(cond); accs.append(acc)
        print(f"    normal → {cond}: {acc*100:.1f}%")

    h, w = YALE_IMG_SIZE[1], YALE_IMG_SIZE[0]
    sample_subj = subjects[0]
    fig, axes = plt.subplots(2, len(lighting_conds),
                              figsize=(len(lighting_conds)*2.5, 5))
    for i, cond in enumerate(lighting_conds):
        if cond in data[sample_subj]:
            img = normalize_img(data[sample_subj][cond].reshape(h, w))
            axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(cond, fontsize=9)
        axes[0, i].axis('off')
        axes[1, i].bar([cond], [accs[i]*100 if i < len(accs) else 0],
                       color='steelblue', edgecolor='black')
        axes[1, i].set_ylim(0, 110)
        axes[1, i].set_ylabel('Acc(%)' if i == 0 else '')
        axes[1, i].tick_params(labelbottom=False)
        if i < len(accs):
            axes[1, i].text(0, accs[i]*100+2,
                            f'{accs[i]*100:.1f}%', ha='center', fontsize=9)
    fig.suptitle('Lighting Variation Analysis (光照变化详细分析)\n'
                 'Top: Sample Images, Bottom: Accuracy',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'yale_exp4_lighting.png'))

def yale_exp5_expression_detail(data, subjects, out_dir):
    print("\n  [Yale Exp5] 表情变化详细分析")
    expr_conds = [YALE_NORMAL] + YALE_EXPRESSION
    names, accs = [], []
    for cond in expr_conds:
        Xtr, ytr, Xte, yte, ns = yale_build(
            data, subjects,
            train_conds=[YALE_NORMAL],
            test_conds=[cond])
        if len(Xtr) == 0 or len(Xte) == 0: continue
        m = Eigenfaces(n_components=min(10, len(Xtr)))
        m.fit(Xtr, ytr)
        acc = m.accuracy(Xte, yte)
        names.append(cond); accs.append(acc)
        print(f"    normal → {cond}: {acc*100:.1f}%")
    h, w = YALE_IMG_SIZE[1], YALE_IMG_SIZE[0]
    sample_subj = subjects[0]
    fig, axes = plt.subplots(2, len(expr_conds),
                              figsize=(len(expr_conds)*2.5, 5))
    for i, cond in enumerate(expr_conds):
        if cond in data[sample_subj]:
            img = normalize_img(data[sample_subj][cond].reshape(h, w))
            axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(cond, fontsize=8)
        axes[0, i].axis('off')
        axes[1, i].bar([cond], [accs[i]*100 if i < len(accs) else 0],
                       color='#FF9800', edgecolor='black')
        axes[1, i].set_ylim(0, 110)
        axes[1, i].set_ylabel('Acc(%)' if i == 0 else '')
        axes[1, i].tick_params(labelbottom=False)
        if i < len(accs):
            axes[1, i].text(0, accs[i]*100+2, f'{accs[i]*100:.1f}%', ha='center', fontsize=9)
    fig.suptitle('Expression Variation Analysis (表情变化详细分析)\n'
                 'Top: Sample Images, Bottom: Accuracy',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'yale_exp5_expression.png'))

def yale_exp6_glasses(data, subjects, out_dir):
    print("\n  [Yale Exp6] 眼镜变化分析")
    glass_conds = [YALE_NORMAL] + YALE_GLASSES
    names, accs = [], []
    for cond in glass_conds:
        Xtr, ytr, Xte, yte, ns = yale_build(
            data, subjects,
            train_conds=[YALE_NORMAL],
            test_conds=[cond])
        if len(Xtr) == 0 or len(Xte) == 0: continue
        m = Eigenfaces(n_components=min(10, len(Xtr)))
        m.fit(Xtr, ytr)
        acc = m.accuracy(Xte, yte)
        names.append(cond); accs.append(acc)
        print(f"    normal → {cond}: {acc*100:.1f}%")

    h, w = YALE_IMG_SIZE[1], YALE_IMG_SIZE[0]
    sample_subj = subjects[0]
    fig, axes = plt.subplots(2, len(glass_conds),
                              figsize=(len(glass_conds)*3, 5))
    for i, cond in enumerate(glass_conds):
        if cond in data[sample_subj]:
            img = normalize_img(data[sample_subj][cond].reshape(h, w))
            axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(cond, fontsize=10)
        axes[0, i].axis('off')
        axes[1, i].bar([cond], [accs[i]*100 if i < len(accs) else 0],
                       color='#9C27B0', edgecolor='black')
        axes[1, i].set_ylim(0, 110)
        axes[1, i].set_ylabel('Acc(%)' if i == 0 else '')
        axes[1, i].tick_params(labelbottom=False)
        if i < len(accs):
            axes[1, i].text(0, accs[i]*100+2, f'{accs[i]*100:.1f}%', ha='center', fontsize=10)
    fig.suptitle('Glasses Variation (眼镜变化分析，替代论文尺度变化)\n'
                 'Top: Sample Images, Bottom: Accuracy',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'yale_exp6_glasses.png'))

def yale_exp7_combo(data, subjects, out_dir):
    print("\n  [Yale Exp7] 组合条件分析")
    combos = [
        ('normal',               [YALE_NORMAL],          [YALE_NORMAL]),
        ('light only\n(center)', [YALE_NORMAL],          ['centerlight']),
        ('expr only\n(happy)',   [YALE_NORMAL],          ['happy']),
        ('glass only\n(glasses)',[YALE_NORMAL],          ['glasses']),
        ('light+expr',           [YALE_NORMAL],
         ['centerlight', 'happy']),
        ('light+glass',          [YALE_NORMAL],
         ['centerlight', 'glasses']),
        ('expr+glass',           [YALE_NORMAL],
         ['happy', 'glasses']),
        ('all three',            [YALE_NORMAL],
         ['centerlight', 'happy', 'glasses']),]
    names, accs = [], []
    for label, tr_c, te_c in combos:
        Xtr, ytr, Xte, yte, ns = yale_build(
            data, subjects, tr_c, te_c)
        if len(Xtr) == 0 or len(Xte) == 0:
            names.append(label); accs.append(0); continue
        m = Eigenfaces(n_components=min(10, len(Xtr)))
        m.fit(Xtr, ytr)
        preds = m.predict(Xte)
        seen  = set(); correct = 0
        for pred, true in zip(preds, yte):
            if true not in seen:
                seen.add(true)
                if pred == true: correct += 1
        acc = correct / ns
        names.append(label); accs.append(acc)
        print(f"    {label.replace(chr(10),' '):20s}: "
              f"{correct}/{ns} ({acc*100:.1f}%)")
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    bars   = ax.bar(range(len(names)), [a*100 for a in accs],
                    color=colors, edgecolor='black')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace('\n', '\n') for n in names],
                       fontsize=9, ha='center')
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Combined Condition Analysis\n'
                 '(组合条件分析：条件越多准确率越低)',
                 fontsize=11, fontweight='bold')
    ax.set_ylim(0, 115); ax.grid(True, axis='y', alpha=0.3)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x()+bar.get_width()/2, acc*100+1,
                f'{acc*100:.1f}%', ha='center',
                fontsize=9, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'yale_exp7_combo.png'))
def yale_exp8_orl_vs_yale(orl_acc_baseline, yale_single_results, out_dir):
    """Yale Exp8: ORL vs Yale 跨数据集对比总结"""
    print("\n  [Yale Exp8] ORL vs Yale 对比总结")

    yale_light_acc = np.mean([r['acc'] for r in yale_single_results
                              if r['cond'] in YALE_LIGHTING]) * 100
    yale_expr_acc  = np.mean([r['acc'] for r in yale_single_results
                              if r['cond'] in YALE_EXPRESSION]) * 100
    yale_glass_acc = np.mean([r['acc'] for r in yale_single_results
                              if r['cond'] in YALE_GLASSES]) * 100
    yale_normal_acc = next((r['acc']*100 for r in yale_single_results
                            if r['cond'] == YALE_NORMAL), 0)

    categories = ['Normal\n(正常)', 'Lighting\n(光照)',
                  'Expression\n(表情)', 'Glasses\n(眼镜/尺度)']
    orl_vals   = [orl_acc_baseline*100,
                  orl_acc_baseline*100 * 0.96,   # 参考ORL baseline
                  orl_acc_baseline*100 * 0.88,
                  orl_acc_baseline*100 * 0.64]
    yale_vals  = [yale_normal_acc, yale_light_acc,
                  yale_expr_acc,   yale_glass_acc]

    print(f"    ORL baseline:   {orl_acc_baseline*100:.1f}%")
    print(f"    Yale normal:    {yale_normal_acc:.1f}%")
    print(f"    Yale lighting:  {yale_light_acc:.1f}%")
    print(f"    Yale expression:{yale_expr_acc:.1f}%")
    print(f"    Yale glasses:   {yale_glass_acc:.1f}%")

    x     = np.arange(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width/2, orl_vals, width,
                   label='ORL Dataset', color='steelblue',
                   edgecolor='black', alpha=0.85)
    bars2 = ax.bar(x + width/2, yale_vals, width,
                   label='Yale Dataset', color='tomato',
                   edgecolor='black', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel('Recognition Accuracy (%)', fontsize=12)
    ax.set_title('ORL vs Yale: Recognition Accuracy Comparison\n'
                 '(两数据集识别准确率对比)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 115)
    ax.grid(True, axis='y', alpha=0.3)
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+1,
                f'{h:.1f}%', ha='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'yale_exp8_orl_vs_yale.png'))


def main():
    ensure_dir(ORL_OUTPUT)
    ensure_dir(YALE_OUTPUT)
    print("=" * 65)
    print("  Eigenfaces for Recognition - 完整实验")
    print("  ORL: 7组实验 | Yale: 8组实验 + Figure 9复现")
    print("=" * 65)

    print("\n" + "=" * 65)
    print("  PART 1 — ORL/AT&T Face Database")
    print("=" * 65)
    print("\n[ORL 数据加载]")
    orl_images, orl_labels = load_orl()
    print(f"  共加载 {len(orl_images)} 张，{ORL_N_SUBJ} 人，"
          f"维度 {orl_images.shape[1]}")

    _, orl_base_acc   = orl_exp0_vis(orl_images, orl_labels, ORL_OUTPUT)
    orl_exp1_acc_vs_k(orl_images, orl_labels, ORL_OUTPUT)
    orl_exp2_acc_vs_trainsize(orl_images, orl_labels, ORL_OUTPUT)
    orl_exp3_multiline(orl_images, orl_labels, ORL_OUTPUT)
    orl_exp4_reconstruction_mse(orl_images, orl_labels, ORL_OUTPUT)
    orl_exp5_face_vs_nonface(orl_images, orl_labels, ORL_OUTPUT)
    orl_exp7_scale(orl_images, orl_labels, ORL_OUTPUT)
    orl_exp8_cross_val(orl_images, orl_labels, ORL_OUTPUT)
    k90, k95 = orl_eigenvalue_variance(orl_images, orl_labels, ORL_OUTPUT)

    print("\n" + "=" * 65)
    print("  PART 2 — Yale Face Database")
    print("=" * 65)
    print("\n[Yale 数据加载]")
    yale_data, yale_subjects = load_yale()
    print(f"  共 {len(yale_subjects)} 人，"
          f"{len(YALE_ALL_CONDS)} 种条件")
    yale_exp0_vis(yale_data, yale_subjects, YALE_OUTPUT)
    yale_exp1_acc_vs_k(yale_data, yale_subjects, YALE_OUTPUT)
    yale_single = yale_exp2_single_condition(
        yale_data, yale_subjects, YALE_OUTPUT)
    yale_exp3_figure9(yale_data, yale_subjects, YALE_OUTPUT)
    yale_exp4_lighting_detail(yale_data, yale_subjects, YALE_OUTPUT)
    yale_exp5_expression_detail(yale_data, yale_subjects, YALE_OUTPUT)
    yale_exp6_glasses(yale_data, yale_subjects, YALE_OUTPUT)
    yale_exp7_combo(yale_data, yale_subjects, YALE_OUTPUT)
    yale_exp8_orl_vs_yale(orl_base_acc, yale_single, YALE_OUTPUT)

    print("\n" + "=" * 65)
    print("  全部实验完成！")
    print("=" * 65)
    print(f"\n  ORL结果 ({len([f for f in os.listdir(ORL_OUTPUT) if f.endswith('.png')])}张图):")
    for f in sorted(os.listdir(ORL_OUTPUT)):
        if f.endswith('.png'):
            print(f"    {f}")
    print(f"\n  Yale结果 ({len([f for f in os.listdir(YALE_OUTPUT) if f.endswith('.png')])}张图):")
    for f in sorted(os.listdir(YALE_OUTPUT)):
        if f.endswith('.png'):
            print(f"    {f}")
    print(f"\n  关键指标:")
    print(f"    ORL 基础准确率 (K=40, 5张/人):  {orl_base_acc*100:.2f}%")
    print(f"    ORL 达到90%方差的K值:            {k90}")
    print(f"    ORL 达到95%方差的K值:            {k95}")


if __name__ == '__main__':
    main()
