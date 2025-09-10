import os
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from zipfile import ZipFile

zip_path = r"C:\Users\Hp\Downloads\tae-10-fold (1).zip"
extract_dir = r"C:\Users\Hp\Desktop\tae-10-fold"   

if not os.path.exists(extract_dir):
    from zipfile import ZipFile
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)



def read_keel(filepath):
    data = []
    with open(filepath, "r") as f:
        lines = f.readlines()
        data_section = False
        for line in lines:
            line = line.strip()
            if not line or line.startswith("@") or line.startswith("%"):
                if line.lower().startswith("@data"):
                    data_section = True
                continue
            if data_section:
                data.append(line.split(","))
    df = pd.DataFrame(data)
    # Poslednja kolona je klasa
    X = df.iloc[:, :-1].astype(float).values
    y = df.iloc[:, -1].values
    return X, y


train_files = sorted([f for f in os.listdir(extract_dir) if f.endswith("tra.dat")])
test_files  = sorted([f for f in os.listdir(extract_dir) if f.endswith("tst.dat")])

print("Nađeno foldova:", len(train_files))

# 4) Evaluacija po foldovima
acc_folds = []
cm_total = None

for tr_file, te_file in zip(train_files, test_files):
    Xtr, ytr = read_keel(os.path.join(extract_dir, tr_file))
    Xte, yte = read_keel(os.path.join(extract_dir, te_file))

    # Normalizacija po train skali
    xmin, xmax = Xtr.min(axis=0), Xtr.max(axis=0)
    Xtr = (Xtr - xmin) / (xmax - xmin + 1e-9)
    Xte = (Xte - xmin) / (xmax - xmin + 1e-9)

    # Definiši MLP 
    mlp = MLPClassifier(hidden_layer_sizes=(50,20),
                        activation="logistic",
                        solver="sgd",
                        learning_rate_init=0.01,
                        max_iter=500,
                        random_state=157)

    mlp.fit(Xtr, ytr)
    ypred = mlp.predict(Xte)

    acc = accuracy_score(yte, ypred)
    acc_folds.append(acc)

    cm = confusion_matrix(yte, ypred, labels=np.unique(ytr))
    if cm_total is None:
        cm_total = cm
    else:
        cm_total += cm

    print(f"{tr_file} → acc = {acc:.3f}")

# 5) Rezime
print("\nProsečna tačnost (10-fold):", np.mean(acc_folds))
print("Standardna devijacija:", np.std(acc_folds))

# 6) Crtanje konfuzione matrice
plt.figure(figsize=(6,5))
sns.heatmap(cm_total, annot=True, fmt="d", cmap="Blues",
            xticklabels=np.unique(ytr), yticklabels=np.unique(ytr))
plt.xlabel("Predikcija")
plt.ylabel("Original")
plt.title("Konfuziona matrica – 10-fold CV (zbirno)")
plt.show()

