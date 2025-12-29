import os
import json
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def create_output_dir(base_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(base_dir, timestamp)
    os.makedirs(output_path, exist_ok=True)
    return output_path

def save_config(config, output_path):
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

def plot_loss_curve(train_losses, val_losses, output_path):
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o", label="Treino")
    plt.plot(range(1, len(val_losses) + 1), val_losses, marker="s", label="Validação")
    plt.title("Loss por Época")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_path, "loss_curve.png"))
    plt.close()

def plot_confusion_matrix(cm, output_path):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=["Ausência", "Achado"],
                yticklabels=["Ausência", "Achado"])
    plt.title("Matriz de Confusão (Teste)")
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.savefig(os.path.join(output_path, "confusion_matrix.png"))
    plt.close()

def plot_roc_curve(y_true, y_probs, output_path):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlabel("Falso Positivo")
    plt.ylabel("Verdadeiro Positivo")
    plt.title("Curva ROC (Teste)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(output_path, "roc_curve.png"))
    plt.close()