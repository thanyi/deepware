from dataset.dataset import DeepfakeDataset
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as cal_auc
from sklearn.metrics import confusion_matrix
import numpy as np
from torch.utils import data

def evaluate(model, normal_root,malicious_root,csv_root, mode='valid',):

    my_dataset = DeepfakeDataset(normal_root=normal_root, malicious_root=malicious_root, mode=mode, resize=299,
                                 csv_root=csv_root)
    malicious_name = malicious_root.split('/')[-1]
    print("This is the {} {} dataset!".format(malicious_name,mode))
    print("dataset size:{}".format(len(my_dataset)))

    bz = 64
    # torch.cache.empty_cache()
    with torch.no_grad():
        y_true, y_pred = [], []

        dataloader = torch.utils.data.DataLoader(
            dataset=my_dataset,
            batch_size=bz,
            shuffle=True,
            num_workers=0
        )

        device = torch.device("cuda")
        correct = 0
        total = len(dataloader)

        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            output = model(x)
            y_pred.extend(output.sigmoid().flatten().tolist())
            y_true.extend(y.flatten().tolist())

            if total % 10 == 0:
                print(f"当前进度：{i}/{total}...")

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)

        AUC = cal_auc(fpr, tpr)

        for i in range(len(y_pred)):
            if y_pred[i] < 0.5:
                y_pred[i] = 0
            else:
                y_pred[i] = 1

        r_acc = accuracy_score(y_true, y_pred)
        con_mat = confusion_matrix(y_true,y_pred)

    return r_acc, AUC, con_mat