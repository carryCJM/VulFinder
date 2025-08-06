import torch
import math
from sklearn.metrics import classification_report, accuracy_score
# metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, PrecisionRecallDisplay
from sklearn.metrics import auc


class Utilities():
    def get_index_by_coarse_label(self, out, label):
        out = torch.cat(out).cpu().numpy()
        label = torch.cat(label).cpu().numpy()

        print("out:",out)
        print("label:",label)
        
        recall = recall_score(label, out) 
        precision = precision_score(label, out)
        accuracy = accuracy_score(label, out)
        f1 = f1_score(label, out)
        
        
        return {
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "f1_score": f1
        }

