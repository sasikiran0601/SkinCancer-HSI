import torch
import yaml
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from torch.amp import autocast
from src.cli import get_model
from src.hsi_dataset import make_dataloader
from tqdm import tqdm

def evaluate(checkpoint_path="best_finetuned_self_attention.pth"):
    with open("configs/finetune.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = get_model(
        config.get("model", "self_attention"),
        dropout=config.get("dropout", 0.4),
        attn_dim=config.get("attn_dim", 128)
    )
    
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint {checkpoint_path} not found!")
        return
        
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    for split_name in ["val", "test"]:
        loader = make_dataloader(
            split=split_name,
            batch_size=config.get("batch_size", 32),
            label_mode="multi",
            normalization=config.get("normalization", "per_pixel"),
            data_dir=config.get("data_dir", "../preprocessing/outputs"),
            shuffle=False
        )
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data in tqdm(loader, desc=f"{split_name.upper()} Eval"):
                x, y = batch_data[0].to(device), batch_data[1].to(device)
                outputs = model(x)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        
        print(f"\n{'='*50}")
        print(f"{split_name.upper()} SET EVALUATION")
        print(f"{'='*50}")
        print("\nConfusion Matrix (Rows: True [BE, BM, ME, MM], Cols: Pred [BE, BM, ME, MM]):")
        cm = confusion_matrix(all_labels, all_preds)
        print(cm)
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=["BE", "BM", "ME", "MM"], zero_division=0, digits=4))

import sys

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "best_finetuned_self_attention.pth"
    print(f"Evaluating checkpoint: {ckpt}")
    evaluate(checkpoint_path=ckpt)
