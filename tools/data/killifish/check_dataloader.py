import sys
from src.data.behavior.killifish_dataset import KillifishContinuousDataset
from torch.utils.data import DataLoader

def main():
    print("Instantiating dataset...")
    dataset = KillifishContinuousDataset(
        metadata_csv="data/killifish/data/a1_20241119/26441580/df_reformat_10_20241119_join_edit.csv",
        kinematics_dir="data/killifish/data/p3_20230526/test/standardization/"
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        print("Creating dataloader...")
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        batch = next(iter(dataloader))
        print(f"Batch shape: {batch.shape}")
        print(f"Batch dtype: {batch.dtype}")
    else:
        print("Dataset is empty. Something went wrong.")

if __name__ == "__main__":
    main()
