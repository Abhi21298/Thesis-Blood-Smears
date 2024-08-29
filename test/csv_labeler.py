import os
import pandas as pd

def csv_labeller(path):
    
    for root, subdirs, files in os.walk(path):
        if files == [] or files == ():
            continue

        csv_file = os.path.join(root, str(root.rsplit("\\", maxsplit = 1)[1]) + ".csv")
        
        df = pd.read_csv(csv_file, header=0)

        if 'label' in df.columns:
            continue
        
        print(csv_file)
        labels = []
        for rows in df['area']:
            if rows > 300:
                labels.append("RBC")
            else:
                labels.append("")
            
        df['label'] = labels
        df.to_csv(csv_file, index= False)
        
        

if __name__ == "__main__":
    csv_labeller(r"D:\UCC\Thesis\segment-anything-main\assets\Mask_sub_folders")
