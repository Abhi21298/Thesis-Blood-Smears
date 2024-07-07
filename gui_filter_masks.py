
import pandas as pd
import os
import shutil
import cv2 
import numpy as np
import matplotlib.pyplot as plt

def edit_csv_v1(path):
    
    excluded_imgs = os.path.join(os.path.dirname(path), r"excluded")
    os.makedirs(excluded_imgs, exist_ok=True)

    for root, subdirs, files in os.walk(path):
        if files == [] or files == ():
            continue
        
        try:
            #csv_file = os.path.join(root, str(root.rsplit("\\", maxsplit = 1)[1]) + ".csv")
            csv_file = os.path.join(root, str(os.path.basename(root)) + ".csv")
            #print(csv_file)

            df = pd.read_csv(csv_file, header=0).to_dict('records')
            new_df = []

            for rows in df:
                # if rows['label'] == "RBC" and (rows["area"] > 1500 or rows["area"] < 300):
                #     rows["label"] = ""
                #     os.remove(os.path.join(root, str(rows['id'] + '.png')))
                if rows["area"] < 350 or rows["area"] > 1900:
                    rm_path = os.path.join(root, str(rows['id']) + '.png')
                    #print(rm_path)
                    if os.path.exists(rm_path):
                        shutil.move(rm_path, os.path.join(excluded_imgs, str(rows['id']) + '.png'))
                else:    
                    new_df.append(rows)
        except:
            continue


        print(csv_file)
        df = pd.DataFrame(new_df)
        df.to_csv(csv_file, index= False)

def edit_csv(path, cell_area_threshold = 300):
    
    excluded_imgs = os.path.join(os.path.dirname(path), r"excluded")
    os.makedirs(excluded_imgs, exist_ok=True)
    tracker = np.zeros((512,512), dtype=np.uint8)
    print("inside function")
    
    for root, subdirs, files in os.walk(path):
        if files == [] or files == ():
            continue
        
        try:
            #csv_file = os.path.join(root, str(root.rsplit("\\", maxsplit = 1)[1]) + ".csv")
            csv_file = os.path.join(root, str(os.path.basename(root)) + ".csv")
            df = pd.read_csv(csv_file, header=0).sort_values(by="area").to_dict('records')
            
            new_df = {}
            ids = []
            bbox_x = []
            bbox_y = []
            bbox_w = []
            bbox_h = []
            
            for rows in df:
                img_path = os.path.join(root, str(rows['id']) + '.png')
                #print(img_path)
                if (int(rows["area"]) < cell_area_threshold or int(rows['area'] > 3000)) and os.path.exists(img_path):
                    shutil.move(img_path, os.path.join(excluded_imgs, str(rows['id']) + '.png'))
                    continue
                
                img = cv2.imread(img_path, 0)
                prev = tracker.copy()
                tracker = cv2.add(tracker, img)
                diff = cv2.subtract(tracker, prev)
                _, final_diff = cv2.threshold(diff, 1, 255, cv2.THRESH_BINARY)

                ## find if there are more than 1 masks despite getting difference mask 
                num_labels, labels = cv2.connectedComponents(final_diff)

                for label in range(1, num_labels):

                    component_mask = (labels == label).astype(np.uint8) * 255
                    area = (labels==label).sum()
                    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if contours and area > 500:
                        x,y,w,h = cv2.boundingRect(contours[0])

                        #if w*h < connected_com_thresh:
                        #    continue
                        
                        bbox_x.append(x)
                        bbox_y.append(y)
                        bbox_w.append(w)
                        bbox_h.append(h)
                        #print(w, h, w*h)
                        
                        id = str(rows['id']) + '_' + str(label)
                        ids.append(id)
                        image_name = os.path.join(root, id + '.png')
                        
                        if not os.path.exists(image_name):
                            cv2.imwrite(image_name, component_mask)
                
                os.remove(img_path)
        except Exception as e:
            print(e)            
        
        new_df['id'] = ids
        new_df['bbox_x0'] = bbox_x
        new_df['bbox_y0'] = bbox_y
        new_df['bbox_w'] = bbox_w
        new_df['bbox_h'] = bbox_h
        tracker = np.zeros((512,512), dtype=np.uint8)
        print(csv_file)
        df = pd.DataFrame(new_df)
        df.to_csv(csv_file, index= False)

# if __name__ == "__main__":
#     edit_csv(r"D:\UCC\Thesis\segment-anything-main\test\0_0")



        
