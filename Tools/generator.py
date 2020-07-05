import os
import csv
import numpy as np
import pandas as pd
import skimage.io as io
import random 
import Tools.wearmask

random.seed(2333)

class CelebA_Pre():
    def __init__(self, root, list_bbox, list_ldmk, list_parts):
        """
        `root`: dir, the path of images\\
        `list_*`, file path, csv files
        """
        assert os.path.isdir(root)
        self.root = root
        self.list_bbox = list_bbox
        self.list_ldmk = list_ldmk
        self.list_parts = list_parts

        self._init_DFs()


    def _init_DFs(self):
        """
        get the dataframes. 
        """
        self.DF_bbox = pd.read_csv(self.list_bbox, index_col=0)
        self.DF_ldmk = pd.read_csv(self.list_ldmk, index_col=0)
        self.DF_parts = pd.read_csv(self.list_parts)

    def get_parts(self, mode='eval'):
        """
        splitting all files into train/eval/test 
        Keep same partition as kaggle.
        ### Params
        `mode`: 'train', 'eval' or 'test'
        """
        mode_map = {
            'train':0,
            'eval':1,
            'test':2
        }

        ids = self.DF_parts[self.DF_parts['partition'] == mode_map[mode]]['image_id'].tolist()
        return ids

    def read_annote(self, id):
        """
        Get the annotation of given id. 

        ### Parames
        `id`: str, the id of target img, like '000003.jpg'
        
        ### Return
        `img_path`: str, the path for img,\\
        `bbox`: list of int, [x_1, y_1, width, height]\\
        `ldmk`: list of int, five point landmark, with shape (10, )\\
                lefteye_x  lefteye_y  righteye_x  righteye_y  nose_x  nose_y  leftmouth_x  leftmouth_y  rightmouth_x  rightmouth_y
        """
        img_path = os.path.join(self.root, id)
        bbox = self.DF_bbox.loc[id].to_numpy()
        ldmk = self.DF_ldmk.loc[id].to_numpy()

        return img_path, bbox, ldmk

    def cross_val_folds(self, img_root, train_ids, eval_ids, test_ids, num_folds=5):
        """
        Generating the cross validation. 
        1. legitimate check
        2. split
        
        ### Return 
        `legi_folds`: including the path of legitimate images (wear mask)\\
        `eval_folds`: list of list, each element is a list of path for evaluation. 
        """
        train_ids.extend(eval_ids)
        legi_folds = []
        for id in train_ids:
            id = id.split('.')[0] + '.png'
            f = os.path.join(img_root, id)
            if os.path.isfile(f):
                legi_folds.append(f)
            else:
                pass
        
        total_len = len(legi_folds)

        chuck_size = total_len // num_folds
        # cross validation
        random.shuffle(legi_folds)
        train_folds = []
        eval_folds = [legi_folds[i*chuck_size:(i+1)*chuck_size] if (i+1) != num_folds else legi_folds[i*chuck_size:] for i in range(num_folds)]
        legi_folds = set(legi_folds)
        # XOR operation to get train_folds
        for fold in eval_folds:
            set_eval = set(fold)
            train_folds.append(list(legi_folds.symmetric_difference(set_eval)))

        return legi_folds, train_folds, eval_folds
        

        

        


    
# Example:
if __name__ == "__main__":
    
    celeba = CelebA_Pre('../data/CelebA/img_align_celeba/', 
                            './Dataset/CelabA/list_bbox_celeba.csv',
                            './Dataset/CelabA/list_landmarks_align_celeba.csv',
                            './Dataset/CelabA/list_eval_partition.csv')
    
    train_ids = celeba.get_parts('train')
    eval_ids = celeba.get_parts('eval')
    test_ids = celeba.get_parts('test')

    # Just a example for cross validation. 
    legi_folds, eval_folds = celeba.cross_val_folds('../data/CelebA/img_align_celeba/', train_ids, eval_ids, num_folds=5)
    # quick show
    print("*"*10)
    print("some img path")
    print(legi_folds[0])
    print(eval_folds[0][0])
    print("*"*10)

    # read some 
    id = train_ids[0]
    img_path, bbox, ldmk = celeba.read_annote(id)
    img_path = '000001.jpg'
    img = io.imread(img_path)
    ##......
    ## Adding draw annotations (omit)
    ## Adding Facial Mask (omit)
    flag, img = wearmask.FaceMasker(img, True, 'hog').mask() # flag denotes whether we can add mask on the img
    ##......
    if flag:
        io.imsave('train_0.jpg', img)
