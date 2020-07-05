import os
import json
import torch

class Args():
    """
    Reading json file and using the __dict__
    """

    def __init__(self, json_path):
        self.json_path=json_path
        self.update(json_path)

    def save_as(self,json_path):
        with open(json_path,'w') as f:
            json.dump(self.__dict__, f, indent=4)
    
    def save(self):
        with open(self.json_path,'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self,json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def merge_params(self, b):
        self.__dict__.update(b.__dict__)

    @property
    def dict(self):
        return self.__dict__


def get_files(path, extension=None):
    assert os.path.isdir(path), "Path not exist"
    files = []
    ext_cond = lambda f: f.endswith(extension) if extension is not None else True
    
    for dir_path, _, file_names in os.walk(path):
        for file in file_names:
            if ext_cond(file):
                files.append(os.path.join(dir_path, file))

    return files


def save_model(path, name, model, optimizer, epoch, metric):
    """
    Save the model ckpt format
    """
    if not os.path.isdir(path):
        os.mkdir(path)

    model_path = os.path.join(path, name)
    ckpt = {
        'epoch': epoch,
        'metric': metric,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(ckpt, model_path)

def load_model(path, name, model, optimizer):
    """
    Load the model ckpt format
    """
    model_path = os.path.join(path, name)
    assert os.path.isfile(model_path), "The model file {} does not exist."

    if not torch.cuda.is_available():
        ckpt = torch.load(model_path, map_location='cpu')
    else:
        ckpt = torch.load(model_path)

    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    epoch = ckpt['epoch']
    metric = ckpt['metric']

    return model, optimizer, epoch, metric

    