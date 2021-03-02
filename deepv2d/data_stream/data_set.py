import torch.utils.data.Dataset as Dataset



class DatasetAdatper(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_obj):
        """
        数据集封装类
        """
        self.data_obj = data_obj

    def __len__(self):
        return len(self.data_obj)

    def __getitem__(self, idx):
       return 