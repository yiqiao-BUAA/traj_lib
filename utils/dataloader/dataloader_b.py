from torch.utils.data import Dataset, DataLoader
from traj_lib.utils.register import register_dataset
from traj_lib.utils.logger import get_logger
logger = get_logger(__name__)
dataset_name = 'b'

class MyDataset(Dataset):
    def __init__(self):
        self.data = [2, 2, 2, 2, 2]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
@register_dataset(dataset_name)
class MyDataLoader(DataLoader):
    def __init__(self):
        dataset = MyDataset()
        super(MyDataLoader, self).__init__(dataset, batch_size=1, shuffle=False, num_workers=0)
        logger.debug(f"MyDataLoader  with dataset: {dataset_name}")