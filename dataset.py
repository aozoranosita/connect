class ConnectomicsDataset(Dataset):
    def __init__(self, h5_file, transform=None):
        self.h5_file = h5_file
        self.transform = transform
        with h5py.File(h5_file, 'r') as f:
            self.data = f['raw'][:]
            self.labels = f['label'][:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
