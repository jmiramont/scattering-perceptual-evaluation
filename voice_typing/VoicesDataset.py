from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np


class VoicesDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset_name, p= ''):
        """
        Returns signals and classification from the desired dataset.
        """

        if dataset_name == 'all':
            meei = VoicesDataset('meei')
            svd = VoicesDataset('svd')

            self.data = np.concatenate([meei.data, svd.data])
            self.tipos = np.concatenate([meei.tipos, svd.tipos])


        else:

            self.data = pd.read_csv(p + 'signals_' + dataset_name + '.csv', header=None)
            self.data = self.data.to_numpy()
            self.tipos = pd.read_csv(p + 'labels_' + dataset_name + '.csv', header=None)
            self.tipos = self.tipos.to_numpy()
            #self.tipos[self.tipos == 4] = 3
            self.tipos = self.tipos - 1
            self.tipos = self.tipos.T

    def __len__(self):
        return len(self.tipos)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # img_name = os.path.join(self.root_dir,
        #                        self.landmarks_frame.iloc[idx, 0])
        # image = io.imread(img_name)
        # landmarks = self.landmarks_frame.iloc[idx, 1:]
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        # print(self.tipos.shape)
        labels = self.tipos[idx]
        signals = self.data[idx, :]
        # sample = {'signal': signals, 'label': labels}
        return signals, labels


# Si se corre el archivo independientemente del resto ejecuta lo que sigue.
if __name__ == '__main__':
    voces = VoicesDataset('')
    print(len(voces))
    voces_data = DataLoader(voces, batch_size=100, shuffle=True)

    for datos, etiquetas in voces_data:
        print(datos)
        print(etiquetas)
