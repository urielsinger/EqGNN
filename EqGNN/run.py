import numpy as np
import torch

from EqGNN import config
from EqGNN.datasets import GraphDataModule
from EqGNN.models import GraphModel

hparams = config.parser.parse_args()

np.random.seed(hparams.seed)
torch.manual_seed(hparams.seed)

dm = GraphDataModule(dataset_name=hparams.dataset, sensitive_attribute=hparams.sensitive_attribute)
dm.prepare_data()
dm.setup()

model = GraphModel(hparams)
model.fit(datamodule=dm)
model.test(datamodule=dm)