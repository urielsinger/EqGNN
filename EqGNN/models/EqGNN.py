import os
from datetime import datetime
import pytz
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import wandb
import torch
from torch import nn
from torch.distributions.categorical import Categorical
from scipy.stats import entropy
from sklearn.metrics import f1_score

from EqGNN.models import GNN


class GraphModule(pl.LightningModule):
    def __init__(self, hparams, in_feats, out_feats, out_label, in_label=None, dropout=0.5, type='cce', lr=None):
        super(GraphModule, self).__init__()
        self.hparams = hparams
        self.in_label = in_label
        self.out_label = out_label
        self.lr = lr if lr is not None else hparams.learning_rate

        self.model = GNN(in_feats, hparams.embedding_size, hparams.embedding_size, dropout, embedding=in_label is None)
        self.last_layer = nn.Linear(hparams.embedding_size, out_feats)

        self.type = type
        if self.type not in ['cce', 'bce', 'regression']:
            raise Exception('no such type', type)


    def apply_attribute_mask(self, y_proba, y_true, attribute_mask=None):
        if self.type == 'cce' and attribute_mask is not None:
            y_true = y_true[:, attribute_mask]
            y_proba = y_proba[:, attribute_mask]
        return y_proba, y_true

    def loss(self, y_proba, y_true):
        if self.type == 'cce':
            weight = 1 / y_true.sum(axis=0)
            loss = torch.nn.CrossEntropyLoss(weight)(y_proba, torch.argmax(y_true, axis=1))
            values = torch.argmax(y_proba, axis=1).cpu().detach().numpy()
            true_values = torch.argmax(y_true, axis=1).cpu().detach().numpy()
        elif self.type == 'bce':
            num_true = y_true.sum()
            num_samples = len(y_true)
            pos_weight = float(num_samples - num_true) / num_true
            loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)(y_proba, y_true)
            values = torch.sigmoid(y_proba).cpu().detach().numpy().squeeze()
            true_values = y_true.cpu().detach().numpy().squeeze()
        elif self.type == 'regression':
            loss = torch.nn.L1Loss()(y_proba, y_true)
            values = y_proba.cpu().detach().numpy().squeeze()
            true_values = y_true.cpu().detach().numpy().squeeze()
        else:
            raise Exception(f'no such type: {self.type}')
        return loss, values, true_values

    def forward(self, graph, inputs=None):
        if inputs is None:
            node_indices = graph.nodes()
            h = self.model(graph, node_indices)
        else:
            h = self.model(graph, inputs)

        return self.last_layer(h), h

    def predict(self, batch):
        graph = batch[0]['graph']
        y_true = batch[0][self.out_label].float()
        mask = batch[0]['mask']
        inputs = None

        if not self.in_label:
            y_proba, h = self.forward(graph)
        else:
            inputs = batch[0][self.in_label].float()
            y_proba, h = self.forward(graph, inputs=inputs)
            if inputs.shape[1] > 1:
                inputs = torch.argmax(inputs, axis=1)

        return y_proba, h, y_true, mask, inputs

    def sample(self, batch, return_true=False):
        y_proba, _, y_true, mask, X = self.predict(batch)

        attribute2mask = batch[0]['attribute2mask']
        if self.out_label in attribute2mask:
            y_proba = y_proba[:, attribute2mask[self.out_label]]

        if self.hparams.graph_sampler:
            if self.type == 'bce':
                y_proba = torch.sigmoid(y_proba)
                y_proba = torch.cat([1-y_proba, y_proba], axis=1)
            elif self.type == 'cce':
                y_proba = torch.softmax(y_proba, axis=1)
            sample_index = Categorical(probs=y_proba).sample()
        else:
            sample_index = Categorical(probs=torch.Tensor(self.matrix_values[:, X.cpu()].T)).sample()
        y_sample = np.zeros_like(y_true.cpu())
        if self.type == 'bce':
            indices = sample_index == 1
            y_sample[np.arange(len(y_sample))[indices.cpu()]] = 1
        elif self.type == 'cce':
            y_sample[np.arange(len(y_sample)), sample_index.cpu()] = 1

        if self.out_label in attribute2mask:
            y_sample[y_true.cpu()[:, ~attribute2mask[self.out_label]].squeeze() == 1] = np.array(~attribute2mask[self.out_label], dtype=np.int)
        y_sample = torch.Tensor(y_sample).float().to(self.device)

        if return_true:
            return y_sample, y_true
        return y_sample

    def step(self, batch: dict, optimizer_idx: int = None, name: str = 'loss'):
        attribute_mask = batch[0]['attribute2mask'][self.out_label] if self.out_label in batch[0]['attribute2mask'] else None
        y_proba, _, y_true, mask, X = self.predict(batch)
        y_proba_clean, y_true_clean = self.apply_attribute_mask(y_proba, y_true, attribute_mask)
        loss, values, true_values = self.loss(y_proba_clean[mask], y_true_clean[mask])
        self.log(f'{self.out_label}/{name}_loss', loss)
        self.log(f'{self.out_label}/{name}_pred_histogram', wandb.Histogram(values), reduce_fx=lambda x: x[0])
        self.log(f'{self.out_label}/{name}_true_histogram', wandb.Histogram(true_values), reduce_fx=lambda x: x[0])

        if X is not None and self.current_epoch == self.trainer.max_epochs - 1:
            X = X.cpu().detach().numpy()
            x_labels, y_labels = list(np.sort(np.unique(X))), list(np.sort(np.unique(y_true.cpu().detach().numpy())))
            if self.type == 'bce':
                values = np.array(values >= 0.5, dtype=np.int).squeeze()
            matrix_values = np.array([[sum(values[X[mask] == x_label] == y_label) for x_label in x_labels] for y_label in y_labels])
            matrix_values = matrix_values/np.sum(matrix_values, axis=0, keepdims=True)
            matrix_values[matrix_values != matrix_values] = 0
            self.log(f'{self.out_label}/{name}_pred_heatmap', wandb.plots.HeatMap(x_labels, y_labels, matrix_values), reduce_fx=lambda x: x[0])
            matrix_values = np.array([[sum(true_values[X[mask] == x_label] == y_label) for x_label in x_labels] for y_label in y_labels])
            matrix_values = matrix_values / np.sum(matrix_values, axis=0, keepdims=True)
            matrix_values[matrix_values != matrix_values] = 0
            # self.log(f'{self.out_label}/{name}_true_heatmap', wandb.plots.HeatMap(x_labels, y_labels, matrix_values), reduce_fx=lambda x: x[0])
            self.matrix_values = matrix_values
            self.matrix_values[0, np.sum(self.matrix_values, axis=0) == 0] = 1

        return loss

    def training_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None):
        loss = self.step(batch, optimizer_idx, name='train')
        return {'loss': loss}

    def validation_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None):
        self.step(batch, optimizer_idx, name='val')

    def test_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None):
        self.step(batch, optimizer_idx, name=f'test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.last_layer.parameters()), lr=self.lr)
        return [optimizer]


class AdversarialGraphModule(GraphModule):
    def __init__(self, hparams, in_feats, out_feats, out_label, disc_label, disc_feats, model_P_A, in_label=None, dropout=0.5, type='cce'):
        super(AdversarialGraphModule, self).__init__(hparams, in_feats, out_feats, out_label, in_label=in_label, dropout=dropout, type=type)

        self.disc_label = disc_label
        self.model_P_A = model_P_A

        hidden_size = hparams.embedding_size if self.hparams.use_hidden else 0
        if self.hparams.discriminator_loss == 'permutation':
            dims = (2 * (out_feats + disc_feats) + hidden_size, int(out_feats + disc_feats + hidden_size/2), 1)
        elif self.hparams.discriminator_loss == 'FairGNN':
            dims = (hparams.embedding_size, disc_feats)
        elif self.hparams.discriminator_loss == 'Debias':
            dims = (2 * out_feats + hidden_size, int(out_feats + hidden_size/2), disc_feats)
        elif self.hparams.discriminator_loss in ['paired', 'unpaired']:
            dims = (2 * out_feats + disc_feats + hidden_size, int(out_feats + (disc_feats + hidden_size) / 2), 1)
        elif self.hparams.discriminator_loss == 'None':
            dims = None
        else:
            raise Exception(f'no such discriminator_loss: {self.hparams.discriminator_loss}')

        if dims is not None:
            if self.hparams.NN_discriminator or self.hparams.discriminator_loss == 'FairGNN':
                layers = [nn.Linear(dims[0], dims[1])]
                for i in range(1, len(dims) - 1):
                    layers.append(nn.ReLU())
                    layers.append(nn.Linear(dims[i], dims[i + 1]))
                self.discriminator = nn.Sequential(*layers)
            else:
                self.discriminator = GNN(dims[0], dims[1], dims[2], dropout=0)

        self.bce_loss = torch.nn.BCELoss()
        self.bce_withlogits_loss = torch.nn.BCEWithLogitsLoss()

    def discriminator_step(self, y_proba, y_true, h, attributes, dummy_attributes, graph, mask, flag=True):
        def discriminate(samples, graph):
            if self.hparams.NN_discriminator:
                return self.discriminator(samples)
            else:
                return self.discriminator(graph, samples)
        real_samples = torch.cat([y_proba, attributes, y_true], axis=1)
        fake_samples = torch.cat([y_proba, dummy_attributes, y_true], axis=1)

        if self.hparams.discriminator_loss == 'permutation':
            true_permutations = np.random.choice([0, 1], size=len(y_proba))
            both_attributes = torch.stack([attributes, dummy_attributes])
            attributes_permute = torch.cat([both_attributes[list(true_permutations), range(len(true_permutations))],
                                            both_attributes[list(1 - true_permutations), range(len(true_permutations))]], axis=1)
            if self.hparams.use_hidden:
                paires = torch.cat([y_proba, attributes_permute, y_true, h], axis=1).to(self.device)
            else:
                paires = torch.cat([y_proba, attributes_permute, y_true], axis=1).to(self.device)
            pred_permutations = discriminate(paires, graph).squeeze()
            true_permutations = torch.Tensor(true_permutations).to(self.device)
            cov = self.covariance(fake_samples[mask], real_samples[mask])
            if flag:
                d_loss = self.bce_withlogits_loss(pred_permutations[mask], true_permutations[mask])
            else:
                d_loss = self.bce_withlogits_loss(pred_permutations[mask], 1-true_permutations[mask])

        elif self.hparams.discriminator_loss == 'FairGNN':
            pred_attributes = self.discriminator(h)
            proba_attributes = torch.sigmoid(pred_attributes)
            A = torch.cat((proba_attributes[mask], y_proba[mask]), axis=1)
            cov = self.covariance(A)
            if flag:
                d_loss = self.bce_withlogits_loss(pred_attributes, attributes)
            else:
                d_loss = self.bce_withlogits_loss(pred_attributes, 1 - attributes)

        elif self.hparams.discriminator_loss == 'Debias':
            if self.hparams.use_hidden:
                paires = torch.cat([y_proba, y_true, h], axis=1).to(self.device)
            else:
                paires = torch.cat([y_proba, y_true], axis=1).to(self.device)
            pred_attributes = discriminate(paires, graph)
            cov = 0
            if flag:
                d_loss = self.bce_withlogits_loss(pred_attributes, attributes)
            else:
                d_loss = self.bce_withlogits_loss(pred_attributes, 1 - attributes)

        elif self.hparams.discriminator_loss == 'paired':
            cov = self.covariance(fake_samples[mask], real_samples[mask])
            if self.hparams.use_hidden:
                real_samples = torch.cat([real_samples, h], axis=1)
                fake_samples = torch.cat([fake_samples, h], axis=1)
            real_proba = torch.sigmoid(discriminate(real_samples, graph).squeeze())[mask]
            fake_proba = torch.sigmoid(discriminate(fake_samples, graph).squeeze())[mask]
            diff = (real_proba - fake_proba)
            d_loss = diff.mean()
            if flag:
                d_loss = - d_loss

        elif self.hparams.discriminator_loss == 'unpaired':
            cov = self.covariance(fake_samples[mask], real_samples[mask])
            if self.hparams.use_hidden:
                real_samples = torch.cat([real_samples, h], axis=1)
                fake_samples = torch.cat([fake_samples, h], axis=1)
            real_proba = self.discriminator(graph, real_samples).squeeze()[mask]
            fake_proba = self.discriminator(graph, fake_samples).squeeze()[mask]
            if flag:
                real_fake = torch.cat([torch.ones(len(real_proba)), torch.zeros(len(fake_proba))]).to(self.device)
            else:
                real_fake = torch.cat([torch.zeros(len(real_proba)), torch.ones(len(fake_proba))]).to(self.device)
            real_fake_proba = torch.cat([real_proba, fake_proba])
            d_loss = self.bce_withlogits_loss(real_fake_proba, real_fake)

        return d_loss, cov

    def step(self, batch: dict, optimizer_idx: int = None, name='train'):
        results = {}
        attribute_mask = batch[0]['attribute2mask'][self.out_label] if self.out_label in batch[0]['attribute2mask'] else None

        # Classifier
        if optimizer_idx == 0:
            # sample dummy_attributes
            self.dummy_attributes, self.attributes = self.model_P_A.sample(batch, return_true=True)

            # predict
            self.y_proba, self.h, self.y_true, self.mask, X = self.predict(batch)
            y_proba, y_true = self.apply_attribute_mask(self.y_proba, self.y_true, attribute_mask=attribute_mask)
            g_loss, values, true_values = self.loss(y_proba[self.mask], y_true[self.mask])
            if self.type == 'cce':
                self.y_proba = torch.softmax(self.y_proba, axis=1)
            elif self.type == 'bce':
                self.y_proba = torch.sigmoid(self.y_proba)

            # discriminate
            if self.hparams.discriminator_loss != 'None':
                d_loss, cov = self.discriminator_step(self.y_proba, self.y_true, self.h, self.attributes, self.dummy_attributes, batch[0]['graph'], self.mask, flag=False)
            else:
                d_loss, cov = 0, 0

            # final loss
            loss = g_loss + self.hparams.lmb * (d_loss + self.hparams.gamma * cov)
            results[f'generator/{name}_g_loss'] = g_loss
            results[f'generator/{name}_d_loss'] = d_loss
            results[f'generator/{name}_loss'] = loss
            self.log(f'{self.out_label}/{name}_pred_histogram', wandb.Histogram(values), reduce_fx=lambda x: x[0])
            self.log(f'{self.out_label}/{name}_true_histogram', wandb.Histogram(true_values), reduce_fx=lambda x: x[0])

            # fair metrics
            if self.type == 'bce':
                values = np.array(values >= 0.5, dtype=np.int).squeeze()
            attributes = self.attributes[self.mask].squeeze().detach().cpu().numpy()
            fair_results = self.fair_metrics(values, true_values, attributes)
            for k in fair_results:
                results[f'fairness/{name}_{k}'] = fair_results[k]

            results[f'generator/{name}_accuracy'] = np.mean(values == true_values)
            results[f'generator/{name}_f1_macro'] = f1_score(true_values, values, average='macro')
            results[f'generator/{name}_f1_micro'] = f1_score(true_values, values, average='weighted')

        # Discriminator
        if optimizer_idx == 1:
            self.y_proba = self.y_proba.detach()
            self.h = self.h.detach()
            loss, _ = self.discriminator_step(self.y_proba, self.y_true, self.h, self.attributes, self.dummy_attributes, batch[0]['graph'], self.mask, flag=True)
            results[f'discriminator/{name}_loss'] = loss

        for k, v in results.items():
            self.log(k, v)

        return loss, results

    @staticmethod
    def covariance(Z, W=None, scale=1.0):
        if W is None:
            W = Z
        # Center X,Xk
        mZ = Z - torch.mean(Z, 0, keepdim=True)
        mW = W - torch.mean(W, 0, keepdim=True)
        # Compute covariance matrices
        SZZ = torch.mm(torch.t(mZ), mZ) / mZ.shape[0]
        SWW = torch.mm(torch.t(W), mW) / mW.shape[0]

        # Compute loss
        T = (SZZ - SWW).pow(2).sum() / scale
        return T

    def training_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None):
        if self.hparams.discriminator_loss == 'None':
            loss, results = self.step(batch, 0, name='train')
        else:
            loss, results = self.step(batch, optimizer_idx, name='train')
        return {'loss': loss, 'prog': results}

    def validation_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None):
        results = {}
        if self.hparams.discriminator_loss == 'None':
            results.update(self.step(batch, 0, name='val')[1])
        else:
            for i in range(len(self.optimizers())):
                results.update(self.step(batch, i, name='val')[1])
        return results

    def test_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None):
        results = {}
        if self.hparams.discriminator_loss == 'None':
            results.update(self.step(batch, 0, name='test')[1])
        else:
            for i in range(len(self.optimizers())):
                results.update(self.step(batch, i, name='test')[1])
        return results

    def configure_optimizers(self):
        optimizer1 = super(AdversarialGraphModule, self).configure_optimizers()[0]
        optimizer1.weight_decay = self.hparams.weight_decay
        if self.hparams.discriminator_loss == 'None':
            return [optimizer1]

        optimizer2 = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return [optimizer1, optimizer2]

    def fair_metrics(self, y_proba, y_true, attributes):
        parity, equality, weight = [], [], []

        unique_labels = np.unique(y_true)
        unique_attributes = np.unique(attributes)
        for label in unique_labels:
            label_indices = y_true == label
            pred_y = y_proba == label

            label_parity, label_equality = [], []
            for attribute in unique_attributes:
                attribute_indices = attributes == attribute
                attribute_label_indices = attribute_indices * label_indices

                cur_parity = sum(pred_y[attribute_indices]) / sum(attribute_indices)
                cur_equality = sum(pred_y[attribute_label_indices]) / sum(attribute_label_indices)

                label_parity.append(cur_parity)
                label_equality.append(cur_equality)

            if len(unique_attributes) == 2:
                parity.append(abs(label_parity[1]-label_parity[0]))
                equality.append(abs(label_equality[1]-label_equality[0]))
            else:
                norm = 1 / entropy([1] * len(unique_attributes), base=2)
                parity.append(norm * entropy(label_parity, base=2))
                equality.append(norm * entropy(label_equality, base=2))
            weight.append(sum(label_indices))

        parity = np.array(parity)
        equality = np.array(equality)
        weight = np.array(weight) / sum(weight)

        results = {'parity': np.max(parity), 'equality': np.max(equality)}
        return results

class GraphModel:
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        name = f"{hparams.discriminator_loss}-{hparams.lmb}-{hparams.gamma}"
        self.logger = WandbLogger(name=name, save_dir=hparams.log_path,
                                  version=datetime.now().strftime('%y%m%d_%H%M%S.%f'),
                                  project='EqGNN', config=hparams)
        self.ckpt_path = os.path.join(self.logger.save_dir, self.logger._project, self.logger.version)

        if hparams.discriminator_loss == 'None':
            hparams.lmb = 0


    def fit(self, datamodule):
        sample = datamodule.train.__getitem__(0)
        num_feats = len(sample['features'][0])
        num_labels = len(sample['labels'][0])
        num_sensitive_attribute = len(sample['sensitive_attribute'][0])

        # Sampler
        self.model_P_A = GraphModule(hparams=self.hparams,
                                     in_feats=num_labels,
                                     out_feats=num_sensitive_attribute,
                                     in_label='labels',
                                     out_label='sensitive_attribute',
                                     dropout=0,
                                     type='bce' if num_sensitive_attribute==1 else 'cce')
        trainer = pl.Trainer(gpus=self.hparams.gpus,
                             max_epochs=3*self.hparams.epochs if self.hparams.graph_sampler else 1,
                             logger=self.logger,
                             log_every_n_steps=1)
        trainer.fit(self.model_P_A, datamodule=datamodule)

        # EqOdd
        self.model = AdversarialGraphModule(hparams=self.hparams,
                                            in_feats=num_feats,
                                            out_feats=num_labels,
                                            in_label='features',
                                            out_label='labels',
                                            dropout=self.hparams.dropout,
                                            type='bce' if num_labels==1 else 'cce',
                                            disc_label = 'sensitive_attribute',
                                            disc_feats = num_sensitive_attribute,
                                            model_P_A = self.model_P_A)
        checkpoint_callback = ModelCheckpoint(save_top_k=1,
                                              verbose=True,
                                              save_last=True,
                                              monitor='generator/val_loss',
                                              mode='min',
                                              dirpath=self.ckpt_path,
                                              filename='fair_best')
        early_stooping = EarlyStopping(monitor='generator/val_loss', patience=50, mode='min')
        self.trainer = pl.Trainer(gpus=self.hparams.gpus,
                                  max_epochs=self.hparams.epochs,
                                  logger=self.logger,
                                  log_every_n_steps=1,
                                  callbacks=[early_stooping, checkpoint_callback])
        self.trainer.fit(self.model, datamodule=datamodule)

    def test(self, datamodule):
        ckpt_path = os.path.join(self.ckpt_path, 'fair_best.ckpt')
        if os.path.exists(ckpt_path):
            self.trainer.test(ckpt_path=ckpt_path, datamodule=datamodule)
        else:
            ckpt_path = os.path.join(self.ckpt_path, 'fair_best-v0.ckpt')
            self.trainer.test(ckpt_path=ckpt_path, datamodule=datamodule)
        self.trainer.logger.experiment.log({}, commit=True)

