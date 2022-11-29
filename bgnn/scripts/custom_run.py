# from catboost import CatboostError
# import sys
# sys.path.append('../')

from bgnn.models.GBDT import GBDTCatBoost, GBDTLGBM
from bgnn.models.MLP import MLP
from bgnn.models.GNN import GNN
from bgnn.models.BGNN import BGNN
from bgnn.models.FRMCC import FRMCC
from bgnn.models.FRMCC import FrcMLP
from bgnn.scripts.utils import NpEncoder

import os
import json
import time
import datetime
from pathlib import Path
from collections import defaultdict as ddict

import torch
import torch.nn as nn
import pandas as pd
import networkx as nx
import random
import numpy as np
import fire
from omegaconf import OmegaConf
from sklearn.model_selection import ParameterGrid


class RunModel:
    def read_input(self, input_folder):
        self.X = pd.read_csv(f'{input_folder}/X.csv')
        self.y = pd.read_csv(f'{input_folder}/y.csv')

        networkx_graph = nx.read_graphml(f'{input_folder}/graph.graphml')
        networkx_graph = nx.relabel_nodes(networkx_graph, {str(i): i for i in range(len(networkx_graph))})
        self.networkx_graph = networkx_graph

        categorical_columns = []
        if os.path.exists(f'{input_folder}/cat_features.txt'):
            with open(f'{input_folder}/cat_features.txt') as f:
                for line in f:
                    if line.strip():
                        categorical_columns.append(line.strip())

        self.cat_features = None
        if categorical_columns:
            columns = self.X.columns
            self.cat_features = np.where(columns.isin(categorical_columns))[0]

            for col in list(columns[self.cat_features]):
                self.X[col] = self.X[col].astype(str)


        if os.path.exists(f'{input_folder}/masks.json'):
            with open(f'{input_folder}/masks.json') as f:
                self.masks = json.load(f)
        else:
            print('Creating and saving train/val/test masks')
            idx = list(range(self.y.shape[0]))
            self.masks = dict()
            for i in range(self.max_seeds):
                random.shuffle(idx)
                r1, r2, r3 = idx[:int(.6*len(idx))], idx[int(.6*len(idx)):int(.8*len(idx))], idx[int(.8*len(idx)):]
                self.masks[str(i)] = {"train": r1, "val": r2, "test": r3}

            with open(f'{input_folder}/masks.json', 'w+') as f:
                json.dump(self.masks, f, cls=NpEncoder)


    def get_input(self, dataset_dir, dataset: str):
        if dataset == 'house':
            input_folder = dataset_dir / 'house'
        elif dataset == 'county':
            input_folder = dataset_dir / 'county'
        elif dataset == 'vk':
            input_folder = dataset_dir / 'vk'
        elif dataset == 'wiki':
            input_folder = dataset_dir / 'wiki'
        elif dataset == 'avazu':
            input_folder = dataset_dir / 'avazu'
        elif dataset == 'vk_class':
            input_folder = dataset_dir / 'vk_class'
        elif dataset == 'house_class':
            input_folder = dataset_dir / 'house_class'
        elif dataset == 'dblp':
            input_folder = dataset_dir / 'dblp'
        elif dataset == 'slap':
            input_folder = dataset_dir / 'slap'
        else:
            input_folder = dataset

        if self.save_folder is None:
            self.save_folder = f'results/{dataset}/{datetime.datetime.now().strftime("%d_%m")}'

        self.read_input(input_folder)
        print('Save to folder:', self.save_folder)


    def run_one_model(self, config_fn, model_name):
        self.config = OmegaConf.load(config_fn)
        grid = ParameterGrid(dict(self.config.hp))
        # print('grid', grid)
        for ps in grid:
            param_string = ''.join([f'-{key}{ps[key]}' for key in ps])
            exp_name = f'{model_name}{param_string}'
            print(f'\nSeed {self.seed} RUNNING:{exp_name}')

            runs = []
            runs_custom = []
            times = []
            for _ in range(self.repeat_exp): ## repeat_exp default is 1
                # print('repeat exp value:', self.repeat_exp)
                start = time.time()
                model = self.define_model(model_name, ps)
                
                inputs = {'X': self.X, 'y': self.y, 'train_mask': self.train_mask,
                          'val_mask': self.val_mask, 'test_mask': self.test_mask, 'cat_features': self.cat_features}
                if model_name in ['gnn', 'resgnn', 'bgnn', 'frcgnn']:
                    inputs['networkx_graph'] = self.networkx_graph

                metrics = model.fit(num_epochs=self.config.num_epochs, patience=self.config.patience,
                           loss_fn=f"{self.seed_folder}/{exp_name}.txt",
                           metric_name='loss' if self.task == 'regression' else 'accuracy', **inputs)
                finish = time.time()
                best_loss = min(metrics['loss'], key=lambda x: x[1])
                best_custom = max(metrics['r2' if self.task == 'regression' else 'accuracy'], key=lambda x: x[1])
                runs.append(best_loss)
                runs_custom.append(best_custom)
                times.append(finish - start)
            self.store_results[exp_name] = (list(map(np.mean, zip(*runs))),
                                       list(map(np.mean, zip(*runs_custom))),
                                       np.mean(times),
                                       )
            # print(f'{"-"*20}exitting{"-"*20}')
            # return ## only run for 1 parameter grid option
        
    def tune_frocc(self, path):
        print(self.train_mask[:10])
        X_train = self.X.to_numpy()[self.train_mask]
        y_train = self.y.to_numpy().reshape(-1)[self.train_mask]
        X_test  = self.X.to_numpy()[self.test_mask]
        y_test  = self.y.to_numpy().reshape(-1)[self.test_mask]
        with open(path, '+w') as f:
            f.write('')
        with open(path, '+a') as f:
            f.write(f'Xtrain shape: {X_train.shape}, ytrain shape: {y_train.shape}\n')
            # f.write(f"dataset: {}\n")
        epsilons = [0.001, 0.01, 0.03]
        dimensions = [5000, 2000, 1000, 500]
        kernels = ['sigmoid']
        for epsilon in epsilons:
            for dimension in dimensions:
                for kernel in kernels:
                    print(f"epsilon: {epsilon}, dimension: {dimension}, kernel: {kernel}")
                    with open(path, '+a') as f:
                        f.write(f"epsilon: {epsilon}, dimension: {dimension}, kernel: {kernel}\n")
                    multi_class_frocc = FRMCC(epsilon=epsilon, dimension=dimension, kernel=kernel)
                    multi_class_frocc.fit( X_train, y_train, print_details = True)
                    frc_pred, _ = multi_class_frocc.predict(X_train, y_train, print_acc=True, path=path)
                    frc_pred, _ = multi_class_frocc.predict(X_test, y_test, print_acc=True, path=path)
                    with open(path, '+a') as f:
                        f.write('\n')
        ## Just trying to fit frocc
        exit()

    def define_model(self, model_name, ps):
        if model_name == 'catboost':
            return GBDTCatBoost(self.task, **ps)
        elif model_name == 'lightgbm':
            return GBDTLGBM(self.task, **ps)
        elif model_name == 'mlp':
            return MLP(self.task, **ps)
        elif model_name == 'gnn':
            return GNN(self.task, **ps)
        elif model_name == 'resgnn':
            print('got inside res gnn')
            print(f'task: {self.task}, X shape: {self.X.shape}, y shape: {self.y.shape}')
            gbdt = GBDTCatBoost(self.task)
            gbdt.fit(self.X, self.y, self.train_mask, self.val_mask, self.test_mask,
                     cat_features=self.cat_features,
                     num_epochs=1000, patience=100,
                     # plot=False, verbose=False, loss_fn=None,
                     metric_name='loss' if self.task == 'regression' else 'accuracy')
            return GNN(task=self.task, gbdt_predictions=gbdt.model.predict(self.X), **ps)
        elif model_name == 'frcgnn':
            prediction_options = ['1d pred', '2d clf', '1d mlp']
            use_predection = '1d mlp'
            X_train = self.X.to_numpy()[self.train_mask]
            y_train = self.y.to_numpy().reshape(-1)[self.train_mask]
            X_test  = self.X.to_numpy()[self.test_mask]
            y_test  = self.y.to_numpy().reshape(-1)[self.test_mask]
            print('got inside frc gnn')
            print(f'task: {self.task}, X shape: {self.X.shape}, y shape: {self.y.shape}')
            
            # path = 'frc_result_hc2.txt'
            # self.tune_frocc(path)
            epsilon = 0.01
            dimension = 500
            kernel = 'sigmoid'
            print(f"FROCC specs=> epsilon: {epsilon}, dimension: {dimension}, kernel: {kernel}")
            multi_class_frocc = FRMCC(epsilon=epsilon, dimension=dimension, kernel=kernel)
            multi_class_frocc.fit(X_train, y_train, print_details = True)
            frc_pred, frc_clf_pred = multi_class_frocc.predict(self.X.to_numpy(), self.y.to_numpy(), print_acc=True)
            if use_predection == '1d pred':
                new_data = frc_pred
            if use_predection == '2d clf':
                new_data = frc_clf_pred
            elif use_predection == '1d mlp':
                in_features = frc_clf_pred.shape[1]
                frc_clf_pred = torch.from_numpy(frc_clf_pred).float()
                mlp_ip = frc_clf_pred[self.train_mask]
                mlp_op = torch.from_numpy(self.y.to_numpy()[self.train_mask]).float()
                mlp_test_ip = frc_clf_pred[self.test_mask]
                mlp_test_op = torch.from_numpy(self.y.to_numpy()[self.test_mask]).float()
                print(f"in features: {in_features}, ip shape: {mlp_ip.shape}, op shape: {mlp_op.shape}")
                frc_mlp_model = FrcMLP(
                    input_features=in_features,
                    hidden_features=2, output_dimension=1)
                frc_mlp_model.fit(mlp_ip, mlp_op, 300, 0.03, mlp_test_ip, mlp_test_op)
                new_data = frc_mlp_model(frc_clf_pred).detach().numpy().reshape(-1)
                print(f"new_data/frc mlp op shape:{new_data.shape}")
            return GNN(task=self.task, gbdt_predictions= new_data, **ps)
        elif model_name == 'bgnn':
            return BGNN(self.task, **ps)

    def create_save_folder(self, seed):
        self.seed_folder = f'{self.save_folder}/{seed}'
        os.makedirs(self.seed_folder, exist_ok=True)

    def split_masks(self, seed):
        self.train_mask, self.val_mask, self.test_mask = self.masks[seed]['train'], \
                                                         self.masks[seed]['val'], self.masks[seed]['test']

    def save_results(self, seed):
        self.seed_results[seed] = self.store_results
        with open(f'{self.save_folder}/seed_results.json', 'w+') as f:
            json.dump(self.seed_results, f)

        self.aggregated = self.aggregate_results()
        with open(f'{self.save_folder}/aggregated_results.json', 'w+') as f:
            json.dump(self.aggregated, f)

    def get_model_name(self, exp_name: str, algos: list):
        # get name of the model (for gnn-like models (eg. gat))
        if 'name' in exp_name:
            model_name = '-' + [param[4:] for param in exp_name.split('-') if param.startswith('name')][0]
        else:
            model_name = ''

        # get a model used a MLP (eg. MLP-GNN)
        if 'gnn' in exp_name and 'mlpTrue' in exp_name:
            model_name += '-MLP'

        # algo corresponds to type of the model (eg. gnn, resgnn, bgnn)
        for algo in algos:
            if exp_name.startswith(algo):
                return algo + model_name
        return 'unknown'

    def aggregate_results(self):
        algos = ['catboost', 'lightgbm', 'mlp', 'gnn', 'resgnn', 'fcrgnn', 'bgnn']
        model_best_score = ddict(list)
        model_best_time = ddict(list)

        results = self.seed_results
        for seed in results:
            model_results_for_seed = ddict(list)
            for name, output in results[seed].items():
                model_name = self.get_model_name(name, algos=algos)
                if self.task == 'regression': # rmse metric
                    val_metric, test_metric, time = output[0][1], output[0][2], output[2]
                else: # accuracy metric
                    val_metric, test_metric, time = output[1][1], output[1][2], output[2]
                model_results_for_seed[model_name].append((val_metric, test_metric, time))

            for model_name, model_results in model_results_for_seed.items():
                if self.task == 'regression':
                    best_result = min(model_results) # rmse
                else:
                    best_result = max(model_results) # accuracy
                model_best_score[model_name].append(best_result[1])
                model_best_time[model_name].append(best_result[2])

        aggregated = dict()
        for model, scores in model_best_score.items():
            aggregated[model] = (np.mean(scores), np.std(scores),
                                 np.mean(model_best_time[model]), np.std(model_best_time[model]))
        return aggregated

    def run(self, dataset: str, *args,
            save_folder: str = None,
            task: str = 'classification',
            repeat_exp: int = 1,
            max_seeds: int = 1,
            dataset_dir: str = None,
            config_dir: str = None
            ):
        start2run = time.time()
        self.repeat_exp = repeat_exp
        self.max_seeds = max_seeds
        print(dataset, args, task, repeat_exp, max_seeds, dataset_dir, config_dir)

        dataset_dir = Path(dataset_dir) if dataset_dir else Path(__file__).parent.parent / 'datasets'
        config_dir = Path(config_dir) if config_dir else Path(__file__).parent.parent / 'configs' / 'model'
        print(dataset_dir, config_dir)

        self.task = task
        self.save_folder = save_folder
        self.get_input(dataset_dir, dataset)

        self.seed_results = dict()
        for ix, seed in enumerate(self.masks):
            print(f'{dataset} Seed {seed}')
            self.seed = seed

            self.create_save_folder(seed)
            self.split_masks(seed)

            self.store_results = dict()
            for arg in args:
                if arg == 'all':
                    self.run_one_model(config_fn=config_dir / 'catboost.yaml', model_name="catboost")
                    self.run_one_model(config_fn=config_dir / 'lightgbm.yaml', model_name="lightgbm")
                    self.run_one_model(config_fn=config_dir / 'mlp.yaml', model_name="mlp")
                    self.run_one_model(config_fn=config_dir / 'gnn.yaml', model_name="gnn")
                    self.run_one_model(config_fn=config_dir / 'resgnn.yaml', model_name="resgnn")
                    self.run_one_model(config_fn=config_dir / 'frcgnn.yaml', model_name="frcgnn")
                    self.run_one_model(config_fn=config_dir / 'bgnn.yaml', model_name="bgnn")
                    break
                elif arg == 'catboost':
                    self.run_one_model(config_fn=config_dir / 'catboost.yaml', model_name="catboost")
                elif arg == 'lightgbm':
                    self.run_one_model(config_fn=config_dir / 'lightgbm.yaml', model_name="lightgbm")
                elif arg == 'mlp':
                    self.run_one_model(config_fn=config_dir / 'mlp.yaml', model_name="mlp")
                elif arg == 'gnn':
                    self.run_one_model(config_fn=config_dir / 'gnn.yaml', model_name="gnn")
                elif arg == 'resgnn':
                    self.run_one_model(config_fn=config_dir / 'resgnn.yaml', model_name="resgnn")
                elif arg == 'frcgnn':
                    self.run_one_model(config_fn=config_dir / 'frcgnn.yaml', model_name="frcgnn")
                elif arg == 'bgnn':
                    self.run_one_model(config_fn=config_dir / 'bgnn.yaml', model_name="bgnn")

            self.save_results(seed)
            if ix+1 >= max_seeds:
                break

        print(f'Finished {dataset}: {time.time() - start2run} sec.')

if __name__ == '__main__':
    fire.Fire(RunModel().run)