import os
import pickle
import random
from collections import defaultdict
from pathlib import Path

import hydra
import networkx as nx
import numpy as np
import torch as th
import torch.multiprocessing as mp
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
import torch_geometric

import graph_generation as gg

# class CustomGraphDataset(Dataset):
#     def __init__(self, sub_30, sub_100, sub_300, sub_1000):
#         self.sub_30 = sub_30
#         self.sub_100 = sub_100
#         self.sub_300 = sub_300
#         self.sub_1000 = sub_1000
#         self.batch_sizes = [15, 5, 2, 1] 
#         self.total_batches = min(
#             len(sub_30) // 15,
#             len(sub_100) // 5,
#             len(sub_300) // 2,
#             len(sub_1000) // 1
#         )

#     def __len__(self):
#         return self.total_batches * 4  # Each cycle has 4 batches

#     def __getitem__(self, idx):
#         batch_type = idx % len(self.batch_sizes)
#         cycle = idx // len(self.batch_sizes)

#         if batch_type == 0:  # sub_30 graphs
#             start = cycle * self.batch_sizes[0]
#             end = start + self.batch_sizes[0]
#             return self.sub_30[start:end]
#         elif batch_type == 1:  # sub_100 graphs
#             start = cycle * self.batch_sizes[1]
#             end = start + self.batch_sizes[1]
#             return self.sub_100[start:end]
#         elif batch_type == 2:  # sub_300 graphs
#             start = cycle * self.batch_sizes[2]
#             end = start + self.batch_sizes[2]
#             return self.sub_300[start:end]
#         elif batch_type == 3:  # sub_1000 graphs
#             return [self.sub_1000[cycle]]
    

# class CustomGraphDataLoader:
#     def __init__(self, dataset, shuffle=False, pin_memory=False, collate_fn=None, num_workers=0, multiprocessing_context=None):
#         self.dataset = dataset
#         self.shuffle = shuffle
#         self.pin_memory = pin_memory
#         self.collate_fn = collate_fn
#         self.num_workers = num_workers
#         self.multiprocessing_context = multiprocessing_context
#         self.batch_size_cycle = [15, 5, 2, 1]  # Corresponds to sub_30, sub_100, sub_300, sub_1000

#     def __iter__(self):
#         self.current_cycle = 0
#         return self._data_iterator()

#     def _data_iterator(self):
#         while True:
#             # Determine the batch size for the current cycle
#             batch_size = self.batch_size_cycle[self.current_cycle % len(self.batch_size_cycle)]
#             self.current_cycle += 1
            
#             # Create a DataLoader for the current batch size
#             # offset data set to ignore batch that has already been yield before --> DIDN'T WORK
#             # pad with none to make each graph's interval's length==15? TBD
#             loader = DataLoader(
#                 self.dataset,
#                 batch_size=batch_size,
#                 shuffle=self.shuffle,
#                 pin_memory=self.pin_memory,
#                 collate_fn=self.collate_fn,
#                 num_workers=self.num_workers,
#                 multiprocessing_context=self.multiprocessing_context
#             )
            
#             # Yield each batch from the loader
#             for batch in loader:
#                 yield batch
#                 break

def get_expansion_items(cfg: DictConfig, train_graphs):
    # Spectral Features
    spectrum_extractor = (
        gg.spectral.SpectrumExtractor(
            num_features=cfg.spectral.num_features,
            normalized=cfg.spectral.normalized_laplacian,
        )
        if cfg.spectral.num_features > 0
        else None
    )

    # Train Dataset
    red_factory = gg.reduction.ReductionFactory(
        contraction_family=cfg.reduction.contraction_family,
        cost_type=cfg.reduction.cost_type,
        preserved_eig_size=cfg.reduction.preserved_eig_size,
        sqrt_partition_size=cfg.reduction.sqrt_partition_size,
        weighted_reduction=cfg.reduction.weighted_reduction,
        min_red_frac=cfg.reduction.min_red_frac,
        max_red_frac=cfg.reduction.max_red_frac,
        red_threshold=cfg.reduction.red_threshold,
        rand_lambda=cfg.reduction.rand_lambda,
    )

    if cfg.reduction.num_red_seqs > 0:
        train_dataset = gg.data.FiniteRandRedDataset(
            adjs=[nx.to_scipy_sparse_array(G, dtype=np.float64) for G in train_graphs],
            red_factory=red_factory,
            spectrum_extractor=spectrum_extractor,
            num_red_seqs=cfg.reduction.num_red_seqs,
        )
    else:
        train_dataset = gg.data.InfiniteRandRedDataset(
            adjs=[nx.to_scipy_sparse_array(G, dtype=np.float64) for G in train_graphs],
            red_factory=red_factory,
            spectrum_extractor=spectrum_extractor,
        )

    # Dataloader
    is_mp = cfg.reduction.num_red_seqs < 0  # if infinite dataset
    # if cfg.dataset.name in ["NDSSL"]:
    #     train_dataloader = CustomGraphDataLoader(
    #         train_dataset,
    #         shuffle=False,
    #         pin_memory=True,
    #         collate_fn=Batch.from_data_list,
    #         num_workers=min(mp.cpu_count(), cfg.training.max_num_workers) * is_mp,
    #         multiprocessing_context="spawn" if is_mp else None,
    #     )
    # else:
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=Batch.from_data_list,
        num_workers=min(mp.cpu_count(), cfg.training.max_num_workers) * is_mp,
        multiprocessing_context="spawn" if is_mp else None,
    )

    # Model
    if cfg.spectral.num_features > 0:
        sign_net = gg.model.SignNet(
            num_eigenvectors=cfg.spectral.num_features,
            hidden_features=cfg.sign_net.hidden_features,
            out_features=cfg.model.emb_features,
            num_layers=cfg.sign_net.num_layers,
        )
    else:
        sign_net = None

    features = 2 if cfg.diffusion.name == "discrete" else 1
    if cfg.model.name == "ppgn":
        model = gg.model.SparsePPGN(
            node_in_features=features * (1 + cfg.diffusion.self_conditioning),
            edge_in_features=features * (1 + cfg.diffusion.self_conditioning),
            node_out_features=features,
            edge_out_features=features,
            emb_features=cfg.model.emb_features,
            hidden_features=cfg.model.hidden_features,
            ppgn_features=cfg.model.ppgn_features,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
        )
    elif cfg.model.name == "gine":
        model = gg.model.GINE(
            node_in_features=features * (1 + cfg.diffusion.self_conditioning),
            edge_in_features=features * (1 + cfg.diffusion.self_conditioning),
            node_out_features=features,
            edge_out_features=features,
            emb_features=cfg.model.emb_features,
            hidden_features=cfg.model.hidden_features,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
        )
    else:
        raise ValueError(f"Unknown model name: {cfg.model.name}")

    # Diffusion
    if cfg.diffusion.name == "discrete":
        diffusion = gg.diffusion.sparse.DiscreteGraphDiffusion(
            self_conditioning=cfg.diffusion.self_conditioning,
            num_steps=cfg.diffusion.num_steps,
        )
    elif cfg.diffusion.name == "edm":
        diffusion = gg.diffusion.sparse.EDM(
            self_conditioning=cfg.diffusion.self_conditioning,
            num_steps=cfg.diffusion.num_steps,
        )
    else:
        raise ValueError(f"Unknown diffusion name: {cfg.diffusion.name}")

    # Method
    method = gg.method.Expansion(
        diffusion=diffusion,
        spectrum_extractor=spectrum_extractor,
        emb_features=cfg.model.emb_features,
        augmented_radius=cfg.method.augmented_radius,
        augmented_dropout=cfg.method.augmented_dropout,
        deterministic_expansion=cfg.method.deterministic_expansion,
        min_red_frac=cfg.reduction.min_red_frac,
        max_red_frac=cfg.reduction.max_red_frac,
        red_threshold=cfg.reduction.red_threshold,
    )

    return {
        "train_dataloader": train_dataloader,
        "method": method,
        "model": model,
        "sign_net": sign_net,
    }


def get_one_shot_items(cfg: DictConfig, train_graphs):
    # Train Dataset
    train_dataset = gg.data.DenseGraphDataset(
        adjs=[nx.to_numpy_array(G, dtype=bool) for G in train_graphs]
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    # Model
    features = 2 if cfg.diffusion.name == "discrete" else 1
    model = gg.model.PPGN(
        in_features=features * (1 + cfg.diffusion.self_conditioning),
        out_features=features,
        emb_features=cfg.model.emb_features,
        hidden_features=cfg.model.hidden_features,
        ppgn_features=cfg.model.ppgn_features,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
    )

    # Diffusion
    if cfg.diffusion.name == "discrete":
        diffusion = gg.diffusion.dense.DiscreteGraphDiffusion(
            self_conditioning=cfg.diffusion.self_conditioning,
            num_steps=cfg.diffusion.num_steps,
        )
    elif cfg.diffusion.name == "edm":
        diffusion = gg.diffusion.dense.EDM(
            self_conditioning=cfg.diffusion.self_conditioning,
            num_steps=cfg.diffusion.num_steps,
        )
    else:
        raise ValueError(f"Unknown diffusion name: {cfg.diffusion.name}")

    # Method
    method = gg.method.OneShot(diffusion=diffusion)

    return {"train_dataloader": train_dataloader, "method": method, "model": model}


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.debugging:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Fix random seeds
    random.seed(0)
    np.random.seed(0)
    th.manual_seed(0)

    # Graphs
    # if cfg.dataset.name in ["NDSSL"]:
    #     print("Use customized dataset")
    #     pickle_file_paths = ['data/sub_30_graphs.pkl','data/sub_100_graphs.pkl', 
    #                 'data/sub_300_graphs.pkl', 'data/sub_1000_graphs.pkl']
    #     with open(pickle_file_paths[0], 'rb') as file:
    #         sub_30 = pickle.load(file)
    #         sub_30 = [v for k, v in sub_30.items()]
    #     with open(pickle_file_paths[1], 'rb') as file:
    #         sub_100 = pickle.load(file)
    #         sub_100 = [v for k, v in sub_100.items()]
    #     with open(pickle_file_paths[2], 'rb') as file:
    #         sub_300 = pickle.load(file)
    #         sub_300 = [v for k, v in sub_300.items()]
    #     with open(pickle_file_paths[3], 'rb') as file:
    #         sub_1000 = pickle.load(file)
    #         sub_1000 = [v for k, v in sub_1000.items()]
        #sub30:sub100:sub300:sub1000 = 11 : 3 : 1 : 1 = sum as 16
        #train_group = 7
        #val_group = 1
        #test_group = 1
        # train_graphs = sub_30[:112] + sub_100[:21] + sub_300[:7] + sub_1000[:7]
        # validation_graphs = sub_30[112:123] + sub_100[21:24] + sub_300[7:8] + sub_1000[7:8]
        # test_graphs = sub_30[123:134] + sub_100[24:28] + sub_300[8:9] + sub_1000[8:9]

        
    if cfg.dataset.load:
        with open(Path("./data") / f"{cfg.dataset.name}.pkl", "rb") as f:
            dataset = pickle.load(f)
        print("use default dataset")
        train_graphs = dataset["train"]
        validation_graphs = dataset["val"]
        test_graphs = dataset["test"]
        

    elif cfg.dataset.name in ["planar", "tree"]:
        graph_generator = (
            gg.data.generate_planar_graphs
            if cfg.dataset.name == "planar"
            else gg.data.generate_tree_graphs
        )

        train_graphs = graph_generator(
            num_graphs=cfg.dataset.train_size,
            min_size=cfg.dataset.min_size,
            max_size=cfg.dataset.max_size,
            seed=0,
        )
        validation_graphs = graph_generator(
            num_graphs=cfg.dataset.val_size,
            min_size=cfg.dataset.min_size,
            max_size=cfg.dataset.max_size,
            seed=1,
        )
        test_graphs = graph_generator(
            num_graphs=cfg.dataset.test_size,
            min_size=cfg.dataset.min_size,
            max_size=cfg.dataset.max_size,
            seed=2,
        )

    elif cfg.dataset.name == "sbm":
        train_graphs = gg.data.generate_sbm_graphs(
            num_graphs=cfg.dataset.train_size,
            min_num_communities=cfg.dataset.min_num_communities,
            max_num_communities=cfg.dataset.max_num_communities,
            min_community_size=cfg.dataset.min_community_size,
            max_community_size=cfg.dataset.max_community_size,
            seed=0,
        )
        validation_graphs = gg.data.generate_sbm_graphs(
            num_graphs=cfg.dataset.val_size,
            min_num_communities=cfg.dataset.min_num_communities,
            max_num_communities=cfg.dataset.max_num_communities,
            min_community_size=cfg.dataset.min_community_size,
            max_community_size=cfg.dataset.max_community_size,
            seed=1,
        )
        test_graphs = gg.data.generate_sbm_graphs(
            num_graphs=cfg.dataset.test_size,
            min_num_communities=cfg.dataset.min_num_communities,
            max_num_communities=cfg.dataset.max_num_communities,
            min_community_size=cfg.dataset.min_community_size,
            max_community_size=cfg.dataset.max_community_size,
            seed=2,
        )

    else:
        raise ValueError(f"Unknown dataset name: {cfg.dataset.name}")

    # keep only largest connected component for train graphs
    # for G in train_graphs:
    #     print(G)
    train_graphs = [
        G.subgraph(max(nx.connected_components(G), key=len)) for G in train_graphs
    ]
    # import pdb; pdb.set_trace()

    # Metrics
    validation_metrics = [
        gg.metrics.NodeNumDiff(),
        gg.metrics.NodeDegree(),
        gg.metrics.ClusteringCoefficient(),
        gg.metrics.OrbitCount(),
        gg.metrics.Spectral(),
        gg.metrics.Wavelet(),
        gg.metrics.Ratio(),
        gg.metrics.Uniqueness(),
        gg.metrics.Novelty(),
    ]

    if "planar" in cfg.dataset.name:
        validation_metrics += [
            gg.metrics.ValidPlanar(),
            gg.metrics.UniqueNovelValidPlanar(),
        ]
    elif "tree" in cfg.dataset.name:
        validation_metrics += [
            gg.metrics.ValidTree(),
            gg.metrics.UniqueNovelValidTree(),
        ]
    elif "sbm" in cfg.dataset.name:
        validation_metrics += [
            gg.metrics.ValidSBM(),
            gg.metrics.UniqueNovelValidSBM(),
        ]

    # Method
    if cfg.method.name == "expansion":
        method_items = get_expansion_items(cfg, train_graphs)
    elif cfg.method.name == "one_shot":
        method_items = get_one_shot_items(cfg, train_graphs)
    else:
        raise ValueError(f"Unknown method name: {cfg.method.name}")
    method_items = defaultdict(lambda: None, method_items)

    # Trainer
    th.set_float32_matmul_precision("high")
    trainer = gg.training.Trainer(
        sign_net=method_items["sign_net"],
        model=method_items["model"],
        method=method_items["method"],
        train_dataloader=method_items["train_dataloader"],
        train_graphs=train_graphs,
        validation_graphs=validation_graphs,
        test_graphs=test_graphs,
        metrics=validation_metrics,
        cfg=cfg,
    )

    print("Starting training...")
    if cfg.testing:
        trainer.test()
    else:
        trainer.train()
    print("Finished training!")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
