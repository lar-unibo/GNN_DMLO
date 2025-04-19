import os, glob, copy, torch
from tqdm import tqdm
import numpy as np

from gnn_dmlo.core.graph_generation import GraphGenerationWh, ReadPickle
from gnn_dmlo.core.utils import get_data_pyg, compute_dir_smooth, create_new_folder, to_numpy


class GraphDataProcessing:
    def __init__(self, data_path: str, out_dataset_path: str):
        self.bp_flag = True
        self.augmentation = True
        # folder where to save graphs

        augmentation_str = "augmentation" if self.augmentation else "no_augmentation"
        self.graphs_path = os.path.join(
            out_dataset_path,
            os.path.basename(data_path) + f"_LP_{augmentation_str}",
        )
        create_new_folder(self.graphs_path)

        self.rp = ReadPickle()
        self.graph_gen = GraphGenerationWh()

        self.preprocess(data_path)
        self.files = sorted(glob.glob(os.path.join(self.graphs_path, "*.pt")))

    def preprocess(self, path: str) -> None:
        for f in tqdm(sorted(glob.glob(os.path.join(path, "*")))):
            idx = int(f.split("/")[-1].split(".pickle")[0])

            if self.augmentation:
                self.repetitions = 2
            else:
                self.repetitions = 1

            self.mask_deg_flag = False
            for i in range(self.repetitions):
                try:
                    dict_data = self.rp.read(f, show=False)

                    data_list = self.compute_sample_aug(dict_data, f)
                    for it, datum in enumerate(data_list):
                        torch.save(datum, os.path.join(self.graphs_path, str(idx).zfill(4) + f"_{it}_{i}.pt"))

                        # self.plot_sample_pyg(datum)

                except Exception as e:
                    print("IDX: {}, Error: {}".format(idx, e))
                    continue

                self.mask_deg_flag = not self.mask_deg_flag

    def generate_graph(self, mask: np.ndarray, points_gt_dict: dict, path: str):
        graph_data = self.graph_gen.exec(
            mask,
            points_gt_dict,
            gt_data_flag=True,
            bp_flag=self.bp_flag,
            mask_deg_flag=self.mask_deg_flag,
        )
        return self.get_data_basic(graph_data, path, bps=graph_data["bps"])

    def compute_sample_aug(self, data, path):
        def aug_vertical(points_dict, mask_shape):
            points_dict_out = {}
            for k, points in points_dict.items():
                points = copy.deepcopy(points)
                points[:, 1] = np.fabs(points[:, 1] - mask_shape[0])
                points[np.where(points[:, 1] == mask_shape[0]), 1] -= 1

                points_dict_out[k] = points

            return points_dict_out

        def aug_horizontal(points_dict, mask_shape):
            points_dict_out = {}
            for k, points in points_dict.items():
                points = copy.deepcopy(points)
                points[:, 0] = np.fabs(points[:, 0] - mask_shape[1])
                points[np.where(points[:, 0] == mask_shape[1]), 0] -= 1

                points_dict_out[k] = points
            return points_dict_out

        def aug_both(points_dict, mask_shape):
            points_dict_out = {}
            for k, points in points_dict.items():
                points = copy.deepcopy(points)
                points[:, 1] = np.fabs(points[:, 1] - mask_shape[0])
                points[np.where(points[:, 1] == mask_shape[0]), 1] -= 1

                points[:, 0] = np.fabs(points[:, 0] - mask_shape[1])
                points[np.where(points[:, 0] == mask_shape[1]), 0] -= 1

                points_dict_out[k] = points
            return points_dict_out

        mask = data["mask"]
        points_gt_dict = data["gt"]

        # normal
        data = self.generate_graph(mask, points_gt_dict, path)

        # vertical
        points_gt_dict_v = aug_vertical(points_gt_dict, mask.shape)
        data_v = self.generate_graph(np.flipud(mask), points_gt_dict_v, path)

        # horizontal
        points_gt_dict_h = aug_horizontal(points_gt_dict, mask.shape)
        data_h = self.generate_graph(np.fliplr(mask), points_gt_dict_h, path)

        # both
        points_gt_dict_f = aug_both(points_gt_dict, mask.shape)
        data_f = self.generate_graph(np.flip(mask), points_gt_dict_f, path)

        return [data, data_v, data_h, data_f]

    def get_data_basic(self, graph_data, path, bps):
        edges_knn = graph_data["edges_knn"]
        edges_gt = graph_data["edges_gt"]

        data = get_data_pyg(graph_data["nodes"], edges_knn, graph_data["dist_img"], path)
        data.bps = torch.from_numpy(bps).type(torch.FloatTensor)

        # ******************************************************
        # NODES ANGLES
        data.node_dir = torch.from_numpy(graph_data["nodes_dir"]).type(torch.FloatTensor)
        data.node_angle_gauss = torch.from_numpy(compute_dir_smooth(graph_data["nodes_dir"])).type(torch.FloatTensor)

        # ******************************************************
        # POSITIVE AND NEGATIVE EDGES
        edges_gt_list = edges_gt.tolist()
        edge_neg = np.array(
            [e for e in edges_knn.tolist() if (e not in edges_gt_list) and (e[::-1] not in edges_gt_list)]
        )

        data.edge_index = torch.from_numpy(edges_knn.T).type(torch.LongTensor)  # (2, num_edges)
        data.edge_pos = torch.from_numpy(edges_gt).type(torch.LongTensor)  # (num_pos_edges, 2)
        data.edge_neg = torch.from_numpy(edge_neg).type(torch.LongTensor)  # (num_neg_edges, 2)

        return data

    def plot_sample_pyg(self, data_pyg):
        dist_img = to_numpy(data_pyg.dist).squeeze()
        nodes = to_numpy(data_pyg.x) * np.array([dist_img.shape[0], dist_img.shape[1]])

        edges_knn = to_numpy(data_pyg.edge_index).T
        edges_gt = to_numpy(data_pyg.edge_pos)

        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(dist_img)
        plt.plot(nodes[:, 1], nodes[:, 0], "r.")

        for e0, e1 in edges_knn:
            pos_e0, pos_e1 = nodes[e0], nodes[e1]
            plt.plot([pos_e0[1], pos_e1[1]], [pos_e0[0], pos_e1[0]], color="black", linewidth=1)

        for e0, e1 in edges_gt:
            pos_e0, pos_e1 = nodes[e0], nodes[e1]
            plt.plot([pos_e0[1], pos_e1[1]], [pos_e0[0], pos_e1[0]], color="white", linewidth=1)

        plt.show()


class GraphDatasetLinkPred(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str) -> None:
        self.files = sorted(glob.glob(os.path.join(dataset_path, "*.pt")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx])


if __name__ == "__main__":
    proj_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    out_dataset_path = os.path.join(proj_path, "datasets")

    data_path = "/home/alessio/Downloads/gnn_dlo/example_synthetic_dataset"
    g = GraphDataProcessing(data_path, out_dataset_path)
