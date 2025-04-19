import torch, torch_geometric, torch_sparse
import numpy as np
import random, os, shutil, itertools
import networkx as nx
import scipy.sparse
from termcolor import cprint
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2

from shapely import geometry


def update_log(input_dict, log_dict):
    for k, v in input_dict.items():
        if isinstance(v, torch.Tensor):
            log_dict[k] += v.item()
        else:
            log_dict[k] += v


def print_log(log):
    train = {}
    val = {}
    score = {}
    for k, v in log.items():
        if "train" in k:
            train[k] = round(v, 5)
        elif "val" in k:
            val[k] = round(v, 5)
        elif k != "epoch":
            score[k] = round(v, 5)
    cprint(f"epoch: {log['epoch']}", "yellow")
    print("train: ", train)
    print("val: ", val)
    print("score: ", score)


def set_seeds(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_new_folder(path):
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)


def compute_gt_graph_nx(data):
    mask = to_numpy(data.mask.squeeze())
    nodes_arr = to_numpy(data.x) * mask.shape[:2]
    return create_nx_graph_angles(nodes_arr, to_numpy(data.edge_pos), to_numpy(data.node_angle_gauss))


def compute_input_graph_nx(data):
    mask = to_numpy(data.mask.squeeze())
    nodes_arr = to_numpy(data.x) * mask.shape[:2]
    return create_nx_graph_angles(nodes_arr, to_numpy(data.edge_index.T), to_numpy(data.node_angle_gauss))


def create_nx_graph(nodes, edges):
    g = nx.Graph()
    g.add_nodes_from([(it, {"pos": np.array(xx)}) for it, xx in enumerate(nodes)])
    g.add_edges_from(edges)
    return g


def create_nx_graph_labels(nodes: np.ndarray, edges: np.ndarray, labels: np.ndarray, angles: np.ndarray):
    g = nx.Graph()
    g.add_nodes_from(
        [
            (it, {"pos": np.array(xx), "label": ll, "angle": angle})
            for it, (xx, ll, angle) in enumerate(zip(nodes, labels, angles))
        ]
    )
    g.add_edges_from(edges)
    return g


def create_nx_graph_edgesw_angles(nodes: np.ndarray, edges: np.ndarray, edges_w: np.ndarray, angles: np.ndarray):
    g = nx.Graph()
    g.add_nodes_from(
        [(it, {"pos": np.array(xx), "angle": angle}) for it, (xx, angle) in enumerate(zip(nodes, angles))]
    )
    g.add_edges_from([(e[0], e[1], {"weight": edges_w[it]}) for it, e in enumerate(edges)])
    return g


def create_nx_graph_angles(nodes: np.ndarray, edges: np.ndarray, angles: np.ndarray):
    g = nx.Graph()
    g.add_nodes_from(
        [(it, {"pos": np.array(xx), "angle": angle}) for it, (xx, angle) in enumerate(zip(nodes, angles))]
    )
    g.add_edges_from(edges)
    return g


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_numpy(x):
    return x.detach().cpu().numpy()


def positional_encodings_rwpe(nodes, edges, pos_enc_dim=16):
    """
    Initializing positional encoding with RWPE
    """
    g = create_nx_graph(nodes, edges)

    # Geometric diffusion features with Random Walk
    A = nx.adjacency_matrix(g)
    D = np.array([d[1] for d in g.degree()])
    Dinv = scipy.sparse.diags(D.clip(1) ** -1.0, dtype=float)
    RW = A * Dinv

    # Iterate
    pos_enc = [RW.diagonal()]
    M_power = RW
    for _ in range(pos_enc_dim - 1):
        M_power = M_power * RW
        pos_enc.append(M_power.diagonal())
    return np.stack(pos_enc, axis=-1)


def adjecency_matrix_from_edge_list(edges, num_nodes):
    edge_index = torch_geometric.utils.to_undirected(torch.from_numpy(edges).T)
    A = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes)).to_dense()
    return A


def compute_dir_smoothxxx(nodes_dir):
    def gaussian_label(label, num_class, u=0, sig=4.0):
        start = int(np.floor(-num_class / 2))
        stop = int(np.ceil(num_class / 2))
        x = np.arange(start, stop, 1).astype(int)
        y_sig = np.exp(-((x - u) ** 2) / (2 * sig**2))
        return np.concatenate([y_sig[stop - label :], y_sig[: stop - label]], axis=0)

    x = []
    for dir in nodes_dir:
        a = np.arctan2(dir[0], dir[1])
        if a < 0:
            a += np.pi
        x.append(gaussian_label(int(np.round(np.degrees(a))), 180, sig=4))
    return np.array(x)


def compute_dir_smooth(nodes_dir, num_class=180, u=0, sig=4.0):
    start = int(np.floor(-num_class / 2))
    stop = int(np.ceil(num_class / 2))
    x = np.arange(start, stop, 1).astype(int)
    y_sig = np.exp(-((x - u) ** 2) / (2 * sig**2))

    angles = np.arctan2(nodes_dir[:, 0], nodes_dir[:, 1])
    angles[angles < 0] += np.pi
    labels = np.round(np.degrees(angles)).astype(int)
    labels[labels < 0] += 180
    labels[labels >= 180] -= 180

    conc_point = stop - labels
    y_sig_out = np.array([np.concatenate([y_sig[cp:], y_sig[:cp]], axis=0) for cp in conc_point])
    return y_sig_out


def compute_edges_dirs(nodes, edges):
    nodes_0 = nodes[edges[:, 0]]
    nodes_1 = nodes[edges[:, 1]]
    edge_dirs = (nodes_1 - nodes_0).astype(np.float32)
    edge_norms = np.linalg.norm(edge_dirs, axis=1)
    edge_dirs[:, 0] = edge_dirs[:, 0] / edge_norms
    edge_dirs[:, 1] = edge_dirs[:, 1] / edge_norms
    edge_dirs = edge_dirs.reshape(-1, 2)
    return edge_dirs, edge_norms


def compute_data_subgraph(graph, dist_img, nodes_arr, nodes_z, nodes_mask_enc, node_id):
    nodes = list(graph.neighbors(node_id))
    nodes.append(node_id)
    nodes = np.array(nodes)

    edges = [e for e in list(graph.edges(nodes)) if e[0] in nodes and e[1] in nodes]
    edges = np.array(edges)

    if len(edges) == 0:
        return False, None

    mapping = {n: i for i, n in enumerate(nodes)}
    sg_edges_mp = np.array([(mapping[e0], mapping[e1]) for e0, e1 in edges])

    sg_nodes_pos = nodes_arr[nodes]
    sg_nodes_z = nodes_z[nodes]
    sg_nodes_mask_enc = nodes_mask_enc[nodes]

    data_new = get_data_pyg_subgraph(sg_nodes_pos, sg_edges_mp, dist_img)
    data_new.mask_enc = torch.from_numpy(sg_nodes_mask_enc).type(torch.FloatTensor)
    data_new.z = torch.from_numpy(sg_nodes_z).type(torch.FloatTensor)

    return True, data_new


def closest_point_on_segment(segment_start, segment_end, target_point):
    # calculate the vector from the start point of the segment to the target point
    v1 = target_point - segment_start

    # calculate the vector representing the segment
    v2 = segment_end - segment_start

    # calculate the dot product of the two vectors
    dot_product = np.dot(v1, v2)

    # calculate the length of the segment
    segment_length = np.linalg.norm(v2)

    # calculate the projection of the vector from step 1 onto the vector from step 2
    projection = dot_product / segment_length**2

    # if the projection is less than 0, the closest point is the start point of the segment
    if projection < 0:
        return segment_start, 0

    # if the projection is greater than the length of the segment, the closest point is the end point of the segment
    if projection > 1:
        return segment_end, 1

    # otherwise, the closest point is the start point of the segment plus the projection of the vector from step 1 onto the vector from step 2, scaled by the length of the segment
    closest_point = segment_start + projection * v2
    return closest_point, projection


def randomize_nodes_indices(graph_data: dict):
    nodes_arr = graph_data["nodes"]

    g = nx.Graph()
    g.add_nodes_from(
        [(it, {"pos": np.array(xx), "dir": graph_data["nodes_dir"][it, :]}) for it, xx in enumerate(nodes_arr)]
    )

    nodes_list = list(g.nodes())
    random.shuffle(nodes_list)
    node_mapping = {it: k for it, k in enumerate(nodes_list)}

    g_gt = g.copy()
    g_gt.add_edges_from(graph_data["edges_gt"])
    g_gt_mapped = nx.relabel_nodes(g_gt, node_mapping)

    g_knn = g.copy()
    g_knn.add_edges_from(graph_data["edges_knn"])
    g_knn_mapped = nx.relabel_nodes(g_knn, node_mapping)

    nodes_mapped = np.zeros((len(nodes_arr), 2))
    dirs_mapped = np.zeros((len(nodes_arr), 2))
    for k, v in g_knn_mapped.nodes(data=True):
        nodes_mapped[k, :] = v["pos"]
        dirs_mapped[k, :] = v["dir"]

    edges_knn_mapped = np.array(list(g_knn_mapped.edges()))
    edges_gt_mapped = np.array(list(g_gt_mapped.edges()))

    return nodes_mapped, dirs_mapped, edges_knn_mapped, edges_gt_mapped


def randomize_nodes_indices_online(graph_data: dict):
    nodes_arr = graph_data["nodes"]

    g = nx.Graph()
    g.add_nodes_from([(it, {"pos": np.array(xx)}) for it, xx in enumerate(nodes_arr)])

    nodes_list = list(g.nodes())
    random.shuffle(nodes_list)
    node_mapping = {it: k for it, k in enumerate(nodes_list)}

    g_knn = g.copy()
    g_knn.add_edges_from(graph_data["edges_knn"])
    g_knn_mapped = nx.relabel_nodes(g_knn, node_mapping)

    nodes_mapped = np.zeros((len(nodes_arr), 2))
    for k, v in g_knn_mapped.nodes(data=True):
        nodes_mapped[k, :] = v["pos"]

    edges_knn_mapped = np.array(list(g_knn_mapped.edges()))

    print(nodes_mapped.shape)
    print(edges_knn_mapped.shape)

    return nodes_mapped, edges_knn_mapped


def get_data_pyg(nodes_arr, edges_knn, dist_img, path):
    data = torch_geometric.data.Data(num_nodes=nodes_arr.shape[0], path=path)

    # ***************************
    # NODES
    data.x = torch.from_numpy(nodes_arr / np.array([dist_img.shape[0], dist_img.shape[1]])).type(torch.FloatTensor)

    # MASK
    data.mask = torch.from_numpy(dist_img / np.max(dist_img)).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor)
    data.dist = torch.from_numpy(dist_img).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor)
    data.rows = torch.from_numpy((nodes_arr[:, 0] // 2)).type(torch.LongTensor)
    data.cols = torch.from_numpy((nodes_arr[:, 1] // 2)).type(torch.LongTensor)

    # ***************************
    # EDGES
    edge_dir, edge_norm = compute_edges_dirs(nodes_arr, edges_knn)

    data.edge_dir = torch.from_numpy(edge_dir).type(torch.FloatTensor)  # (num_edges, 2)
    data.edge_norm = torch.from_numpy(edge_norm / np.max(edge_norm)).reshape(-1, 1).type(torch.FloatTensor)
    data.edge_index = torch.from_numpy(edges_knn.T).type(torch.LongTensor)  # (2, num_edges)

    data.edge_dir_smooth = torch.from_numpy(compute_dir_smooth(edge_dir)).type(torch.FloatTensor)

    num_edges = edges_knn.shape[0]

    adj_fake = np.zeros((nodes_arr.shape[0], num_edges))  # useful matrix num_nodes x num_edges
    adj_fake[edges_knn[:, 0], np.arange(num_edges)] = 1
    adj_fake[edges_knn[:, 1], np.arange(num_edges)] = -1  # accounting for the edge direction

    npad = (
        (0, 0),
        (0, 5000 - adj_fake.shape[1]),
    )  # npad is a tuple of (n_before, n_after) for each dimension # 2000 is the maximum number of edges
    adj_fake_padded = np.pad(adj_fake, pad_width=npad, mode="constant", constant_values=0.0)
    data.adj_fake = torch.from_numpy(adj_fake_padded).type(torch.FloatTensor)  # (num_nodes, 2000)

    return data


def get_data_pyg_subgraph(nodes_arr, edges_knn, dist_img):
    data = torch_geometric.data.Data(num_nodes=nodes_arr.shape[0])

    # ***************************
    # NODES
    data.x = torch.from_numpy(nodes_arr / np.array([dist_img.shape[0], dist_img.shape[1]])).type(torch.FloatTensor)

    # MASK
    data.rows = torch.from_numpy((nodes_arr[:, 0] // 2)).type(torch.LongTensor)
    data.cols = torch.from_numpy((nodes_arr[:, 1] // 2)).type(torch.LongTensor)

    # ***************************
    # EDGES
    edge_dir, edge_norm = compute_edges_dirs(nodes_arr, edges_knn)

    data.edge_dir = torch.from_numpy(edge_dir).type(torch.FloatTensor)  # (num_edges, 2)
    data.edge_norm = torch.from_numpy(edge_norm / np.max(edge_norm)).reshape(-1, 1).type(torch.FloatTensor)
    data.edge_index = torch.from_numpy(edges_knn.T).type(torch.LongTensor)  # (2, num_edges)

    data.edge_dir_smooth = torch.from_numpy(compute_dir_smooth(edge_dir)).type(torch.FloatTensor)

    return data


###############################################################################


def mask_branchpoints(dist_img, branch_points):
    mask_bp = np.zeros_like(dist_img)
    for bp in branch_points:
        x_values = np.arange(bp[1] - 10, bp[1] + 10)
        y_values = np.arange(bp[0] - 10, bp[0] + 10)
        dist_values = []
        for x in x_values:
            for y in y_values:
                if x > 0 and x < mask_bp.shape[1] and y > 0 and y < mask_bp.shape[0]:
                    dist_values.append(dist_img[int(y), int(x)])

        if len(dist_values) == 0:
            continue

        dist_th = max(np.max(dist_values), 5) * 3
        cv2.circle(mask_bp, tuple([int(bp[1]), int(bp[0])]), int(dist_th), 255, -1)
    return mask_bp


def labels_cls_from_graph3(data):
    mask = to_numpy(data.mask.squeeze())
    dist_img = to_numpy(data.dist.squeeze())
    nodes_arr = to_numpy(data.x) * mask.shape[:2]
    edges_gt = to_numpy(data.edge_pos)

    graph_gt = create_nx_graph_angles(nodes_arr, to_numpy(data.edge_pos), to_numpy(data.node_angle_gauss))
    graph_mp = create_nx_graph_angles(nodes_arr, to_numpy(data.edge_index.T), to_numpy(data.node_angle_gauss))

    labels = np.zeros(data.x.shape[0], dtype=int)  # 0: normal, 1: branchpoint
    train_mask = np.zeros(data.x.shape[0])

    nodes_bp = [k for k, v in graph_gt.degree(graph_gt.nodes()) if v == 3]
    branch_points_pos = [graph_gt.nodes[k]["pos"] for k in nodes_bp]
    mask_bp = mask_branchpoints(dist_img, branch_points_pos)

    # ------------------------------
    problematic_node = [(k, v) for k, v in graph_gt.degree(graph_gt.nodes()) if v > 2]
    for k, v in problematic_node:
        if v == 3:
            labels[k] = 1
            train_mask[k] = 1
        else:
            pos = graph_gt.nodes[k]["pos"]
            if mask_bp[int(pos[0]), int(pos[1])] > 0:
                labels[k] = 1
                train_mask[k] = 1
            else:
                labels[k] = 0
                train_mask[k] = 1

    problematic_node_mp = [(k, v) for k, v in graph_mp.degree(graph_mp.nodes()) if v > 2]
    for k, v in problematic_node_mp:
        pos = graph_mp.nodes[k]["pos"]
        v = mask_bp[int(pos[0]), int(pos[1])]
        if train_mask[k] == 0 and v == 0:
            nn = list(graph_mp.neighbors(k))

            found = False
            for n in nn:
                if labels[n] == 1:
                    found = True
                    break
            if not found:
                labels[k] = 0
                train_mask[k] = 1

    # intersection points
    nodes_int = []
    combs = list(itertools.combinations(edges_gt.tolist(), 2))
    for c1, c2 in combs:
        if c1[0] != c2[0] and c1[0] != c2[1] and c1[1] != c2[0] and c1[1] != c2[1]:
            pos10 = graph_gt.nodes[c1[0]]["pos"]
            pos11 = graph_gt.nodes[c1[1]]["pos"]
            pos20 = graph_gt.nodes[c2[0]]["pos"]
            pos21 = graph_gt.nodes[c2[1]]["pos"]
            line1 = geometry.LineString([pos10, pos11])
            line2 = geometry.LineString([pos20, pos21])
            if line1.intersects(line2):
                nodes_int.extend([c1[0], c1[1], c2[0], c2[1]])

    for k in nodes_int:
        if k in problematic_node:
            continue

        pos = graph_gt.nodes[k]["pos"]
        if mask_bp[int(pos[0]), int(pos[1])] > 0:
            continue

        labels[k] = 0
        train_mask[k] = 1

    return labels, train_mask


def labels_cls_from_graph2(data):
    int_label = 0  # same as normal points

    mask = to_numpy(data.mask.squeeze())
    dist_img = to_numpy(data.dist.squeeze())
    nodes_arr = to_numpy(data.x) * mask.shape[:2]
    edges_gt = to_numpy(data.edge_pos)
    graph = create_nx_graph_angles(nodes_arr, to_numpy(data.edge_pos), to_numpy(data.node_angle_gauss))

    labels = np.zeros(data.x.shape[0], dtype=int)  # 0: normal, 1: branchpoint, 2: intersectionpoint
    train_mask = np.zeros(data.x.shape[0])

    # branch points
    degrees_tuples = graph.degree(graph.nodes())
    nodes_bp = [k for k, v in degrees_tuples if v == 3]

    branch_points_pos = [graph.nodes[k]["pos"] for k in nodes_bp]
    mask_bp = mask_branchpoints(dist_img, branch_points_pos)

    for k in graph.nodes():
        bp_pos = graph.nodes[k]["pos"]
        if mask_bp[int(bp_pos[0]), int(bp_pos[1])] > 0:
            labels[k] = 1
            train_mask[k] = 1

    counter = len(nodes_bp)

    # intersection points
    nodes_int = []
    edges_gt_list = edges_gt.tolist()
    combs = list(itertools.combinations(edges_gt_list, 2))
    for c1, c2 in combs:
        if c1[0] != c2[0] and c1[0] != c2[1] and c1[1] != c2[0] and c1[1] != c2[1]:
            pos10 = graph.nodes[c1[0]]["pos"]
            pos11 = graph.nodes[c1[1]]["pos"]
            pos20 = graph.nodes[c2[0]]["pos"]
            pos21 = graph.nodes[c2[1]]["pos"]
            line1 = geometry.LineString([pos10, pos11])
            line2 = geometry.LineString([pos20, pos21])
            if line1.intersects(line2):
                nodes_int.extend([c1[0], c1[1], c2[0], c2[1]])

    for k in nodes_int:
        labels[k] = int_label
        train_mask[k] = 1
    counter += len(nodes_int)

    # normal points
    nodes_normal = [k for k, v in degrees_tuples if v == 2]
    random.shuffle(nodes_normal)
    for k in nodes_normal:
        if counter == 0:
            break

        check_distance = True
        for node in nodes_bp:
            if nx.has_path(graph, source=k, target=node):
                if nx.shortest_path_length(graph, source=k, target=node) < 3:
                    check_distance = False
                    break
        if check_distance:
            labels[k] = 0
            train_mask[k] = 1
            counter -= 1

    return labels, train_mask


def labels_cls_from_graph(data, dense_bp=False):
    int_label = 0  # same as normal points

    mask = to_numpy(data.mask.squeeze())
    nodes_arr = to_numpy(data.x) * mask.shape[:2]
    edges_gt = to_numpy(data.edge_pos)
    graph = create_nx_graph_angles(nodes_arr, to_numpy(data.edge_pos), to_numpy(data.node_angle_gauss))

    labels = np.zeros(data.x.shape[0], dtype=int)  # 0: normal, 1: branchpoint, 2: intersectionpoint
    train_mask = np.zeros(data.x.shape[0])

    # branch points
    degrees_tuples = graph.degree(graph.nodes())
    nodes_bp = [k for k, v in degrees_tuples if v == 3]
    for k in nodes_bp:
        labels[k] = 1
        train_mask[k] = 1

        if dense_bp:
            for n in graph.neighbors(k):
                labels[int(n)] = 1
                train_mask[int(n)] = 1

    counter = len(nodes_bp)

    # intersection points
    nodes_int = []
    edges_gt_list = edges_gt.tolist()
    combs = list(itertools.combinations(edges_gt_list, 2))
    for c1, c2 in combs:
        if c1[0] != c2[0] and c1[0] != c2[1] and c1[1] != c2[0] and c1[1] != c2[1]:
            pos10 = graph.nodes[c1[0]]["pos"]
            pos11 = graph.nodes[c1[1]]["pos"]
            pos20 = graph.nodes[c2[0]]["pos"]
            pos21 = graph.nodes[c2[1]]["pos"]
            line1 = geometry.LineString([pos10, pos11])
            line2 = geometry.LineString([pos20, pos21])
            if line1.intersects(line2):
                nodes_int.extend([c1[0], c1[1], c2[0], c2[1]])

    for k in nodes_int:
        labels[k] = int_label
        train_mask[k] = 1
    counter += len(nodes_int)

    nodes_bp_int = nodes_bp

    # normal points
    nodes_normal = [k for k, v in degrees_tuples if v == 2]
    random.shuffle(nodes_normal)
    for k in nodes_normal:
        if counter == 0:
            break

        check_distance = True
        for node in nodes_bp_int:
            if nx.has_path(graph, source=k, target=node):
                if nx.shortest_path_length(graph, source=k, target=node) < 3:
                    check_distance = False
                    break
        if check_distance:
            labels[k] = 0
            train_mask[k] = 1
            counter -= 1
    return labels, train_mask


def draw_graph_dataset(graph, axs, train_mask, labels, node_size=100, font_size=8):
    color_map = [0.5, 0.5, 0.5, 1.0] * np.ones((graph.number_of_nodes(), 4))
    for i in range(len(train_mask)):
        if train_mask[i] == 1:
            color = [0.0, 0.0, 0.0, 1.0]
            color[labels[i]] = 1.0
            color_map[i] = color
    plt.sca(axs)
    nx.draw(
        graph,
        pos={k: (n["pos"][1], n["pos"][0]) for k, n in graph.nodes(data=True)},
        node_size=node_size,
        node_color=color_map,
    )
    for k, n in graph.nodes(data=True):
        axs.text(n["pos"][1], n["pos"][0], k, fontsize=font_size, ha="center", va="center", color="red")


def draw_graph_axs(graph, axs, node_size=100, font_size=8):
    color_map = [0.5, 0.5, 0.5, 1.0] * np.ones((graph.number_of_nodes(), 4))

    plt.sca(axs)
    nx.draw(
        graph,
        pos={k: (n["pos"][1], n["pos"][0]) for k, n in graph.nodes(data=True)},
        node_size=node_size,
        node_color=color_map,
    )
    for k, n in graph.nodes(data=True):
        axs.text(n["pos"][1], n["pos"][0], k, fontsize=font_size, ha="center", va="center", color="red")


def draw_graph_colordeg(graph, axis, node_size=100, font_size=8, colors=None, color_by_degree=False):
    plt.sca(axis)

    if color_by_degree:
        color_map = [d[1] for d in graph.degree()]
        color_map = np.array(color_map) / np.max(color_map)
        color_map = plt.cm.get_cmap("jet")(color_map)
    elif colors is not None:
        color_map = colors
    else:
        color_map = "lightblue"

    nx.draw(
        graph,
        pos={k: (n["pos"][1], n["pos"][0]) for k, n in graph.nodes(data=True)},
        node_size=node_size,
        node_color=color_map,
    )
    for k, n in graph.nodes(data=True):
        axis.text(
            n["pos"][1], n["pos"][0], k, fontsize=font_size, ha="center", va="center", color="black", weight="bold"
        )
    axis.axis("off")


def draw_graph_cls(graph, axis, node_cls, node_size=100, font_size=8):
    plt.sca(axis)

    if node_cls.shape[1] == 1:
        cmap = plt.cm.get_cmap("jet")
        colors = [cmap(p) for p in node_cls]
    else:
        colors = np.argmax(node_cls, axis=1)
        colors_values = np.max(node_cls, axis=1) / 2
        cmaps = [plt.cm.get_cmap("Blues"), plt.cm.get_cmap("Reds"), plt.cm.get_cmap("Greens")]
        colors = [cmaps[c](p) for c, p in zip(colors, colors_values)]

    nx.draw(
        graph,
        pos={k: (n["pos"][1], n["pos"][0]) for k, n in graph.nodes(data=True)},
        node_size=node_size,
        node_color=colors,
    )
    for k, n in graph.nodes(data=True):
        axis.text(
            n["pos"][1], n["pos"][0], k, fontsize=font_size, ha="center", va="center", color="white", weight="bold"
        )
    axis.axis("off")


def simple_draw_graph(node_pos, edges, axis, node_size=100, font_size=8, colors="lightblue"):
    axis.scatter(node_pos[:, 1], node_pos[:, 0], s=node_size, c=colors)

    pos_0 = node_pos[edges[:, 0]]
    pos_1 = node_pos[edges[:, 1]]
    axis.plot([pos_0[:, 1], pos_1[:, 1]], [pos_0[:, 0], pos_1[:, 0]], c="black", linewidth=0.5)

    for i, pos in enumerate(node_pos):
        text_fixed_size(axis, str(i), (float(pos[1]), float(pos[0])), h=font_size, color="black")
    axis.axis("off")


def text_fixed_size(ax, text, pos, w=None, h=None, auto_trans=True, color="k"):
    assert not (w is None and h is None)

    tp = mpl.textpath.TextPath((0.0, 0.0), text, size=1, prop={"weight": "bold"})
    x0, y0 = np.amin(np.array(tp.vertices), axis=0)
    x1, y1 = np.amax(np.array(tp.vertices), axis=0)
    hax = -np.subtract(*ax.get_xlim())
    wax = -np.subtract(*ax.get_ylim())

    if w is None:
        w = h / (y1 - y0) * (x1 - x0)
    if h is None:
        h = w / (x1 - x0) * (y1 - y0)
    if auto_trans:
        w *= np.sign(hax)
        h *= np.sign(wax)

    verts = []
    for vert in tp.vertices:
        vx = vert[0] * w / (x1 - x0) + pos[0]
        vy = vert[1] * h / (y1 - y0) + pos[1]
        verts += [[vx, vy]]
    verts = np.array(verts)

    tp = mpl.path.Path(verts, tp.codes)
    ax.add_patch(mpl.patches.PathPatch(tp, facecolor=color, lw=0))


def line_from_angle(angle, pos, length=8):
    pos_x, pos_y = pos
    x_new = pos_x + length * np.cos(angle)
    y_new = pos_y - length * np.sin(angle)
    x_new2 = pos_x - length * np.cos(angle)
    y_new2 = pos_y + length * np.sin(angle)
    return [x_new, x_new2], [y_new, y_new2]
