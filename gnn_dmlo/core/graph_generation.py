import cv2, pickle, os
import numpy as np
import torch, torch_geometric
import networkx as nx
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True


class ReadPickle:
    def __init__(self):
        self.K = np.array([525, 0.0, 340, 0.0, 525, 230, 0.0, 0.0, 1.0]).reshape(3, 3)
        self.H = 480
        self.W = 640

    def compute_spline(self, points, num_points=20):
        tck, u = splprep(np.array(points).T, u=None, k=3, s=0)
        u_new = np.linspace(u.min(), u.max(), num_points)
        new_points = np.array(splev(u_new, tck, der=0)).T
        return new_points[
            np.where(
                (new_points[:, 0] < self.W)
                & (new_points[:, 0] >= 0)
                & (new_points[:, 1] < self.H)
                & (new_points[:, 1] >= 0)
            )
        ]

    def project_2D(self, pose, points_3d):
        T = np.linalg.inv(pose)
        rvec, _ = cv2.Rodrigues(T[:3, :3])
        point2d = cv2.projectPoints(np.array(points_3d), rvec, T[0:3, 3], self.K, np.zeros(5))[0].squeeze().astype(int)
        return point2d[
            np.where((point2d[:, 0] < self.W) & (point2d[:, 0] >= 0) & (point2d[:, 1] < self.H) & (point2d[:, 1] >= 0))
        ]

    def read(self, path, show=False):
        data = pickle.load(open(path, "rb"))
        pose = np.array(data["pose"])
        color = np.array(data["color"])
        instances = np.array(data["instances"])
        labels = np.unique(instances)
        mask = np.array(instances != labels[-1]).astype(np.uint8)
        gt_data = data["gt"]

        inst_masks = {}
        for i in range(len(labels) - 1):
            inst_masks[labels[i]] = np.array(instances == labels[i]).astype(np.uint8)

        # GT
        gt_out_dict = {}
        avg_points_dist = None
        if "curve_0" in gt_data.keys():
            points_2d = self.project_2D(pose, gt_data["curve_0"])
            points_2d_s = self.compute_spline(points_2d, num_points=20)
            avg_points_dist = np.mean(np.linalg.norm(points_2d_s[1:] - points_2d_s[:-1], axis=1))
            gt_out_dict["curve_0"] = points_2d_s

        for k, points in gt_data.items():
            if k != "curve_0":
                if avg_points_dist == None:
                    num_points = 20
                else:
                    num_points = np.sum(np.linalg.norm(points_2d[1:] - points_2d[:-1], axis=1)) / avg_points_dist

                points_2d = self.project_2D(pose, points)
                gt_out_dict[k] = self.compute_spline(points_2d, num_points=int(num_points))

        if show:
            fig, axs = plt.subplots(1, 3, figsize=(14, 6))
            fig.suptitle(path, fontsize=16)
            axs[0].imshow(color)
            axs[0].axis("off")
            axs[0].set_title("color")
            axs[1].axis("off")
            axs[1].imshow(mask)
            axs[1].set_title("mask")
            axs[2].imshow(color)
            for k, points in gt_out_dict.items():
                axs[2].scatter(points[:, 0], points[:, 1], label="{}".format(k))

            axs[2].axis("off")
            axs[2].set_title("points")
            axs[2].legend()

            plt.tight_layout()
            plt.show()

        # OUTPUT
        out_dict = {
            "color": color,
            "mask": mask,
            "inst_masks": inst_masks,
            "gt": gt_out_dict,
        }

        return out_dict


class GraphGenerationWh:
    def __init__(self, n_knn=8, debug=False):
        self.n_knn = n_knn
        self.debug = debug

    def exec(self, mask, gt_data=None, gt_data_flag=False, bp_flag=True, mask_deg_flag=True):
        mask_copy = mask.copy()

        ########################################
        # DISTANCE IMAGE
        mask_copy = cv2.GaussianBlur(mask_copy, (3, 3), 0)
        dist_img = cv2.distanceTransform(mask_copy, cv2.DIST_L2, 3)

        indices = np.where(dist_img < 1)
        dist_img[indices] = 0
        mask_copy[indices] = 0

        ########################################
        # NODES
        nodes, points_max = self.compute_nodes(mask_copy, dist_img)

        # MASK DEGRADATION
        if mask_deg_flag and gt_data_flag:
            mask_copy = mask_paths_augmentation(mask_copy, gt_data)
            mask_copy = cv2.dilate(mask_copy, np.ones((3, 3)), iterations=2)

            splines_data = self.compute_spline_from_gt_data(gt_data)
            mask0 = np.zeros_like(mask_copy)
            spline0 = splines_data["curve_0"]["points"]
            for point in spline0:
                dist_value = dist_img[int(point[1]), int(point[0])]
                cv2.circle(mask0, (int(point[0]), int(point[1])), int(dist_value), 1, -1)
            mask0 = cv2.dilate(mask0, np.ones((3, 3)), iterations=4)

            mask_copy[mask0 > 0] = 1

            mask_copy, dist_img, nodes, points_max = self.mask_degradation_nodes_distimg(mask_copy)

        out_dict = {"nodes": nodes, "dist_img": dist_img, "points_max": points_max}

        ########################################
        # EDGES
        out_dict["edges_knn"] = self.compute_edges_knn(out_dict["nodes"])

        ########################################
        # GT
        if gt_data_flag is True:
            # NODES ORDERING BASED ON SPLINE GT
            splines_data = self.compute_spline_from_gt_data(gt_data)
            nodes_ids_dict = self.nodes_ordering_from_gt(out_dict["nodes"], splines_data)
            out_dict["nodes_ids_dict"] = nodes_ids_dict

            # NODES DIRS
            out_dict["nodes_dir"] = self.compute_nodes_dir_from_spline_data(out_dict["nodes"], splines_data)

            # EDGE GT
            out_dict["edges_gt"] = self.compute_basic_edges_gt(nodes_ids_dict)

            # EDGES GT BRANCHES
            if bp_flag is True:
                edges_gt_bp = self.compute_edges_gt_branches_simple(
                    nodes_ids_dict, out_dict["nodes"], out_dict["edges_knn"]
                )
                out_dict["edges_gt"] = np.concatenate([out_dict["edges_gt"], edges_gt_bp])

            # BRANCH POINTS LOCATIONS
            bp_locations = []
            for k, v in gt_data.items():
                if k != "curve_0":
                    bp_locations.append(v[0][::-1])
            out_dict["bps"] = np.array(bp_locations)

            # self.viz(mask, out_dict["nodes"], out_dict["edges_knn"], out_dict["edges_gt"], splines_data, bp_locations)

        return out_dict

    def compute_nodes(self, mask_img, dist_img, target_mean_dist=15):
        # local maximus
        max_image = cv2.dilate(dist_img, np.ones((3, 3)), iterations=1)
        maxmask = (dist_img == max_image) & mask_img
        points_maxmask = np.column_stack(np.nonzero(maxmask))

        points_dist = cdist(points_maxmask, points_maxmask)
        points_dist[points_dist == 0] = np.inf
        points_dist = np.min(points_dist, axis=1)

        ratio = np.mean(points_dist) / target_mean_dist
        points = torch.Tensor(points_maxmask)
        indices_fps = torch_geometric.nn.fps(points, ratio=ratio)
        nodes = points[indices_fps].detach().cpu().numpy().astype(int)
        return nodes, points_maxmask

    def nodes_ordering_from_gt(self, nodes, splines_data):
        def closer_point(target, points):
            dist = np.linalg.norm(points - target, axis=1)
            idx = np.argmin(dist)
            return dist[idx], idx

        keys = list(splines_data.keys())
        indices_dict = {k: [] for k in keys}
        nodes_ids_dict = {k: [] for k in keys}
        for it, p in enumerate(nodes):
            tmp_dist, tmp_idx = [], []
            for v in splines_data.values():
                dist, idx = closer_point(p[::-1], v["points"])
                tmp_idx.append(idx)
                tmp_dist.append(dist)

            if np.min(tmp_dist) > 10:
                continue

            if "curve_0" in keys:
                if tmp_dist[keys.index("curve_0")] < 5:
                    idx_min_dist = tmp_idx[keys.index("curve_0")]
                    k_min_dist = keys[keys.index("curve_0")]
                else:
                    idx_min_dist = tmp_idx[np.argmin(tmp_dist)]
                    k_min_dist = keys[np.argmin(tmp_dist)]
            else:
                idx_min_dist = tmp_idx[np.argmin(tmp_dist)]
                k_min_dist = keys[np.argmin(tmp_dist)]

            nodes_ids_dict[k_min_dist].append(it)
            indices_dict[k_min_dist].append(idx_min_dist)

        for k in keys:
            indices_ordered = np.argsort(indices_dict[k])
            nodes_ids_dict[k] = np.array(nodes_ids_dict[k])[indices_ordered].tolist()

        return nodes_ids_dict

    def compute_spline_from_gt_data(self, gt_data):
        splines_data = {}
        for k, v in gt_data.items():
            spline, spline_der = self.compute_spline_with_der(v)
            splines_data[k] = {"points": spline, "der": spline_der}
        return splines_data

    def compute_nodes_dir_from_spline_data(self, nodes, spline_data):
        def closer_point_with_der(node, points, points_der=None):
            distances = np.linalg.norm(points - node, axis=1)
            idx = np.argmin(distances)
            if points_der is None:
                return distances[idx]
            else:
                return distances[idx], points_der[idx] / np.linalg.norm(points_der[idx])

        nodes_dir = []
        for node in nodes:
            tmp_dist, tmp_dir = [], []
            for v in spline_data.values():
                dist, dir = closer_point_with_der(node[::-1], v["points"], v["der"])
                tmp_dist.append(dist)
                tmp_dir.append(dir)

            nodes_dir.append(tmp_dir[np.argmin(tmp_dist)])
        nodes_dir = np.array(nodes_dir)
        return nodes_dir[:, [1, 0]]

    def mask_degradation_nodes_distimg(self, mask):
        """
        splines_data = self.compute_spline_from_gt_data(gt_data)
        nodes_ids_dict = self.nodes_ordering_from_gt(nodes, splines_data)

        # random circles on some nodes
        for k, v in nodes_ids_dict.items():
            for i in range(10):
                random_id = np.random.randint(0, len(v))
                point = nodes[v[random_id]]
                min_r = dist_img[point[0], point[1]]
                max_r = np.max(dist_img) * 1
                r = np.random.uniform(min_r, max_r)
                cv2.circle(mask, (point[1], point[0]), int(r), 1, -1)

        # random rectangles on some nodes
        for n in nodes:
            random_dir = np.random.uniform(-1, 1, 2)
            scaled_dir = dist_img[n[0], n[1]] * random_dir * 1
            x1, y1 = n + scaled_dir
            x2, y2 = n - scaled_dir

            cv2.rectangle(mask, (int(y1), int(x1)), (int(y2), int(x2)), 1, -1)

        ########################################
        # RANDOM BLOBS
        for i in range(10):
            random_node = nodes[np.random.randint(0, len(nodes))]
            random_noise = np.random.uniform(-20, 20, 2)
            point = random_node + random_noise

            random_dir = np.random.uniform(-1, 1, 2)
            rand_x = np.random.uniform(1, 10, 1)
            x1, y1 = point + random_dir * np.random.uniform(1, 10, 1)
            x2, y2 = point - random_dir * rand_x / 2

            cv2.rectangle(mask, (int(y1), int(x1)), (int(y2), int(x2)), 1, -1)
        """

        ###
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        dist_img = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        dist_img = cv2.GaussianBlur(dist_img, (3, 3), 0)

        nodes, points_maxmask = self.compute_nodes(mask, dist_img)
        return mask, dist_img, nodes, points_maxmask

    def compute_edges_gt_branches_simple(self, nodes_ids_dict, nodes_arr, edges_knn):
        def get_node_id(node_pos, nodes_arr):
            return np.argmin(np.linalg.norm(nodes_arr - node_pos, axis=1))

        def cos_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        def candidate_knn_nodes(node_id, edges_knn):
            tmp1 = edges_knn[np.where(edges_knn[:, 0] == node_id)[0], 1]
            tmp2 = edges_knn[np.where(edges_knn[:, 1] == node_id)[0], 0]
            return np.unique(np.concatenate([tmp1, tmp2]))

        def compute_edge_bp(curve_ids, nodes_arr, edges_knn):
            curve = nodes_arr[curve_ids]
            curve0 = curve[0]
            dir1 = curve0 - curve[1]
            dir1 = dir1 / np.linalg.norm(dir1)

            node_id = get_node_id(curve0, nodes_arr)
            edges_knn_node = candidate_knn_nodes(node_id, edges_knn)
            scores, norms = [], []
            for node in edges_knn_node:
                dir2 = nodes_arr[node] - curve0
                norm = np.linalg.norm(dir2)
                dir2 = dir2 / norm
                scores.append(cos_sim(dir1, dir2))
                norms.append(norm)
            norms = 1 - (np.array(norms) / np.max(norms))
            scores = np.array(scores) * norms

            if np.max(scores) > 0.0:
                edge_node_id = edges_knn_node[np.argmax(scores)]
                return [node_id, edge_node_id]
            else:
                return []

        edges_gt_bp = []
        edge1 = compute_edge_bp(nodes_ids_dict["curve_1"], nodes_arr, edges_knn)
        if len(edge1) > 0:
            edges_gt_bp.append(edge1)

        edge2 = compute_edge_bp(nodes_ids_dict["curve_12"], nodes_arr, edges_knn)
        if len(edge2) > 0:
            edges_gt_bp.append(edge2)

        edge3 = compute_edge_bp(nodes_ids_dict["curve_3"], nodes_arr, edges_knn)
        if len(edge3) > 0:
            edges_gt_bp.append(edge3)

        edge4 = compute_edge_bp(nodes_ids_dict["curve_4"], nodes_arr, edges_knn)
        if len(edge4) > 0:
            edges_gt_bp.append(edge4)

        return edges_gt_bp

    def compute_basic_edges_gt(self, nodes_ids_dict):
        edges_all = []
        for values in nodes_ids_dict.values():
            for it in range(len(values) - 1):
                edges_all.append([values[it], values[it + 1]])
        return np.array(edges_all)

    def compute_edges_knn(self, nodes):
        edges_knn = torch_geometric.nn.knn_graph(torch.from_numpy(nodes), self.n_knn)
        return edges_knn.detach().numpy().T

    def compute_spline_with_der(self, points, k=3):
        tck, u = splprep(points.T, u=None, k=k, s=0)
        points = np.array(splev(np.linspace(u.min(), u.max(), 500), tck, der=0)).T
        points_der = np.array(splev(np.linspace(u.min(), u.max(), 500), tck, der=1)).T
        return points, points_der

    def viz(self, mask, nodes, edges_knn, edges_gt=None, splines_data=None, bp_locations=None, path=None):
        def nx_graph(nodes_arr, edges):
            G = nx.Graph()
            G.add_nodes_from([(it, {"pos": np.array(xx)}) for it, xx in enumerate(nodes_arr)])
            G.add_edges_from(edges)
            return G

        def draw_graph(g, ax, node_size=50, fontsize=4, colors=None):
            if colors is None:
                colors = "lightblue"

            plt.sca(ax)
            nx.draw(
                g,
                pos={k: (n["pos"][1], n["pos"][0]) for k, n in g.nodes(data=True)},
                node_size=node_size,
                node_color=colors,
            )
            for k, n in g.nodes(data=True):
                ax.text(n["pos"][1], n["pos"][0], k, fontsize=fontsize, ha="center", va="center", color="black")

        #####################
        plt.imshow(mask)
        for k, v in splines_data.items():
            plt.plot(v["points"][:, 0], v["points"][:, 1], "o-", label="{}".format(k))

        for node in nodes:
            plt.scatter(node[1], node[0], c="black", s=20, zorder=150)

        for pos in bp_locations:
            plt.scatter(pos[1], pos[0], c="red", s=20, zorder=150)

        plt.tight_layout()

        if edges_gt is None:
            graph_knn = nx_graph(nodes, edges_knn)
            plt.imshow(mask)
            plt.axis("off")
            ax = plt.gca()
            draw_graph(graph_knn, ax, node_size=100, fontsize=8)
            plt.tight_layout()
            plt.show()
            plt.close()
        else:
            graph_knn = nx_graph(nodes, edges_knn)
            graph_gt = nx_graph(nodes, edges_gt)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 8))
            if path is not None:
                fig.suptitle(path)

            ax1.imshow(mask)
            ax1.axis("off")

            draw_graph(graph_knn, ax1, node_size=75, fontsize=12)
            ax1.set_title("graph_knn")

            ax2.imshow(mask)
            ax2.axis("off")

            # colors based on number of edges

            draw_graph(graph_gt, ax2, node_size=75, fontsize=12)
            ax2.set_title("graph_gt")

            plt.tight_layout()
            plt.show()
            plt.close()


def draw_random_shape(mask, center, scale):
    a = get_random_points(n=9, scale=scale)
    offset = center - np.mean(a, axis=0)
    a = a + offset
    x, y, _ = get_bezier_curve(a, rad=0.2, edgy=0.05)
    cv2.drawContours(mask, [np.column_stack([x, y]).astype(int)], -1, 1, -1)


def mask_paths_augmentation(mask, paths):
    curve3 = paths["curve_3"]
    draw_random_shape(mask, curve3[-1, :], np.random.uniform(30, 60))

    curve4 = paths["curve_4"]
    draw_random_shape(mask, curve4[-1, :], np.random.uniform(30, 60))

    if False:
        plt.imshow(mask)
        plt.show()

    return mask


import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt


bernstein = lambda n, k, t: binom(n, k) * t**k * (1.0 - t) ** (n - k)


def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve


class Segment:
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1
        self.p2 = p2
        self.angle1 = angle1
        self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2 - self.p1) ** 2))
        self.r = r * d
        self.p = np.zeros((4, 2))
        self.p[0, :] = self.p1[:]
        self.p[3, :] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self, r):
        self.p[1, :] = self.p1 + np.array([self.r * np.cos(self.angle1), self.r * np.sin(self.angle1)])
        self.p[2, :] = self.p2 + np.array([self.r * np.cos(self.angle2 + np.pi), self.r * np.sin(self.angle2 + np.pi)])
        self.curve = bezier(self.p, self.numpoints)


def get_curve(points, **kw):
    segments = []
    for i in range(len(points) - 1):
        seg = Segment(points[i, :2], points[i + 1, :2], points[i, 2], points[i + 1, 2], **kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve


def ccw_sort(p):
    d = p - np.mean(p, axis=0)
    s = np.arctan2(d[:, 0], d[:, 1])
    return p[np.argsort(s), :]


def get_bezier_curve(a, rad=0.2, edgy=0):
    """given an array of points *a*, create a curve through
    those points.
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
    p = np.arctan(edgy) / np.pi + 0.5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0, :]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    f = lambda ang: (ang >= 0) * ang + (ang < 0) * (ang + 2 * np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang, 1)
    ang = p * ang1 + (1 - p) * ang2 + (np.abs(ang2 - ang1) > np.pi) * np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var")
    x, y = c.T
    return x, y, a


def get_random_points(n=5, scale=0.8, mindst=None, rec=0):
    """create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or 0.7 / n
    a = np.random.rand(n, 2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1) ** 2)
    if np.all(d >= mindst) or rec >= 200:
        return a * scale
    else:
        return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec + 1)


if __name__ == "__main__":
    if False:
        main_path = "/home/alessio/dev/graph_dlo_learning/gnn_dlo/testing/data_photoneo"
        mask_path = os.path.join(main_path, "wh1/masks/10.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        graph_generator = GraphGenerationWh()

        out_dict = graph_generator.exec(mask, gt_data_flag=False, bp_flag=False, mask_deg_flag=False)

        nodes = out_dict["nodes"]
        points_max = out_dict["points_max"]
        dist_img = out_dict["dist_img"]
        black_img = np.zeros_like(mask)
        print(mask.shape, dist_img.shape, black_img.shape)

        # MASK
        plt.imshow(mask, cmap="gray")
        plt.scatter(nodes[:, 1], nodes[:, 0])
        plt.tight_layout()
        plt.show()

    if False:
        main_path = "/home/alessio/dev/graph_dlo_learning/gnn_dlo/testing/data_photoneo"
        mask_path = os.path.join(main_path, "wh1/masks/6.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        r1, r2 = 100, 400
        c1, c2 = 60, 400
        mask_plot = mask[r1:r2, c1:c2]

        graph_generator = GraphGenerationWh()

        out_dict = graph_generator.exec(mask_plot, gt_data_flag=False, bp_flag=False, mask_deg_flag=False)

        nodes = out_dict["nodes"]
        points_max = out_dict["points_max"]
        dist_img = out_dict["dist_img"]
        black_img = np.zeros_like(mask_plot)
        print(mask.shape, dist_img.shape, black_img.shape)

        # DISTANCE IMAGE
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(dist_img, cmap="gray")
        plt.tight_layout()
        plt.axis("off")
        plt.savefig(os.path.join(main_path, "dist_img.pdf"), bbox_inches="tight")
        plt.close()

        # POINTS MAX
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(black_img, cmap="gray")
        plt.scatter(points_max[:, 1], points_max[:, 0], c="white", s=1)
        plt.tight_layout()
        plt.axis("off")
        plt.savefig(os.path.join(main_path, "points_max.pdf"), bbox_inches="tight")
        plt.close()

        # NODES
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(black_img, cmap="gray")
        plt.scatter(nodes[:, 1], nodes[:, 0], c="red", s=75)
        plt.tight_layout()
        plt.axis("off")
        plt.savefig(os.path.join(main_path, "nodes.pdf"), bbox_inches="tight")
        plt.close()

    if True:  # synthetic data
        folder = "synthetic_data_generation/wh_set3"
        file = "58.pickle"

        path = os.path.join(folder, file)

        reader = ReadPickle()
        data = reader.read(path, show=False)

        mask = data["mask"]
        gt_data = data["gt"]

        cv2.imwrite("mask.png", mask * 255)
        cv2.imwrite("color.jpg", data["color"])

        graph_generator = GraphGenerationWh()

        out_dict = graph_generator.exec(mask, gt_data=gt_data, gt_data_flag=True, bp_flag=True, mask_deg_flag=True)

        nodes = out_dict["nodes"]
        edges = out_dict["edges_gt"]
        dist_img = out_dict["dist_img"]
        bps = out_dict["bps"]

        import itertools
        from shapely import geometry

        def create_nx_graph(nodes, edges):
            g = nx.Graph()
            g.add_nodes_from([(it, {"pos": np.array(xx)}) for it, xx in enumerate(nodes)])
            g.add_edges_from(edges)
            return g

        def make_mask_branch_int_points(nodes, edges, bps, dist_img):
            graph_gt = create_nx_graph(nodes, edges)
            int_pos = compute_intersections(graph_gt)
            positive_pos = np.concatenate((bps, int_pos), axis=0)
            return mask_branchpoints(dist_img, positive_pos)

        def compute_intersections(graph_gt):
            edges_gt = np.array(list(graph_gt.edges))
            positions = []
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
                        int_pos = line1.intersection(line2)
                        int_pos = [p for p in int_pos.coords]
                        # print("intersection: ", int_pos)
                        positions.extend(int_pos)
            return positions

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

        bp_ip_mask = make_mask_branch_int_points(nodes, edges, bps, dist_img)

        g = nx.Graph()
        g.add_nodes_from([(it, {"pos": np.array(xx)}) for it, xx in enumerate(nodes)])
        g.add_edges_from(edges)

        idx_0 = [n for n in g.nodes() if g.degree(n) == 0]
        idx_3 = [n for n in g.nodes() if g.degree(n) == 3]

        print(nodes.shape)
        print(edges.shape)

        white_img = np.ones((mask.shape[0], mask.shape[1], 3), dtype=np.uint8) * 255
        white_img[bp_ip_mask > 0] = [160, 240, 156]
        plt.imshow(white_img)
        for it, node in enumerate(nodes):

            if it in idx_0:
                continue

            v = bp_ip_mask[int(node[0]), int(node[1])]
            if v > 0:
                plt.scatter(node[1], node[0], c="tab:red", s=50, zorder=10)
            else:
                plt.scatter(node[1], node[0], c="tab:blue", s=50, zorder=10)

        for e0, e1 in edges:
            pos_e0 = nodes[e0]
            pos_e1 = nodes[e1]
            plt.plot([pos_e0[1], pos_e1[1]], [pos_e0[0], pos_e1[0]], color="black", linewidth=1)

        plt.xlim(-10, mask.shape[1] + 10)
        plt.ylim(mask.shape[0] + 10, -10)
        plt.axis("off")
        plt.savefig("graph.pdf", bbox_inches="tight")
