import math, itertools
import numpy as np
from scipy.stats import vonmises
import matplotlib.pyplot as plt
import networkx as nx
import shapely
from scipy.interpolate import splprep, splev
import cv2


class PathsAggregator:
    def __init__(self):
        self.predictor = CurvatureVonMisesPredictor(kappa=8)
        self.num_points_sections = 10
        self.debug = False

    def run(self, paths, graph, cluster_dict, dist_img):
        # first filtering
        # paths_f = self.filter_paths_occupancy(paths, dist_img)

        cluster_dict = self.aggregate_close_clusters(graph, cluster_dict, hops=3)

        cluster_dict = self.group_paths_by_cluster(paths, cluster_dict)
        paths_up = self.solve_each_cluster(paths, graph, cluster_dict)

        # second filtering
        paths_up = self.filter_paths_occupancy(paths_up, dist_img)
        paths_up = self.filter_paths_length(paths_up, min_length=4)

        # find intersections and branch points
        out_int, out_bp = self.find_intersections_and_branchpoints(paths_up)
        out_bp = self.find_additional_branchpoints(paths_up, out_bp)

        # merge BPs with only two paths if possible
        paths_up, out_bp = self.merge_branchpoints_with_2_paths(paths_up, out_bp)

        if self.debug:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            for k, path in paths.items():
                points = np.array(path["points"])
                axs[0].plot(points[:, 0], points[:, 1], "o-", label="path" + str(k))
                axs[1].plot(points[:, 0], points[:, 1], "o-", color="black")

            for k, cluster in cluster_dict.items():
                nodes = cluster["nodes_ext"]
                pos = np.array([graph.nodes[n]["pos"] for n in nodes])
                axs[1].scatter(pos[:, 0], pos[:, 1], marker="X", label="cluster" + str(k), zorder=100)

            for k, path in paths_up.items():
                points = np.array(path["points"])
                axs[2].plot(points[:, 0], points[:, 1], "o-", label="path" + str(k))

            for v in out_int:
                point = v["pos"]
                paths = v["path_ids"]
                print("intersection point: ", point, "paths: ", paths)
                axs[2].scatter(point[0], point[1], marker="o", color="red", s=100, zorder=100)

            for v in out_bp:
                point = v["pos"]
                paths = v["path_ids"]
                print("branch point: ", point, "paths: ", paths)
                axs[2].scatter(point[0], point[1], marker="o", color="blue", s=100, zorder=100)

            axs[0].legend()
            axs[1].legend()
            axs[2].legend()
            axs[0].axis("equal")
            axs[1].axis("equal")
            axs[2].axis("equal")
            axs[0].set_title("Original Paths (filtered)")
            axs[1].set_title("Clusters")
            axs[2].set_title("Updated Paths")
            plt.tight_layout()
            plt.show()

        return paths_up, out_int, out_bp

    def merge_branchpoints_with_2_paths(self, paths, out_bp):
        bp_to_delete = []
        mapping_keys = {}
        for bp in out_bp:
            print(bp)
            if len(bp["path_ids"]) != 2:
                continue

            paths_reduced = {}
            for k in bp["path_ids"]:
                if k in paths:
                    paths_reduced[k] = paths[k]
                else:
                    if k in mapping_keys:
                        k = mapping_keys[k]
                        if k in paths:
                            paths_reduced[k] = paths[k]

            if len(paths_reduced) != 2:
                continue
            cluster_center = bp["pos"]
            solutions = self.solve_paths_4(paths_reduced, cluster_center)
            if len(solutions) != 1:
                continue

            solution = solutions[0]
            if solution["score"] < 0.0001:
                continue

            # update paths and clusters
            p0, p1 = solution["pair"]

            # update paths
            points_new = np.concatenate((np.flip(solution["points1"], axis=0), solution["points2"]), axis=0)
            nodes_new = np.concatenate((np.flip(solution["nodes1"], axis=0), solution["nodes2"]), axis=0)

            paths[p0]["points"] = points_new
            paths[p0]["nodes"] = nodes_new
            del paths[p1]

            # update bps
            bp_to_delete.append(bp)
            mapping_keys[p1] = p0

        out_bp = [bp for bp in out_bp if bp not in bp_to_delete]
        return paths, out_bp

    def filter_paths_length(self, paths, min_length=4):
        paths_f = {}
        for k, path in paths.items():
            points = np.array(path["points"])
            if len(points) < min_length:
                continue
            paths_f[k] = path
        return paths_f

    def filter_paths_occupancy(self, paths, dist_img, occupancy_th=0.5):
        mask = np.zeros_like(dist_img).astype(np.uint8)

        paths_f = {}
        for k, path in sorted(paths.items(), key=lambda x: len(x[1]["points"]), reverse=True):
            points = np.array(path["points"]).astype(np.int32)
            values = dist_img[points[:, 0], points[:, 1]]
            thickness = np.mean(values) + np.std(values)

            mask_tmp = np.zeros_like(mask)
            cv2.polylines(mask_tmp, [points], False, 255, thickness=int(thickness * 2))

            # intersection percentage over mask_tmp
            mask_int = cv2.bitwise_and(mask, mask_tmp)
            ratio = np.sum(mask_int) / np.sum(mask_tmp)
            if ratio > occupancy_th:
                continue

            mask[mask_tmp > 0] = 255
            paths_f[k] = path

        # plt.imshow(mask)
        # plt.show()

        return paths_f

    def find_intersections_and_branchpoints(self, paths):
        keys = list(paths.keys())
        combs = list(map(list, itertools.combinations(keys, 2)))

        out_int, out_bp = [], []
        for it0, it1 in combs:
            points0 = paths[it0]["points"]
            points1 = paths[it1]["points"]

            line0 = shapely.geometry.LineString(points0)
            line1 = shapely.geometry.LineString(points1)
            intersect_point = line0.intersection(line1)

            intersect_points = None
            if type(intersect_point) == shapely.geometry.multipoint.MultiPoint:
                intersect_points = [(point.x, point.y) for point in intersect_point.geoms]
            elif type(intersect_point) == shapely.geometry.point.Point:
                intersect_points = [(intersect_point.x, intersect_point.y)]

            if intersect_points is None:
                continue

            for point in intersect_points:
                idx0 = np.argmin(np.linalg.norm(np.array(points0) - np.array(point), axis=1))
                idx1 = np.argmin(np.linalg.norm(np.array(points1) - np.array(point), axis=1))
                if (idx0 != 0 and idx0 != len(points0) - 1) and (idx1 != 0 and idx1 != len(points1) - 1):
                    # fully inside both paths -> intersection
                    out_int.append({"pos": (point[0], point[1]), "path_ids": (it0, it1)})
                else:
                    # branch point
                    out_bp.append({"pos": (point[0], point[1]), "path_ids": (it0, it1)})

        return out_int, out_bp

    def find_additional_branchpoints(self, paths, out_bp, dist_th=20):  # distance threshold in pixels
        splines = {k: self.spline_interpolation(path["points"]) for k, path in paths.items()}
        already_found = [bp["path_ids"] for bp in out_bp]
        for k, path in paths.items():
            points = path["points"]
            for k2, spline2 in splines.items():
                if k == k2:
                    continue

                if (k, k2) in already_found or (k2, k) in already_found:
                    continue

                # closest point on spline
                d0 = np.min(np.linalg.norm(spline2 - points[0], axis=1))
                if d0 < dist_th:
                    out_bp.append({"pos": (points[0][0], points[0][1]), "path_ids": (k, k2)})

                d1 = np.min(np.linalg.norm(spline2 - points[-1], axis=1))
                if d1 < dist_th:
                    out_bp.append({"pos": (points[-1][0], points[-1][1]), "path_ids": (k, k2)})

        return out_bp

    def spline_interpolation(self, points):
        points = np.array(points)
        if len(points) <= 3:
            return points

        tck, u = splprep(points.T, u=None, s=0.0)
        u_new = np.linspace(u.min(), u.max(), points.shape[0] * 10)
        x_new, y_new = splev(u_new, tck, der=0)
        return np.vstack([x_new, y_new]).T

    def aggregate_close_clusters(self, graph, cluster_dict, hops):
        to_delete_keys = []
        for k1, cluster1 in cluster_dict.items():
            if k1 in to_delete_keys:
                continue

            nodes1 = cluster1["nodes_ext"]
            for k2, cluster2 in cluster_dict.items():
                if k2 == k1:
                    continue

                nodes2 = cluster2["nodes_ext"]
                combs = list(itertools.product(nodes1, nodes2))
                for n1, n2 in combs:
                    if nx.has_path(graph, n1, n2):
                        path = nx.shortest_path(graph, n1, n2)
                        if len(path) <= hops:
                            print("merge clusters: ", k1, k2)
                            # merge the two clusters
                            cluster1["nodes_ext"].extend(cluster2["nodes_ext"])
                            cluster1["nodes_nn"].extend(cluster2["nodes_nn"])
                            to_delete_keys.append(k2)
                            break

        cluster_dict = {k: v for k, v in cluster_dict.items() if k not in to_delete_keys}
        return cluster_dict

    def solve_each_cluster(self, paths, graph, cluster_dict):
        paths_up = paths.copy()

        if False:
            plt.figure()
            for k, path in paths_up.items():
                points = np.array(path["points"])
                plt.plot(points[:, 0], points[:, 1], "o-", label="path" + str(k))

            plt.legend()
            plt.axis("equal")
            plt.tight_layout()
            plt.show()

        mapping = {}
        for k, cluster in cluster_dict.items():
            cluster_paths = cluster["paths"]
            cluster_center = cluster["center_pos"]
            nodes = cluster["nodes_ext"]
            print("cluster", k, "center: ", cluster_center, "paths:", cluster_paths)

            paths_reduced = {}
            for k in cluster_paths:
                if k in paths_up:
                    paths_reduced[k] = paths_up[k]
                else:
                    if k in mapping:
                        k = mapping[k]
                        if k in paths_up:
                            paths_reduced[k] = paths_up[k]

            if len(paths_reduced) < 3:
                continue

            paths_reduced_copy = paths_reduced.copy()

            solutions = self.solve_paths_4(paths_reduced, cluster_center)

            # update paths and clusters
            for solution in solutions:
                p0, p1 = solution["pair"]

                # update paths
                points_new = np.concatenate((np.flip(solution["points1"], axis=0), solution["points2"]), axis=0)
                nodes_new = np.concatenate((np.flip(solution["nodes1"], axis=0), solution["nodes2"]), axis=0)

                paths_up[p0]["points"] = points_new
                paths_up[p0]["nodes"] = nodes_new
                del paths_up[p1]
                paths_reduced[p0]["points"] = points_new
                paths_reduced[p0]["nodes"] = nodes_new
                del paths_reduced[p1]
                print("update: ", p0, p1)

                # update clusters
                for k2, cluster2 in cluster_dict.items():
                    if k == k2:
                        continue

                    if p1 in cluster2["paths"]:
                        idx = cluster2["paths"].index(p1)
                        cluster2["paths"][idx] = p0
                        mapping[p1] = p0

            if False:
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                for k, path in paths_reduced_copy.items():
                    points = np.array(path["points"])
                    axs[0].plot(points[:, 0], points[:, 1], "o-", label="path" + str(k))
                    axs[1].plot(points[:, 0], points[:, 1], "o-", color="black")

                pos = np.array([graph.nodes[n]["pos"] for n in nodes])
                axs[1].scatter(pos[:, 0], pos[:, 1], marker="X", label="cluster" + str(k), zorder=100)

                for k, path in paths_reduced.items():
                    points = np.array(path["points"])
                    axs[0].plot(points[:, 0], points[:, 1], "o-", label="path" + str(k))
                    axs[1].plot(points[:, 0], points[:, 1], "o-", color="black")

                axs[0].legend()
                axs[1].legend()
                axs[0].axis("equal")
                axs[1].axis("equal")
                plt.tight_layout()
                plt.show()

            if False:
                plt.figure()
                for k, path in paths_up.items():
                    points = np.array(path["points"])
                    plt.plot(points[:, 0], points[:, 1], "o-", label="path" + str(k))

                plt.legend()
                plt.axis("equal")
                plt.tight_layout()
                plt.show()

        return paths_up

    def solve_paths_2(self, paths, cluster_nodes):
        pass

    def solve_paths_4(self, paths, cluster_center):
        combinations = list(itertools.combinations(paths.keys(), 2))
        # print("combinations: ", combinations)
        scores_dict = {}
        for comb in combinations:
            # print("comb: ", comb)
            path1 = paths[comb[0]]
            path2 = paths[comb[1]]

            points1 = np.array(path1["points"])
            points2 = np.array(path2["points"])
            nodes1 = path1["nodes"]
            nodes2 = path2["nodes"]

            if np.linalg.norm(points1[0] - cluster_center) > np.linalg.norm(points1[-1] - cluster_center):
                points1 = np.flip(points1, axis=0)
                nodes1 = np.flip(nodes1, axis=0)

            if np.linalg.norm(points2[0] - cluster_center) > np.linalg.norm(points2[-1] - cluster_center):
                points2 = np.flip(points2, axis=0)
                nodes2 = np.flip(nodes2, axis=0)

            if nodes1[0] == nodes2[0]:
                nodes1 = nodes1[1:]
                points1 = points1[1:]

            if len(points1) > self.num_points_sections:
                points1_reduced = points1[: self.num_points_sections]
            else:
                points1_reduced = points1

            if len(points2) > self.num_points_sections:
                points2_reduced = points2[: self.num_points_sections]
            else:
                points2_reduced = points2

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            dist = np.linalg.norm(points1_reduced[0] - points2_reduced[0])
            if dist > 100:
                continue
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            points1_reduced = np.flip(points1_reduced, axis=0)[:-1]
            points2_reduced = points2_reduced[1:]
            points = np.concatenate((points1_reduced, points2_reduced), axis=0)

            scores_dict[comb] = {
                "pair": comb,
                "score": self.predictor.computeScore(points),
                "points1": points1[1:],
                "points2": points2[1:],
                "nodes1": nodes1[1:],
                "nodes2": nodes2[1:],
            }

            if False:
                print("pair: ", comb, "score ", scores_dict[comb]["score"])
                for k, v in paths.items():
                    plt.plot(np.array(v["points"])[:, 0], np.array(v["points"])[:, 1], "o-", label="path" + str(k))

                plt.scatter(points[:, 0], points[:, 1], label="Points", s=100)
                plt.scatter(cluster_center[0], cluster_center[1], marker="X", color="red")
                plt.legend()
                plt.show()

        ################
        # LAST STEP
        scores_dict = {k: v for k, v in sorted(scores_dict.items(), key=lambda item: item[1]["score"], reverse=True)}
        best_pairs = self.select_best_combs(scores_dict)

        if self.debug:
            for k, s in scores_dict.items():
                print(k, s["score"])
            print(" ******* solution: ", best_pairs)

        return [scores_dict[pair] for pair in best_pairs]

    def select_best_combs(self, scores):
        counter = 0
        pairs, done = [], []
        for ke, values in scores.items():
            # if values["score"] < 0.0001:
            #    break

            if counter > 2:
                break

            if ke[0] not in done and ke[1] not in done:
                pairs.append(ke)
                done.extend([ke[0], ke[1]])
                counter += 1

        return pairs

    def group_paths_by_cluster(self, paths, cluster_dict):
        for k, cluster in cluster_dict.items():
            nodes = cluster["nodes_ext"]
            nodes_nn = cluster["nodes_nn"]
            all_nodes = nodes + nodes_nn

            list_paths_in_cluster = []
            for path_k, path_v in paths.items():
                nodes_path = path_v["nodes"]
                if set(all_nodes).intersection(set(nodes_path)):
                    list_paths_in_cluster.append(path_k)

            cluster["paths"] = list_paths_in_cluster
            cluster["nodes_all"] = all_nodes
        return cluster_dict


####################################################################################################
####################################################################################################
####################################################################################################


class VonMisesBuffer(object):
    def __init__(self, k=3):
        self.vonmises_range = np.arange(0, math.pi * 2, 0.0001)
        self.vonmises_base = vonmises(k)
        self.vonmises_values = self.vonmises_base.pdf(self.vonmises_range)

    def pdf(self, x):
        i = int(math.fabs(x) * 10000)
        if i >= 0 and i < len(self.vonmises_values):
            return self.vonmises_values[i]
        return 0.0


class CurvatureVonMisesPredictor:
    def __init__(self, kappa=12, max_angle=math.pi / 2):
        self.vonmises = VonMisesBuffer(k=kappa)
        self.max_angle = max_angle

    def computeScore(self, points):
        #######################################
        # Single Node Path
        #######################################
        if len(points) <= 2:
            return 1.0

        #######################################
        # Normal Path
        #######################################
        directions = []
        for i in range(1, len(points)):
            p1 = np.array(points[i - 1])
            p2 = np.array(points[i])
            direction = p2 - p1
            direction = direction / np.linalg.norm(direction)
            directions.append(direction)

        thetas = []
        for i in range(1, len(directions)):
            d1 = directions[i - 1]
            d2 = directions[i]
            a1 = math.atan2(d1[1], d1[0])
            a2 = math.atan2(d2[1], d2[0])
            angle = a1 - a2
            angle = math.acos(math.cos(angle))

            thetas.append(angle)

        # print("thetas: ", thetas)
        if math.fabs(thetas[-1]) > self.max_angle:
            return 0.0

        if len(thetas) == 1:
            prob = self.vonmises.pdf(thetas[0])
        elif len(thetas) > 1:
            probs = [self.vonmises.pdf(theta) for theta in thetas]
            prob = np.prod(probs)

        return prob
