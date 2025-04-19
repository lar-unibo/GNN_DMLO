# GNN Topology Representation Learning for Deformable Multi-Linear Objects Dual-Arm Robotic Manipulation

Abstract:
Deformable Multi-Linear Objects (DMLOs), or Branched Deformable Linear Objects (BDLOs), are flexible objects that possess a linear structure similar to DLOs but also feature branching or bifurcation points where the object’s path diverges into multiple sections. The representation of complex DMLOs, such as wiring harnesses, poses significant challenges in various applications, including robotic systems’ perception and manipulation planning. This paper proposes an approach to address the robust and efficient estimation of a topological representation for DMLOs leveraging a graph-based description of the scene obtained via graph neural networks. Starting from a binary mask of the scene, graph nodes are sampled along the objects’ estimated centerlines. Then, a data-driven pipeline is employed to learn the assignment of graph edges between nodes and to characterize the node’s type based on their local topology and orientation. Finally, by utilizing the learned information, a solver combines the predictions and generates a coherent representation of the objects in the scene. The approach is experimentally evaluated using a test set of complex real-world DMLOs. Within an offline evaluation, the proposed approach achieves a Dice score exceeding 90% in predicting graph edges. Similarly, the identification accuracy of branch and intersection points in the graph topology is above 90%. Additionally, the method demonstrates efficient performance, achieving a runtime of over 20 FPS. In an online assessment employing a dual-arm robotic setup, the approach is successfully applied to disentangle three automotive wiring harnesses, demonstrating the effectiveness of the proposed approach in a real-world scenario.

**Published in**:  
IEEE Transactions on Automation Science and Engineering 📄 [Open Access Publication](https://ieeexplore.ieee.org/document/10970007)

---

## Installation

```
pip install -e .
```

tested on:
```
      - matplotlib==3.9.4
      - networkx==3.2.1
      - numpy==1.26.3
      - opencv-python==4.11.0.86
      - pillow==11.0.0
      - pyg-lib==0.4.0+pt26cpu
      - scipy==1.13.1
      - shapely==2.0.7
      - torch==2.6.0+cpu
      - torch-cluster==1.6.3+pt26cpu
      - torch-geometric==2.6.1
      - torch-scatter==2.1.2+pt26cpu
      - torch-sparse==0.6.18+pt26cpu
      - torch-spline-conv==1.2.2+pt26cpu
      - torchaudio==2.6.0+cpu
      - torchvision==0.21.0+cpu
```

---

# Citation
If you find this work useful, please consider citing:

```
@ARTICLE{10970007,
  author={Caporali, Alessio and Galassi, Kevin and Zanella, Riccardo and Palli, Gianluca},
  journal={IEEE Transactions on Automation Science and Engineering}, 
  title={GNN Topology Representation Learning for Deformable Multi-Linear Objects Dual-Arm Robotic Manipulation}, 
  year={2025},
  doi={10.1109/TASE.2025.3562231}}
```
