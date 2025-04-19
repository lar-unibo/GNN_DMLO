import os
import cv2
import glob
from gnn_dmlo.core.core import FullPredictor


if __name__ == "__main__":
    ############################################

    CHECKPOINT_LP_PATH = "../checkpoints/linkpred.pth"
    CHECKPOINT_CLS_PATH = "../checkpoints/nodecls.pth"

    SAMPLES_PATH = "../example_real_samples"

    ############################################
    predictor = FullPredictor(CHECKPOINT_LP_PATH, CHECKPOINT_CLS_PATH)

    files = sorted(glob.glob(os.path.join(SAMPLES_PATH, "*.png")))
    print(files)

    for mask_path in files:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask > 127] = 255
        mask[mask <= 127] = 0

        paths_final, out_int, out_bp, data_save = predictor.run(mask, plot=False)

        print(f"Paths: {paths_final}")

        quit()
