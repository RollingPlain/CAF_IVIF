import cv2
import numpy as np
def Get_color(Vis_dir, F_dir):

        vis_YCrCb = cv2.cvtColor(Vis_dir, cv2.COLOR_BGR2YCrCb)

        if len(F_dir.shape) == 2:
            F_dir = F_dir[:, :, np.newaxis]

        vis_YCrCb_1 = np.transpose(vis_YCrCb, (2, 0, 1))
        F_1 = np.transpose(F_dir, (2, 0, 1))

        vis_YCrCb_1[0] = F_1[0]

        vis_YCrCb_2 = np.transpose(vis_YCrCb_1, (1, 2, 0))
        F_fin = cv2.cvtColor(vis_YCrCb_2, cv2.COLOR_YCrCb2BGR)
        return F_fin



