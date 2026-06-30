
import os
import numpy as np

from abc import ABC, abstractmethod

class Efforce(ABC):

    def __init__(self):

        self.prev_gt = None
        self.prev_tr = None
        self.traces  = {}

        self.files_open = {}


    def evaluate(self, v, ud, ut, detector, tracker):

        self.v  = v
        self.ud = ud
        self.ut = ut
        
        self.K  = len(v)

        self.detector = detector
        self.tracker  = tracker


        intra = self.intra_frame()
        inter = self.inter_frame()

        values = self.join_metrics(intra, inter)

        return values



    @abstractmethod
    def cost_matrix(self, v1, v2):
        pass

    @abstractmethod
    def names(self):
        pass

    @abstractmethod
    def intra_frame(self):
        pass

    @abstractmethod
    def inter_frame(self):
        pass

    @abstractmethod
    def join_metrics(self, intra, inter):
        pass



    def IDSW(self, tr, gt):

        cost, row, col, _ = self.cost_matrix(tr[:, 1:], gt[:, 1:])

        matched_gt_ids      = gt[col, 0].astype(int)
        matched_tracker_ids = tr[row, 0].astype(int)

        prev_matched_tracker_ids = self.prev_traces[matched_gt_ids]

        is_idsw = (np.logical_not(np.isnan(prev_matched_tracker_ids))) & (
                  np.not_equal(matched_tracker_ids, prev_matched_tracker_ids))

        self.prev_traces[matched_gt_ids] = matched_tracker_ids


        # print('->', len(gt), len(col), ':', len(col) / len(gt))


        return np.sum(is_idsw), cost, row, col


    



    @staticmethod
    def coord2center(v):

        v2 = v.copy()

        v2[:, 0] = v2[:, 0] + v2[:, 2] / 2
        v2[:, 1] = v2[:, 1] + v2[:, 3] / 2

        return v2[:, :2]


    @staticmethod
    def coord2corner(v):

        v2 = v.copy()

        v2[:, 2] = v2[:, 0] + v2[:, 2]
        v2[:, 3] = v2[:, 1] + v2[:, 3]

        return v2


    @staticmethod
    def get_iou(v1, v2):
    

        xA = max(v1[0], v2[0])
        yA = max(v1[1], v2[1])
        xB = min(v1[2], v2[2])
        yB = min(v1[3], v2[3])

        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))

        boxAArea = abs((v1[2] - v1[0]) * (v1[3] - v1[1]))
        boxBArea = abs((v2[2] - v2[0]) * (v2[3] - v2[1]))

        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou


    @staticmethod
    def iou(v1, v2):

        matrix = np.zeros((len(v1), len(v2)))

        for i, v1_bbx in enumerate(v1):

            for j, v2_bbx in enumerate(v2):

                matrix[i, j] = 1 - Efforce.get_iou(v1_bbx, v2_bbx)


        return matrix


        