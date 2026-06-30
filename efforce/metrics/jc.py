
import numpy as np


from scipy.optimize import linear_sum_assignment

from metrics.Efforce import Efforce


class JC(Efforce):

    def __init__(self):

        super().__init__()


    def cost_matrix(self, v1, v2):
        
        v1 = self.coord2corner(v1)
        v2 = self.coord2corner(v2)

        matrix = self.iou(v1, v2)

        row, col = linear_sum_assignment(matrix)
        
        cost = matrix[row, col].sum()

        return cost



    def names(self):
        return ['MOT Acc.', 'Intra Frame', 'Inter Frame', 'Intra det.', 'Intra trk.', 'Inter det.', 'Inter trk.']


    def intra_frame(self):

        # Create variables.
        E  = np.zeros((self.K))
        Ed = np.zeros((self.K))
        Et = np.zeros((self.K))

        V  = np.zeros((self.K))
        Ud  = np.zeros((self.K))
        Ut  = np.zeros((self.K))
        Ad = np.zeros((self.K))
        At = np.zeros((self.K))

        # print(self.ut.keys())
        # print('----------------')
        # print(self.ut.values())

        for k in range(self.K):

            # Check values.
            if not k+1 in self.ut:  self.ut[k+1] = np.zeros((0, 5))
            if not k+1 in self.ud:  self.ud[k+1] = np.zeros((0, 5))

            # Variables.
            V[k]  = len(self.v[k + 1])
            Ud[k] = len(self.ud[k + 1])
            Ut[k] = len(self.ut[k + 1])

            # Cost of the Hungarian Model
            Ad[k] = self.cost_matrix(self.ud[k + 1][:, 1:], self.v[k + 1][:, 1:])
            At[k] = self.cost_matrix(self.ut[k + 1][:, 1:], self.v[k + 1][:, 1:])

            Ed[k] = Ad[k] + abs(Ud[k] - V[k])
            Et[k] = At[k] + abs(Ut[k] - V[k])


            E[k] = abs(Ed[k] - Et[k])


        Sd = sum(Ed) / self.K
        St = sum(Et) / self.K

        S = (1 / self.K) * sum([(1 - ((E[k] / Ed[k]) if Ed[k] != 0 else 0)) for k in range(self.K)])


        return S, Sd, St


    def inter_frame(self):
        
        # Create variables.
        E  = np.zeros((self.K))
        Ed = np.zeros((self.K))
        Et = np.zeros((self.K))

        V  = np.zeros((self.K))
        Ud  = np.zeros((self.K))
        Ut  = np.zeros((self.K))
        Ad = np.zeros((self.K))
        At = np.zeros((self.K))



        for k in range(self.K - 1):

            # Variables.
            V[k]  = len(self.v[k + 1])
            Ud[k] = len(self.ud[k + 1])
            Ut[k] = len(self.ut[k + 1])

            # Cost of the Hungarian Model
            Ad[k] = self.cost_matrix(self.ud[k + 1][:, 1:], self.v[k + 2][:, 1:])
            At[k] = self.cost_matrix(self.ut[k + 1][:, 1:], self.v[k + 2][:, 1:])

            Ed[k] = Ad[k] + abs(Ud[k] - V[k])
            Et[k] = At[k] + abs(Ut[k] - V[k])


            E[k] = abs(Ed[k] - Et[k])


        Sd = sum(Ed) / (self.K - 1)
        St = sum(Et) / (self.K - 1)

        S = (1 / (self.K - 1)) * sum([(1 - ((E[k] / Ed[k]) if Ed[k] != 0 else 0)) for k in range(self.K - 1)])


        return S, Sd, St


    def join_metrics(self, intra, inter, alfa=0.5):

        S_intra, Sd_intra, St_intra = intra
        S_inter, Sd_inter, St_inter = inter

        return (alfa * S_intra) + ((1 - alfa) * S_inter), S_intra, S_inter, Sd_intra, St_intra, Sd_inter, St_inter


