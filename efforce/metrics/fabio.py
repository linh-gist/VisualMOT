
import numpy as np

from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment


from metrics.Efforce import Efforce


class Fabio(Efforce):

    def __init__(self):

        super().__init__()


    def cost_matrix(self, v1, v2):
        
        matrix = distance.cdist(v1, v2, 'cosine')

        row, col = linear_sum_assignment(matrix)

        cost = matrix[row, col].sum()

        return cost, row, col, len(row)


    def names(self):
        return ['Det. Efforce', 'Trk. Efforce', 'Trk. over Det.', 'MOT Acc']


    def intra_frame(self):
        '''
        Fabio, intra-frame complexity 1.

        '''

        # Create variables.
        E  = np.zeros((self.K))
        Ed = np.zeros((self.K))
        Et = np.zeros((self.K))
        yd = np.zeros((self.K))
        yt = np.zeros((self.K))
        I  = np.zeros((self.K))
        v  = np.zeros((self.K))


        for k in range(self.K):

            # Check values.
            if not k+1 in self.ut:  self.ut[k+1] = np.zeros((0, 5))
            if not k+1 in self.ud:  self.ud[k+1] = np.zeros((0, 5))

            Ad, _, _, yd[k] = self.cost_matrix(self.ud[k + 1][:, 1:], self.v[k + 1][:, 1:])
            At, _, _, yt[k] = self.cost_matrix(self.ut[k + 1][:, 1:], self.v[k + 1][:, 1:])

            e = .1   # ?

            Ud = len(self.ud[k + 1])
            Ut = len(self.ut[k + 1])
            V  = len(self.v[k + 1])

            # Calculate efforce for detection / tracking.
            Ed[k] = (0.5 * ((Ad / (yd[k] + e)) + (abs(min(Ud, V) - yd[k]) / (min(Ud, V) + e)))) + (abs(Ud - V) / max(Ud, V))
            Et[k] = (0.5 * ((At / (yt[k] + e)) + (abs(min(Ut, V) - yt[k]) / (min(Ut, V) + e)))) + (abs(Ut - V) / max(Ut, V))

            # Efforce combination.
            E[k] = Ed[k] - Et[k]

            # TODO
            I[k], _, _, _ = self.IDSW(self.ut[k + 1], self.v[k + 1])
            v[k] = V


        E_a = sum(E) / self.K

        # S = (1 / K) * sum(1 - (E[k] / Ed[k]) + y * (I[k] / v[k]))
        S = (1 / self.K) * sum([(1.0 - (E[k] / Ed[k]) + yt[k] * (I[k] / v[k])) for k in range(self.K)])
        # print(S)

        return S, E_a

        


    def inter_frame(self):
        return (0,)


    def join_metrics(self, intra, inter):

        S, E_a = intra

        return (S,)



# # Fabio (intra-frame complexity 1)

    # Ed[k] = (0.5 * ((Ad[k] / (Y + e)) + ((min(Ud[k], V[k]) - y) / (min(Ud[k], V[k]) + e)))) + (|Ud[k], V[k]| / max(Ud[k], V[k]))
    # Et[k] = (0.5 * ((At[k] / (Y + e)) + ((min(Ut[k], V[k]) - y) / (min(Ut[k], V[k]) + e)))) + (|Ut[k], V[k]| / max(Ut[k], V[k]))


    # E[k] = Ed[k] - Et[k]


    # S = (1 / K) * sum(1 - (E[k] / Ed[k]) + y * (I[k] / v[k]))

