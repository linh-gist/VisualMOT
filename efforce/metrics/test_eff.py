
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

from efforce.metrics.Efforce import Efforce


class Test_eff(Efforce):

    def __init__(self):

        super().__init__()

        self.alfa = 0.5
        self.beta = 0.5

        self.e = 0.0000001

        self.plot = False


    def cost_matrix(self, v1, v2):


        # x, y, w, h ---> x1, y1, x2, y2
        v1m = self.coord2corner(v1)
        v2m = self.coord2corner(v2)

        # Cost matrix based on IoU
        matrix = self.iou(v1m, v2m)


        row, col = linear_sum_assignment(matrix)
        
        cost = matrix[row, col].sum()

        return cost, row, col, matrix



    def names(self):
        # return ['Det. Efforce', 'Trk. Efforce', 'Trk. over Det.', 'MOT Acc', 'IDs Efforce']
        # return ['Eintra', 'Sf', 'Sd', 'St', 'Y', 'Nd', 'Nt', 'Id', 'It', 'Einter', 'Sf', 'Sd', 'St', 'Y', 'Nd', 'Nt', 'Id', 'It']
        return ['E', 'Eintra', 'Qd', 'Qt', 'Nd', 'Nt', 'Id', 'It', 'Einter', 'Id', 'It', 'Y', 'C', 'IDSW']
        


    def intra_frame(self):

        gt_ids = self.gt_ids_counter()
        self.prev_traces = np.nan * np.zeros(len(gt_ids))

        # Association scores
        Id = np.zeros((self.K))
        It = np.zeros((self.K))
        # Ad = np.zeros((self.K))
        # At = np.zeros((self.K))

        # Cardinality scores
        Nd = np.zeros((self.K))
        Nt = np.zeros((self.K))
        Ud = np.zeros((self.K))
        Ut = np.zeros((self.K))
        Ld = np.zeros((self.K))
        Lt = np.zeros((self.K))
        V  = np.zeros((self.K))

        # Final scores
        Qd = np.zeros((self.K))
        Qt = np.zeros((self.K))
        Qd_sum = np.zeros((self.K))
        Qt_sum = np.zeros((self.K))

        Eintra = np.zeros((self.K))
        Eintra_sum = np.zeros((self.K))

        IDSW = np.zeros((self.K))


        for k in range(self.K):

            kp = k + 1

            # Check values.
            if not kp in self.ut:  self.ut[kp] = np.zeros((0, 5))
            if not kp in self.ud:  self.ud[kp] = np.zeros((0, 5))


            # Calculate association costs and IDSW
            Ad, row_d, _, _    = self.cost_matrix(self.ud[kp][:, 1:], self.v[kp][:, 1:])
            # At, row_t, _, _    = self.cost_matrix(self.ut[kp][:, 1:], self.v[kp][:, 1:])
            idsw, At, row_t, _ = self.IDSW(self.ut[kp], self.v[kp])


            # Cardinaliry variables
            V[k]  = len(self.v[kp])
            Ud[k] = len(self.ud[kp])
            Ut[k] = len(self.ut[kp])
            Ld[k] = len(row_d)
            Lt[k] = len(row_t)

            # IDSW for inter-frame complexity
            IDSW[k] = (1 - (idsw / (Lt[k] + self.e)))
            
            # Association scores
            Id[k] = 1 - (Ad / (Ld[k] + self.e))
            It[k] = 1 - (At / (Lt[k] + self.e))

            # Cardinality comparision
            Nd[k] = 1 - (abs(Ud[k] - V[k]) / max(V[k], Ud[k]))
            Nt[k] = 1 - (abs(Ut[k] - V[k]) / max(V[k], Ut[k]))

            # Quality of bounding boxes metric
            Qd[k] = Id[k] * Nd[k]
            Qt[k] = It[k] * Nt[k]

            Qd_sum[k] = (self.alfa * Id[k]) + ((1 - self.alfa) * Nd[k])
            Qt_sum[k] = (self.beta * It[k]) + ((1 - self.beta) * Nt[k])



            Eintra[k] = Qt[k] - Qd[k]
            Eintra_sum[k] = Qt_sum[k] - Qd_sum[k]



        if self.plot == True:
            self.plotDet(Qd, Qt, Eintra)
            self.plotDet_basic(Nd, Nt, Id, It, Qd, Qt)
            self.plotDet_cardinality(Nd, Nt, Ud, Ut, V)
            

        

        Eintra_list = Eintra

        Id_list = Id
        It_list = It
        Nd_list = Nd
        Nt_list = Nt
        Qd_list = Qd
        Qt_list = Qt

        # Association scores
        Id = sum(Id) / self.K
        It = sum(It) / self.K
        # Ad = sum(Ad) / self.K
        # At = sum(At) / self.K

        # Cardinality scores
        Nd = sum(Nd) / self.K
        Nt = sum(Nt) / self.K
        # Ud = sum(Ud) / self.K
        # Ut = sum(Ut) / self.K
        # V  = sum(V) / self.K
        # Ld = sum(Ld) / self.K
        # Lt = sum(Lt) / self.K
        
        # Final scores
        Qd = sum(Qd) / self.K
        Qt = sum(Qt) / self.K

        self.IDSW = IDSW

        # self.save_list_file('intra.csv', Eintra)

        Eintra = sum(Eintra) / self.K


        return Eintra, Qd, Qt, Nd, Nt, Id, It, Eintra_list, Qd_list, Qt_list, Id_list, Nd_list, It_list, Nt_list




    def inter_frame(self):


        # Association scores
        Id = np.zeros((self.K))
        It = np.zeros((self.K))
        # Bd = np.zeros((self.K))
        # Bt = np.zeros((self.K))

        # Cardinality scores
        C = np.zeros((self.K))
        Ud = np.zeros((self.K))
        Ut = np.zeros((self.K))
        Ld = np.zeros((self.K))
        Lt = np.zeros((self.K))
        V  = np.zeros((self.K))

        # IDSW = np.zeros((self.K))

        # Final scores
        Y  = np.zeros((self.K))

        Einter = np.zeros((self.K))



        for k in range(self.K-1):

            kp = k + 1

            # Check values.
            if not kp in self.ut:  self.ut[kp] = np.zeros((0, 5))
            if not kp in self.ud:  self.ud[kp] = np.zeros((0, 5))


            # Calculate association costs and IDSW
            Bd, row_d, _, _ = self.cost_matrix(self.ud[kp][:, 1:], self.ud[kp + 1][:, 1:])
            Bt, row_t, _, _ = self.cost_matrix(self.ut[kp][:, 1:], self.ut[kp + 1][:, 1:])

            # idsw, _, _, _ = self.IDSW(self.ut[kp], self.v[kp])


            GT_IDs = len(np.union1d(self.v[kp][:, 0], self.v[kp+1][:, 0]))


            # Cardinaliry variables
            V[k]  = len(self.v[kp])
            Ud[k] = len(self.ud[kp])
            Ut[k] = len(self.ut[kp])
            Ld[k] = len(row_d)
            Lt[k] = len(row_t)
            
            # Association scores
            Id[k] = 1 - (Bd / (Ld[k] + self.e))
            It[k] = 1 - (Bt / (Lt[k] + self.e))
            Y[k]  = It[k] - Id[k]


            # IDSW[k] = (1 - (idsw / (Lt[k] + self.e)))

            # Cardinality comparision
            C[k] = 1 - (abs(Lt[k] - GT_IDs) / (Lt[k] + GT_IDs))



            Einter[k] = C[k] * self.IDSW[k] + Y[k]
        

        if self.plot == True:
            self.plotInterAss(Id, It, Y)
            self.plotInterID(Einter, self.IDSW, C, Y)

        
        Einter_list = Einter
        Y_list = Y
        C_list = C
        IDSW_list = self.IDSW




        # Association scores
        Id = sum(Id) / self.K
        It = sum(It) / self.K

        # Cardinality scores
        C = sum(C) / self.K
        # Ud = sum(Ud) / self.K
        # Ut = sum(Ut) / self.K
        # Ld = sum(Ld) / self.K
        # Lt = sum(Lt) / self.K
        # V  = sum(V) / self.K
        
        IDSW = sum(self.IDSW) / self.K

        # Final scores
        Y  = sum(Y) / self.K

        Einter  = sum(Einter) / self.K



        return Einter, Id, It, Y, C, IDSW, Einter_list, Y_list, C_list, IDSW_list





    def gt_ids_counter(self):

        list_ids = []

        for k in range(self.K):

            k_list = self.v[k + 1]

            list_ids = np.union1d(list_ids, k_list)

        return list_ids




    def calc_score_trace(self):

        for v in self.traces_id:
            pass



    def join_metrics(self, intra, inter):

        Eintra = intra[0]
        Einter = inter[0]

        E = Eintra + Einter

        # self.plotMetric(E, Eintra, Einter)

        return tuple([E] + list(intra[:7]) + list(inter[:6]) + list(intra[7:]) + list(inter[6:]))





    #####################################################################
    #
    #   Plots
    #
    #
    #####################################################################


    def plotDet(self, Qd, Qt, Eintra):

        plt.figure(figsize=(10, 6))

        # Detection compare
        plt.plot(Qd,  label='Qd') # (Quality of Bboxes from detector)
        plt.plot(Qt,  label='Qt') # (Quality of Bboxes from tracker)
        plt.plot(Eintra,  label='Eintra') # (Tracking Efforce Intra-frame)


        plt.ylabel('Score')
        plt.xlabel('Frame number')
        plt.title('%s / %s performance bounding boxes' % (self.detector, self.tracker))
        plt.legend()
        # plt.show()

        plt.savefig('outputs/figs/Intra_%s_%s.jpg' % (self.detector, self.tracker), dpi=230)
        plt.clf()
        plt.close()


    def plotDet_basic(self, Nd, Nt, Id, It, Qd, Qt):

        plt.figure(figsize=(10, 6))

        # Detection compare
        plt.plot(Nd,  label='Nd') # (Cardinality BBoxes)
        plt.plot(Nt,  label='Nt') # (Cardinality BBoxes)
        plt.plot(Id,  label='Id') # (Overlap with BBoxes)
        plt.plot(It,  label='It') # (Overlap with BBoxes)
        plt.plot(Qd,  label='Qd') # (Quality of Bboxes from detector)
        plt.plot(Qt,  label='Qt') # (Quality of Bboxes from tracker)


        plt.ylabel('Score')
        plt.xlabel('Frame number')
        plt.title('%s / %s performance bounding boxes' % (self.detector, self.tracker))
        plt.legend()
        # plt.show()

        plt.savefig('outputs/figs/Intra_basic_%s_%s.jpg' % (self.detector, self.tracker), dpi=230)
        plt.clf()
        plt.close()
        # plt.close('all')


    def plotDet_cardinality(self, Nd, Nt, Ud, Ut, V):

        plt.figure(figsize=(10, 6))

        # Detection compare
        plt.plot(Nd,  label='Nd') # (Cardinality BBoxes)
        plt.plot(Nt,  label='Nt') # (Cardinality BBoxes)
        plt.plot(Ud,  label='Ud') # (Number of BBoxes by detector)
        plt.plot(Ut,  label='Ut') # (Number of BBoxes by tracker)
        plt.plot(V,   label='V') # (Number of ground truth BBoxes)


        plt.ylabel('Score')
        plt.xlabel('Frame number')
        plt.title('%s / %s performance bounding boxes' % (self.detector, self.tracker))
        plt.legend()
        # plt.show()

        plt.savefig('outputs/figs/Intra_cardin_%s_%s.jpg' % (self.detector, self.tracker), dpi=230)
        plt.clf()
        plt.close()
        # plt.close('all')


    def plotInterAss(self, Id, It, Y):

        plt.figure(figsize=(10, 6))

        # Detection compare
        plt.plot(Id, label='Id') # (BBoxes overlap consecutive frames)
        plt.plot(It, label='It') # (BBoxes overlap consecutive frames)
        plt.plot(Y,  label='Difference between Id and It  (It - Id)')


        plt.ylabel('Score')
        plt.xlabel('Frame number')
        plt.title('%s / %s performance bounding boxes' % (self.detector, self.tracker))
        plt.legend()
        # plt.show()

        plt.savefig('outputs/figs/IntrerAss_%s_%s.jpg' % (self.detector, self.tracker), dpi=230)
        plt.clf()
        plt.close()
        # plt.close('all')


    def plotInterID(self, Einter, IDSW, C, Y):

        plt.figure(figsize=(10, 6))

        # Detection compare
        plt.plot(Einter, label='Einter') # (Tracking Efforce Inter-frame)
        plt.plot(IDSW, label='IDSW_score') # (IDSW score)
        plt.plot(C,    label='C') # (Cardinality metric)
        plt.plot(Y,  label='Y') # (Difference between detection and tracking)


        plt.ylabel('Score')
        plt.xlabel('Frame number')
        plt.title('%s / %s performance bounding boxes' % (self.detector, self.tracker))
        plt.legend()
        # plt.show()

        plt.savefig('outputs/figs/Inter_%s_%s.jpg' % (self.detector, self.tracker), dpi=230)
        plt.clf()
        plt.close()
        # plt.close('all')


    def plotMetric(self, E, Eintra, Einter):

        plt.figure(figsize=(10, 6))

        # Detection compare
        plt.plot(E,  label='E') # (Tracking Efforce)
        plt.plot(Eintra,  label='Eintra') # (Tracking Efforce Intra-frame)
        plt.plot(Einter, label='Einter') # (Tracking Efforce Inter-frame)


        plt.ylabel('Score')
        plt.xlabel('Frame number')
        plt.title('%s / %s performance bounding boxes' % (self.detector, self.tracker))
        plt.legend()
        # plt.show()

        plt.savefig('outputs/figs/metric_%s_%s.jpg' % (self.detector, self.tracker), dpi=230)
        plt.clf()
        plt.close()
        # plt.close('all')
