import os
import numpy as np

from multiprocessing import Pool
from functools import partial

# from metrics.fabio import Fabio
# from metrics.jc import JC
from efforce.metrics.test_eff import Test_eff


def load_file(path, type, labels=[1, 2, 3, 4, 5, 6, 7]):
    '''
    Load tracking file from path.

    Inputs:
        - path : path were the file is stored.
        - type : select from list: ['gt', 'det', 'public', 'trc']

    Outputs:
        - dict where key is the frame number and value detections
          in the frame.
    '''

    file = np.loadtxt(path, delimiter=',')

    unique = np.unique(file[:, 0])

    frame = {}

    for u in unique:

        a = np.where(file[:, 0] == u)

        # frame[u] = file[a][:, 1:]
        aux_frame = file[a][:, 1:]

        if type == 'gt':

            idx = np.where(np.isin(aux_frame[:, 6], labels))
            aux_frame = aux_frame[idx]

            aux_frame = aux_frame[:, :5]


        elif type == 'det':

            aux_frame = aux_frame[:, :5]


        elif type == 'public':

            aux_frame = aux_frame[:, :5]


        elif type == 'trc':

            aux_frame = aux_frame[:, :5]


        else:
            assert "Error, incorrect type"

        frame[u] = aux_frame

    return frame, len(unique)


def run_metrics(metric_obj, gt_file, det_file, track_file, K, detector, tracker):
    values = metric_obj.evaluate(gt_file, det_file, track_file, detector, tracker)

    return values


def pretty_print(content, type, header=None, f=None):
    if header:

        for el in header:

            print('%-16.15s' % el, end='')
            if not f is None:  f.write('%s,' % el)

    for el in content[:-1]:

        if type == 'str':
            print('%-16.15s' % el, end='')
            if not f is None:  f.write('%s,' % el)

        elif type == 'flt':
            print('%-16.2f' % el, end='')
            if not f is None:  f.write('%.2f,' % el)

    # Print file
    if type == 'str':
        print('%-16.15s' % content[-1], end='')
        if not f is None:  f.write('%s' % content[-1])

    elif type == 'flt':
        print('%-16.2f' % content[-1], end='')
        if not f is None:  f.write('%.2f' % content[-1])

    print()
    if not f is None:  f.write('\n')


def set_file_name(path, names):
    files_open = {}

    for n in names:
        files_open[n] = open(os.path.join(path, n), "w")

    return files_open


def save_list_file(f, data):
    f.write('%.4f' % data[0])

    for e in data[1:]:
        f.write(', %.4f' % e)

    f.write('\n')





if __name__ == '__main__':
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from mot_evaluator import mot17

    detector = "bytetrack"
    tracker = "BYTETrack"
    conf = 0.3
    files_name_list = ['intra.csv', 'Qd.csv', 'Qt.csv', 'Id.csv', 'Nd.csv', 'It.csv', 'Nt.csv', 'inter.csv', 'Y.csv',
                       'C.csv', 'IDSW.csv']
    files_n = len(files_name_list)
    tracker_dir = "../results/detector_bytetrack/BYTETrack"
    det_dir = "../dets/detector_bytetrack/"
    gt_dir = "/media/ubuntu/2715608D71CBF6FC/datasets/mot/"

    train_dir, test_dir, seqs_train, seqs_test = mot17(gt_dir)
    metric_obj = Test_eff()
    pretty_print(metric_obj.names(), 'str', header=['Detector', 'Tracker', 'Sequence'])
    print('-------------------------------------------------------------------------------------------------------------------')
    all_results = np.zeros((len(seqs_train), len(metric_obj.names())))
    tem = np.zeros((len(seqs_train)))
    for idx, s_name in enumerate(seqs_train):
        metric_obj = Test_eff()
        #
        gt_path = os.path.join(train_dir, s_name, 'gt/gt.txt')
        det_path = os.path.join('/media/ubuntu/2715608D71CBF6FC/datasets/mot/MOT17/train', s_name, 'det/det.txt')
        track_path = os.path.join(tracker_dir, s_name + '.txt')
        # load files
        gt_file, K = load_file(gt_path, 'gt')
        track_file, _ = load_file(track_path, 'trc')
        npz_lines = np.load(det_dir + "/" + s_name.replace('7', '6')[0:8] + ".npz")
        n_frames = int(len(npz_lines.files) / 2)
        det_file = {}
        for frame_id in range(n_frames):
            try:
                bboxs, reidfeat = npz_lines[str(frame_id) + '_det'], npz_lines[str(frame_id) + '_feat']
            except:
                bboxs, reidfeat = np.empty((0, 4)), np.empty((0, 128))  # no detection
            bboxs = bboxs[bboxs[:, 4] > conf]
            bboxs[:, 2:4] -= bboxs[:, 0:2]  # xyxy to xywh
            bboxs[:, 1:5] = bboxs[:, 0:4]
            bboxs[:, 0] = -1
            det_file[frame_id + 1] = bboxs
        # compute TEM (Tracking Effort Measure)
        out = metric_obj.evaluate(gt_file, det_file, track_file, detector, tracker)
        tem[idx] = out[0]
        #
        values = out[:-files_n]
        all_results[idx] = np.array(out[:-files_n])
        pretty_print(values, 'flt', header=[detector, tracker, s_name])
    print("TEM (Tracking Effort Measure)", np.average(all_results[:, 0]), np.average(tem), np.std(tem))
    # END
