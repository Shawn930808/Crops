import torch
import math
from .utils import nonmaximalsuppression as nms

class AccuracyMeter():
    def __init__(self, channel_count):
        self.channel_count = channel_count
        self.tp = [0] * channel_count
        self.fp = [0] * channel_count
        self.fn = [0] * channel_count
        self.tn = [0] * channel_count

    def update(self, channel_results):
        for i in range(self.channel_count):
            self.tp[i] += channel_results[i][0]
            self.fp[i] += channel_results[i][1]
            self.fn[i] += channel_results[i][2]

    def f1(self):
        results = []

        for i in range(self.channel_count):
            tp, fp, fn = self.tp[i], self.fp[i], self.fn[i]
            precision, recall = 0.0, 0.0
        
            if tp + fp > 0:
                precision = tp / float(tp + fp)

            if tp + fn > 0:
                recall = tp / float(tp + fn)

            f1 = 0.0
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            results.append(f1)
        return results

def _distance_squared(a, b):
    return pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2)

def get_distances(pt, gt):
    d = torch.zeros(len(pt))

    for i in range(len(pt)):
        mn = 10000
        for j in range(len(gt)):
            dist = _distance_squared(pt[i], gt[j])
            if dist < mn:
                mn = dist
        d[i] = math.sqrt(mn)

    return d

def evaluate_heatmap(heatmap, gtpoints, nmsthreshold, distancethreshold):
    prpoints = nms(heatmap, nmsthreshold)

    if len(prpoints) == 0 or len(gtpoints) == 0:
        # Empty tensor, either early in the training process, or an empty image
        if len(prpoints) == len(gtpoints):
            # No predicted or target points
            return 0, 0, 0
        elif len(prpoints) == 0:
            # No predicted points, all points are false negatives
            return 0, 0, len(gtpoints)
        else:
            # No grond truth points, all false positives
            return 0, len(prpoints), 0

    prdist = get_distances(prpoints, gtpoints).le_(distancethreshold)
    gtdist = get_distances(gtpoints, prpoints).le_(distancethreshold)

    tp = int(prdist.sum())
    fp = int((1 - prdist).sum())
    fn = int((1 - gtdist).sum())

    return tp, fp, fn

def evaluate_heatmap_batch(heatmap, gtpoints, nmsthreshold, distancethresholds):
    batch_count = heatmap.size(0)
    channel_count = heatmap.size(1)

    if distancethresholds.size(1) != channel_count:
        raise Exception("Incorrect number of distance thresholds")

    channel_results = []

    for channel in range(0,channel_count):
        ctp, cfp, cfn = 0, 0, 0
        for batch in range(0,batch_count):
            if len(gtpoints[batch]) != channel_count:
                raise Exception("Number of ground truth channels does not match heatmap channels")

            tp, fp, fn = evaluate_heatmap(heatmap[batch][channel], gtpoints[batch][channel], nmsthreshold, distancethresholds[batch][channel])

            ctp += tp
            cfp += fp
            cfn += fn

        channel_results.append([ctp, cfp, cfn])

    return channel_results
