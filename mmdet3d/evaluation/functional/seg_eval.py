# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmengine.logging import print_log
from terminaltables import AsciiTable


def fast_hist(preds, labels, num_classes):
    """Compute the confusion matrix for every batch.

    Args:
        preds (np.ndarray):  Prediction labels of points with shape of
        (num_points, ).
        labels (np.ndarray): Ground truth labels of points with shape of
        (num_points, ).
        num_classes (int): number of classes

    Returns:
        np.ndarray: Calculated confusion matrix.
    """

    # NOTE: We must filter both labels and preds to avoid negative indices
    # (e.g., when some points are ignored or mapped to -1 for evaluation).
    k = ((labels >= 0) & (labels < num_classes) & (preds >= 0)
         & (preds < num_classes))
    bin_count = np.bincount(
        num_classes * labels[k].astype(int) + preds[k],
        minlength=num_classes**2)
    return bin_count[:num_classes**2].reshape(num_classes, num_classes)


def per_class_iou(hist):
    """Compute the per class iou.

    Args:
        hist(np.ndarray):  Overall confusion martix
        (num_classes, num_classes ).

    Returns:
        np.ndarray: Calculated per class iou
    """

    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def get_acc(hist):
    """Compute the overall accuracy.

    Args:
        hist(np.ndarray):  Overall confusion martix
        (num_classes, num_classes ).

    Returns:
        float: Calculated overall acc
    """

    return np.diag(hist).sum() / hist.sum()


def get_acc_cls(hist):
    """Compute the class average accuracy.

    Args:
        hist(np.ndarray):  Overall confusion martix
        (num_classes, num_classes ).

    Returns:
        float: Calculated class average acc
    """

    return np.nanmean(np.diag(hist) / hist.sum(axis=1))


def seg_eval(gt_labels,
             seg_preds,
             label2cat,
             ignore_index,
             logger=None,
             label_mapping=None):
    """Semantic Segmentation  Evaluation.

    Evaluate the result of the Semantic Segmentation.

    Args:
        gt_labels (list[torch.Tensor]): Ground truth labels.
        seg_preds  (list[torch.Tensor]): Predictions.
        label2cat (dict): Map from label to category name.
        ignore_index (int): Index that will be ignored in evaluation.
        logger (logging.Logger | str, optional): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.
        label_mapping (dict[int, int] | np.ndarray | None, optional):
            Optional mapping from the original label space to a new label space
            for evaluation-only reporting (e.g., merging fine-grained classes
            into super-classes). Any label not included in the mapping will be
            treated as ignored (-1).
            Defaults to None.

    Returns:
        dict[str, float]: Dict of results.
    """
    assert len(seg_preds) == len(gt_labels)
    num_classes = len(label2cat)

    hist_list = []
    map_arr = None
    if label_mapping is not None:
        if isinstance(label_mapping, dict):
            if len(label_mapping) == 0:
                raise ValueError('label_mapping must not be empty when given.')
            max_k = int(max(label_mapping.keys()))
            map_arr = np.ones(max_k + 1, dtype=np.int64) * (-1)
            for k, v in label_mapping.items():
                map_arr[int(k)] = int(v)
        else:
            map_arr = np.asarray(label_mapping, dtype=np.int64)

    for i in range(len(gt_labels)):
        gt_seg = gt_labels[i].astype(np.int64)
        pred_seg = seg_preds[i].astype(np.int64)

        # filter out ignored points
        pred_seg[gt_seg == ignore_index] = -1
        gt_seg[gt_seg == ignore_index] = -1

        if map_arr is not None:
            # Map labels for evaluation-only reporting.
            # Unmapped labels become -1 and are ignored by fast_hist.
            gt_mapped = np.ones_like(gt_seg, dtype=np.int64) * (-1)
            pred_mapped = np.ones_like(pred_seg, dtype=np.int64) * (-1)

            gt_valid = (gt_seg >= 0) & (gt_seg < len(map_arr))
            pred_valid = (pred_seg >= 0) & (pred_seg < len(map_arr))
            gt_mapped[gt_valid] = map_arr[gt_seg[gt_valid]]
            pred_mapped[pred_valid] = map_arr[pred_seg[pred_valid]]

            gt_seg = gt_mapped
            pred_seg = pred_mapped

        # calculate one instance result
        hist_list.append(fast_hist(pred_seg, gt_seg, num_classes))

    iou = per_class_iou(sum(hist_list))
    # if ignore_index is in iou, replace it with nan
    if ignore_index < len(iou):
        iou[ignore_index] = np.nan
    miou = np.nanmean(iou)
    acc = get_acc(sum(hist_list))
    acc_cls = get_acc_cls(sum(hist_list))

    header = ['classes']
    for i in range(len(label2cat)):
        header.append(label2cat[i])
    header.extend(['miou', 'acc', 'acc_cls'])

    ret_dict = dict()
    table_columns = [['results']]
    for i in range(len(label2cat)):
        ret_dict[label2cat[i]] = float(iou[i])
        table_columns.append([f'{iou[i]:.4f}'])
    ret_dict['miou'] = float(miou)
    ret_dict['acc'] = float(acc)
    ret_dict['acc_cls'] = float(acc_cls)

    table_columns.append([f'{miou:.4f}'])
    table_columns.append([f'{acc:.4f}'])
    table_columns.append([f'{acc_cls:.4f}'])

    table_data = [header]
    table_rows = list(zip(*table_columns))
    table_data += table_rows
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    print_log('\n' + table.table, logger=logger)

    return ret_dict
