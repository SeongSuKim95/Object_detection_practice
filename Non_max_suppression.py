import torch
from iou import intersection_over_union
# Goal : Cleaning up Bounding Boxes by thresholding them with IoU value.

# Non-max suppression algorithm : Start with discarding all bounding boxes < probability threshold

# While Bounding Boxes:
#  -Take out the largest probability box
#  -Remove all other boxes with IoU < threshold

# (And we do this for each class)

def non_max_suppression(
    bboxes,
    iou_threshold,
    prob_threshold,
    box_format = "corners"
):
 
    # predictions = [[class, predictions, x1, y1, x2,y2],[class,predictions,...],...]
    # 6 elements per each list

    assert type(bboxes) == list
    bboxes = [box for box in bboxes if box[1] > prob_threshold]
    bboxes = sorted(bboxes, key = lambda x: x[1], reverse = True) # Sorting the bounding boxes from highest probability at the beginning 
    
    bboxes_after_nms = []
    
    while bboxes :
        chosen_box = bboxes.pop(0) # Choose highest score
        bboxes = [
            box for box in bboxes # Compute IoU between bboxes, if IoU if higher than threshold we can remove boxes
            if box[0] != chosen_box[0] or intersection_over_union(torch.tensor(chosen_box[2:]),torch.tensor(box[2:]),box_format = box_format) < iou_threshold
            # Should check if class of each bbox is equal. We should do this for each class separately
        ]
        bboxes_after_nms.append(chosen_box)
    
    return bboxes_after_nms



