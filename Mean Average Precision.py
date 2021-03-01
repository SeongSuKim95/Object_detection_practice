import torch
from collections import Counter
from iou import intersection_over_union

# Goal : To understand and implement the most common metric used in Deep Learning to evalutate object detection models.
# Concretely, if we read "mAP@0.5:0.05:0.95" in a research paper we should know exactly what that means, and know how to do that evalutation on our own model.

# Precision and Recall

# TP: Correct bounding box ( high IoU) FP: Incorrect bounding box ( low Iou)
# FN: No bounding box which it should be.
# Precision = TP/(TP+FP) : Of all bounding box predictions, what fraction was actually correct?
# Recall = TP/(TP+FN) : Of all "target" bounding boxes, what fraction did we correctly detect?

# Step 

# 1. Get "all" bounding box predictions on our test set
# 2. Sort by descending confidence score
# 3. Calculate the precision and recall as we go through all outputs
# 4. Plot the Precision-Recall graph
# 5. Calculate Area under PR curve
# 6. We need to calculate for all classes. If there is two classes,
#    class1 AP = 0.74
#    class2 AP = 0.533
#    mAP = (0.533 + 0.74)/2 = 0.636
# 7. All this was calculated given a specific IoU threshold of certain value, we need to redo all computations for many IoUs,
#    example:0.5,0.55,0.6,...,0.95. Then average this and this will be out final result.
#    This is what is meant by mAP@0.5:0.05:0.95

def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold = 0.5, box_format = "corners", num_classes = 20
):

    #pred_boxes (list) : [[train_idx, class_pred, prob_scoer, x1, y1, x2, y2],...]
    
    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes): # for just Single IoU threshold

        detections = [] 
        ground_truths = []

        for detection in pred_boxes: # If class_pred of pred_boxes is c --> Append to detections
            if detection[1] == c:  # detection[1] = class
                detections.append(detection)
        
        for true_box in true_boxes: # Count class c in ground truth label
            if true_box[1] == c:
                ground_truths.append(true_box)
        
        amount_bboxes = Counter([gt[0] for gt in ground_truths]) # gt[0] --> Training index, Count index of ground truth boxes

        # Ex
        # img 0 has 3 bboxes (for ground truth)
        # img 1 has 5 bboxes
        # amount_bboxes(dictionary) = {0:3, 1:5}

        for key, val in amount_bboxes.items():

            amount_bboxes[key] = torch.zeros(val) # {0:3, 1:5} --> {0:torch.tensor([0,0,0]), 1: torch.tensor([0,0,0,0,0]))}
            # amount_bboxes = {0:torch.tensor([0,0,0]), 1: torch.tesnor([0,0,0,0,0])}
        
        detections.sort(key = lambda x: x[2], reverse = True) # Sort by confidence score
        
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))

        total_true_bboxes = len(ground_truths) # Num of Ground truth bboxes

        for detection_idx, detection in enumerate(detections):
            ground_truths_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]] # Choose bboxes in ground truth whose class is equal to current class 
            num_gts = len(ground_truths_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truths_img): # Computate iou with gt and detections
                iou = intersection_over_union(
                    torch.tensor(detection[3:]), #[x1,y1,x2,y2]
                    torch.tensor(gt[3:]),
                    box_format = "corner"
                    )
                 
                if iou > best_iou :
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0 :

                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                
                else :
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1
                
        # [1,1,0,1,0] -> cumsum -> [1,2,2,3,3]
        TP_cumsum = torch.cumsum(TP,dim = 0)
        FP_cumsum = torch.cumsum(FP,dim = 0)

        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))

        precisions = torch.cat((torch.tensor([1]),precisions))
        recalls = torch.cat((torch.tensor([0]),recalls))
        
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


