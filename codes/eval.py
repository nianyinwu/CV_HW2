"""Evaluate a digit recognition model """

import torch
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.cuda.amp import autocast

def evaluation(
    device: torch.device,
    model: torch.nn.Module,
    valid_loader: torch.utils.data.DataLoader,
    gt_json: str,
    threshold: float
) -> tuple[torch.Tensor, float]:
    """
    Evaluate the model on the validation dataset.

    Args :
        device : Evaluation device.
        model : Trained model to evaluate.
        valid_loader : DataLoader for the validation set.
        gt_json : The path of ground truth annotation file.
        threshold : The score threshold to filter prediction. 

    Returns:
        mean_ap : Mean Average Precision.
        acc : Accuracy.
    """

    # Evaluation mAP, Accuracy
    model.eval()
    results = []
    correct = 0
    with torch.no_grad():
        for images, targets in (tqdm(valid_loader, ncols=120)):
            images = [img.to(device) for img in images]

            with autocast():
                outputs = model(images)

            for output, target in zip(outputs, targets):
                image_id = target["image_id"].item() if "image_id" in target else 0

                # Accuracy
                gt_boxes = target["boxes"].cpu().tolist()
                gt_labels = target["labels"].cpu().tolist()

                gt_digits = sorted(zip(gt_boxes, gt_labels), key=lambda x: x[0][0])
                gt_digits = [str(label - 1) for _, label in gt_digits]
                gt_label = ''.join(gt_digits)

                pred_digits = [
                    (box.tolist(), label.item()) for box, label, score in zip(
                        output['boxes'], output['labels'], output['scores']
                    ) if score.item() >= threshold
                ]

                # Sort the result by x-axis
                pred_digits = sorted(pred_digits, key=lambda x: x[0][0])
                pred_digits = [str(label - 1) for _, label in pred_digits]

                pred_label = ''.join(pred_digits)
                if pred_label == gt_label:
                    correct += 1

                # mAP
                for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
                    x1, y1, x2, y2 = box.tolist()
                    results.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [x1, y1, x2-x1, y2-y1],
                        "score": float(score)
                    })


    coco_gt = COCO(gt_json)
    coco_pred = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_pred, iouType='bbox')

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mean_ap = coco_eval.stats[0]
    acc = correct / len(valid_loader.dataset)

    return mean_ap, acc
