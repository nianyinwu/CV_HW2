""" Inference script for digit recognition """

import os
import json
import csv
import argparse
import warnings
from typing import List, Tuple
from tqdm import tqdm
from utils import tqdm_bar

import torch
from model import get_model
from dataloader import dataloader


# ignore warnings
warnings.filterwarnings('ignore')

def get_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Inference")

    parser.add_argument(
        '--device',
        type=str,
        choices=["cuda", "cpu"],
        default="cuda"
    )
    parser.add_argument(
        '--data_path',
        '-d',
        type=str,
        default='../data',
        help='Path to input data'
    )
    parser.add_argument(
        '--weights',
        '-w',
        type=str,
        default='./best.pth',
        help='Path to model weights'
    )
    parser.add_argument(
        '--save_path',
        '-s',
        type=str,
        default='./saved_model',
        help='the path of save the training model'
    )
    parser.add_argument(
        '--batch_size',
        '-b',
        type=int,
        default=1,
        help='Batch size for inference'
    )

    return parser.parse_args()

def test(
    args: argparse.Namespace,
    test_model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
) -> List[Tuple[str, str]]:
    """
    Perform inference on the test set.

    Returns:
        List of tuples (image_name, predicted_class)
    """

    test_model.eval()
    test_model.to(args.device)

    pred_box = []
    pred_label = []

    with torch.no_grad():
        for images, file_names in (pbar := tqdm(data_loader, ncols=120)):
            images = [img.to(args.device) for img in images]
            outputs = test_model(images)

            for file_name, output in zip(file_names, outputs):
                image_id = int(os.path.splitext(file_name)[0])

                # pred.json
                for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
                    if float(score) < 0.5:
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    w = x2 - x1
                    h = y2 - y1
                    pred_box.append({
                        "image_id": image_id,
                        "bbox": [x1, y1, w, h],
                        "score": float(score),
                        "category_id": int(label)
                    })

                # pred.csv
                if len(output['labels']) == 0:
                    label = -1
                else:
                    paired = list(zip(
                        output['boxes'].tolist(),
                        output['labels'].tolist(),
                        output['scores'].tolist()
                    ))
                    paired.sort(key=lambda x: x[0][0])
                    result = []
                    for _, label, score in paired:
                        if score < 0.5:
                            continue
                        result.append(str(label-1))

                    label = ''.join(result)
                    if len(label) == 0:
                        pred_label = -1
                pred_label.append((image_id, label))

                tqdm_bar('Test', pbar)

    return pred_box, pred_label


def make_json(save_path: str, predictions: list[tuple[str, str]]) -> None:
    """
    Generate prediction JSON file.
    """

    predictions = sorted(predictions, key=lambda x: x["image_id"])
    with open(f"{save_path}/pred.json", "w", newline='', encoding='utf-8') as file:
        json.dump(predictions, file)

    print('Save pred.json !!!')

def make_csv(save_path: str, predictions: list[tuple[str, str]]) -> None:
    """
    Generate prediction CSV file.
    """
    predictions.sort(key=lambda x: x[0])
    with open(f"{save_path}/pred.csv", "w", newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['image_id', 'pred_label'])  # header
        writer.writerows(predictions)

    print('Save pred.csv !!!')

if __name__ == "__main__":
    opt = get_args()
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")

    test_loader = dataloader(opt, 'test')

    # Load model
    model = get_model(num_classes=11)
    model.load_state_dict(torch.load(opt.weights))

    # Run inference
    pred_json, pred_csv = test(opt, model, test_loader)

    make_json(opt.save_path, pred_json)
    make_csv(opt.save_path, pred_csv)

    print("Saved pred.json and pred.csv !")
