import torch
from torch._C import Def
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm

from torch.utils.data.dataloader import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils import(
    intersection_over_union,
    non_max_suppression,
    mean_average_precision,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint
)

from loss import YoloLoss

seed = 123
torch.manual_seed(seed)
# Commit change 

# Device

device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparams

batch_size = 16
learning_rate = 2e-5
weight_decay = 0 # No regularization
epochs = 100
num_workers = 2
pin_memory = True
load_model = False
# load_model = True
load_model_file = "overfit.pth.tar"

IMG_DIR = "data/images"
LABEL_DIR = "data/labels"

class Compose(object):

    def __init__(self,transforms):
        self.transforms = transforms
    
    def __call__(self,img,bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes   

        return img, bboxes


transform = Compose([transforms.Resize((448,448)), transforms.ToTensor()])

def train_fn(train_loader, model, optimizer, loss_fn):

    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x,y) in enumerate(loop):
        x,y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out,y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the progress bar
        loop.set_postfix(loss = loss.item())
    
    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

def main():
    model = Yolov1(split_size = 7, num_boxes = 2, num_classes = 20).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr = learning_rate, weight_decay= weight_decay
    )

    loss_fn = YoloLoss()

    if load_model:

        load_checkpoint(torch.load(load_model_file), model, optimizer)

    train_dataset = VOCDataset(
        "data/8examples.csv", # For sanity check
        transform = transform,
        img_dir = IMG_DIR,
        label_dir = LABEL_DIR,
    )

    test_dataset = VOCDataset(
        "data/test.csv", # For sanity check
        transform = transform,
        img_dir = IMG_DIR,
        label_dir = LABEL_DIR,
    )

    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory = pin_memory,
        shuffle = True,
        drop_last = False,
    )

    test_loader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory = pin_memory,
        shuffle = True,
        drop_last = False,
    )

    for epoch in range(epochs):

        # for x, y in train_loader :
        #     x = x.to(device)
        #     for idx in range(8):
        #         bboxes = cellboxes_to_boxes(model(x))
        #         bboxes = non_max_suppression(bboxes[idx], iou_threshold = 0.5, threshold = 0.5, box_format = "midpoint")
        #         plot_image(x[idx].permute(1,2,0).to("cpu"),bboxes)

        #         import sys
        #         sys.exit()
            
        pred_boxes, target_boxes = get_bboxes(
            train_loader,model,iou_threshold = 0.5, threshold = 0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold = 0.5, box_format = "midpoint"
        )

        print(f"Train mAP: {mean_avg_prec}")
        
        if mean_avg_prec > 0.9:
            checkpoint = {
                "state_dict" : model.state_dict(),
                "optimizer"  : optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename = load_model_file)
            import time
            time.sleep(10)
        train_fn(train_loader, model, optimizer, loss_fn)

if __name__ == "__main__":
    main()