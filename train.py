from dataset_ import YOLODataset
from config import config
from torchinfo import summary
from models import YOLOv3
from loss import YoloLoss
import torch 
import torch.optim as optim
from tqdm import tqdm 
from utils import utils, plot, metric_utils
from torch.utils.data import DataLoader
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

def train(model,optimizer,criterion,dataIter,scaled_anchors):
    losses = []
    for idx,(img,tar) in enumerate(dataIter):
        img = img.to(config.DEVICE)
        out = model(img)
#         pdb.set_trace()
        loss_ = [criterion(out[id],tar[id],scaled_anchors[id]) for id in range(len(scaled_anchors))]
        loss = sum([x["total_loss"] for x in loss_])
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mean_loss = sum(losses)/len(losses)
        dataIter.set_postfix(loss=mean_loss)


def main():
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)
    
    model = YOLOv3(config=config.model_net,n_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    criterion = YoloLoss(config)
    train_dataset = YOLODataset("/mnt/e/cat_dog/lol01.txt",config.ANCHORS,transform=config.train_transforms)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )
    test_dataset = YOLODataset("/mnt/e/cat_dog/lol01.txt",config.ANCHORS,transform=config.train_transforms)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )
#     utils.plot_couple_examples(model, train_loader, 0.6, 0.5, scaled_anchors) 

    for epoch in range(config.NUM_EPOCHS):
        dataIter = tqdm(train_loader,leave=True)
        print(f'Epoch {epoch}')
        train(model,optimizer, criterion, dataIter, scaled_anchors=scaled_anchors)

        if config.SAVE_MODEL:
            utils.save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")

        if epoch >= 0 and epoch % 5 == 0:
            print("On Test Eval loader:")
            print("On Test loader:")
            metric_utils.check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
            pred_boxes, true_boxes = utils.get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
            mapval = metric_utils.mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                num_classes=config.NUM_CLASSES,
            )
            print(f"MAP: {mapval.item()}")
            model.train()
    return model


if __name__ == "__main__":
    model_ = main()