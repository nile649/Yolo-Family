import config 
import torch 
import torch.optim as optim
from model import YOLOv3
from tqdm import tqdm 
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)
from loss import YoloLoss
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True
def train(model,trainloader,optimizer,criterion, scaled_anchors):
    loop = tqdm(trainloader,leave=True)
    losses = []

    for idx,(img,tar) in enumerate(loop):
        # breakpoint()
        # print(tar)
        img = img.to(config.DEVICE)
        out = model(img)
        # breakpoint()
        loss = [criterion(out[id],tar[id],scaled_anchors[id]) for id in range(len(scaled_anchors))]
        loss = sum(loss)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_loss = sum(losses)/len(losses)
        loop.set_postfix(loss=mean_loss)



def main():
    model = YOLOv3(n_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    criterion = YoloLoss(config)
    train_loader, test_loader, eval_loader = get_loaders(
        train_csv_path=config.path, test_csv_path=config.path, eval_csv_path=config.path
    )

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)
    for epoch in range(config.NUM_EPOCHS):
        #plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors) 
        train(model,train_loader, optimizer, criterion=criterion, scaled_anchors=scaled_anchors)

        #if config.SAVE_MODEL:
        #    save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")

        #print(f"Currently epoch {epoch}")
        #print("On Train Eval loader:")
        #print("On Train loader:")
        #check_class_accuracy(model, train_loader, threshold=config.CONF_THRESHOLD)

        if epoch > 0 and epoch % 3 == 0:
            check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print(f"MAP: {mapval.item()}")
            model.train()


if __name__ == "__main__":
    main()

if __name__=="__main__":
    main()