import os
import numpy as np
import torch
import torchvision
import argparse

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# Modified SimCLR with Keypoint Detection and Description
from modified_simclr import ModifiedSimCLR  # Ensure you have this modified module
from simclr.modules import get_resnet
from simclr.modules.transformations import TransformsSimCLR
from simclr.modules.sync_batchnorm import convert_model

from model import load_optimizer, save_model  # These may need adjustments
from utils import yaml_config_hook  # Adjust if necessary

def train(args, train_loader, model, criterion, optimizer, writer):
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.to(args.device, non_blocking=True)
        x_j = x_j.to(args.device, non_blocking=True)

        # Forward pass through the modified model
        prob_i, prob_j, descriptors_i, descriptors_j = model(x_i, x_j)

        # Assume 'criterion' can handle the new outputs; this may require custom loss functions
        loss = criterion(prob_i, prob_j, descriptors_i, descriptors_j)
        loss.backward()
        optimizer.step()

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))

        if args.nr == 0 and step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        if args.nr == 0:
            writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
            args.global_step += 1

        loss_epoch += loss.item()
    return loss_epoch

# The main function remains largely the same, but ensure you instantiate the modified model correctly
def main(gpu, args):
    rank = args.nr * args.gpus + gpu

    if args.nodes > 1:
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(gpu)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load your custom dataset
    train_dataset = CustomDataset(
        args.dataset_dir,
        transform=YourCustomTransforms(size=args.image_size),
    )

    if args.nodes > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True
        )
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
    )

    # Initialize ResNet
    encoder = get_resnet(args.resnet, pretrained=False)
    n_features = encoder.fc.in_features  # Get dimensions of fc layer

    # Initialize the modified model instead of SimCLR
    model = ModifiedSimCLR(encoder, descriptor_size=256)  # Adjust parameters as needed
    if args.reload:
        model_fp = os.path.join(
            args.model_path, "checkpoint_{}.tar".format(args.epoch_num)
        )
        model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    model = model.to(args.device)

    # Optimizer / Loss
    optimizer, scheduler = load_optimizer(args, model)
    # Update or implement a custom loss function as needed
    criterion = YourCustomLossFunction(args.batch_size, args.temperature, args.world_size)

    # DDP / DP
    if args.dataparallel:
        model = convert_model(model)
        model = DataParallel(model)
    else:
        if args.nodes > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[gpu])

    model = model.to(args.device)

    writer = None
    if args.nr == 0:
        writer = SummaryWriter()

    args.global_step = 0
    args.current_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(args, train_loader, model, criterion, optimizer, writer)

        if args.nr == 0 and scheduler:
            scheduler.step()

        if args.nr == 0 and epoch % 10 == 0:
            save_model(args, model, optimizer)

        if args.nr == 0:
            writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
            writer.add_scalar("Misc/learning_rate", lr, epoch)
            print(
                f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}"
            )
            args.current_epoch += 1

    ## End training
    save_model(args, model, optimizer)


if __name__ == "__main__":
    # argparse and config setup remains the same

    args = parser.parse_args()
    # Setup for distributed training, device selection, etc., remains the same

    # Main changes would be in how you instantiate the modified model:
    encoder = get_resnet(args.resnet, pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer
    
    # Initialize the modified model instead of SimCLR
    model = ModifiedSimCLR(encoder, descriptor_size=256)  # Adjust parameters as needed
    # Rest of the model setup, DDP/DataParallel setup, and training loop invocation remains the same
