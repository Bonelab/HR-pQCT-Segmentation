import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset.SingleImageDataset import SingleImageDataset
from contextlib import nullcontext


def traintest(
        args,
        model,
        device,
        dataset,
        dataloader_kwargs,
        image_dataset_kwargs,
        image_dataloader_kwargs,
        optimizer,
        losses,
        testing_functions,
        logger,
        train=True
):
    if train:
        model.train()
    else:
        model.eval()

    logger.set_field_value('train/test', 0 if train else 1)

    loader = DataLoader(dataset, **dataloader_kwargs)

    image_dataset = SingleImageDataset(**image_dataset_kwargs)

    for idx, image in enumerate(loader):

        logger.set_field_value('idx', idx)
        logger.set_field_value('name', image['name'])

        image_dataset.setup_new_image(image)

        traintest_image(
            args, model, device, image_dataset, image_dataloader_kwargs,
            optimizer, losses, testing_functions, logger, train=train
        )

        logger.log()

        if args.dry_run:
            break


def traintest_image(
        args,
        model,
        device,
        image_dataset,
        image_dataloader_kwargs,
        optimizer,
        losses,
        testing_functions,
        logger,
        train=True
):
    context = nullcontext() if train else torch.no_grad()

    for loss in losses.values():
        loss['values'] = []

    test_vals = {}
    for k in testing_functions.keys():
        test_vals[k] = []

    image_dataloader = DataLoader(image_dataset, **image_dataloader_kwargs)

    for idx, batch in enumerate(image_dataloader):

        data = batch['image'].to(device)
        labels = batch['labels'].to(device)

        with context:
            embeddings = model(data)

        total_loss = 0

        for loss in losses.values():
            loss_temp = loss['coefficient'] * loss['function'](embeddings, labels)
            loss['values'].append(loss_temp.item())
            total_loss += loss_temp

        for k, f in testing_functions.items():
            test_vals[k].append(f(embeddings, labels))

        if train:
            optimizer.zero_grad()
            total_loss.backward()
            if not args.no_clamp_gradients:
                for parameter in model.parameters():
                    parameter.grad.data.clamp(-1, 1)
            optimizer.step()

        if args.dry_run:
            break

    for k, loss in losses.items():
        logger.set_field_value(k, np.asarray(loss['values']).mean())

    for k in testing_functions.keys():
        logger.set_field_value(k, np.asarray(test_vals[k]).mean())
