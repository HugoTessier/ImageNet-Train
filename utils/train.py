import os
import sys

import torch


def _save_results(name, message_head, dataset, e, epochs, global_loss, metrics, output_path, results):
    if not os.path.isdir(output_path):
        try:
            os.mkdir(output_path)
        except OSError:
            print(f'Failed to create the folder {output_path}')
        else:
            print(f'Created folder {output_path}')
    with open(os.path.join(output_path, name + '_results.txt'), 'a') as f:
        message = f'{message_head}: epoch {e}/{epochs} -> '
        message += f'CrossEntropy loss = {float(global_loss) / (len(dataset["test"]) * dataset["test"].batch_size)}, '
        for k, m in enumerate(metrics):
            message += f'{m.name} = {float(results[k]) / (len(dataset["test"]) * dataset["test"].batch_size)} '
        message += '\n'
        f.write(message)


def _test(checkpoint, criterion, dataset, device, metrics):
    checkpoint.model.eval()
    with torch.no_grad():
        results = [0 for _ in range(len(metrics))]
        global_loss = 0
        for i, (data, target) in enumerate(dataset['test']):
            data, target = data.to(device), target.to(device)

            output = checkpoint.model(data)
            loss = criterion(output, target.long())

            for k, m in enumerate(metrics):
                results[k] += m(output, target)
            global_loss += loss.item()

            message = f'\rTest ({i + 1}/{len(dataset["test"])}) -> '
            message += f'CrossEntropy loss = {round(float(global_loss) / ((i + 1) * dataset["test"].batch_size), 3)}, '
            for k, m in enumerate(metrics):
                message += f'{m.name} = {round(float(results[k]) / ((i + 1) * dataset["test"].batch_size), 3)}\t'
            message += '           '
            sys.stdout.write(message)
    return global_loss, results


def _train(checkpoint, criterion, dataset, device, metrics):
    results = [0 for _ in range(len(metrics))]
    global_loss = 0
    checkpoint.model.train()
    for i, (data, target) in enumerate(dataset['train']):
        data, target = data.to(device), target.to(device)
        checkpoint.optimizer.zero_grad()

        output = checkpoint.model(data)
        loss = criterion(output, target.long())
        loss.backward()
        checkpoint.optimizer.step()

        for k, m in enumerate(metrics):
            results[k] += m(output, target)
        global_loss += loss.item()

        message = f'\rTrain ({i + 1}/{len(dataset["train"])}) -> '
        message += f'CrossEntropy loss = {round(float(global_loss) / ((i + 1) * dataset["train"].batch_size), 3)}, '
        for k, m in enumerate(metrics):
            message += f'{m.name} = {round(float(results[k]) / ((i + 1) * dataset["train"].batch_size), 3)}\t'
        message += '           '
        sys.stdout.write(message)


def train_model(name, checkpoint, dataset, epochs, criterion, metrics, output_path, device):
    e = 0
    while e < epochs:
        e = checkpoint.store_model(e)
        if e >= epochs:
            break
        e += 1
        print(f'\nEpoch {e}/{epochs}')

        _train(checkpoint, criterion, dataset, device, metrics)
        print()
        global_loss, results = _test(checkpoint, criterion, dataset, device, metrics)

        _save_results(checkpoint.name, name, dataset, e, epochs, global_loss, metrics, output_path, results)

        checkpoint.scheduler.step()
    checkpoint.store_model(e)
