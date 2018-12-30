import numpy as np

def validation(model, criterion, valid_loader):
    model.eval()
    losses = []
    iou = []
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs = inputs.permute(0,3,1,2).type(torch.FloatTensor).to(device)
            truth_image = get_true_target(targets)
            targets = targets.permute(0,3,1,2).type(torch.FloatTensor).to(device)
            outputs, outputs_pixel, outputs_image = model(inputs)
            loss = criterion(outputs, outputs_pixel, outputs_image, targets, truth_image)
            losses += [loss.item()]
            iou += [my_iou_metric_pad(get_numpy(targets), get_numpy(outputs))]

        valid_loss = np.mean(losses)  # type: float

        valid_iou = np.mean(iou)

        metrics = {'val_loss': valid_loss, 'val_iou': valid_iou}
    return metrics

def train(model, criterion, train_loader, optimizer, epoch, 
          scheduler, report_each=10, valid_iou=0):
        model.train()
        random.seed()
        scheduler.step(valid_iou)
        lr = optimizer.param_groups[0]['lr']
        print('Current learning rate: {:.4f}'.format(lr))
        losses = []
        ious = []
        tl = train_loader
        
        try:
            mean_loss = 0
            mean_iou = 0
            for i, (inputs, targets) in enumerate(tl):
                inputs = inputs.permute(0,3,1,2).type(torch.FloatTensor).to(device)
                truth_image = get_true_target(targets)
                targets = targets.permute(0,3,1,2).type(torch.FloatTensor).to(device)
                outputs, outputs_pixel, outputs_image = model(inputs)
                loss = criterion(outputs, outputs_pixel, outputs_image, targets, truth_image)
                optimizer.zero_grad()
                batch_size = inputs.size(0)
                loss.backward()
                optimizer.step()
                losses += [loss.item()]
                ious += [my_iou_metric_pad(get_numpy(targets), get_numpy(outputs))]
                mean_loss = np.mean(losses[-report_each:])
                mean_iou = np.mean(ious[-report_each:])

                if i % report_each == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Loss {loss:.4f} ({loss_avg:.4f})\t'
                          'IOU {iou:.3f} ({iou_avg:.3f})'.format(
                           epoch, i, len(tl), loss=losses[-1], loss_avg=mean_loss, iou=ious[-1], iou_avg=mean_iou))

            metrics = {'train_loss': mean_loss, 'train_iou': mean_iou}
            return metrics
        
        except KeyboardInterrupt:
            print('Ctrl+C, saving snapshot')
            print('done.')
            return