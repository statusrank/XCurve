def adjust_learning_rate(optimizer, epoch, lr, epoch_to_adjust=30):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_new = lr * (0.1**(epoch // epoch_to_adjust))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new