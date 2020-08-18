def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=30000, power=0.9,):
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer

    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr*(1 - iter/max_iter)**power

def resnet_lr_scheduler(optimizer, new_init_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_init_lr

def adjust_learning_rate(optimizer, init_lr, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr