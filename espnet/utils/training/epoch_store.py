from chainer.training.extension import Extension


class EpochStore(Extension):
    """A tensorboard logger extension"""

    def __init__(self, epoch=0):
        self._epoch = epoch

    def __call__(self, trainer):
        # self._epoch = trainer.updater.get_iterator('main').epoch
        self._epoch += 1
    
    def get_epoch(self):
        return self._epoch