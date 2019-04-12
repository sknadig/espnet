from chainer.training.extension import Extension
import torch
import numpy as np
class TensorboardLogger(Extension):
    """A tensorboard logger extension"""

    def __init__(self, logger, model, idim, device, converter, data, grad_clip, att_reporter=None, entries=None, epoch=0):
        """Init the extension

        :param SummaryWriter logger: The logger to use
        :param PlotAttentionReporter att_reporter: The (optional) PlotAttentionReporter
        :param entries: The entries to watch
        :param int epoch: The starting epoch
        """
        self._entries = entries
        self._att_reporter = att_reporter
        self._logger = logger
        self._epoch = epoch
        self.model = model
        self.grad_clip = grad_clip
        self.idim = idim
        self.device = device
        # self.dummy_input = (torch.rand(1, 200, self.idim), [200], [[1,2,3]])
        self.x = torch.rand(1,200,self.idim).to(self.device)
        self.ilens = torch.tensor([200]).to(self.device)
        self.ys_pad = torch.tensor([[2,3,4]]).to(self.device)
        # self.data = converter(data, device)
        self.data = (self.x, self.ilens, self.ys_pad)
    def __call__(self, trainer):
        """Updates the events file with the new values

        :param trainer: The trainer
        """
        observation = trainer.observation
        for k, v in observation.items():
            if (self._entries is not None) and (k not in self._entries):
                continue
            if k is not None and v is not None:
                if 'cupy' in str(type(v)):
                    v = v.get()
                if 'cupy' in str(type(k)):
                    k = k.get()
                self._logger.add_scalar(k, v, trainer.updater.iteration)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self._logger.add_scalar("Gradient clipping", grad_norm, trainer.updater.iteration)

        for name, param in self.model.named_parameters():
            self._logger.add_histogram(name, param.clone().cpu().data.numpy(), trainer.updater.iteration)
        
        

        # self._logger.add_graph(self.model, self.data)

        if self._att_reporter is not None and trainer.updater.get_iterator('main').epoch > self._epoch:
            self._epoch = trainer.updater.get_iterator('main').epoch
            self._att_reporter.log_attentions(self._logger, trainer.updater.iteration)
            self._logger.add_scalar("Epoch", trainer.updater.get_iterator('main').epoch, trainer.updater.iteration)

            