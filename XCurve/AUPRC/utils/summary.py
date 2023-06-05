import os
from tensorboardX import SummaryWriter

class TensorboardSummary(object):
        
    def create_summary(self, directory,args):
        writer = SummaryWriter(log_dir=os.path.join(directory))
        self.args = args
        return writer
