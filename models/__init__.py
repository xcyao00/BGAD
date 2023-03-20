from .fc_flow import flow_model, conditional_flow_model
from .modules import positionalencoding2d

__all__ = ['load_flow_model', 'positionalencoding2d']


def load_flow_model(args, in_channels):
    if args.flow_arch == 'flow_model':
        model = flow_model(args, in_channels)
    elif args.flow_arch == 'conditional_flow_model':
        model = conditional_flow_model(args, in_channels)
    else:
        raise NotImplementedError('{} is not supported Normalizing Flow!'.format(args.flow_arch))
    
    return model