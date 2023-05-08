import torch
import nflows
from nflows import flows, transforms, distributions
import numpy as np
from torch.nn import functional as F
from utils import *


def sampler(model, num_samples, batch_size=10000):
    model.eval()
    with torch.no_grad(): 
        num_batches = num_samples // batch_size + (1 if num_samples % batch_size != 0 else 0)
        samples=[]
        for i in range(num_batches):
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            batch_samples = model.sample(num_samples=current_batch_size)
            samples.append(batch_samples)
        samples = torch.cat(samples, dim=0)
    return samples.cpu().detach()


def Normalizing_Flow(args,
                     RQS_tail_bound=12, 
                     num_bins=20, 
                     use_residual_blocks=True, 
                     random_mask=False, 
                     save_architecture=True):
   
    dim = args.flow_dim

    flow_model = args.flow_type+'_'+args.flow_func
    list_transforms = []
    
    for _ in range(args.num_flows):   

        if flow_model=='MAF_affine':
            
            flow=nflows.transforms.autoregressive.MaskedAffineAutoregressiveTransform(
                features=dim,
                hidden_features=args.hidden_dims,
                context_features=args.context_dim,
                num_blocks=args.num_blocks,
                use_residual_blocks=use_residual_blocks,
                random_mask=random_mask,
                activation=args.activation,
                dropout_probability=args.dropout,
                use_batch_norm=args.batch_norm
                    )

        elif flow_model=='MAF_RQSpline':
            
            use_residual_blocks=None
            
            flow=nflows.transforms.autoregressive.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=dim,
                hidden_features=args.hidden_dims,
                context_features=args.context_dim,
                num_blocks=args.num_blocks,
                use_residual_blocks=use_residual_blocks,
                random_mask=random_mask,
                activation=args.activation,
                dropout_probability=args.dropout,
                use_batch_norm=args.batch_norm,
                num_bins=num_bins,
                tails='linear',
                tail_bound=RQS_tail_bound
                )

        elif flow_model=='coupling_RQSpline':
            
            mask = torch.ones(dim)
            if args.coupl_mask=='checkers': mask[::2]=-1
            elif args.coupl_mask=='mid-split': mask[int(dim/2):]=-1  # 2006.08545
            
            def resnet(in_features, out_features):
                return nflows.nn.nets.ResidualNet(
                    in_features,
                    out_features,
                    context_features=args.context_dim,
                    hidden_features=args.hidden_dims,
                    num_blocks=args.num_blocks,
                    activation=args.activation,
                    dropout_probability=args.dropout,
                    use_batch_norm=args.batch_norm,
                )
            
            flow=nflows.transforms.coupling.PiecewiseRationalQuadraticCouplingTransform(
                                            mask=mask,
                                            transform_net_create_fn=resnet,
                                            num_bins=num_bins,
                                            tails='linear',
                                            tail_bound=RQS_tail_bound
                                            )

        perm=nflows.transforms.permutations.Permutation(permutation_layer(args))
        list_transforms.append(flow)
        list_transforms.append(perm)

    transform = nflows.transforms.base.CompositeTransform(list_transforms).to(args.device) 
    base_dist = nflows.distributions.normal.StandardNormal(shape=[dim])
    model = nflows.flows.base.Flow(transform, base_dist).to(args.device)

    if save_architecture:
        with open(args.workdir+'/model_architecture.txt', 'w') as file: file.write('model = {}\n'.format(model))
   
    return model


def permutation_layer(args):
    dim=args.dim
    if args.permutation:
        if 'cycle' in args.permutation:
            n=int(args.permutation.split('-')[0]) 
            if n<dim:
                p=list(range(dim)[-n:])+list(range(dim)[:-n]) # 3-cyclic : [0,1,2,3,4,5,6,7,8,9] -> [7,8,9,0,1,2,3,4,5,6]    
            else:
                raise ValueError('n-cycle must be a positive integer smaller than dim')
        elif args.permutation=='inverse':
            p=list(range(dim))
            p.reverse()
        else:
            raise ValueError('wrong permutation arg. Use [n]-cycle or inverse')
    else:
        p=list(range(dim))
    return torch.tensor(p)

