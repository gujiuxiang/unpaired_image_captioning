import torch
import torch.nn as nn
from torch.autograd import Variable
from onmt.modules.Util import BottleLinear
from onmt.modules import aeq
from onmt.modules.activations import Softmax, Sparsemax, ConstrainedSoftmax, \
    ConstrainedSparsemax
import pdb
import numpy as np

class GlobalAttention(nn.Module):
    """
    Luong Attention.

    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.


        H_1 H_2 H_3 ... H_n
          q   q   q       q
            |  |   |       |
              \ |   |      /
                      .....
                  \   |  /
                      a

    Constructs a unit mapping.
    $$(H_1 + H_n, q) => (a)$$
    Where H is of `batch x n x dim` and q is of `batch x dim`.

    Loung Attention (dotprod):
    $$\tanh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)$$.:

    Bahdanau Attention (mlp):
    $$c = \sum_{j=1}^{SeqLength}\a_jh_j$$.
    The Alignment-function $$a$$ computes an alignment as:
    $$a_j = softmax(v_a^T \tanh(W_a q + U_a h_j) )$$.

    """
    def __init__(self, dim, coverage=False, attn_type="dotprod",
                 attn_transform="softmax", c_attn=0.0):
        super(GlobalAttention, self).__init__()

        self.dim = dim
        self.attn_type = attn_type
        assert (self.attn_type in ["dotprod", "mlp"]), (
                "Please select a valid attention type.")

        if self.attn_type == "dotprod":
            self.linear_in = nn.Linear(dim, dim, bias=False)
            self.linear_out = nn.Linear(dim*2, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = BottleLinear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=False)
            self.v = BottleLinear(dim, 1, bias=False)
            # Modify initialization of self.v to have high variance
            # self.v.weight.data.normal_(0, 1000)
        if attn_transform == 'softmax':
            self.sm = nn.Softmax()
        elif attn_transform == 'sparsemax':
            self.sm = Sparsemax()
        elif attn_transform == 'constrained_softmax':
            self.sm = ConstrainedSoftmax()
        elif attn_transform == 'constrained_sparsemax':
            self.sm = ConstrainedSparsemax()
        else:
            raise NotImplementedError
        self.attn_transform = attn_transform
        
        self.tanh = nn.Tanh()
        self.mask = None
        self.c_attn = c_attn

        if coverage:
            self.linear_cover = nn.Linear(1, dim, bias=False)

    def applyMaskNone(self):
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, input, context, coverage=None, upper_bounds=None):
        """
        input (FloatTensor): batch x dim
        context (FloatTensor): batch x sourceL x dim
        coverage (FloatTensor): batch x sourceL
        upper_bounds (FloatTensor): batch x sourceL
        """
        # Check input sizes
        batch, sourceL, dim = context.size()
        batch_, dim_ = input.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        aeq(self.dim, dim)
        if coverage is not None:
            batch_, sourceL_ = coverage.size()
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)

        if self.mask is not None:
            beam_, batch_, sourceL_ = self.mask.size()
            aeq(batch, batch_*beam_)
            aeq(sourceL, sourceL_)

        if coverage:
            context += self.linear_cover(coverage.view(-1).unsqueeze(1)) \
                           .view_as(context)
            context = self.tanh(context)
        # Alignment/Attention Function
        if self.attn_type == "dotprod":
            # batch x dim x 1
            targetT = self.linear_in(input).unsqueeze(2)
            # batch x sourceL
            attn = torch.bmm(context, targetT).squeeze(2)
        elif self.attn_type == "mlp":
            # batch x dim x 1
            wq = self.linear_query(input).unsqueeze(1)
            # batch x sourceL x dim
            uh = self.linear_context(context.contiguous())
            # batch x sourceL x dim
            wquh = uh + wq.expand_as(uh)
            # batch x sourceL x dim
            wquh = self.tanh(wquh)
            # batch x sourceL
            #print("self.v: ", self.v.weight)
            attn = self.v(wquh.contiguous()).squeeze()
 
        # EXPERIMENTAL
        
        if upper_bounds is not None and 'constrained' in self.attn_transform and self.c_attn!=0.0:
            indices = torch.arange(0,upper_bounds.size(1)-1).cuda().long()
            uu = torch.index_select(upper_bounds.data, 1, indices) 
            attn = attn + self.c_attn * Variable(torch.cat((uu, torch.zeros(upper_bounds.size(0)).cuda()), 1))


        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        if self.attn_transform == 'constrained_softmax':
            if upper_bounds is None:
                attn = nn.Softmax()(attn)
            else:
                # assert round(np.sum(upper_bounds.cpu().data.numpy()), 5) >= 1.0, pdb.set_trace()
                attn = self.sm(attn, upper_bounds)
        elif self.attn_transform == 'constrained_sparsemax':
            if upper_bounds is None:
                attn = Sparsemax()(attn)
            else:
                attn = self.sm(attn, upper_bounds)
        else:
            attn = self.sm(attn)
            #if upper_bounds is None:
            #    attn = self.sm(attn)
            #else:
            #    attn = self.sm(attn - upper_bounds)
        
        # Compute context weighted by attention.
        # batch x 1 x sourceL
        attn3 = attn.view(attn.size(0), 1, attn.size(1))
        # batch x dim
        weightedContext = torch.bmm(attn3, context).squeeze(1)
        # Concatenate the input to context (Luong only)
        if self.attn_type == "dotprod":
            weightedContext = torch.cat((weightedContext, input), 1)
            weightedContext = self.linear_out(weightedContext)
            weightedContext = self.tanh(weightedContext)

        # Check output sizes
        batch_, sourceL_ = attn.size()
        aeq(batch, batch_)
        aeq(sourceL, sourceL_)
        batch_, dim_ = weightedContext.size()
        aeq(batch, batch_)
        aeq(dim, dim_)

        return weightedContext, attn
