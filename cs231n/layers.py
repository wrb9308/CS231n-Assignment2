from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    num_train = x.shape[0]
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    x_reshape = x.reshape(num_train,-1)
    out = np.dot(x_reshape,w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    num_train = x.shape[0]
    x_reshape = x.reshape(num_train,-1)
    
    dw = x_reshape.T.dot(dout)
    dx = dout.dot(w.T).reshape(x.shape)
    db = np.sum(dout,axis=0)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0,x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    x_temp = np.zeros(x.shape)
    x_temp[x>0] = 1
    dx = x_temp*dout
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training time forward pass for batch norm.      #                         
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        
        # step 1 : mu- shape of mu (D,)
        mu = 1/float(N)*np.sum(x,axis = 0)
        
        # step 2 : xmu - shape of xmu (N, D) 
        xmu = x - mu
        
        # step 3 : sq - shape of sq (N, D)
        sq = xmu**2
        
        # step 4 : var - shape of var (D,)
        var = 1/float(N)*np.sum(sq, axis = 0)
        
        # step 5 : sqrtvar - shape of sqrtvar (D,)
        sqrtvar = np.sqrt(var + eps)
        
        # step 6 : ivar - shape of ivar (D,)
        ivar = 1./sqrtvar
        
        # step 7 : x_hat - shape of x_hat (N, D)
        x_hat = xmu*ivar
        
        # step 8 : gammax_hat : shape of gammax_hat (N, D)
        gammax_hat = x_hat*gamma
        
        # step 9 : betax_hat : shape of betax_hat (N, D)
        betax_hat = gammax_hat + beta
        
        running_mean = running_mean*momentum + (1 -momentum)*mu
        running_var = running_var*momentum + (1 - momentum)*var
        
        out = betax_hat
        cache = (mu, xmu, sq, var, sqrtvar, ivar, x_hat, gammax_hat, gamma, beta, x, bn_param)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        x_hat = (x - running_mean)/np.sqrt(running_var+eps)
        out = gamma*x_hat + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    mu, xmu, sq, var, sqrtvar, ivar, x_hat, gammax_hat, gamma, beta, x, bn_param = cache
    eps = bn_param.get('eps', 1e-5)
    N, D = dout.shape
    
    # step 1 : dx_hat - shape of dx_hat (N, D)
    dbeta = np.sum(dout, axis = 0)
    dx_hat = dout*gamma
    dgamma = np.sum(dout * x_hat, axis = 0)
    
    # step 2 : dxmu1 - shape of dxmu1 (N ,D)
    dxmu1 = dx_hat*ivar
    
    # step 3 : divar - shape of divar (D,)
    divar = np.sum(dx_hat * xmu, axis = 0)
    
    # step 4 : dsqrtvar - shape of dsqrtvar (D, )
    dsqrtvar = -1./(sqrtvar**2)*divar
    
    # step 5 : dvar - shape of dvar (D, )
    dvar = 0.5*dsqrtvar/np.sqrt(var+eps)
    
    # step 6 : dsq - shape of dsq (N, D)
    dsq = 1/float(N)*np.ones((N,D))*dvar
    
    # step 7 : dxmu2 - shape of dxmu2 (N, D)
    dxmu2 = 2*xmu*dsq
    
    # step 8 : dx1 - shape of dx1 (N, D)
    dx1 = dxmu1 + dxmu2
    
    # step 9 : dmu - shape of dmu(D, )
    dmu = -np.sum(dxmu1 + dxmu2, axis = 0)
    
    # step10 : dx2 - shape of dx2 (N, D)
    dx2 = 1/float(N)*np.ones((N,D))*dmu
    
    # step11 : dx - shape of dx (N, D)
    dx = dx1 + dx2
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    mu, xmu, sq, var, sqrtvar, ivar, x_hat, gammax_hat, gamma, beta, x, bn_param = cache
    eps = bn_param.get('eps', 1e-5)
    N, D = dout.shape
    
    dgamma = np.sum(dout*x_hat,axis  = 0 )
    dbeta = np.sum(dout, axis = 0)
    dx = 1/float(N)*ivar*(N*dout*gamma
                          -np.sum(dout*gamma, axis = 0) 
                          -x_hat*np.sum(dout*gamma*x_hat, axis = 0) )
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = (np.random.rand(*x.shape)<(1-p))/(1-p)
        out = mask*x
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout*mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # Parameters of the conv
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    pad = conv_param['pad']
    stride = conv_param['stride']
    
    H_out = int(1 + (H + 2 * pad - HH) / stride)
    W_out = int(1 + (W + 2 * pad - WW) / stride)
    
    # padding of x
    x_padded = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),mode = 'constant')
    
    # the mask of im2col
    k_mask = np.repeat(np.arange(C), HH * WW).reshape(-1,1)
    i_mask_ver = np.tile(np.repeat(np.arange(HH), WW), C).reshape(-1,1)
    i_mask_hon = stride * np.repeat(np.arange(H_out), W_out).reshape(1,-1)
    j_mask_ver = np.tile(np.tile(np.arange(WW), HH), C).reshape(-1,1)
    j_mask_hon = stride * np.tile(np.arange(W_out),H_out).reshape(1,-1)
    i_mask = i_mask_ver + i_mask_hon
    j_mask = j_mask_ver + j_mask_hon
    
    # calculate the reshaped x(HH*WW*C,H_out*W_out*N) and w(F,HH*WW*C)
    x_col = x_padded[:,k_mask,i_mask,j_mask].transpose((1,2,0)).reshape(HH*WW*C,-1)
    w_col = w.reshape(F,-1)
    
    # calculate out
    out = (w_col.dot(x_col) + b.reshape(-1,1)).reshape(F,H_out,W_out,N).transpose((3,0,1,2))
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, x_col,conv_param,i_mask,j_mask,k_mask)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # parameters of conv_backward
    x, w, b, x_col,conv_param,i_mask,j_mask,k_mask = cache
    F, C, HH, WW = w.shape
    N, _, H, W = x.shape
    pad = conv_param['pad']
    stride = conv_param['stride']
    H_padded, W_padded = H + 2*pad, W + 2*pad
    
    # calculate db
    db = np.sum(dout, axis = (0, 2, 3))
    
    # calculate dw
    dout_reshaped = dout.transpose((1,2,3,0)).reshape(F, -1)
    dw_col = dout_reshaped.dot(x_col.T)
    dw = dw_col.reshape(w.shape)
    
    #calculate dx
    w_reshaped = w.reshape(F,-1)
    dx_col = (w_reshaped.T).dot(dout_reshaped)
    x_padded = np.zeros((N,C,H_padded,W_padded),dtype=dx_col.dtype)
    dx_reshaped = dx_col.reshape(C*HH*WW,-1,N).transpose((2,0,1))
    np.add.at(x_padded, (slice(None),k_mask,i_mask,j_mask),dx_reshaped)
    if pad ==0:
        dx = x_padded
    else:
        dx = x_padded[:, :, pad:-pad, pad:-pad]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    N,C,H,W = x.shape
    HP = pool_param['pool_height']
    WP = pool_param['pool_width']
    Stride = pool_param['stride']
    H_out = int((H - HP)/Stride + 1)
    W_out = int((W - WP)/Stride + 1)
    
    i_mask_hon = np.repeat(np.arange(HP),WP).reshape(1,-1)
    i_mask_ver = Stride * np.repeat(np.arange(H_out),W_out).reshape(-1,1)
    j_mask_hon = np.tile(np.arange(WP),HP).reshape(1,-1)
    j_mask_ver = Stride * np.tile(np.arange(W_out),H_out).reshape(-1,1)
    i_mask = i_mask_hon + i_mask_ver
    j_mask = j_mask_hon + j_mask_ver
    x_col = x[:,:,i_mask,j_mask].reshape(-1,HP*WP).transpose((1,0))
    
    max_idx = np.argmax(x_col,axis = 0)
    out_col = x_col[max_idx,range(max_idx.size)]
    
    out = out_col.reshape(N,C,H_out,W_out)
    
    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param, i_mask, j_mask, x_col, max_idx, H_out, W_out)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    (x, pool_param, i_mask, j_mask, x_col, max_idx, H_out, W_out) = cache
    N,C,H,W = x.shape
    HP = pool_param['pool_height']
    WP = pool_param['pool_width']
    Stride = pool_param['stride']
 
    dx_col = np.zeros_like(x_col)
    dx = np.zeros_like(x)
    
    dout_col = dout.reshape(1,-1)
    dx_col[max_idx,range(max_idx.size)] = dout_col
    dx_col = dx_col.transpose((1,0)).reshape(N,C,-1,HP*WP)
    np.add.at(dx,(slice(None),slice(None),i_mask,j_mask),dx_col)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    N, C, H, W = x.shape
    x_reshaped = x.transpose((0,2,3,1)).reshape(-1,C)
    out_reshaped, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)
    out = out_reshaped.reshape(N,H,W,C).transpose((0,3,1,2))
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    N, C, H, W = dout.shape
    dout_reshaped = dout.transpose((0,2,3,1)).reshape(-1,C)
    dx_reshaped, dgamma, dbeta = batchnorm_backward_alt(dout_reshaped, cache)
    dx = dx_reshaped.reshape(N,H,W,C).transpose((0,3,1,2))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
