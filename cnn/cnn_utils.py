import numpy as np
from absl import logging as logger

class Conv3x3:
    """Convolution layer using 3x3x3 filters"""
    
    def __init__(self, num_filters):
        
        self.num_filters = num_filters
        
        # A filter of size 3 x 3 applied to an input containing 3 channels, aka RGB
        # is a 3 x 3 x 3 filter volumn 
        # generate filter 
        # divide by 9 to reduce variance 
        self.filters = np.random.randn(self.num_filters, 3, 3, 3) / 9
        
    @staticmethod
    def zero_padding(image):
        """
        Zero pad the input image using same padding (padding = 1)
        - image is a 3d numpy array               
        """
        # height, width, channel of input image 
        h,w,c = image.shape
        # zero padding 
        padded_image = np.zeros((h+2,w+2,3))
        padded_image[1:h+1, 1:w+1] = image
        
        return padded_image
        
    def iterate_local_region(self, image):
        """
        Generate 3x3x3 local image using half padding 
        - image is a 3d numpy array                
        """
        # height, width, channel of input image 
        h,w,c = image.shape
        
        for i in range(h-2):
            for j in range(w-2):
                
                local_im = image[i:(i+3), j:(j+3)]
                
                yield local_im, i, j
                
    def forward(self, input):
        """
        Performs a forward pass of the convolutional layer using the given input
        with Stride 1 
        Returns a 3d numpy array 
        - input is a 3d numpy array 
        """
        
        self.input = input 
        h, w, c = input.shape
        output = np.zeros((h-2, w-2, self.num_filters))
        
        # iterate over local image and perform conv operation 
        for local_im, i, j in self.iterate_local_region(input):
            
            output[i,j] = np.sum(local_im * self.filters, axis = (1,2,3))
            
        return output 
    
    def backprop(self, dL_dout, lr):
        """
        Performs back propagation of the convolutional layer using the gradient w.r.t its output 
        
        """
        dL_dk = np.zeros(self.filters.shape)
        
        for local_im, i, j in self.iterate_local_region(self.input):
            for d in range(self.num_filters):
                # dout_dk = d_out/d_filter gradient of feature map w.r.t kernel 
                dout_dk = local_im
                dL_dk[d] += dL_dout[i,j,d] * dout_dk
                
        self.filters -= lr * dL_dk
                
        return None 

    
class MaxPool2x2:
    """max pooling of 2x2 region of input"""
    
    def iterate_pool_region(self, input):
        
        """
        Generate 2x2 region for pooling without overlapping 
        - input is a 3d numpy array 
        """
        
        h, w, _ = input.shape
        pool_h = h // 2  
        pool_w = w // 2 
        
        for i in range(pool_h):
            for j in range(pool_w):
                
                pool_region = input[(i*2):(i*2 + 2), (j*2):(j*2 + 2)]
                
                yield pool_region, i,j
                
    def forward(self, input):
        """
        Performs forward pass of the max pooling operation over input 
        Returns a 3d numpy array 
        """
        
        # cache input for backprop 
        self.input = input
        
        h, w, num_filters = input.shape 
        pool_h = h // 2  
        pool_w = w // 2 
        
        # Initialise output 
        output = np.zeros((pool_h, pool_w, num_filters))
        
        for pool_region, i, j in self.iterate_pool_region(input):
            
            # Max Pooling
            output[i,j] = np.amax(pool_region, axis=(0,1))
            
        return output 
    
    def backprop(self, dL_dout):
        """
        Performs back propagation of the max pooling operation given gradient of loss w.r.t. the output 
        """
        
        dL_dinput = np.zeros(self.input.shape)
        
        for im_region, i, j in self.iterate_pool_region(self.input):
            
            h, w, d = im_region.shape
            local_max_value = np.amax(im_region, axis=(0,1))
            
            for _i in range(h):
                for _j in range(w):
                    for _d in range(d):
                        
                        if im_region[_i, _j, _d] == local_max_value[_d]:
                            dL_dinput[(i*2 + _i),(j*2 + _j) , _d] = dL_dout[i,j,_d] 
            
        return dL_dinput      
    

class Softmax:
    """A standard fully-connected layer with softmax activation"""  
    
    def __init__(self, input_len, num_hidden):
        
        # Initilise weights and biases 
        # divide by length to reduce variance (hack)
        self.weights =  np.random.randn(input_len, num_hidden) / input_len
        self.biases  =  np.zeros(num_hidden) # can set as random values too 
        
    def forward(self, input):
        """
        Performs a forward pass of softmax layer over the input 
        Returns a 1d numpy array containing the probability of respective classes 
        - input is a numpy array with any dimension (normally a volumn)
        """
        # cache input shape 
        self.input_shape = input.shape
        
        # Flatten input array (as we are handling fully connected layer -> squeeze volumn to vector) 
        input = input.flatten()
        # cache input for backprop
        self.input = input 
        
        # Weight multiplication 
        inner_prod = input @ self.weights + self.biases 
        # cache values passed to softmax layer 
        self.totals = inner_prod
        
        exp = np.exp(inner_prod)
        
        softmax_prob = exp / np.sum(exp, axis=0)
        
        return softmax_prob
    
    def backprop(self, dL_dout, lr):
        """
        Receives backward pass from loss and performs backward pass of the softmax layer.
        Returns the gradient of loss w.r.t. the input to this layer 
        - dL_dout is the gradient of loss w.r.t the output of this layer, 1d numpy array 
        """
        
        # Get -1/p_i 
        for i, gradient in enumerate(dL_dout):
            if gradient == 0:
                continue 
            
            # dp/dt
            # S = \sigma_i^M exp(t_i)
            exp_t = np.exp(self.totals)
            S = np.sum(exp_t) 
            
            dp_dt = - exp_t[i] * (S**-2) * exp_t
            dp_dt[i] = (exp_t[i] * (S - exp_t[i])) / (S ** 2)
        
            # dt/dw
            # To be more precise,dt_dw corresponds to dt_i/dw_i where t = (t_i), W = (w_i) 
            # for all i, dt_i/dw_i = x
            dt_dw = self.input 
            # dt/db 
            # dt_db corresponds to dt_i / db_i 
            # for all i, dt_i / db_i = 1 
            dt_db = 1 
            # dt/dx 
            dt_dx = self.weights
            
            # combine gradients using chain rule 
            # dL/dt = dL/dp x dp/dt (shape = (num_hidden, ))
            dL_dt = gradient * dp_dt 
            # dL/dw = dL/dt x dt/dw 
            dL_dw = dt_dw[np.newaxis].T @ dL_dt[np.newaxis]
            # dL/db = dL/dt x dt/db 
            dL_db = dL_dt * dt_db
            # dL/dx = dL/dt x dt/dx 
            dL_dx = dt_dx @ dL_dt 
            
            # use SGD to update weights and biases in softmax layer 
            self.weights -= lr * dL_dw
            self.biases -= lr * dL_db
            
            return dL_dx.reshape(self.input_shape)
        

        
class trainNetwork:
    
    
    def __init__(self, conv_layer, maxpool, softmax):
        
        """Initialise layers of CNN"""
        
        self.conv_layer = conv_layer
        self.maxpool = maxpool
        self.softmax = softmax
    
    def forward(self, image, label):
    
        """
        Performs a forward pass of CNN and calculates accuracy and cross-entropy loss 
        - image is a 3d numpy array 
        - lable is a digit (int)
        """
        # Rescale image to [-1, 1] 
        image = (image - 0.5)*2 
        # Zero padding 
        out = self.conv_layer.zero_padding(image)
        # Conv layer 
        out = self.conv_layer.forward(out)
        # Max pooling layer 
        out = self.maxpool.forward(out)
        # Softmax layer 
        out = self.softmax.forward(out) 

        # Compute cross-entropy loss and accuracy 
        loss = - np.log(out[label])
        # binary classification 
        # return index of the largest probability 
        acc = 1 if np.argmax(out) == label else 0

        return out, loss, acc 
    
    
    def train(self, image, label, lr, class_num):
        """Performs a training step for given image and label 
        Returns cross-entropy loss and accuracy 
        """
        output, loss, acc = self.forward(image, label)

        # dL/dp_c 
        gradient = np.zeros(class_num)
        gradient[label] = -1/output[label]

        # back propagrate thru softmax layer 
        gradient = self.softmax.backprop(gradient, lr)
        gradient = self.maxpool.backprop(gradient)
        gradient = self.conv_layer.backprop(gradient, lr)

        return loss, acc 
