from IPython.display import clear_output
import numpy as np

class PatchesExtractor:
    
    def __init__(self, s=11, stride_1=1, stride_2=1, max_batch_size=20, verbose=False):
        
        self.s = s
        self.stride_1 = stride_1
        self.stride_2 = stride_2
        
        self.max_batch_size = max_batch_size
        self.verbose = verbose
        
    def extract(self, X, **args):
        
        if 'msg' in args.keys():
            msg = args['msg']+"\n"
        else:
            msg = ""
        
        n, m1, m2, ch = X.shape
        
        assert m1 == m2, "The input tensors must be square."
        assert m1 >= self.s, "The input images side must be at least of the set patch size."
        
        # Extracting the set of patches:
        
        P = np.arange(self.s)[None,None,:] + m2*np.arange(self.s)[None,:,None] + ch*m2*np.arange(ch)[:,None,None]
        T = np.arange(0, m1-self.s+1, self.stride_1)[None,:] + m2*np.arange(0, m2-self.s+1, self.stride_2)[:,None]
        I = T[None,:,:,None,None,None]+P[None,None,None,:,:,:]

        i = 0
        for i in range(n//self.max_batch_size):

            PX = np.take_along_axis(X[i*self.max_batch_size:(i+1)*self.max_batch_size,:,:,:]\
                                     .reshape(self.max_batch_size,-1,1,1,1,1), np.repeat(I, self.max_batch_size, 0), 1) # [self.max_batch_size, m1-s, m2-s, ch, s, s]
            PX = PX.transpose(0,4,5,3,1,2) # [self.max_batch_size, s, s, ch, m1-s, m2-s]
            # PX = PX.reshape(self.max_batch_size,self.s,self.s,ch,-1)

            yield PX
            
            if self.verbose:
                clear_output(wait=True)
                print(msg+f"Batch: {i+1}/{n//self.max_batch_size} processed")
            
        try: 
            
            PX = np.take_along_axis(X[i*self.max_batch_size:i*self.max_batch_size+n%self.max_batch_size,:,:,:]\
                                         .reshape(n%self.max_batch_size,-1,1,1,1,1), np.repeat(I, n%self.max_batch_size, 0), 1)
            PX = PX.transpose(0,4,5,3,1,2)
            # PX = PX.reshape(n%self.max_batch_size,self.s,self.s,ch,-1) 
        
            yield PX
            
        except ValueError:
            
            pass
        
        if self.verbose:
            print(f"Number of extracted patch per input image: {((m1-self.s+1)//self.stride_1+(m1-self.s+1)%self.stride_1)*((m2-self.s+1)//self.stride_2+(m2-self.s+1)%self.stride_2)}")
            print(f"Number of extracted patches: {n*((m1-self.s+1)//self.stride_1+(m1-self.s+1)%self.stride_1)*((m2-self.s+1)//self.stride_2+(m2-self.s+1)%self.stride_2)}")