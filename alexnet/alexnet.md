Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 32, 32, 64)        23296     
_________________________________________________________________
activation (Activation)      (None, 32, 32, 64)        0         
_________________________________________________________________
batch_normalization (BatchNo (None, 32, 32, 64)        256       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 30, 30, 64)        0         
_________________________________________________________________
dropout (Dropout)            (None, 30, 30, 64)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 15, 15, 128)       401536    
_________________________________________________________________
activation_1 (Activation)    (None, 15, 15, 128)       0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 15, 15, 128)       512       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 7, 7, 128)         0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 7, 7, 128)         0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 7, 7, 192)         221376    
_________________________________________________________________
activation_2 (Activation)    (None, 7, 7, 192)         0         
_________________________________________________________________
batch_normalization_2 (Batch (None, 7, 7, 192)         768       
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 7, 7, 256)         442624    
_________________________________________________________________
activation_3 (Activation)    (None, 7, 7, 256)         0         
_________________________________________________________________
batch_normalization_3 (Batch (None, 7, 7, 256)         1024      
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 7, 7, 256)         590080    
_________________________________________________________________
activation_4 (Activation)    (None, 7, 7, 256)         0         
_________________________________________________________________
batch_normalization_4 (Batch (None, 7, 7, 256)         1024      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 3, 3, 256)         0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 3, 3, 256)         0         
_________________________________________________________________
flatten (Flatten)            (None, 2304)              0         
_________________________________________________________________
dense (Dense)                (None, 2304)              5310720   
_________________________________________________________________
activation_5 (Activation)    (None, 2304)              0         
_________________________________________________________________
batch_normalization_5 (Batch (None, 2304)              9216      
_________________________________________________________________
dropout_3 (Dropout)          (None, 2304)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 576)               1327680   
_________________________________________________________________
activation_6 (Activation)    (None, 576)               0         
_________________________________________________________________
batch_normalization_6 (Batch (None, 576)               2304      
_________________________________________________________________
dropout_4 (Dropout)          (None, 576)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 200)               115400    
_________________________________________________________________
activation_7 (Activation)    (None, 200)               0         
=================================================================
Total params: 8,447,816
Trainable params: 8,440,264
Non-trainable params: 7,552
_________________________________________________________________