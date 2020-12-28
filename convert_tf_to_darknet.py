import tensorflow as tf
import re
import numpy as np

def write_bn_and_weights(weightfile,layer_name):
    write_bn(weightfile,layer_name)
    write_convweights(weightfile,layer_name)

def write_bias_and_weights(weightfile,layer_name):
    write_bias(weightfile,layer_name)
    write_convweights(weightfile,layer_name)

def write_bias(weightfile,layer_name):
    weightfile.write(reader.get_tensor(layer_name + '/bias').tobytes())


def write_bn(weightfile,layer_name):
    weightfile.write(reader.get_tensor(layer_name + '/batch_normalization/beta').tobytes())
    weightfile.write(reader.get_tensor(layer_name + '/batch_normalization/gamma').tobytes())
    weightfile.write(reader.get_tensor(layer_name + '/batch_normalization/moving_mean').tobytes())
    weightfile.write(reader.get_tensor(layer_name + '/batch_normalization/moving_variance').tobytes())

def write_convweights(weightfile,layer_name):
    # 需要将(height, width, in_dim, out_dim)转换成(out_dim, in_dim, height, width)
    conv_weights = np.transpose(reader.get_tensor(layer_name + '/weight'),[3,2,0,1])
    weights.write(conv_weights.tobytes())

def write_residual(weightfile,layer_name):
    write_bn_and_weights(weightfile,layer_name + '/conv1')
    write_bn_and_weights(weightfile,layer_name + '/conv2')

reader = tf.train.NewCheckpointReader('checkpoint/83/yolov3_test_loss=11.0151.ckpt-12')

global_variables = reader.get_variable_to_shape_map()

#result = open('readerResults.txt','w')
weights = open('test3.weight','wb')
#keys = list(global_variables.keys())
#sortedkeys = sorted(keys)
#for variable_name in sortedkeys:
    #if str(variable_name).startswith('darknet/conv0'):
#    result.write('{}:{}\n'.format(variable_name,global_variables[variable_name]))
#result.write('darknet/residual1/conv1/weight:{}'.format(reader.get_tensor('darknet/residual1/conv1/weight')))
#weights.write(reader.get_tensor('darknet/residual1/conv1/weight'))

numpy_data = np.ndarray(shape=(3,),dtype='int32',buffer = np.array([0,2,0],dtype='int32'))
weights.write(numpy_data.tobytes())
weights.flush()
numpy_data = np.ndarray(shape=(1,),
                          dtype='int64',
                          buffer=np.array([320000],dtype='int64'))
weights.write(numpy_data.tobytes())
weights.flush()


write_bn_and_weights(weights,'darknet/conv0')
write_bn_and_weights(weights,'darknet/conv1')
write_residual(weights,'darknet/residual0')
write_bn_and_weights(weights,'darknet/conv4')
write_residual(weights,'darknet/residual1')
write_residual(weights,'darknet/residual2')
write_bn_and_weights(weights,'darknet/conv9')
for i in range(8):
    write_residual(weights,'darknet/residual' + str(3 + i))
write_bn_and_weights(weights,'darknet/conv26')
for i in range(8):
    write_residual(weights,'darknet/residual' + str(11 + i))
write_bn_and_weights(weights,'darknet/conv43')
for i in range(4):
    write_residual(weights,'darknet/residual' + str(19 + i))
for i in range(5):
    write_bn_and_weights(weights,'conv' + str(52 + i))
write_bn_and_weights(weights,'yolo1')
write_bias_and_weights(weights,'feature_map_1')
write_bn_and_weights(weights,'conv57')
################upsample and route here########################
for i in range(5):
    write_bn_and_weights(weights,'conv' + str(58 + i))
write_bn_and_weights(weights,'yolo2')
write_bias_and_weights(weights,'feature_map_2')
write_bn_and_weights(weights,'conv63')
################upsample and route here########################
for i in range(5):
    #print(i)
    write_bn_and_weights(weights,'conv' + str(64 + i))
write_bn_and_weights(weights,'yolo3')
write_bias_and_weights(weights,'feature_map_3')
