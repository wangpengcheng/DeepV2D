from tensorflow.python import pywrap_tensorflow
import pprint # 使用pprint 提高打印的可读性 
import os

checkpoint_path=os.path.join('models/nyu.ckpt')
reader=pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map=reader.get_variable_to_shape_map()
a  = reader.debug_string().decode("utf-8")
#pprint.pprint(a)
for key in var_to_shape_map:
    if 'stereo/encoder' in key:
        print('tensor_name: {} \n'.format(key))
        print('val: \n {} \n'.format(reader.get_tensor(key)))
