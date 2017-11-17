### 目录说明：  
    1. data/TSD-Signal目录需要放入图片文件夹，用来生成record文件，输出到data/records  
    2. data/TSD-Signal-GT目录需要放入图片标记的xml文件，用来生成record文件  
    3. data/test_samples目录需要放入图片文件夹，inference.py生成带标记框的测试结果用。  

### 使用说明：  
    注：   
        1. 下面所有脚本都在src目录下运行。（src/scripts目录下的脚本是工具脚本，可以单独执行）  
            在src路径下设置环境变量：  
                protoc object_detection/protos/*.proto --python_out=.
                export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
                
        2. models/research/object_detection这个文件夹复制到了src目录下，并将一部分必要的文件也复制到了这个目录. 因此本项目可以放到任意地方。   
        3. 如果模型不一样，下面的名称需要更改。 路径一般不用更改，都是相对路径。  
    
    1. xml转换为record：
            python ./create_pet_tf_record.py  \
            --data_dir=/home/cabbage/Desktop/CarND-Capstone/ros/src/tl_detector/train/sample_annotation\
            --output_dir=/home/cabbage/Desktop/CarND-Capstone/ros/src/tl_detector/train/sample_annotation/output
            
    2. 运行train.p训练模型：  
    
        python train.py --logtostderr \
        --pipeline_config_path=../config/ssd_mobilenet_v1.config \
        --train_dir=./model_cpkt
        
    3. 将模型转为pb文件：  
        需要将 *./model_cpkt/model.ckpt-0* 更改为要转换的ckpt文件  
        
        CUDA_VISIBLE_DEVICES="1" python ./object_detection/export_inference_graph.py   \
        --input_type image_tensor     \
        --pipeline_config_path ../config/ssd_mobilenet_v1.config  \
        --trained_checkpoint_prefix ./model_cpkt/model.ckpt-0     \
        --output_directory ./model_pb/
        
    4. 运行inference.py测试模型结果，得到带标注框的图片。
        python inference.py
        
### 模型配置文件更改：  
    注：   
        1. 如果使用新的模型，只需要更改fine_tune_checkpoint   
        2. 模型配置文件在./config/目录下   
    
    1. 分类个数：
        num_classes: 77
        
    2. 模型路径：   
        fine_tune_checkpoint: "./object_detection/checkpoints/ssd_mobilenet_v1_coco_2017_11_08/model.ckpt"    
        
    3. 训练次数：  
        num_steps: 20000  
        
    4. train input路径：  
       train_input_reader: {  
          tf_record_input_reader {  
            input_path: "../data/records/train.record"  
          }  
          label_map_path: "../config/traffic.pbtxt"  
        }  
 
    5. eval路径：  
        eval_input_reader: {
          tf_record_input_reader {
            input_path: "../data/records/val.record"
          }
          label_map_path: "../data/traffic.pbtxt"
          shuffle: false
          num_readers: 1
        }

