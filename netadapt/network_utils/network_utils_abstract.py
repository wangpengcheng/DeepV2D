from abc import ABC, abstractmethod


class NetworkUtilsAbstract(ABC):
    """
    网络工具接口类
    Args:
        ABC ([type]): [description]
    """

    def __init__(self):
        super().__init__()


    @abstractmethod
    def get_network_def_from_model(self, model):
        '''
            network_def contains information about each layer within a model
            网络定义包含有关模型中每个层的信息
            Input: 
                `model`: pytorch model (e.g. nn.Sequential())
                
            Output:
                `network_def`: 网络定义包含分层信息（例如输出/输入通道数）。

                network_def 将用于计算资源和指导剪枝模型。
            
            please refer to def get_network_def_from_model() in functions.py 
            请参阅functions.py中的get_network_def_from_model() 了解更多接口
        '''
        pass


    @abstractmethod
    def simplify_network_def_based_on_constraint(self, network_def, block, constraint, resource_type,
                                                 lookup_table_path):
        '''
            Derive how much a certain block of layers ('block') should be simplified 
            based on resource constraints.
            根据资源限制，得出某个层块（“块”）应简化多少。
            Input:
                `network_def`: defined in get_network_def_from_model()
                `constraint`: (float) representing the FLOPs/weights/latency constraint the simplied model should satisfy
                （float）表示简化模型应满足的FLOPs/weights/latency约束
                `resource_type`: (string) `FLOPs`, `WEIGHTS`, or `LATENCY`
                `lookup_table_path`: (string) path to latency lookup table. Needed only when resource_type == 'LATENCY'
                延迟查找表路径
            Output:
                `simplified_network_def`: simplified network definition. Indicates how much the network should
                be simplified/pruned.
                简化的网络定义。指示应简化/修剪多少网络。
                `simplified_resource`: (float) the estimated resource consumption of simplified network_def.
                简化网络定义的估计资源消耗。
                
            please refer to def simplify_network_def_based_on_constraint(...) in functions.py   
            to see one implementation.
        '''
        pass


    @abstractmethod
    def simplify_model_based_on_network_def(self, simplified_network_def, model):
        '''
            Choose which filters to perserve
            
            Input:
                `simplified_network_def`: network_def shows how a model will be pruned.
                defined in get_network_def_from_model().
                Get simplified_network_def from the output `simplified_network_def` of 
                self.simplify_network_def_based_on_constraint()
                
                `model`: model to be simplified.
                
            Output:
                `simplified_model`: simplified model.
        
            please refer to def simplify_model_based_on_network_def(...) in functions.py 
            to see one implementation
        '''            
        pass
    

    @abstractmethod
    def extra_history_info(self, network_def):
        '''
            return # of output channels per layer
            
            Input: 
                `network_def`: defined in get_network_def_from_model()
            
            Output:
                `num_filters_str`: (string) show the num of output channels for each layer.
                Or you can define your own log
        '''
        pass
        

    @abstractmethod
    def build_lookup_table(self, network_def, resource_type, lookup_table_path):
        '''
            Build lookup table for layers defined by `network_def`.
        
            Input: 
                `network_def`: defined in get_network_def_from_model()
                `resource_type`: (string) resource type (e.g. 'LATENCY')
                `lookup_table_path`: (string) path to save the file of lookup table
        '''
        pass


    @abstractmethod
    def compute_resource(self, network_def, resource_type, lookup_table_path):
        '''
            compute resource based on resource type
        
            Input:
                `network_def`: defined in get_network_def_from_model()
                `resource_type`: (string) resource type (e.g. 'WEIGHTS'/'LATENCY'/'FLOPS')
                `lookup_table_path`: (string) path to lookup table
        
            Output:
                `resource`: (float)
        '''
        pass


    @abstractmethod
    def get_num_simplifiable_blocks(self):
        '''
            Output:
               `num_splifiable_blocks`: (int) num of blocks whose num of output channels can be reduced.
               Note that simplifiable blocks do not include output layer
        '''
        pass


    @abstractmethod
    def fine_tune(self, model, iterations):
        '''
            short-term fine-tune a simplified model
            
            Input:
                `model`: model to be fine-tuned
                `iterations`: (int) num of short-term fine-tune iterations
                
            Output:
                `model`: fine-tuned model
        '''
        pass


    @abstractmethod
    def evaluate(self, model):
        '''
            Evaluate the accuracy of the model
            
            Input:
                `model`: model to be evaluated
                
            Output:
                `accuracy`: (float) (0~100)
        '''
        pass