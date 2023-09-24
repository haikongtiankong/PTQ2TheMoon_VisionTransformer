# import re
# import pickle
from ptq2theMoon import *

class ViTQuantizer(BaseQuantizer):
    def __init__(
            self,
            graph: BaseGraph,
            verbose: bool = True
    ) -> None:
        super().__init__(graph, verbose)
        self._processor = QuantableGraph(GraphReplacer(self._graph))
        self._mergeProcessor = GraphMerger(self._graph)
        self._mergeProcessor.fuse_matmul_add()
        self._mergeProcessor.fuse_layernorm()
        self._mergeProcessor.fuse_gelu()

    @property
    def quant_operation_types(self) -> set:
        return {'ReduceMean', 'MatMul', 'Mul', 'Conv', 'Add', 'Concat', 'Sub', 'Div', 'Softmax','PPQBiasFusedMatMul',
                'Gelu', 'LayerNormalization', 'Pow', 'Erf'}  # 'Mul', 'MatMul', 'Add','PPQBiasFusedMatMul','LayerNormalization', 'Gelu'

    @property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.PPL_CUDA_INT8

    @staticmethod
    def create_config_for_layernorm(
            op: Operation,
            num_of_bits: int = 8,
            quant_min: Union[int, float] = -127,
            quant_max: Union[int, float] = 128,
            observer_algorithm: str = 'percentile',
            policy: QuantizationPolicy =
            QuantizationPolicy(
                QuantizationProperty.PER_CHANNEL +
                QuantizationProperty.LINEAR +
                QuantizationProperty.SYMMETRICAL),
            rounding: RoundingPolicy = RoundingPolicy.ROUND_HALF_EVEN,
            exponent_bits: int = 0,
            channel_axis: int = 2,
            only_output_per_channel: bool = True
    ):
        socket = op.socket
        input_cfgs, output_cfgs = [], []
        policy_input = policy
        policy_output = policy
        for index in range(op.num_of_input):
            if len(op.inputs[index].shape) == 3:
                # state = QuantizationStates.FP32
                # input_cfgs.append(TensorQuantizationConfig(state=state, policy=policy_input))
                state = QuantizationStates.INITIAL
                input_cfgs.append(TensorQuantizationConfig(
                    policy=policy_input, rounding=rounding, channel_axis=2,
                    num_of_bits=num_of_bits, scale=None, offset=None,
                    exponent_bits=exponent_bits, quant_min=quant_min, quant_max=quant_max,
                    observer_algorithm=observer_algorithm, state=state))
            else:
                state = QuantizationStates.INITIAL
                input_cfgs.append(TensorQuantizationConfig(
                    policy=policy_input, rounding=rounding, channel_axis=0,
                    num_of_bits=num_of_bits, scale=None, offset=None,
                    exponent_bits=exponent_bits, quant_min=quant_min, quant_max=quant_max,
                    observer_algorithm=observer_algorithm, state=state))
                # state = QuantizationStates.FP32
                # input_cfgs.append(TensorQuantizationConfig(state=state, policy=policy_input))

        for index in range(op.num_of_output):
            # state = QuantizationStates.FP32
            # output_cfgs.append(TensorQuantizationConfig(state=state, policy=policy_output))
            state = QuantizationStates.INITIAL
            if index < len(socket.out_plat):
                target_plat = socket.out_plat[index]
                if target_plat == TargetPlatform.FP32:
                    state = QuantizationStates.FP32
                if target_plat == TargetPlatform.SOI:
                    state = QuantizationStates.FP32
            output_cfgs.append(TensorQuantizationConfig(
                policy=policy_input, rounding=rounding, channel_axis=2,
                num_of_bits=num_of_bits, scale=None, offset=None,
                exponent_bits=exponent_bits, quant_min=quant_min, quant_max=quant_max,
                observer_algorithm=observer_algorithm, state=state))

        return OperationQuantizationConfig(input_cfgs, output_cfgs)


    @staticmethod
    def create_config_for_fuseMatMULAdd(
            op: Operation,
            num_of_bits: int = 8,
            quant_min: Union[int, float] = -127,
            quant_max: Union[int, float] = 128,
            observer_algorithm: str = 'percentile',
            policy: QuantizationPolicy =
            QuantizationPolicy(
                QuantizationProperty.PER_CHANNEL +
                QuantizationProperty.LINEAR +
                QuantizationProperty.SYMMETRICAL),
            rounding: RoundingPolicy = RoundingPolicy.ROUND_HALF_EVEN,
            exponent_bits: int = 0,
            channel_axis: int = 2,
            only_output_per_channel: bool = True):
        socket = op.socket
        input_cfgs, output_cfgs = [], []
        policy_input = policy
        policy_output = policy
        for index in range(op.num_of_input):
            if len(op.inputs[index].shape) == 2:
                state = QuantizationStates.INITIAL
                input_cfgs.append(TensorQuantizationConfig(
                    policy=policy_input, rounding=rounding, channel_axis=0,
                    num_of_bits=num_of_bits, scale=None, offset=None,
                    exponent_bits=exponent_bits, quant_min=quant_min, quant_max=quant_max,
                    observer_algorithm=observer_algorithm, state=state))
            elif len(op.inputs[index].shape) == 3:
                state = QuantizationStates.INITIAL
                input_cfgs.append(TensorQuantizationConfig(
                    policy=policy_input, rounding=rounding, channel_axis=2,
                    num_of_bits=num_of_bits, scale=None, offset=None,
                    exponent_bits=exponent_bits, quant_min=quant_min, quant_max=quant_max,
                    observer_algorithm=observer_algorithm, state=state))
            else:
                state = QuantizationStates.FP32
                input_cfgs.append(TensorQuantizationConfig(state=state, policy=policy_input))

        for index in range(op.num_of_output):
            # state = QuantizationStates.FP32
            # output_cfgs.append(TensorQuantizationConfig(state=state, policy=policy_output))
            state = QuantizationStates.INITIAL
            if index < len(socket.out_plat):
                target_plat = socket.out_plat[index]
                if target_plat == TargetPlatform.FP32:
                    state = QuantizationStates.FP32
                if target_plat == TargetPlatform.SOI:
                    state = QuantizationStates.FP32
            output_cfgs.append(TensorQuantizationConfig(
                    policy=policy_input, rounding=rounding, channel_axis=2,
                    num_of_bits=num_of_bits, scale=None, offset=None,
                    exponent_bits=exponent_bits, quant_min=quant_min, quant_max=quant_max,
                    observer_algorithm=observer_algorithm, state=state))

        return OperationQuantizationConfig(input_cfgs, output_cfgs)



    @staticmethod
    def create_default_quant_config(
            op: Operation,
            num_of_bits: int = 8,
            quant_min: Union[int, float] = -127,
            quant_max: Union[int, float] = 128,
            observer_algorithm: str = 'percentile',
            policy: QuantizationPolicy =
            QuantizationPolicy(
                QuantizationProperty.PER_TENSOR +
                QuantizationProperty.LINEAR +
                QuantizationProperty.SYMMETRICAL),
            rounding: RoundingPolicy = RoundingPolicy.ROUND_HALF_EVEN,
            exponent_bits: int = 0,
            channel_axis: int = None,
            only_output_per_channel: bool = True
    ) -> OperationQuantizationConfig:
        socket = op.socket
        input_cfgs, output_cfgs = [], []
        policy_input = policy
        policy_output = policy
        for index in range(op.num_of_input):
            state = QuantizationStates.INITIAL
            # for those unexpected inputs and outputs
            # ppq just initilize them as normal variable.
            if index < len(socket.in_plat):
                target_plat = socket.in_plat[index]
                if target_plat == TargetPlatform.FP32:
                    state = QuantizationStates.FP32
                if target_plat == TargetPlatform.SOI:
                    state = QuantizationStates.FP32
            input_cfgs.append(TensorQuantizationConfig(
                policy=policy_input, rounding=rounding, channel_axis=channel_axis,
                num_of_bits=num_of_bits, scale=None, offset=None,
                exponent_bits=exponent_bits, quant_min=quant_min, quant_max=quant_max,
                observer_algorithm=observer_algorithm, state=state))

        for index in range(op.num_of_output):
            state = QuantizationStates.INITIAL
            # for those unexpected inputs and outputs
            # ppq just initilize them as normal variable.
            if index < len(socket.out_plat):
                target_plat = socket.out_plat[index]
                if target_plat == TargetPlatform.FP32:
                    state = QuantizationStates.FP32
                if target_plat == TargetPlatform.SOI:
                    state = QuantizationStates.FP32
            output_cfgs.append(TensorQuantizationConfig(
                policy=policy_output, rounding=rounding, num_of_bits=num_of_bits, scale=None, offset=None,
                exponent_bits=exponent_bits, quant_min=quant_min, quant_max=quant_max, channel_axis=channel_axis,
                observer_algorithm=observer_algorithm, state=state))

        return OperationQuantizationConfig(input_cfgs, output_cfgs)

    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:
        # #观察点
        # if operation.name.split('/')[-1] == 'Softmax':
        # print(operation.name)
        config = self.create_default_quant_config(
            op=operation,
            num_of_bits=8,
            quant_max=127,
            quant_min=-128,
            observer_algorithm='percentile',
            policy=QuantizationPolicy(
                QuantizationProperty.PER_TENSOR +
                QuantizationProperty.LINEAR +
                QuantizationProperty.SYMMETRICAL +
                QuantizationProperty.DYNAMIC),
            rounding=RoundingPolicy.ROUND_HALF_EVEN)  # ROUND默认half_even


        #在进入NormLayer之前，对Add算子进行channel-wise的量化
        if len(operation.name.split("/")) >= 2 and \
                re.search(r'blocks\.(1[0-1]|[0-9])', operation.name.split("/")[-2]) and operation.type == 'Add':
            print(operation.name + ': is changing to channel-wise quant(fc2)')
            config = self.create_default_quant_config(
                op=operation,
                num_of_bits=8,
                quant_max=127,
                quant_min=-128,
                observer_algorithm='minmax',

                policy=QuantizationPolicy(
                    QuantizationProperty.PER_CHANNEL +
                    QuantizationProperty.LINEAR +
                    QuantizationProperty.SYMMETRICAL),
                rounding=RoundingPolicy.ROUND_HALF_EVEN,
                channel_axis=2)

        # and 0 <= int(operation.name.split('/')[-4].split('.')[-1]) <= 5
        if operation.type == 'PPQBiasFusedMatMul' and operation.name.split('/')[-2] == 'fc2' and \
                5 <= int(operation.name.split('/')[-4].split('.')[-1]) <= 6:
            print(operation.name + ': is channel-wised (fused)')
            config = self.create_config_for_fuseMatMULAdd(
                op=operation,
                num_of_bits=8,
                quant_max=127,
                quant_min=-128,
                observer_algorithm='minmax',

                policy=QuantizationPolicy(
                    QuantizationProperty.PER_CHANNEL +
                    QuantizationProperty.LINEAR +
                    QuantizationProperty.SYMMETRICAL +
                    QuantizationProperty.DYNAMIC),
                rounding=RoundingPolicy.ROUND_HALF_EVEN)

        if operation.type == 'LayerNormalization':
            print(operation.name + ': is channel-wised (fused norm)')
            config = self.create_config_for_layernorm(
                op=operation,
                num_of_bits=8,
                quant_max=127,
                quant_min=-128,
                observer_algorithm='minmax',

                policy=QuantizationPolicy(
                    QuantizationProperty.PER_CHANNEL +
                    QuantizationProperty.LINEAR +
                    QuantizationProperty.SYMMETRICAL +
                    QuantizationProperty.DYNAMIC),
                rounding=RoundingPolicy.ROUND_HALF_EVEN
            )

        return config

    def build_quant_pipeline(self, setting: QuantizationSetting) -> QuantizationOptimizationPipeline:
        pipeline = []
        pipeline.append(ParameterQuantizePass())
        pipeline.append(RuntimeCalibrationPass())
        if setting.bias_correct:
            bias_correct_setting = setting.bias_correct_setting
            pipeline.append(BiasCorrectionPass(
                block_size=bias_correct_setting.block_size,
                interested_layers=bias_correct_setting.interested_layers,
                steps=bias_correct_setting.steps,
                collecting_device=bias_correct_setting.collecting_device
            ))
        return QuantizationOptimizationPipeline(pipeline)