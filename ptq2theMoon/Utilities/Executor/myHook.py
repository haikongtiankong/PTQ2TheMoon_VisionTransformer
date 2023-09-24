from ptq2theMoon import *


class MyHook(RuntimeHook):
    def __init__(self, operation: Operation, pickle_path: str, type: int) -> None:
        super().__init__(operation)
        self.op = operation
        self.pickle_path = pickle_path
        self.type = type

    def pre_forward_hook(self, inputs: List[torch.Tensor], **kwargs) -> list:
        # some operations got none as its input
        # therefore we have to create meta for those none input value manually.
        for tensor, var in zip(inputs, self._hook_to.inputs):
            if tensor is None:
                ppq_warning(
                    f'Unexpected input value of operation {self._hook_to.name}, '
                    f'recieving "None" at its input {self._hook_to.inputs.index(var)}')
            else:
                var.shape = tensor.shape
                var.dtype = tensor.dtype

        # customized block1
        pattern = r'/blocks/blocks\.(1[0-1]|[0-9])/mlp/fc[1-2]/MatMul'
        pattern = r'/blocks/blocks\.(1[0-1]|[0-9])/attn/(qkv|proj)/MatMul'
        match = re.search(pattern, self.op.name)
        if match:
            existing_data = {}
            layer_name = self.op.name.split('/')[-2]
            num = self.op.name.split('/')[-4].split('.')[-1]
            pickle_path = r'activations_' + layer_name + '_b' + num + '.pickle'
            pickle_path = os.path.join(self.pickle_path, pickle_path)
            if os.path.exists(pickle_path):
                with open(pickle_path, 'rb') as f:
                    existing_data = pickle.load(f)
                data_input = self.op.inputs[0].value
                data_output = self.op.outputs[0].value
                existing_data['input'].append(data_input)
                existing_data['output'].append(data_output)
                with open(pickle_path, 'wb') as f:
                    pickle.dump(existing_data, f)
            else:
                existing_data['input'] = []
                existing_data['output'] = []
                data_input = self.op.inputs[0].value
                data_output = self.op.outputs[0].value
                existing_data['input'].append(data_input)
                existing_data['output'].append(data_output)
                with open(pickle_path, 'wb') as f:
                    pickle.dump(existing_data, f)

        # if self.op.name == '/blocks/blocks.11/attn/MatMul_1':
        #     pickle_path = r'attention_fp.pickle'
        #     pickle_path = os.path.join(self.pickle_path, pickle_path)
        #     existing_data = self.op.inputs[0].value
        #     with open(pickle_path, 'wb') as f:
        #         pickle.dump(existing_data, f)
        return inputs

    def post_forward_hook(self, outputs: List[torch.Tensor], **kwargs) -> list:
        for tensor, var in zip(outputs, self._hook_to.outputs):
            if tensor is not None:
                var.shape = tensor.shape
                var.dtype = tensor.dtype

        #customize block
        if self.op.name == '/blocks/blocks.11/attn/Softmax':
            pickle_path = r''
            if self.type == 0:
                pickle_path = r'attention_fp_.pickle'
            elif self.type == 1:
                pickle_path = r'attention_normC.pickle'
            else:
                pickle_path = r'attention_normB56Fc2C_.pickle'
            pickle_path = os.path.join(self.pickle_path, pickle_path)
            if os.path.exists(pickle_path):
                with open(pickle_path, 'rb') as f:
                        existing_data = pickle.load(f)
                        existing_data.append(outputs[0])
                        with open(pickle_path, 'wb') as f:
                            pickle.dump(existing_data, f)
            else:
                existing_data = []
                existing_data.append(outputs[0])
                with open(pickle_path, 'wb') as f:
                    pickle.dump(existing_data, f)

        return outputs
