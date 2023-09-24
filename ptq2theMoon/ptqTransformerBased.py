import pickle

from PIL import Image
from torchvision.transforms import transforms

from .Utilities.Heatmap.attention_map import visualize_attention, plot_attention
from .Utilities.Imagenet import load_imagenet_from_directory, evaluate_ppq_module_with_imagenet
from .Quantizer.ViTQuantizer import ViTQuantizer
import torch
from ppq import *
from ppq.api import *
from .ptqBase import PtqBase
from .Model.Vision_transformer import vit_base_patch16_224, vit_base_patch32_224, vit_large_patch16_224
from .Model.Dei_Transformer import deit_base_patch16_224, deit_tiny_patch16_224, deit_small_patch16_224
from .Model.Swin_Transformer import swin_small_patch4_window7_224, swin_tiny_patch4_window7_224, swin_base_patch4_window7_224
from torch.profiler import profile, record_function, ProfilerActivity
import torch.profiler
from tqdm import tqdm
from .Utilities.Executor.BobExecutor import BobbyExecutor

weight_url = {
'vit_base_patch16_224': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
'vit_large_patch16_224': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
'swin_tiny_patch4_window7_224': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',
'swin_small_patch4_window7_224': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth',
'swin_base_patch4_window7_224': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth',
'deit_tiny_patch16_224': 'https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth',
'deit_small_patch16_224': 'https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
'deit_base_patch16_224': 'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',
'mobile_vit_xx_small_224': 'https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xxs.pt'
}

model_dict = {
'vit_base_patch16_224': vit_base_patch16_224(),
'vit_large_patch16_224': vit_large_patch16_224(),
'swin_tiny_patch4_window7_224': swin_tiny_patch4_window7_224(),
'swin_small_patch4_window7_224': swin_small_patch4_window7_224(),
'swin_base_patch4_window7_224': swin_base_patch4_window7_224(),
'deit_tiny_patch16_224': deit_tiny_patch16_224(),
'deit_small_patch16_224': deit_small_patch16_224(),
'deit_base_patch16_224': deit_base_patch16_224()
}

'''
Post training quantization for transformer-based model 
'''
class PTQForTransformerBased(PtqBase):
    def __init__(self, calibDir: str, pthDir=None):
        """
        :param calibDir: the path to the calibration dataset
        :param pthDir: the path to the weight .pth file. If no, use pretrained model of ILSVRC2012
        """
        self.models = ['vit_base_patch16_224',
                       'vit_large_patch16_224',
                       'swin_tiny_patch4_window7_224',
                       'swin_small_patch4_window7_224',
                       'swin_base_patch4_window7_224',
                       'deit_tiny_patch16_224',
                       'deit_small_patch16_224',
                       'deit_base_patch16_224',
                       'mobile_vit_xx_small_224']
        self.platform = TargetPlatform.PPL_CUDA_INT8
        self.bias_correction = False
        self.calib_dir = calibDir
        self.pth = pthDir

    def ptq(self, batchsize: int, model_name: str, dummy_input=None) -> BaseGraph:
        """
        perform post-training quantization
        :param batchsize: batchsize of calibration
        :param model_name: model name
        :return: a ppq BaseGraph of quantized model
        """
        if model_name not in self.models:
            assert 'Invalid model inputs, please make sure you pick a model in ' + self.models

        input_width = int(model_name.split('_')[-1])
        subset = 1024
        model = None

        if self.pth == None:
            model = model_dict[model_name]
            checkpoint = torch.hub.load_state_dict_from_url(
                url=weight_url[model_name],
                map_location="cpu", check_hash=True
            )
            #print(checkpoint)
            if model_name.split('_')[0] == 'vit' or model_name.split('_')[0] =='mobile':
                model.load_state_dict(checkpoint, strict=False)
            else:
                model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model = model_dict[model_name]
            model.load_state_dict(torch.load(self.pth, map_location='cpu'))
        model = model.to('cuda')

        register_network_quantizer(ViTQuantizer, platform=self.platform)
        setting = QuantizationSettingFactory.pplcuda_setting()
        setting.quantize_parameter_setting.baking_parameter = False
        setting.bias_correct = False


        dataloader = load_imagenet_from_directory(
            directory=self.calib_dir, batchsize=batchsize,
            shuffle=True, subset=subset, require_label=False,
            num_of_workers=0)
        print('dataloader prepared')
        tracing = torch.zeros((batchsize, 3, 224, 224)).to('cuda')
        tracing[0] = dataloader.dataset.__getitem__(0)[0]
        tracing[1] = dataloader.dataset.__getitem__(1)[0]

        ppq_quant_ir = quantize_torch_model(
            model=model, calib_dataloader=dataloader, input_shape=[batchsize, 3, input_width, input_width],
            calib_steps=subset / batchsize, collate_fn=lambda x: x.to('cuda'), verbose=0,
            device='cuda', platform=self.platform, setting=setting, onnx_export_file=model_name + "_onnx.model",
            do_quantize=True, inputs=dummy_input)
        print('model quantization finished')

        return ppq_quant_ir


    def report(self, type:str, graph: BaseGraph):
        """
        :param type: quantization error analysis type: 'layerwise' or 'graphwise'
        :param graph: quantized Base graph
        :return: report
        """
        if type not in ['layerwise', 'graphwise']:
            assert 'Invalid report type, please choose a type in ["layerwise", "graphwise"]'

        dataloader = load_imagenet_from_directory(
            directory=self.calib_dir, batchsize=2,
            shuffle=True, subset=32, require_label=False,
            num_of_workers=0)

        if type == 'layerwise':
            return layerwise_error_analyse(graph=graph, running_device='cuda',
                                           collate_fn=lambda x: x.to('cuda'),
                                           dataloader=dataloader, verbose=True)
        else:
            return graphwise_error_analyse(graph=graph, running_device='cuda',
                                           collate_fn=lambda x: x.to('cuda'),
                                           dataloader=dataloader, verbose=True)

    def evaluation(self, val_dir: str, graph: BaseGraph, batchsize: int):
        """
        perform evaluation
        :param val_dir: validation data path
        :param graph: the quantized BaseGraph
        :param batchsize: batchsize
        :return: evluation report
        """
        ppq_eval_report = evaluate_ppq_module_with_imagenet(
                            model=graph, imagenet_validation_dir=val_dir,
                            batchsize=batchsize, device='cpu', verbose=True)
        return ppq_eval_report


    def profiler_table(self, out_path: str, model_name: str, graph: BaseGraph, verbose=True):
        """
        generate profiler table, take a look at what are the time consuming of different operations
        :param out_path: logs output path
        :param model_name: model name
        :param graph: the quantized BaseGraph
        :param verbose: whether to print the result table
        :return:
        """
        executor = None
        device = torch.device('cuda')
        sample_input = [torch.rand(2, 3, 224, 224).to(device) for i in range(32)]
        if graph is not None:
            executor = TorchExecutor(graph)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         record_shapes=True,
                         with_stack= True,
                         on_trace_ready=torch.profiler.tensorboard_trace_handler(
                             dir_name=out_path),
                         ) as prof:
                with torch.no_grad():
                    for batch_idx in tqdm(range(32), desc='Profiling...'):
                        executor.forward(sample_input[batch_idx])
                        prof.step()
            # 输出分析结果
            if verbose:
                print(prof.key_averages().table())
        elif model_name is not None:
            executor = model_dict[model_name]
            executor = executor.to(device)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         record_shapes=True,
                         on_trace_ready=torch.profiler.tensorboard_trace_handler(
                             dir_name=out_path),
                         ) as prof:
                with torch.no_grad():
                    for batch_idx in tqdm(range(32), desc='Profiling...'):
                        output = executor(sample_input[batch_idx])
                        prof.step()
            # 输出分析结果
            if verbose:
                print(prof.key_averages().table())
        else:
            assert 'please enter a model name or a quantized graph!'

    def hook_layer(self, graph: BaseGraph, val_dir: str, batchsize: int, pickle_path: int, type: int = 1):
        executor = BobbyExecutor(graph=graph, device='cuda', type=type, pickle_path=pickle_path)
        dataloader = load_imagenet_from_directory(val_dir, batchsize=batchsize, shuffle=True, num_of_workers=0)
        for batch_idx, (batch_input, batch_label
                        ) in tqdm(enumerate(dataloader),
                                  desc='Evaluating Model...', total=len(dataloader)):
            executor.tracing_operation_meta(inputs=batch_input)

    def attn_map_visual(self, pickle_path: str, img_idx: int, data_path: str):
        pickle_path1 = os.path.join(pickle_path, 'attention_fp.pickle')
        pickle_path2 = os.path.join(pickle_path, 'attention_normC.pickle')
        pickle_path3 = os.path.join(pickle_path, 'attention_normB56Fc2C_.pickle')
        class_idx = img_idx / 2 + 1
        img_seq = img_idx % 2

        path_list = os.listdir(data_path)
        class_file = os.path.join(data_path, path_list[class_idx])
        img_list = os.listdir(class_file)
        image_file = img_list[img_seq]

        input_image = Image.open(image_file)

        transform = transforms.Compose([transforms.Resize(224),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])

        tensor_image = transform(input_image)
        crop_image = input_image.resize((224, 224))

        attentions = []  # list, ele: attns [2, 12, 197, 197]
        for path in [pickle_path1, pickle_path2, pickle_path3]:
            with open(path, 'rb') as f:
                attentions = pickle.load(f)
                attentions = [ele.cpu() for ele in attentions]

            process_attentions = visualize_attention(tensor_image, 16, class_idx)
            # heat_map = np.array(crop_image) * np.mean(process_attentions[0])
            plot_attention(crop_image, img_seq)
