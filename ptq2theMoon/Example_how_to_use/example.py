from ptq2theMoon.ptqTransformerBased import PTQForTransformerBased


# The calibration data path
calib = r'E:\semester2\5005B\code\data\ImageNet12_1k\ILSVRC2012_img_calib'
# The validation data path
val = r'E:\semester2\5005B\code\data\ImageNet12_1k\ILSVRC2012_img_val_tiny'
hook_dir = r'E:\semester2\5005B\code\data\ImageNet12_1k\sample_for_visual'
# Your dir path to store profiler result that can be visualized by tensorboard
profiler_path = 'logs'
# pickle path, to store intermediate activations in picle files
pickle_path = r'D:\python_code\ppq2\pickle'

ptq_tool = PTQForTransformerBased(calibDir=calib)

# model name, choose in ['vit_base_patch16_224', 'vit_large_patch16_224', 'mobile_vit_xx_small', 'mobile_vit_xx_small_224']
model_name = 'deit_tiny_patch16_224'

if __name__ == '__main__':
    # post training quantization
    quantized_graph = ptq_tool.ptq(batchsize=2, model_name=model_name)

    # layer hooking, to extract intermediate activations in pickle files, recommend that data for hooking less that 400 images
    # ptq_tool.hook_layer(graph=quantized_graph, val_dir=hook_dir, batchsize=2, pickle_path=pickle_path)

    # show profiler table of different operation
    # ptq_tool.profiler_table(out_path=profiler_path, model_name=model_name, graph=None)
    # ptq_tool.profiler_table(out_path=profiler_path, graph=quantized_graph, model_name=None)

    # layerwise report by ppq
    report = ptq_tool.report('layerwise',  quantized_graph)

    # perform evaluation
    eval_report = ptq_tool.evaluation(val_dir=val, graph=quantized_graph, batchsize=2)

    # attention map visualization, you can change ViTQuantizer, ensure that u have generated 3 types of pickles before
    # u use this magical function
    '''
    our protocol in use:
    
    '''
    ptq_tool.attn_map_visual(pickle_path=pickle_path, img_idx=0, data_path=hook_dir)

'''
    ablation_study: vit(or m-vit)
    vit-fp: * Prec@1 75.850 Prec@5 92.990 Inference_time 0.014 avg_loss 0.4833
    vit-vanilla: * Prec@1 0.520 Prec@5 2.270 Inference_time 0.197 avg_loss nan
    vit-vanilla-gf (对比推理时间):  * Prec@1 0.550 Prec@5 1.810 Inference_time 0.123 avg_loss 3.4037
    vit-normC:  * Prec@1 64.060 Prec@5 84.960 Inference_time 0.287 avg_loss 0.7864
    vit-matmulC: 不需要了，没有norm一定差到底
    vit-normC-matmulC(fc2): * Prec@1 71.500 Prec@5 90.180 Inference_time 0.451 avg_loss 0.5982
    vit-normC-matmulC: * Prec@1 72.000 Prec@5 90.230 Inference_time 0.849 avg_loss 0.5920
    vit-normC-matmulC(fc1): * Prec@1 64.260 Prec@5 85.090 Inference_time 0.383 avg_loss 0.7821
    vit-normC-matmulC(fc2,0-5): * Prec@1 70.900 Prec@5 89.720 Inference_time 0.368 avg_loss 0.6195
    vit-normC-matmul(qkv): * Prec@1 64.290 Prec@5 84.930 Inference_time 0.363 avg_loss 0.7832
    vit-normC-matmul(proj): * Prec@1 64.420 Prec@5 85.090 Inference_time 0.328 avg_loss 0.7836
    vit-normC-matmulC(fc2,5-6): * Prec@1 70.970 Prec@5 89.580 Inference_time 0.288 avg_loss 0.6196
    vit-normC-matmulC(all except fc2):  * Prec@1 64.530 Prec@5 85.270 Inference_time 0.562 avg_loss 0.7787

    deit-tiny-fp:      * Prec@1 72.250 Prec@5 91.060 Inference_time 0.010 avg_loss 0.6108
    deit-tiny-quant:   * Prec@1 71.260 Prec@5 90.420 Inference_time 0.149 avg_loss 0.6440
    deit-small-fp:     * Prec@1 79.470 Prec@5 95.050 Inference_time 0.009 avg_loss 0.4432
    deit-small-quant:  * Prec@1 78.680 Prec@5 94.550 Inference_time 0.223 avg_loss 0.4756
    deit-base-fp:      * Prec@1 80.770 Prec@5 95.220 Inference_time 0.313 avg_loss 0.4347
    deit-base-quant:   * Prec@1 81.710 Prec@5 95.640 Inference_time 0.015 avg_loss 0.4127
'''

