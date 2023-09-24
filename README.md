# PTQ2TheMoon_VisionTransformer
## Description
The package for Posting training quantization on Vision Transformer, based on Sense time's framework for quantization:ppq  
Just run example.py to start the quantization pipeline  
## Tool Version
onnx 1.14.0  
torch 2.0.1  
protobuf 3.20.3  
## Diagram  
The diagram for implementing the ppq framework on quantization
![diagram](img/diagram.png)
## Graph fusion for quantized model  
Graph fusion refers to the merging of multiple computational operators into one larger computational operator. The main purpose of graph fusion is to reduce the overhead of computation and memory access to improve the inference speed and resource utilization of quantized models.  

**Fused Operators**
Fused Operators | Base Operators
---- | ----
LayerNorm | ReduceMean(1) --- Sub(2) --- Pow(3) --- ReduceMean(4) --- Add(5) --- Sqrt(6) --- Div(7) --- Mul(8) --- (Add)(9)
Gelu | Div(1) --- Erf(2)--- Add(3) --- Mul(4) --- Mul(5)
PPQBiasFusedMatMul (MatMulAdd) | MatMul(1) --- Add(2)  

![graph fusion](img/graphFusion.png)  

## Properties for vanilla quantizers
Num of bits | Quant max value  | Quant min vlaue | Obeserver algorithm | Policy | Rounding Principle
 ---- | ----- | ------ | ------- | -------- | ---------
8  | 127 | -128 | 'percentile' for per-tensor | per-tensor & linear & symmetric | round half even

## Performing Channel-wise quantization on different blocks of Vision Transformer
![channel-wise quant](img/channel_wise_on_vit.png)  

## Result of customized quantizer
An interesting finding is that FC2 of Block 5 and Block 6 seem to have the largest per-tensor SNR error, while we have shown that some channels of FC2 in Block 5 and Block 6 have the largest channel range value among all situations. Therefore, we can think that the maximum channel range value will have a serious impact on quantization when it rises to a certain extent. In addition, this could happen because unlike other operators, FC2 inputs have a high dimensional feature activated by GELU. The dimension of the feature vector in the MLP block in ViT corresponds to the channels of the image feature, which may cause the channel range value of the FC2 activation value to be different (larger) from other fused MatMul operators.
![result of customized quantizer](img/customized_quantized.png)


