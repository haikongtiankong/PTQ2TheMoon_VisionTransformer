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
### Properties for vanilla quantizers
Num of bits | Quant max value  | Quant min vlaue | Obeserver algorithm | Policy | Rounding Principle
 ---- | ----- | ------ | ------- | -------- | ---------
8  | 127 | -128 | 'percentile' for per-tensor | per-tensor & linear & symmetric | round half even
