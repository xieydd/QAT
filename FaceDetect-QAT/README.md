<!--
 * @Author: xieydd
 * @since: 2020-06-09 11:20:14
 * @lastTime: 2020-07-02 09:41:55
 * @LastAuthor: Do not edit
 * @message: 
--> 
### FaceDetect-QAT



#### APIs
1. torch.quantization.fuse_modules: fuse conv+bn+relu/conv+relu/conv+bn
2. torch.quantization.prepare: insert observer for getting statistics of layer
3. torch.quantization.convert: quantization model
4. torch.quantization.get_default_config('fbgemm'): get best config of quantization for backend(x86 or device edge) Notice: fbgemm support per-channel quantization, qnnpack only support per-layer quantization
5. torch.quantization.prepare_qat: insert fake quantized to module

#### Plan
1. - [x] DataParallel Training  
2. - [x] pytorch inference time/result compare
3. - [ ] Support apex, Train and Test
4. - [ ] pytorch to onnx, onnx to ncnn
5. - [x] Half post-training Quantization
6. - [x] pytorch post-training Quantization
7. - [ ] Full post-training Quantization and compared
8. - [ ] No e2e qat train, fine-tune

#### Levels
1. Level1: 不需要数据和重新训练，方法具有普遍性是和所有模型，post-train quantization 即可
2. Level2: 需要数据做 Calibration 
3. Level3: 通用需要数据做 fine-tune
4. Level4: 需要额外数据，进行 fine-tune 并且针对特定模型

#### Summary 长边 320
| 方案 | Hard | Medium | Easy| WiderFace Speed
| ---- | ---- | ----  | ---- | ---- |
| FaceDetect| 0.31| 0.64| 0.78| |
| FaceDetect-QAT| 0.294| 0.612| 0.749| |
| FaceDetect-QAT-StatAssist-GradBoost-fbgemm| 0.31| 0.64| 0.759 | 0.076s|
| FaceDetect-QAT-StatAssist-GradBoost-qnnpack| 0.307| 0.634| 0.766 | 0.058s|
| FaceDetect-NCNN| 0.304| 0.642| 0.774| |

#### Record
```s
1. Training QAT: python train_quantization.py --resume=./weights_lightnn/model_249.pth -b 64 or change config.py param and python train.py --network=slim-qat 
2. Training normal Model:change config.py param and python train.py --network=slim
3. Test for own dataset: python3 detect_sample.py -m ./weights_lightnn/model_249.pth --network=slim-qat  --cpu
4. evaluate widerface: 
python test_widerface.py --network=slim -m slim_Final.pth
python test_widerface.py --network=slim-qat -m ./weight_lights/slim_Final.pth --cpu
cd widerface_evaluate && python3 evaluation.py -p ./widerface_txt/ -g ./ground_truth
```
#### Site
- [pytorch quantization blog](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)
- [StatAssist & GradBoost: A Study on Optimal INT8 Quantization-aware Training from Scratch](https://github.com/clovaai/StatAssist-GradBoost) 
