# ERA-SESSION27 - Phi2 Model Finetuning with QLoRA on OpenAssistant Conversations Dataset (OASST1)

🤗[**Space Link**](https://huggingface.co/spaces/RaviNaik/ERA-SESSION27) 

This is an implementation of [Phi2](https://huggingface.co/microsoft/phi-2) model finetuning using QLoRA stratergy on [OpenAssistant Conversations Dataset (OASST1)](https://huggingface.co/datasets/OpenAssistant/oasst1) 
             
Dataset used to finetune: [OpenAssistant Conversations Dataset (OASST1)](https://huggingface.co/datasets/OpenAssistant/oasst1) 
ChatML modified OSST Dataset: [RaviNaik/oasst1-chatml](https://huggingface.co/datasets/RaviNaik/oasst1-chatml) 
Finetuned Model: [RaviNaik/Phi2-Osst](https://huggingface.co/RaviNaik/Phi2-Osst) 

### Tasks:
1. :heavy_check_mark: Use OpenAssistant dataset.
2. :heavy_check_mark: Finetune Microsoft Phi2 model.
3. :heavy_check_mark: Use QLoRA stratergy.
4. :heavy_check_mark: Create an App on HF space using finetuned model.

## Phi2 Model Description:
```python
PhiForCausalLM(
  (transformer): PhiModel(
    (embd): Embedding(
      (wte): Embedding(51200, 2560)
      (drop): Dropout(p=0.0, inplace=False)
    )
    (h): ModuleList(
      (0-31): 32 x ParallelBlock(
        (ln): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
        (resid_dropout): Dropout(p=0.1, inplace=False)
        (mixer): MHA(
          (rotary_emb): RotaryEmbedding()
          (Wqkv): Linear4bit(in_features=2560, out_features=7680, bias=True)
          (out_proj): Linear4bit(in_features=2560, out_features=2560, bias=True)
          (inner_attn): SelfAttention(
            (drop): Dropout(p=0.0, inplace=False)
          )
          (inner_cross_attn): CrossAttention(
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (mlp): MLP(
          (fc1): Linear4bit(in_features=2560, out_features=10240, bias=True)
          (fc2): Linear4bit(in_features=10240, out_features=2560, bias=True)
          (act): NewGELUActivation()
        )
      )
    )
  )
  (lm_head): CausalLMHead(
    (ln): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
    (linear): Linear(in_features=2560, out_features=51200, bias=True)
  )
  (loss): CausalLMLoss(
    (loss_fct): CrossEntropyLoss()
  )
)
```
## Training Loss Curve:
![image](https://github.com/RaviNaik/ERA-SESSION27/assets/23289802/b477dd79-acab-48d2-aca7-39baa80dfb5b)
### Training Output
```python
TrainOutput(global_step=500, training_loss=1.4746462078094482, metrics={'train_runtime': 4307.6684, 'train_samples_per_second': 3.714, 'train_steps_per_second': 0.116, 'total_flos': 6.667526640623616e+16, 'train_loss': 1.4746462078094482, 'epoch': 1.62})
```
### Loss vs Steps Logs
![image](https://github.com/RaviNaik/ERA-SESSION27/assets/23289802/f305c4e7-c64d-4501-9b60-ae8f9a266349)

## Sample Results:
![image](https://github.com/RaviNaik/ERA-SESSION27/assets/23289802/e76a1f9c-24a4-40ac-b62a-291eacf1e3de)
![image](https://github.com/RaviNaik/ERA-SESSION27/assets/23289802/72278fa6-6e2e-49ea-8f97-78e3eddff8ae)

## Gradio UI:
![image](https://github.com/RaviNaik/ERA-SESSION27/assets/23289802/4fe7e106-7616-408b-8132-644567f8d0bb)



