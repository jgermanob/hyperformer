from functools import partial
from adapters.lora_modelling import MetaLinearLoraController
from third_party.models.modeling_t5 import T5ForConditionalGeneration


def freeze_model_params(model: T5ForConditionalGeneration):
    for param in model.parameters():
        param.requires_grad = False
    
    return model

def add_lora(model: T5ForConditionalGeneration):
    assign_lora = partial(MetaLinearLoraController)
    for block in model.encoder.block:
        self_attention = block.layer[0].SelfAttention
        
        self_attention.q = assign_lora(self_attention.q)
        self_attention.v = assign_lora(self_attention.v)

    for block in model.decoder.block:
        self_attention = block.layer[0].SelfAttention
        self_attention.q = assign_lora(self_attention.q)
        self_attention.v = assign_lora(self_attention.v)

        #cross_atention = block.layer[1].EncDecAttention
        #cross_atention.q = assign_lora(cross_atention.q)
        #cross_atention.v = assign_lora(cross_atention.v)
        
    return model

def get_summary(model: T5ForConditionalGeneration):
    total_params = 0
    trainbale_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainbale_params += param.numel()
            print(f'Trainable param: {name}')
    print(f'Total parameters: {total_params}')
    print(f'trainable parameters: {trainbale_params}')
