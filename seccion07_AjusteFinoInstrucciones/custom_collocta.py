import torch
def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"):
    batchs_lengths = [len(item) for item in batch]
    batch_max_length = max(batchs_lengths) + 1
    
    inputs_lst, target_lst = [], []

    for item in batch:
        new_item = item.copy()  #copiar para evitar pisar el batch original
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1]) #tensor de entradas (truncado)
        targets = torch.tensor(padded[1:]) #tensor de salidas (shift +1)

        mask = targets == pad_token_id  #Reemplazar  todos  los  tokens  de  relleno  excepto  el  primero  en  los  objetivos  por  ignore_index
        indices = torch.nonzero(mask).squeeze()  #devuelve las posiciones (índices) donde mask es True y lo aplana un poco
        if indices.numel() > 1: #si el número de elementos de un tensor es mayor que 1
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]                  #Truncar  opcionalmente  a  la  longitud  máxima  de  secuencia
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        target_lst.append(targets)
    
    inputs_tensor = torch.stack(inputs_lst).to(device)
    target_tensor = torch.stack(target_lst).to(device)
    return inputs_tensor, target_tensor