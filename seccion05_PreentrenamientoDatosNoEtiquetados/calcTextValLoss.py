import torch

def calc_loss_batch(input_batch, taget_batch, model, device):
    input_batch, target_batch = input_batch.to(device), taget_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)                 #Iterativo sobre el numero de lotes si no se especifica un numero fijo de lotes           
    else:
        num_batches = min(num_batches, len(data_loader))          #reducir el numero de lotes para que coicida con el numero total de lotes en el cargador de datos

    for i, (input_batch, target_batch) in enumerate(data_loader):   
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()                           #suma de perdidas por cada lote
        else:
            break
    return total_loss / num_batches     #promedio de la perdida en todos los lotes