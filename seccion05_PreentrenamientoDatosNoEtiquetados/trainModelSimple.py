import torch 
from seccion04_ImplementacionGPTGeneracionTexto.generateTextSimple import generate_text_simple
from calcTextValLoss import calc_loss_loader, calc_loss_batch

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) 
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss           

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  
    model.train()

def generate(model, idx, max_new_tokens, context_size,
            temperature=1.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):                               #bucle for donde obtener logits y centrarse soli en el útimo paso de tiempo
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None:                                     #filtar logits con muestreo top_k
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )

        if temperature > 0.0:                                         #aplicar escalado de temperatura
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
        else:                                                     #selección condiciosa del siguiente token
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:                                    #detener la generación anticipada si se encuentra untoken de fin de secuencia 
            break   
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):

    train_losses, val_losses, track_tokens_seen = [], [], [] #inicializar listas para rastrear pérdidas y tokens vistos
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs): #inicializar el ciclo de entrenamiento principal
        model.train()
        
        for input_batch, target_batch in train_loader:
            #input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            optimizer.zero_grad() #restablecer los gradientes de pérdida de la iteración del lote anterior
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() #calcular los gradientes de pérdida
            optimizer.step() #actualizar los pesos usando descenso de gradiente
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:                     
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        generate_and_print_sample(                                #imprimir ejemplo de texto
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen

