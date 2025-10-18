import torch
import torch.nn as nn
def generate_text_simple(model, idx, max_new_tokens, context_size): #idx  es  una  matriz  (lote,  n_tokens)  de  índices  en  el  contexto  actual 
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]                         #Recortar el contexto  actual  si  excede  el  tamaño  de  contexto  admitido
                                                                #ejemplo si context_size = 4 : el modelo solo puede ver los últimos 4 tokens para hacer su predicción
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]                                 #Enfocarse solo en el último paso de tiempi de modo que pase (batch, n_token, vocab_size) a (batch, vocab_size)
        probas = torch.softmax(logits, dim=-1)                    
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)     
        idx = torch.cat((idx, idx_next), dim=1)                   #Agregar el indice muestreado a ña secuencia de ejecucion
    return idx