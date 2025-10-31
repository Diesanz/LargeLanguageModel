import torch
from calcLossClasification import calc_loss_batch, calc_loss_loader,calc_accuracy_loader
import matplotlib.pyplot as plt

def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    ax2 = ax1.twiny()   #segundo eje x para los ejemplos vistos
    ax2.plot(examples_seen, train_values, alpha=0)  
    ax2.set_xlabel("Examples seen")
    fig.tight_layout()                             #ajustar el diseño apra hacer espacio           
    plt.savefig(f"{label}-plot.pdf")
    plt.show()
    
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, tokenizer):
    #inicaliar listas para rastrear pérdidas y ejemplos vistos
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1
    #buvle de entrenamiento
    for epoch in range(num_epochs):
        model.train()                                             #modelo en modo entrenamiento
        for input_batch, target_batch in train_loader:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            optimizer.zero_grad()                                 #reestablecer los gradientes de pérdida de la iteración del lote anterior
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()                                       #calcular gradiente de pérdida
            optimizer.step()                                    #actualizar los pesos del modelo
            examples_seen += input_batch.shape[0]                 #ejemplos de seguimiento
            global_step += 1
                                                                    #paso de evaluación (opcional)
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                                                                    #calcular la precisión en cada época
        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter
        )
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
    return train_losses, val_losses, train_accs, val_accs, examples_seen