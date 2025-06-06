<link href="/home/cestien/memoria/template/nota_template/miestilo.css" rel="stylesheet"></link>

# Pseudo código y comentarios

## Recordar la secuencia de `fit()` (train)
Dado:
  - Un conjunto de datos de train $(X,Y)$, separados en batches de tamaño bs
  - Una función vectorial paramétrica de parámetros $W$ (red neuronal) $p = F(x;W)$.  
  - Una función escalar llamada loss $L(W) = \text{Loss}_W(P,Y)$ que mide la distancia entre $P$ e $Y$.
  
  - Para cada epoch:
    - Para cada batch:
      - Paso forward: Calcular las salidas de ese minibatch  $p_i = F(x_i;W)$ con $i=1:bs$
      - Paso de cálculo de loss: Calcular el valor de la  loss para ese minibatch 
        $$
        L(W) = \frac{1}{bs}\sum_i \text{Loss}(p_i,y_i))
        $$
      - Paso backward: Calcular el gradiente de la loss para ese minibatch $\nabla_W L(W)$ 
      - Paso de optimización: Cambiar los parámetros $W$ por los los nuevos parámetros $W -= \eta\nabla_W L(W)$
      - Resetear los gradientes acumulados $\nabla_W F(W) = 0$ 

### Ejemplo para el caso de MNIST
```python
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn,optim,tensor
from torch.utils.data import TensorDataset, DataLoader
import pickle, gzip

dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Entrenamiento y validación
def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        # Entrenamiento
        model.train()
        for xb,yb in train_dl: 
            pb = model(xb)
            loss = loss_func(pb,yb)  
            loss.backward()
            opt.step()
            opt.zero_grad()
        # Validación
        model.eval()
        losses = list()
        nums = list()
        with torch.no_grad():
            for xb,yb in valid_dl:
                pb = model(xb)
                loss  = loss_func(pb,yb)
                num = len(xb)
                losses.append(loss.item())
                nums.append(num)
        val_loss = np.sum(np.multiply(losses,nums)) / np.sum(nums)
        print(epoch, val_loss)

# Modifica las dimensiones de las features x
class Convert_x():
    def __init__(self, dl):
        self.dl = dl
    def __len__(self): return len(self.dl)
    def __iter__(self):
        # batches = iter(self.dl)
        for b in self.dl:
            xp = b[0].view(-1,1,28,28).to(dev)
            yp = b[1].to(dev)
            yield xp, yp
            
# Convierte el dataset en grupos de batches
def get_dataloader(x,y,bs,shuffle):
    ds = TensorDataset(*map(tensor, (x,y)))
    dl = DataLoader(ds, batch_size=bs, shuffle=shuffle)
    dlp = Convert_x(dl)
    return dlp

# Aplana los datos de la feature x. La usa nn.Sequential
class Aplanar(nn.Module):
    def __init__(self):
        super().__init__() # Llamada al constructor de la clase padre
    def forward(self, x):
        return x.view(x.size(0),-1)

# Main
bs=64
lr=0.1
epochs=4
with gzip.open('data/mnist.pkl.gz', 'rb') as f:
    ((train_x, train_y), (valid_x, valid_y), _) = pickle.load(f, encoding='latin-1')
model = nn.Sequential(
    nn.Conv2d(1,  16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    Aplanar()
).to(dev)
loss_func = F.cross_entropy
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
train_dl = get_dataloader(train_x, train_y, bs,   shuffle=True)
valid_dl = get_dataloader(valid_x, valid_y, bs*2, shuffle=False )
fit(epochs, model, loss_func, opt, train_dl, valid_dl)
```
## Reconocedor simple de fonemas usando TIMIT
#### Preparación de los datos
  - A partir de los json determinar los pares (wav, transcripción) del train, valid y test datasets. Cada transcripción es una lista de fonos codificados. 
  - Ordenar los datasets por duración de mayor a menor
  - Usando dataloader separar los datos en batches con el correspondiente padding. Cuando se construye el conjunto de train hay que poner shuffle=True, de esta manera en cada epoch todos los datos de train se reordenan aleatoriamente antes de ser separados nuevamente en batches. El bs de validación puede ser más grande que el de train ya que no hay que tocar el modelo.