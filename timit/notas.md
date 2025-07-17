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

## Reconocedor simple de fonemas usando TIMIT
#### Preparación de los datos
  - A partir de los json determinar los pares (wav, transcripción) del train, valid y test datasets. Cada transcripción es una lista de fonos codificados. 
  - Luego usamos la técnica *shuffle global sort local* que consiste en:
    - **Shuffle global**: Desordenar el dataset globalmente en cada epoch. Esto se hace en pytorch seteando `suffle=True` en el Dataloader
    - **Sort local**: Ordenar las secuencias por longitud dentro de cada batch. y luego padearlas. Ambas cosas se hacen dentro de  la collate_fn. 
  - En el conjunto de validación y de test la parte de shuffle global se desactiva pero no la de sort local.
  - El bs de validación puede ser más grande que el de train ya que no hay que tocar el modelo.
  - Los labels que devuelve el json son codificados en enteros. Para eso se usa el método `load_phoneme_vocabulary` el cual genera dos diccionarios para codificar y decodificar.
  
### Bloque de convolución

#### La clase `torch.nn.Conv1d`
  - La entrada tiene dimensión $(N,CI,LI)$ donde $N$ es el batch size, $LI$ es la longitud de la señal en frames (la máxima si no son iguales y el resto del batch se lleva a la máxima completando con ceros). $CI$ es la cantidad de canales de entrada (por ejemplo el número de filtros en escala mel).
  - La salida tiene dimensión $(N,CO,LO)$. $LO$ es la longitud de salida en frames que podría ser distinta a $LI$ como resultado de la convolución y el polling. $CO$ es la cantidad de canales de salida. 
  - Cada kernel tiene  dimensión $K$, el conjunto de todos los kernel tendrá dimensión $(C0, CI, K)$. 
  - La salida $Y$ del canal $j$ para la entrada $X$ del batch $i$  vendrá dada por:
  $$
  Y(N_i,CO_{j}) = \text{bias}(CO_{j}) + \sum_{k=0}^{CI-1}\text{kernel}(CO_j,k)*X(N_i,k)
  $$  

Para la mayoría de los casos generales y para asegurar una buena generalización, la estrategia de "shuffle global, sort local dentro del batch" (es decir, DataLoader(..., shuffle=True, collate_fn=custom_collate_fn)) es la más recomendada. Permite la eficiencia del padding sin sacrificar la aleatoriedad necesaria para la generalización.
  
1. Desordenar (Shuffle) el Dataset Globalmente en Cada Época:

    Propósito: Generalización y evitar sobreajuste (overfitting). Si el modelo siempre ve los mismos datos en el mismo orden, podría aprender patrones específicos de ese orden en lugar de las características subyacentes de los datos. Al mezclar el orden de las muestras en cada época, el modelo ve diferentes combinaciones de datos, lo que lo ayuda a aprender de manera más robusta y a generalizar mejor a datos no vistos.

    Implementación en PyTorch: Esto se logra configurando shuffle=True en el DataLoader. El DataLoader (o su Sampler subyacente) se encarga de barajar los índices de las muestras del Dataset al inicio de cada época.

2. Ordenar las Secuencias por Longitud Dentro de Cada Lote (Batch):

    Propósito: Eficiencia computacional para RNNs. Como ya discutimos, las RNNs (especialmente con pack_padded_sequence) funcionan de manera mucho más eficiente cuando las secuencias dentro de un lote están ordenadas por longitud (generalmente de forma descendente). Esto minimiza la cantidad de cálculos sobre el "padding" (relleno) y optimiza el uso de la memoria de la GPU.

    Implementación en PyTorch: Esto se realiza dentro de la función collate_fn que le pasas al DataLoader. La collate_fn recibe una lista de muestras individuales que el DataLoader ha seleccionado para el lote actual (después del shuffle global) y es responsable de agruparlas, ordenarlas y acolcharlas.

3. Acolchar (Pad) las Secuencias Ordenadas:

    Propósito: Crear tensores rectangulares. Una vez que las secuencias están ordenadas dentro del lote, se rellenan con ceros (o un valor específico) para que todas tengan la misma longitud (la del elemento más largo del lote).

    Implementación en PyTorch: También se realiza dentro de la collate_fn utilizando torch.nn.utils.rnn.pad_sequence.


## Pseudo código de `core.py`
```python
def fit():
  on_fit_start() # Hace init_optimizer()
  for epoch in epoch_counter:

    #----_fit_train()-------
    on_stage_start() 
    for batch in train_set:
      #---fit_batch()-----
      on_fit_batch_start() # No se usa
      compute_forward()
      compute_backward()
      on_fit_batch_end() # No se usa
      #-------------------
    on_stage_end() 
    #------------------------

    #----_fit_valid()--------
    on_stage_start()
    for batch in valid_set:
      #--evaluate_batch()--
      compute_forward()
      compute_backward()
      #--------------------
    on_stage(end) 

def evaluate():
  on_evaluate_start() # Recobra el mejor modelo para evaluar
  on_stage_start() 
  for batch in test_set:
    #--evaluate_batch--
    compute_forward()
    compute_backward()
    #------------------
  on_state_end() 
```


# Pseudo código de un programa típico

  - Inicializar el modelo (la red neuronal)
  - Inicializar la función de pérdida, eg. `torch.nn.CrossEntropyLoss`
  - Inicializar el optimizador, eg. `torch.optim.SGD(modelo.parameters(), lr=0.01, momentum=0.9)`
  - Inicializar el scheduler, eg. `torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999),...)` 

  - para cada epoch
    - train()
    - test()
    - actualizar lr: scheduler.step()

---

# Pseudo código de un programa típico

  - train()
    - para cada minibatch:
      - Poner a cero los gradientes de los parámetros: optimizer.zero_grad()
      - Paso forward. Determinar la salida de un minibatch de entrada output = model(data)
      - Calcular la pérdida de un minibatch loss = funperdida(output, target), es decir: $\mbox{Loss}(y,d;W)$
      - Calcular el gradiente de un minibatch respecto de los parámetros loss.backward(), es decir: $\nabla_W\mbox{Loss}(W)$
      - Actualizar los parámetros: optimizer.step(), es decir: $W = W -\eta\nabla_W\mbox{Loss}(W)$

---

# Pseudo código de un programa típico
  - test()
    - para todos los datos de test:
        - Determinar la salida de los datos de test: output = model(data)
        - Encontrar el índice que da la máxima salida: pred = output.argmax(dim=1, keepdim=True)
        - Compararlo con el target. Si es igual es correcto sino incorrecto.
    - determinar el accuracy en base al porcentaje de correctos.    

