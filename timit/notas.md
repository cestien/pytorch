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

Notas para poder implementar el código en pytorch.
```python
def comming_soon():
    pass
```
