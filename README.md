# Conditional-DRAGAN
Conditional DRAGAN(cGAN based) code(& model) repo.
- Generator 
  4-layer-512-ReluMLP(last layer is tanh)
- Discriminator 
  4-layer-512-ReluMLP.

NOTE: Not include any normalization.

Question is always welcome, Pls add some issue.
## conditional method
Its cGAN.
Concat Noise z and Label c(onehot), then linear layer(Fully-Coneccted Layer) handle it.

## Train
```
python cdragan -fmnist --gpu 0 --epoch 100 --out result
```

## Label
| Label | Description |
| --- | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## Generated Images
(model is gen_iter_18000.npz)
- T-Shirt/top

![T-shirt/top(label:0)](./vis-preview/image00000000.png)

- Trouser

![Trouser(label:1)](./vis-preview/image00000001.png)

- pullover

![pullover(label:2)](./vis-preview/image00000002.png)

- Dress

![Dress(label:3)](./vis-preview/image00000003.png)

- Coat

![Coat(label:4)](./vis-preview/image00000004.png)

- Sandal

![Sandal(label:5)](./vis-preview/image00000005.png)


- Shirt

![Shirt(label:6)](./vis-preview/image00000006.png)

- Sneaker

![Sneaker(label:7)](./vis-preview/image00000007.png)


- Bag

![Bag(label:8)](./vis-preview/image00000008.png)


- Ankle boot

![Ankle boot(label:9)](./vis-preview/image00000009.png)

# LICENSE
This materials provided as MIT License(Please see `LICENSE` file).
