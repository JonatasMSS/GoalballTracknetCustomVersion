# Cálculo dos centroides e do erro médio

Esta documentação descreve a normalização das saídas do TrackNet V1 e do YOLO para uma coordenada `(x, y)` e o cálculo do erro médio de posição utilizado no notebook [`tracknetv1_yolo.ipynb`](./tracknetv1_yolo.ipynb).

## 1. Objetivo da normalização

Os modelos produzem saídas diferentes:

- o TrackNet V1 produz um heatmap de `640 × 360`;
- o YOLO produz bounding boxes;
- o ground truth fornece diretamente uma coordenada `(x_gt, y_gt)`.

Para que a comparação seja possível, a saída de cada modelo é transformada em um único ponto na resolução original do frame:

```text
TrackNet → centroide ponderado do heatmap → (x, y)
YOLO     → centro geométrico da box       → (x, y)
```

## 2. Coordenada produzida pelo TrackNet V1

### 2.1 Transformação da saída em heatmap

Durante a inferência, a saída do TrackNet é processada assim:

```python
class_map = tracknet(inputs).argmax(dim=1)[0].detach().cpu().numpy()
```

Explicação da linha:

1. `tracknet(inputs)` executa a inferência. A saída possui 256 canais para cada posição do mapa, representando intensidades possíveis entre 0 e 255.
2. `.argmax(dim=1)` seleciona, para cada pixel, o canal com o maior valor. O índice desse canal passa a ser a intensidade predita daquele pixel.
3. `[0]` seleciona o primeiro elemento do batch, pois a inferência é executada com uma amostra por vez.
4. `.detach()` separa o resultado do grafo de cálculo do PyTorch.
5. `.cpu()` transfere o resultado da GPU para a memória principal.
6. `.numpy()` converte o tensor em um array NumPy.

O resultado é um vetor contendo `360 × 640` valores entre 0 e 255.

Na função `tracknet_centroid`, esse vetor é reorganizado como uma imagem:

```python
heatmap = np.asarray(class_map, dtype=np.uint8).reshape(
    TRACKNET_SIZE[1], TRACKNET_SIZE[0]
)
```

- `np.asarray(...)` garante que a saída seja um array NumPy.
- `dtype=np.uint8` representa corretamente valores inteiros entre 0 e 255.
- `TRACKNET_SIZE[1]` corresponde à altura `360`.
- `TRACKNET_SIZE[0]` corresponde à largura `640`.
- `reshape(...)` transforma o vetor em um heatmap de formato `(360, 640)`.

### 2.2 Seleção dos pixels candidatos

```python
active = (heatmap >= TRACKNET_HEATMAP_THRESHOLD).astype(np.uint8)
```

O limiar atual é:

```python
TRACKNET_HEATMAP_THRESHOLD = 127
```

A comparação produz uma máscara binária:

```text
heatmap < 127  → 0 → fundo
heatmap ≥ 127  → 1 → possível região da bola
```

O método `astype(np.uint8)` converte os valores booleanos `True` e `False` em `1` e `0`, permitindo que a máscara seja processada pelo OpenCV.

### 2.3 Separação das regiões do heatmap

```python
count, component_map, stats, _ = cv2.connectedComponentsWithStats(active, 8)
```

Essa linha divide os pixels ativos em componentes conexas:

- `count` é a quantidade de componentes encontradas, incluindo o fundo;
- `component_map` informa a qual componente cada pixel pertence;
- `stats` contém área e limites de cada componente;
- `_` recebe os centroides geométricos calculados pelo OpenCV, que não são utilizados porque o notebook calcula um centroide ponderado;
- o valor `8` habilita conectividade pelas laterais e pelas diagonais.

Depois são mantidas somente as componentes com a área mínima configurada:

```python
candidates = [
    component_id for component_id in range(1, count)
    if stats[component_id, cv2.CC_STAT_AREA] >= TRACKNET_MIN_COMPONENT_AREA
]
```

- `range(1, count)` começa em 1 porque a componente 0 é o fundo;
- `cv2.CC_STAT_AREA` recupera a quantidade de pixels da componente;
- atualmente `TRACKNET_MIN_COMPONENT_AREA = 1`, portanto qualquer componente com pelo menos um pixel pode ser considerada.

Se nenhuma região passar pelo filtro, não existe detecção:

```python
if not candidates:
    return None, None
```

O primeiro `None` representa a ausência da coordenada e o segundo representa a ausência de uma medida de confiança.

### 2.4 Escolha da região mais forte

Quando o heatmap contém várias regiões candidatas, é escolhida aquela cuja soma de intensidades é maior:

```python
best_component = max(
    candidates,
    key=lambda component_id: float(
        heatmap[component_map == component_id].sum()
    ),
)
```

Para cada componente `C`, é calculado:

$$
S_C = \sum_{i \in C} H_i
$$

onde `H_i` é a intensidade do pixel `i` no heatmap. A componente selecionada é:

$$
C_{\mathrm{melhor}} = \underset{C}{\arg\max}\; S_C
$$

Isso significa que a escolha considera simultaneamente a área da região e a força de suas respostas. Não é necessariamente escolhida a região com maior área nem apenas aquela com o pixel máximo.

### 2.5 Cálculo do centroide ponderado

Primeiro é criada uma máscara contendo somente a componente selecionada:

```python
mask = component_map == best_component
```

Em seguida são extraídas as coordenadas de seus pixels:

```python
ys, xs = np.nonzero(mask)
```

- `xs` contém as coordenadas horizontais;
- `ys` contém as coordenadas verticais.

As intensidades originais do heatmap são usadas como pesos:

```python
weights = heatmap[mask].astype(np.float64)
```

O centroide horizontal é calculado por:

```python
x_heatmap = float(np.average(xs, weights=weights))
```

Matematicamente:

$$
x_c = \frac{\displaystyle\sum_i x_i H_i}{\displaystyle\sum_i H_i}
$$

O centroide vertical é calculado por:

```python
y_heatmap = float(np.average(ys, weights=weights))
```

Matematicamente:

$$
y_c = \frac{\displaystyle\sum_i y_i H_i}{\displaystyle\sum_i H_i}
$$

Assim, pixels com maior intensidade exercem maior influência sobre o centroide. Isso é diferente de uma média geométrica simples, na qual todos os pixels teriam o mesmo peso.

### 2.6 Conversão para a resolução original

O centroide calculado ainda está no espaço `640 × 360`. Ele é convertido para a resolução original do frame:

```python
x = x_heatmap * original_width / TRACKNET_SIZE[0]
y = y_heatmap * original_height / TRACKNET_SIZE[1]
```

As fórmulas são:

$$
x_{\mathrm{original}} = x_c \cdot \frac{W_{\mathrm{original}}}{640}
$$

$$
y_{\mathrm{original}} = y_c \cdot \frac{H_{\mathrm{original}}}{360}
$$

Nos frames `1920 × 1080` do GSS_6, os dois fatores são iguais a 3. Por exemplo:

```text
Centroide no heatmap: (400, 160)
Centroide original:   (1200, 480)
```

A função também calcula uma medida auxiliar:

```python
confidence = float(weights.mean() / 255.0)
```

Ela corresponde à intensidade média da componente dividida por 255. Essa medida fica entre 0 e 1, mas **não é uma probabilidade calibrada e não deve ser comparada diretamente com a confiança do YOLO**.

Finalmente, a função retorna:

```python
return (x, y), confidence
```

## 3. Coordenada produzida pelo YOLO

O YOLO retorna bounding boxes. A função começa recuperando as boxes:

```python
boxes = result.boxes
```

Quando não existe nenhuma box, não há detecção:

```python
if boxes is None or len(boxes) == 0:
    return None, None
```

As classes e confianças são convertidas para arrays NumPy:

```python
classes = boxes.cls.detach().cpu().numpy().astype(int)
confidences = boxes.conf.detach().cpu().numpy()
```

São mantidas somente as detecções da classe da bola com confiança estritamente maior que `YOLO_CONFIDENCE`:

```python
candidates = np.flatnonzero(
    (classes == YOLO_BALL_CLASS_ID) & (confidences > YOLO_CONFIDENCE)
)
```

Como `YOLO_BALL_CLASS_ID = 0` e `YOLO_CONFIDENCE = 0.75`, as boxes de outras classes e as detecções com confiança menor ou igual a `0.75` são ignoradas.

Se nenhuma box for da classe bola, a função retorna ausência de detecção:

```python
if len(candidates) == 0:
    return None, None
```

Quando existem várias boxes de bola e o ground truth está visível, são calculados os centros das boxes e é escolhida aquela com a menor distância Euclidiana até o ground truth:

```python
coordinates = boxes.xyxy.detach().cpu().numpy()
centers = (coordinates[:, :2] + coordinates[:, 2:]) / 2.0
distances = np.linalg.norm(
    centers[candidates] - np.asarray(ground_truth), axis=1
)
best = int(candidates[np.argmin(distances)])
```

Quando a bola não está visível no ground truth, não existe uma coordenada verdadeira utilizável. Nesse caso, a função mantém como fallback a candidata de maior confiança. Essa escolha por distância é adequada apenas para análise retrospectiva do benchmark, pois uma inferência real não dispõe do ground truth.

Depois são obtidos os cantos da box selecionada:

```python
x1, y1, x2, y2 = boxes.xyxy[best].detach().cpu().numpy()
```

As coordenadas representam:

```text
(x1, y1) → canto superior esquerdo
(x2, y2) → canto inferior direito
```

O centro geométrico da box é calculado por:

```python
point = (
    float((x1 + x2) / 2.0),
    float((y1 + y2) / 2.0),
)
```

Matematicamente:

$$
x_c = \frac{x_1 + x_2}{2}
$$

$$
y_c = \frac{y_1 + y_2}{2}
$$

Por exemplo, para uma box `(1200, 450, 1260, 510)`:

```text
x = (1200 + 1260) / 2 = 1230
y = (450 + 510) / 2 = 480
```

O ponto final é `(1230, 480)`. As coordenadas fornecidas pelo Ultralytics já estão na resolução original do frame, portanto não precisam do redimensionamento aplicado ao TrackNet.

A função retorna o ponto e a confiança original da box:

```python
return point, float(confidences[best])
```

## 4. Cálculo do erro de posição

O erro somente pode ser calculado quando:

- a bola está visível no ground truth;
- o modelo produziu uma coordenada.

Essa condição é representada por:

```python
if point is not None and sample.gt_visible:
```

O erro em pixels é a distância Euclidiana entre o centroide predito e a coordenada real:

```python
distance_px = float(
    np.hypot(
        point[0] - sample.gt_x,
        point[1] - sample.gt_y,
    )
)
```

Matematicamente:

$$
d_i = \sqrt{
    \left(x_{\mathrm{pred},i} - x_{\mathrm{gt},i}\right)^2
    +
    \left(y_{\mathrm{pred},i} - y_{\mathrm{gt},i}\right)^2
}
$$

Por exemplo:

```text
Ground truth: (1200, 480)
Predição:     (1209, 492)
```

Então:

$$
\begin{aligned}
d
&= \sqrt{(1209 - 1200)^2 + (492 - 480)^2} \\
&= \sqrt{9^2 + 12^2} \\
&= 15\;\mathrm{px}
\end{aligned}
$$

Cada distância válida é armazenada:

```python
errors_px.append(distance_px)
```

### 4.1 Erro normalizado

Também é calculado um erro independente da resolução do frame:

```python
height, width = sample_shapes[sample.sample_id]
normalized_error = distance_px / float(np.hypot(width, height))
```

O denominador é a diagonal da imagem:

$$
D = \sqrt{W^2 + H^2}
$$

O erro normalizado é:

$$
e_i = \frac{d_i}{D_i}
$$

Para uma imagem `1920 × 1080`, a diagonal é aproximadamente `2202,91 px`. Um erro de 30 pixels corresponde a:

$$
\frac{30}{2202{,}91} \approx 0{,}0136 = 1{,}36\%
$$

O valor é armazenado com:

```python
normalized_errors.append(normalized_error)
```

### 4.2 Erro médio

Depois de processar todas as amostras, o erro médio em pixels é calculado por:

```python
'mean_error_px': float(np.mean(errors_px)) if errors_px else float('nan')
```

Matematicamente:

$$
\bar{d} = \frac{1}{N}\sum_{i=1}^{N} d_i
$$

onde `N` é a quantidade de frames nos quais a bola está visível e o modelo produziu uma coordenada.

Se nenhuma distância puder ser calculada, o resultado será `NaN`, evitando que a ausência de dados seja apresentada incorretamente como erro zero.

O erro normalizado médio é calculado de forma equivalente:

```python
'mean_normalized_error': (
    float(np.mean(normalized_errors))
    if normalized_errors
    else float('nan')
)
```

Matematicamente:

$$
\bar{e} = \frac{1}{N}\sum_{i=1}^{N}\frac{d_i}{D_i}
$$

## 5. Relação entre erro e verdadeiro positivo

O erro de posição também é usado para decidir se a localização corresponde ao ground truth:

```python
elif distance_px <= MATCH_RADIUS_PX:
    status = 'TP'
    counts['TP'] += 1
else:
    status = 'FP+FN'
    counts['FP'] += 1
    counts['FN'] += 1
```

Com `MATCH_RADIUS_PX = 30.0`:

- distância menor ou igual a 30 px: a localização é considerada um verdadeiro positivo;
- distância maior que 30 px: a predição não corresponde à bola anotada.

Uma localização fora do raio gera `FP+FN` porque existem simultaneamente:

- uma predição em uma posição incorreta, contabilizada como falso positivo;
- uma bola real sem uma predição correspondente, contabilizada como falso negativo.

## 6. Observação sobre o significado do erro médio

O erro médio representa a qualidade espacial **somente das amostras em que o modelo produziu algum ponto e a bola estava visível**. Frames sem detecção não recebem uma distância artificial.

Por esse motivo, o erro médio deve ser interpretado junto com a cobertura de detecção. Um modelo pode apresentar erro médio baixo por localizar muito bem poucos frames, mas ainda possuir muitos falsos negativos. A cobertura utilizada no notebook é:

```python
'visible_coverage': safe_divide(visible_with_prediction, visible_count)
```

Portanto, na comparação final:

- erro médio menor indica melhor precisão espacial nas localizações realizadas;
- cobertura, Recall e F1 indicam se o modelo consegue realizar essas localizações com frequência.
