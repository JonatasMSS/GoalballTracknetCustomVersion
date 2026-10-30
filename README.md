# Tracknet Goalball

Implementações Python/PyTorch de TrackNet V1 e V2 para rastrear a bola em vídeos de goalball. Os contratos de entrada, alvos e pesos são diferentes entre as versões.

## Ambiente e estrutura

Use um ambiente virtual e instale as dependências:

```powershell
pip install -r requirements.txt
```

`tracknet_v1/` e `tracknet_v2/` contêm modelo, dataset, treino, validação e inferência de cada versão. A V2 também contém loss, helpers e notebook. `dataset/` contém somente dados e instruções; testes ficam em `tests/` e resultados em `runs/`.

`torchsummary` é opcional para `TrackentV2.show_model_summary()`; o notebook não exige esse pacote. A V1 usa `math.dist`, sem dependência de SciPy.

## Treino por YAML

Edite [config.yaml](config.yaml) e execute:

```powershell
python train.py
python train.py --config "caminho com espaços/config.yaml"
```

`model` seleciona `tracknet_v1` ou `tracknet_v2`; `models` contém os perfis dessas versões. Somente o perfil ativo é validado e seu módulo de treino é importado. IDs de modelos desconhecidos são rejeitados mesmo quando inativos. Todos os campos do perfil ativo são obrigatórios; campos desconhecidos geram erro. O lançador chama `run_training(config)` com o dicionário desse perfil, sem adaptar argumentos de CLI.

| Perfil | Campos e tipos |
| --- | --- |
| Ambos | `batch_size`, `num_epochs`, `val_intervals`: inteiros positivos; `lr`: número positivo e finito; `device`: `auto`, `cpu` ou `cuda`; `exps_path`: caminho de diretório; `model_path`: arquivo existente ou `null` |
| V1 | `train_path`, `val_path`: diretórios explícitos; `steps_per_epoch`: inteiro positivo; `exp_id`: nome simples de diretório, sem separadores |
| V2 | `train_dataset_path`, `val_dataset_path`: diretórios explícitos; `logs_path`: caminho de diretório; `stepper_gamma`: número positivo e finito; `original_width`, `original_height`: inteiros positivos; `parallel`: booleano YAML |

Booleanos não são aceitos como inteiros. Caminhos relativos são resolvidos a partir da pasta do YAML; absolutos também são aceitos. Sem `--config`, o padrão é o `config.yaml` ao lado de `train.py`, independentemente da pasta de execução. Treino/validação precisam existir, ser distintos e produzir amostras. Saídas ocupadas por arquivos, inclusive em diretórios ancestrais, são rejeitadas antes do treino.

`device: auto` escolhe CUDA quando disponível e CPU nos demais casos; `cpu` força CPU e `cuda` falha cedo se indisponível. Na V2, `parallel: true` solicita DataParallel somente em CUDA com mais de uma GPU.

| Comportamento | V1 | V2 |
| --- | --- | --- |
| Validação em 10 épocas, intervalo 2 | Épocas 3, 5, 7 e 9; agenda legada, sem validação final obrigatória | Épocas 2, 4, 6, 8 e 10; inclui a última |
| Melhor checkpoint | Salva quando F1 melhora sobre zero | Salva o primeiro F1 finito, inclusive zero, e depois melhorias |
| Último checkpoint | Salva nas épocas de validação | Salva a cada época |
| Saídas padrão | `runs/tracknet_v1/default/model_{best,last}.pt`; logs em `plots/` | `runs/tracknet_v2/default/weights/{best,last}_model.pth`; logs em `runs/tracknet_v2/default/logs/` |

A V1 mantém a parada legada: `steps_per_epoch: N` permite até `N+1` batches. A V2 pondera a loss de época pelo número de amostras e rejeita `exps_path` com best/last já existente; escolha outro diretório para novo experimento.

`model_path` inicializa somente o `state_dict`. Otimizador, scheduler da V2, época e melhor métrica reiniciam; não há retomada completa. Pesos V1 e V2 não são intercambiáveis.

## Dados

Os quatro conjuntos `dataset/tracknet_v{1,2}/{train,val}/` são templates vazios. Preencha-os ou ajuste os caminhos do YAML. Veja a hierarquia em [dataset/README.md](dataset/README.md); [dataset/examples/](dataset/examples/) contém apenas um frame e seu label.

Na V2:

- A primeira linha de cada TXT exige exatamente `x y vis`: coordenadas finitas em pixels da resolução original e visibilidade inteira. `original_width`/`original_height` descrevem essa resolução (padrão 1920/1080). Labels ausentes, vazios ou inválidos geram erro; visibilidade zero é explícita. A validação considera visível apenas `vis == 1`.
- GTs são heatmaps em escala de cinza na resolução de entrada (padrão altura 360, largura 640), sem resize automático. O nome exato do frame é preferido; o fallback exige um único JPG/JPEG/PNG com mesmo nome-base. Ausência ou ambiguidade gera erro.
- Frames usam ordem natural numérica (`frame1`, `frame2`, `frame10`) com desempate pelo nome. Janelas não cruzam clips; lacunas numéricas são permitidas. Clips com menos de três frames não geram amostras. Leitura e dimensões dos GTs são verificadas no acesso à amostra.
- Entrada RGB, resize bilinear via PIL e normalização ImageNet; altura/largura precisam ser múltiplas positivas de oito.

A V1 preserva BGR dividido por 255 e ordem `[t,t-1,t-2]`, com calibração 640×360 e listagem sem ordenação explícita. Não transfira convenções de visibilidade ou coordenadas entre versões. Ambos exigem splits explícitos; confira independência por partida/clip e sobreposição real de frames.

A validação V2 agrega três saídas de janelas sobrepostas; a inferência usa apenas o canal atual. Localização fora da tolerância conta como FP sem FN adicional. Hough, limiar e tolerância precisam de calibração; a reorganização preservou essas regras.

## Inferência de vídeo

```powershell
python -m tracknet_v1.infer --video_paths entrada.mp4 --model_path runs/tracknet_v1/default/model_best.pt --output_path runs/tracknet_v1/inference/saida.mp4
python -m tracknet_v2.infer --video_path entrada.mp4 --model_path runs/tracknet_v2/default/weights/best_model.pth --video_out runs/tracknet_v2/inference/saida.mp4
```

V1 aceita de um a quatro `--video_paths`, com `--device`, `--trail_length`, `--start_time` e `--max_seconds`. A saída padrão é `<raiz do projeto>/runs/tracknet_v1/inference/output.mp4`.

V2 mantém `--input_width`, `--input_height`, `--threshold` em [0,1] e `--trail_length` não negativo. A saída padrão é `<raiz do projeto>/runs/tracknet_v2/inference/output_video.mp4`. Entrada/saída precisam ser arquivos distintos; leitura segue até EOF e a contagem declarada serve somente ao progresso. FPS inválido usa 30; dimensões inválidas geram erro. Captura/writer são liberados em falhas, e o buffer temporal guarda frames limpos antes dos desenhos.

## Notebook e visualização

[tracknet_v2/notebooks/Trackv2.ipynb](tracknet_v2/notebooks/Trackv2.ipynb) localiza a raiz pelos ancestrais e importa os módulos V2. O dataset padrão é `dataset/tracknet_v2/val`. Configure resoluções e checkpoint explícito antes de prever ou validar; o notebook não inicia treino.

[tracknet_v2/helpers.py](tracknet_v2/helpers.py) mostra frame e heatmap centrais, preservando input e modo explícito. A desnormalização retorna cópia; a visualização aceita par/tripla, limita o primeiro batch e retorna um grid sem exibi-lo automaticamente.

## Verificação e logs

```powershell
python -B -m unittest discover -s tests -v
python train.py --help
python -m tracknet_v1.infer --help
python -m tracknet_v2.infer --help
tensorboard --logdir runs/tracknet_v2/default/logs
tensorboard --logdir runs/tracknet_v1/default/plots
```

`runs/`, bytecode e os quatro conjuntos são ignorados pelo Git; exemplos/placeholders são versionados. Testes sintéticos não comprovam qualidade em partidas reais. O [relatório histórico](docs/relatorio%20V2.md) mantém a análise original e evidências posteriores.

Estrutura e documentação atualizadas em 30/10/2026.
