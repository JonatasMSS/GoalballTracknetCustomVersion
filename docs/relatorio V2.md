> **Registro histórico:** a análise abaixo foi preservada integralmente e corresponde ao commit `ca1ccb4`.
> Os números de linha e a afirmação de que nenhum código foi alterado referem-se àquela análise.
> As correções e verificações posteriores estão na seção final **Status em 29/10/2026**.

# Relatório TrackNet V2

Revisão com o agente cientista de dados sênior. Base: código atual, commit `ca1ccb4`. Nenhum código de execução foi alterado.

## Parecer sobre a arquitetura

- **A arquitetura é estruturalmente coerente para 640×360.** Os três níveis de pooling e upsampling fecham as dimensões; as concatenações de 768, 384 e 192 canais estão corretas. Referências: [models/TracknetV2.py:62](models/TracknetV2.py#L62), [models/TracknetV2.py:73](models/TracknetV2.py#L73), [models/TracknetV2.py:86](models/TracknetV2.py#L86).
- **Entrada e saída estão alinhadas:** três frames RGB formam 9 canais; a saída possui três heatmaps, na ordem `t−2, t−1, t`. O canal 2 usado na inferência corresponde ao frame atual. Referências: [dataset:183](datasets/TracknetV2Dataset.py#L183), [dataset:206](datasets/TracknetV2Dataset.py#L206), [modelo:89](models/TracknetV2.py#L89), [predictV2.py:96](predictV2.py#L96).
- **Logits na loss e sigmoid na inferência estão corretos nos scripts.** `model.eval()` controla BatchNorm; `set_mode()` controla a ativação da saída. São mecanismos independentes, usados corretamente pelo fluxo principal. Referências: [modelo:138](models/TracknetV2.py#L138), [loss:30](utils/TracknetV2/loss.py#L30), [validação:119](utils/TracknetV2/train_val.py#L119).
- Conv–BatchNorm–ReLU, AdamW e normalização ImageNet são escolhas válidas. A leitura do código permite avaliar coerência; qualidade de rastreamento exige experimento com dados e pesos reais.

## Falhas e correções prioritárias

- **Alta — Frames desenhados voltam como entrada.** A inferência desenha círculos e trajetória antes de guardar `frame` em `prev_frames`. As próximas previsões recebem essas marcações, ausentes no treino. **Correção:** guardar uma cópia limpa antes dos desenhos. Referências: [predictV2.py:104](predictV2.py#L104), [predictV2.py:110](predictV2.py#L110).

- **Alta — Labels de PNG são perdidos silenciosamente.** A troca `.replace('.jpg', '.txt')` mantém a extensão `.png`; o leitor então devolve `(0, 0, 0)`, mesmo existindo um TXT correto. Isso transforma bolas anotadas em ausências na avaliação. **Correção:** substituir a extensão independentemente do formato e distinguir arquivo ausente de ausência anotada. Referências: [dataset:52](datasets/TracknetV2Dataset.py#L52), [dataset:159](datasets/TracknetV2Dataset.py#L159). Problema reproduzido com o método real de leitura.

- **Média — Heatmaps não acompanham o redimensionamento da entrada.** As imagens viram 360×640, mas os GTs são carregados no tamanho original; se diferirem, a loss recebe formas incompatíveis. O caminho do GT também exige a mesma extensão do frame. **Correção:** validar tamanho/nome dos GTs e redimensionar ou regenerar os mapas conforme sua representação. Referências: [dataset:27](datasets/TracknetV2Dataset.py#L27), [dataset:45](datasets/TracknetV2Dataset.py#L45), [dataset:121](datasets/TracknetV2Dataset.py#L121).

- **Média — Pré-processamento diferente entre treino e inferência.** O treino redimensiona via PIL, com antialias; a inferência usa `F.interpolate` sem antialias. A redução pode representar a bola de maneira diferente. **Correção:** usar o mesmo procedimento nos dois caminhos e comparar os tensores resultantes. Referências: [dataset:25](datasets/TracknetV2Dataset.py#L25), [predictV2.py:19](predictV2.py#L19). Comportamento confirmado nas documentações de [Resize](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html) e [interpolate](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.interpolate.html).

- **Média — Dimensões configuráveis podem quebrar as concatenações.** Altura e largura precisam ser múltiplos de 8; por exemplo, 361×641 falha. A CLI aceita esses tamanhos sem validação. **Correção:** rejeitar dimensões incompatíveis antes da inferência. Referências: [predictV2.py:46](predictV2.py#L46), [modelo:121](models/TracknetV2.py#L121), [modelo:133](models/TracknetV2.py#L133).

- **Média — Validação e melhor checkpoint podem não ocorrer.** Com os padrões, valida nas épocas humanas 3, 5, 7 e 9, deixando a época 10 sem avaliação. Treinos de até duas épocas não validam; se todos os F1 forem zero, `best_model.pth` nunca é salvo. **Correção:** definir a periodicidade usando a época humana, avaliar o encerramento e salvar o primeiro resultado válido. Referências: [trainV2.py:106](trainV2.py#L106), [trainV2.py:117](trainV2.py#L117), [trainV2.py:125](trainV2.py#L125).

- **Média — Carregar pesos não retoma o treinamento completo.** `--model_path` restaura apenas o modelo; AdamW, scheduler, época e melhor F1 reiniciam. **Correção:** salvar/restaurar esses estados ou documentar a opção como inicialização por pesos. Referências: [trainV2.py:66](trainV2.py#L66), [trainV2.py:80](trainV2.py#L80), [trainV2.py:136](trainV2.py#L136).

- **Média — Validação vazia aparenta sucesso.** Sem amostras, `validate()` retorna loss zero e métricas zero, sem avisar que nada foi avaliado. **Correção:** rejeitar dataset de validação vazio antes do treino. Referências: [trainV2.py:109](trainV2.py#L109), [validação:156](utils/TracknetV2/train_val.py#L156).

- **Baixa — Helper de visualização incompatível com V2.** `make_prediction_grid()` repete os três canais de saída para nove e tenta concatená-los com RGB de três canais; também desnormaliza uma visão de `x` no lugar. **Correção:** escolher o heatmap correspondente ao frame exibido e desnormalizar uma cópia. Referências: [utils/helpers.py:20](utils/helpers.py#L20), [utils/helpers.py:45](utils/helpers.py#L45), [utils/helpers.py:56](utils/helpers.py#L56). Não é chamado pelo treino principal.

- **Baixa — Notebook e documentação estão desatualizados.** O notebook inverte os dois primeiros labels e sua última célula usa `val_loader`, `criterion` e `postprocess` não definidos; trata labels como dicionários, embora sejam tuplas. O README usa `--dataset_path` e `--parallel True`, incompatíveis com a CLI atual. **Correção:** importar os módulos atuais no notebook e atualizar os exemplos. Referências: [notebook:418](Notebooks/Trackv2.ipynb#L418), [notebook:644](Notebooks/Trackv2.ipynb#L644), [README:67](README.md#L67), [trainV2.py:23](trainV2.py#L23). O notebook e `show_model_summary()` também dependem de `torchsummary`, ausente em `requirements.txt`.

## Premissas dos dados e da avaliação

- **Ordem temporal:** `sorted()` funciona para nomes com números preenchidos com zeros. Para `frame1`, `frame2`, `frame10`, resulta em `frame1`, `frame10`, `frame2`. Validar nomes, sequência e arquivos legíveis; hoje não há checagem. Referência: [dataset:37](datasets/TracknetV2Dataset.py#L37).
- **Escala e visibilidade:** a conversão pressupõe coordenadas em pixels e uma única resolução original para todo o conjunto. A validação reconhece somente `vis == 1`; outros códigos serão tratados como ausência. Confirmar esses contratos nos dados. Referências: [dataset:171](datasets/TracknetV2Dataset.py#L171), [validação:139](utils/TracknetV2/train_val.py#L139).
- **Separação dos conjuntos:** caminhos diferentes não provam independência. Conferir sobreposição de partidas/clipes e frames entre treino e validação. Não foi constatado vazamento, pois os conjuntos não estão disponíveis. Referência: [trainV2.py:44](trainV2.py#L44).
- **Loss:** a implementação é uma focal BCE estável. Para alvos binários, a ponderação equivale à ponderação separada dos termos positivo/negativo; para heatmaps suaves, não equivale. Definir o contrato dos GTs antes de afirmar reprodução exata da WBCE original. Referência: [loss:37](utils/TracknetV2/loss.py#L37).
- **Métricas:** localização fora da tolerância conta apenas como FP; o recall atual não é a fração de todas as bolas visíveis corretamente localizadas. Existem protocolos TrackNet com essa convenção; documentá-la, separar erros de localização e falsas detecções, e acompanhar acerto sobre todas as bolas visíveis. A validação agrega três saídas de janelas sobrepostas, enquanto a produção usa apenas a atual; medir também o canal 2 para representar esse uso. Referências: [validação:130](utils/TracknetV2/train_val.py#L130), [validação:148](utils/TracknetV2/train_val.py#L148), [validação:152](utils/TracknetV2/train_val.py#L152).
- **Pós-processamento:** Hough seleciona o primeiro círculo com parâmetros fixos; não há evidência de que seja inadequado, mas threshold, raio e tolerância de 10 pixels precisam de calibração no goalball. Referências: [train_val.py:42](utils/TracknetV2/train_val.py#L42), [train_val.py:79](utils/TracknetV2/train_val.py#L79).

## Verificação e limites

- Escopo lido: modelo, dataset, loss, treino, validação, inferência, helpers, testes V2, notebook, README, dependências e inicializadores relacionados.
- Sintaxe dos 20 arquivos Python verificada; checagens leves confirmaram o bug PNG, a ordenação lexicográfica e as restrições dimensionais por cálculo.
- `python -B -m unittest discover -s tests -v` falhou na importação por ausência de `torch`. Os dois testes existentes cobrem ativação e contagem temporal com pós-processamento simulado; não cobrem dataset, loss real ou inferência completa.
- Não foram executados treinamento, forward real ou avaliação com pesos/dados. Portanto, não há conclusão sobre acurácia, convergência ou consumo de GPU.

**Ordem sugerida:** corrigir realimentação dos desenhos e labels; assegurar contratos de GT/pré-processamento; ajustar validação/checkpoints; então medir desempenho com divisão por partida e inferência real.


---

## Status em 29/10/2026

As nove correções do plano e os complementos de validação/visualização foram implementados e revisados em sequência. O texto histórico acima permanece como registro dos defeitos originais; os contratos científicos preservados estão discriminados abaixo.

| Ponto | Resultado atual | Implementação |
| --- | --- | --- |
| 1. Entrada temporal limpa | Uma cópia do frame entra no buffer antes de qualquer desenho; as próximas janelas não recebem a trajetória anotada. | `predictV2.py` |
| 2. Labels e extensões | Caminhos TXT independem da extensão do frame. A primeira linha exige três campos, coordenadas finitas e visibilidade inteira; ausência, vazio e conteúdo inválido geram erros contextualizados. Visibilidade zero continua sendo uma anotação explícita. | `datasets/TracknetV2Dataset.py` |
| 3. Preservação do vídeo de entrada | A CLI rejeita entrada e saída que apontam para o mesmo arquivo, considerando caminhos normalizados e identidade do arquivo quando ambos existem. | `predictV2.py` |
| 4. Separação de diretórios | Treino e validação precisam ser diretórios existentes e fisicamente distintos; aliases da mesma pasta são rejeitados antes de criar artefatos. Isso não substitui auditoria de sobreposição entre partidas/frames. | `trainV2.py` |
| 5. Agenda e checkpoints | Valida a cada intervalo de épocas humanas e na última época. Salva o primeiro F1 finito, inclusive zero, e depois somente melhorias; o último modelo é salvo por época. Épocas não positivas, F1 não finito e best/last já existentes são rejeitados. Um experimento novo exige outro `--exps_path`. | `trainV2.py` |
| 6. Loss média e vazios | Treino e validação ponderam a loss dos batches pelo número de amostras; loaders sem amostras geram erro. Backward e definição das métricas foram preservados. | `utils/TracknetV2/train_val.py` |
| 7. Pré-processamento consistente | Dataset e inferência usam RGB, resize bilinear via PIL e a mesma normalização ImageNet. Um teste com detalhes de alta frequência confirma igualdade dos tensores e falha com o backend anterior substituído somente em memória. | `predictV2.py`, `datasets/TracknetV2Dataset.py` |
| 8. Contratos dos dados | Dimensões originais positivas/finitas e entradas positivas/múltiplas de oito são verificadas. Frames têm ordem natural com desempate pelo nome e lacunas permitidas, sem cruzar clips. GT exato é preferido; fallback por nome-base deve ser único em JPG/JPEG/PNG. Leitura inválida e GT fora da resolução geram erro, sem resize de GT. Datasets vazios são rejeitados antes de criar logs/modelo/otimizador; o construtor não decodifica todas as imagens. | `datasets/TracknetV2Dataset.py`, `trainV2.py`, `predictV2.py` |
| 9. Leitura e recursos de vídeo | Leitura segue até EOF, independentemente da contagem declarada. Contagem válida serve somente ao progresso; FPS inválido usa 30 e dimensões inválidas são rejeitadas. Captura e writer são liberados também nas falhas de abertura, previsão e pós-processamento. | `predictV2.py` |

**Complemento 10 — um forward na validação.** A loss recebe os logits e o pós-processamento recebe `sigmoid(logits)` do mesmo forward. `eval()` e o modo explícito continuam independentes; a validação termina em modo de inferência, e o treino seguinte restaura o modo de treino. Os testes verificam isso com parâmetros e gradientes reais.

**Complemento 11 — helpers e notebook.** A desnormalização copia o tensor. Os grids V2 usam frame e heatmap centrais, preservam inputs e modo explícito, aceitam DataParallel e respeitam o tamanho efetivo do batch. A visualização de batches aceita par/tripla, consome somente o primeiro batch e não exibe automaticamente. O notebook importa os módulos atuais, exige checkpoint antes de previsão/validação, mantém modelo e inputs no mesmo dispositivo e usa `validate()` atual, incluindo FP em frames sem bola. Definições duplicadas, dependência de `torchsummary` e saídas antigas foram removidas.

README, instruções para agentes e ajuda de `--model_path` foram atualizados. A opção inicializa somente o `state_dict`; otimizador, scheduler, época e melhor F1 reiniciam. Não foi implementada retomada completa de treinamento.

### Evidências de verificação

- **Suíte em CPU:** `python -B -m unittest discover -s tests -q` executou 51 testes: 47 passaram e quatro tiveram skip explícito por restrições de hardlink/symlink no Windows.
- **Integração real com dados sintéticos em CPU:** dataset, forward/backward do modelo, um passo AdamW com `wbce_loss`, validação com Hough, gravação/carregamento de `state_dict` e inferência em MP4 de quatro frames foram exercitados. Isso verifica execução integrada, sem representar treinamento de desempenho em partidas.
- **Notebook:** sete células de código executadas com configuração sintética e `state_dict` carregado; dois grids e a validação final foram verificados. A configuração sem checkpoint também foi rejeitada antes de criar o modelo.
- Testes de regressão cobrem janelas limpas, labels/GTs, ordenação e resolução, guards de CLI/diretórios/checkpoints, agenda, médias ponderadas, modos/gradientes, EOF/recursos e visualização. Sintaxe, JSON do notebook e whitespace do diff foram verificados.

### Limites e contratos mantidos

A arquitetura, os formatos dos pesos, a loss existente, a seleção do canal atual na inferência, a regra `vis == 1` e a classificação de erro de localização como FP permanecem preservados. A loss não foi substituída nem declarada reprodução exata da WBCE original para heatmaps suaves. Parâmetros de Hough e tolerância não foram recalibrados.

Não foram avaliados jogos reais, generalização, treinamento longo ou execução em GPU. Independência dos splits e qualidade das anotações reais ainda dependem dos dados; os resultados sintéticos não estabelecem acurácia, convergência ou ganhos de desempenho.


---

## Reorganização modular em 30/10/2026

O texto anterior, inclusive o status de 29/10, foi preservado integralmente. Seus caminhos e números de linha são referências históricas aos arquivos daqueles commits; os destinos atuais estão no mapa abaixo. O prefixo histórico preservado tem SHA256 `6d15e7b5704463f79a0f33737bafed5b9cffec9fbcd00e7651bc1ef987302476`.

| Origem histórica | Destino atual |
| --- | --- |
| `models/TracknetV1.py` | [tracknet_v1/model.py](../tracknet_v1/model.py) |
| `models/TracknetV2.py` | [tracknet_v2/model.py](../tracknet_v2/model.py) |
| `datasets/TracknetV1Dataset.py` | [tracknet_v1/dataset.py](../tracknet_v1/dataset.py) |
| `datasets/TracknetV2Dataset.py` | [tracknet_v2/dataset.py](../tracknet_v2/dataset.py) |
| `utils/TracknetV1/train_val.py` | [tracknet_v1/train_val.py](../tracknet_v1/train_val.py) |
| `utils/TracknetV2/train_val.py` | [tracknet_v2/train_val.py](../tracknet_v2/train_val.py) |
| `utils/TracknetV2/loss.py` | [tracknet_v2/loss.py](../tracknet_v2/loss.py) |
| `utils/helpers.py` | [tracknet_v2/helpers.py](../tracknet_v2/helpers.py) |
| `trainV1.py`, `trainV2.py` | [train.py](../train.py) + [config.yaml](../config.yaml), chamando as APIs `tracknet_v1.train.run_training` e `tracknet_v2.train.run_training` |
| `inference_video.py`, `predictV2.py` | [tracknet_v1/infer.py](../tracknet_v1/infer.py), [tracknet_v2/infer.py](../tracknet_v2/infer.py), executados com `python -m` |
| `Notebooks/Trackv2.ipynb` | [tracknet_v2/notebooks/Trackv2.ipynb](../tracknet_v2/notebooks/Trackv2.ipynb) |
| Exemplos em `Notebooks/` | [dataset/examples/img000007.jpg](../dataset/examples/img000007.jpg), [dataset/examples/img000007.txt](../dataset/examples/img000007.txt), sem alteração de bytes |

### Etapas e verificações

| Etapa | Evidência |
| --- | --- |
| 1. Baseline, commit `e49fb381f8065e941ae947fe6ef235c6554874a7` | Fora do repositório: pesos V1/V2, input/saídas, manifestos e contrato aprovado. Strict load em segunda instância e paridade bit a bit em CPU. Suíte: 51 testes, 47 passaram e 4 skips. |
| 2. Pacote V2 | API nativa `run_training(config)`, seleção de device e inferência modular; strict load de 121 chaves e logits/probabilidades idênticos ao baseline. Modelo, dataset, loss, validação e helpers preservados em bytes. Notebook: sete células com sintaxe válida; imports/configuração conferidos da raiz e de sua nova pasta. Suíte: 54 testes, 50 passaram e 4 skips. |
| 3. Pacote V1 | Splits explícitos sem `random_split`, seleção de device e `math.dist` substituindo SciPy. Arquitetura/dataset preservados em bytes; strict load de 126 chaves e saída idêntica ao baseline. Agenda, checkpoints, Adam, limite legado de batches e calibrações de treino/inferência mantidos. Suíte: 65 testes, 61 passaram e 4 skips. |
| 4. Lançador YAML | `safe_load`, validação do perfil ativo, caminhos relativos ao YAML e dispatch apenas do módulo escolhido, sem importar PyTorch ao importar o lançador. Casos de paths com espaços/outro CWD, tipos/ranges, artefatos e erros CLI verificados. Suíte: 79 testes, 75 passaram e 4 skips. |
| 5. Estrutura e documentação | Exemplos movidos com SHA256 preservado, quatro templates vazios, dados/resultados/bytecode ignorados e fontes legadas já migradas removidas sem wrappers. Removidos seis bytecodes, inicializadores antigos e o esqueleto Metrics não utilizado; pastas removidas somente quando vazias. |
| 6. Integração final após reorganização | Concluída em CPU: suíte final de 79 testes (75 passaram, 4 skips), paridade dos dois modelos, três CLIs, controlador V1 com classificador simulado, componentes V2 reais, inferência real e notebook nos dois diretórios; detalhes e limites abaixo. |

As suítes foram executadas em CPU com `python -B -m unittest discover -s tests -q`; os quatro skips explícitos correspondem às restrições de hardlink/symlink do Windows. Testes V1 agora também exercitam código legado que emite dois avisos `torch.tensor(tensor)`; esse comportamento não foi alterado.

Os perfis completos e os defaults estão em `config.yaml`. `model_path` continua inicializando somente pesos; não foi criado resume. Saídas padrão estão centralizadas em `runs/<modelo>/`. Treino V1 mantém validação nas épocas humanas 3,5,7,9 para 10 épocas/intervalo 2, sem final obrigatório; V2 mantém intervalos humanos e a última época.

Nenhum dataset ou peso real foi movido ou copiado. Os exemplos mantêm SHA256 `332f7b786f6c467c9223aa4c98306c8fbe3be08a17c74a0e1c2bcba24fab0b39` (JPG) e `e56f1f5d39e20a7db08e84ec5a51ca0e5c3fbae03bb490af7615398cda28c3b8` (TXT). Não houve avaliação em partidas reais, treino longo, GPU ou recalibração científica nesta reorganização.


### Evidências finais da etapa 6

A suíte foi repetida após a limpeza com `python -B -m unittest discover -s tests -q`: 79 testes, 75 passaram e quatro skips explícitos, sem falhas. Os dois avisos legados da V1 permaneceram. O script externo `run_tracknet_integration.py` terminou com código zero; `tracknet-reorganization-integration-20261030/results.json` e os logs desse diretório registram a execução em CPU, seed `20261029`, com dados e artefatos sintéticos fora do repositório.

- **Pesos e CLIs:** strict load de 126 chaves V1 e 121 chaves V2, sem chaves ausentes ou inesperadas; saída V1 e logits/probabilidades V2 idênticos bit a bit ao baseline. As ajudas de `train.py`, `tracknet_v1.infer` e `tracknet_v2.infer` terminaram com código zero.
- **Controlador V1:** lançador YAML com caminhos relativos e espaços, dataset, loaders, loops de treino/validação, CrossEntropy e Adam reais em `[1,9,360,640]`, durante três épocas. Somente o classificador foi substituído por um módulo mínimo simulado. Seus parâmetros mudaram e `model_last.pt` foi salvo; F1 zero não criou best, conforme a regra legada. Isso verifica o controlador, sem representar treino da arquitetura V1 completa.
- **Componentes V2:** dataset sintético com duas janelas, arquitetura TrackentV2 real em `[2,9,16,16]`, `wbce_loss`, backward, um passo AdamW com alteração dos pesos, validação/Hough reais com valores finitos e gravação/recarregamento estrito do checkpoint. A execução usou as APIs de componentes; não foi um treino completo pelo lançador YAML na resolução padrão 640×360.
- **Vídeos:** inferências pelas duas CLIs com arquiteturas reais e MP4 decodificado. V1 processou um frame em entrada 640×360, solicitado por `--max_seconds 0.25` no vídeo de 4 FPS; V2 processou quatro frames em entrada 16×16 até EOF. O SHA256 do vídeo de origem permaneceu igual.
- **Notebook:** as sete células de código completas foram executadas da raiz e de `tracknet_v2/notebooks`, com o checkpoint V2 sintético carregado. Cada execução produziu dois grids RGB e validação com valores finitos, sem alterar o notebook no disco.

Os fixtures usam ausência de bola anotada; losses/métricas finitas demonstram execução, não qualidade de localização. Os smokes não equivalem a treinamento completo das duas arquiteturas em 640×360, nem avaliam partidas reais, generalização, convergência, latência ou GPU. Loss, calibrações e protocolo de métricas continuam preservados.
