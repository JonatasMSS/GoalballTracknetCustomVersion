# Orientações para agentes — Tracknet Goalball

## Objetivo e escopo

Projeto Python/PyTorch para rastrear a bola em vídeos de goalball com TrackNet V1 e V2.
Responda em português; preserve os identificadores e as convenções do código.
Faça a menor mudança que resolva a tarefa inteira. Antes de editar, leia os módulos
afetados, procure seus chamadores e identifique testes, CLI e documentação envolvidos.
Não corrija problemas encontrados fora do escopo sem uma tarefa que inclua a correção.

## Mapa do projeto

| Fluxo | Entrada | Implementação |
| --- | --- | --- |
| Treino V1 | `trainV1.py` | `models/TracknetV1.py::BallTrackerNet`, `datasets/TracknetV1Dataset.py`, `utils/TracknetV1/train_val.py` |
| Treino V2 | `trainV2.py` | `models/TracknetV2.py::TrackentV2`, `datasets/TracknetV2Dataset.py`, `utils/TracknetV2/loss.py`, `utils/TracknetV2/train_val.py` |
| Inferência V1 | `inference_video.py` | `get_input_tensor`, `postprocess`, `main`; aceita de 1 a 4 vídeos |
| Inferência V2 | `predictV2.py` | `preprocess_frame`, `build_input`, `draw_trajectory`; usa o canal 2 da saída |
| Testes | `tests/` | `unittest`: dataset, treino, validação, inferência e helpers V2 com fixtures sintéticas |

`Notebooks/Trackv2.ipynb` importa os módulos atuais e exige checkpoint antes de prever
ou validar. `utils/helpers.py` visualiza o frame/heatmap central da V2, preservando
inputs e modo explícito; a desnormalização devolve uma cópia. `utils/Metrics/ModelMetrics.py`
é apenas um esqueleto. `relatorio V2.md` preserva a análise histórica e registra as
correções posteriores; não substitui o código atual.

## Contratos que precisam permanecer alinhados

| Contrato | V1 | V2 |
| --- | --- | --- |
| Entrada por batch | `[B,9,H,W]`, padrão `H=360`, `W=640` | `[B,9,H,W]`, mesmo tamanho padrão |
| Cor e normalização | BGR, dividido por 255 | RGB, dividido por 255 e normalizado com média/desvio ImageNet |
| Ordem dos frames | `[t, t-1, t-2]` | `[t-2, t-1, t]` |
| Saída padrão | `[B,256,H*W]`, classes por pixel | `[B,3,H,W]`, um heatmap por frame na mesma ordem temporal |
| Loss | `CrossEntropyLoss`, alvo inteiro | `wbce_loss`, logits e alvos em `[0,1]` |

- Preserve a grafia pública `TrackentV2` e a compatibilidade dos `state_dict` existentes.
- Na V2, `model.train()`/`eval()` controlam os módulos, incluindo BatchNorm;
  `set_mode("train")` retorna logits e `set_mode("inference")` aplica sigmoid.
  `postprocessing` recebe probabilidades: não aplique sigmoid duas vezes. A validação
  reutiliza um forward de logits por batch e calcula `sigmoid(logits)` para as métricas.
- Mudanças de pré-processamento, canais, ordem temporal ou tamanho devem alcançar
  dataset, modelo, treino, validação, inferência e testes afetados. V1 e V2 têm contratos distintos.
- A V2 tem três níveis de pooling/upsampling com concatenações: use altura e largura
  múltiplas de 8. Confirme dimensões e escala antes de alterar resolução.

## Dados e avaliação

- Estrutura efetiva: `{frames_out,gts,labels}/partida/clip/arquivo` dentro de cada conjunto.
  Labels TXT usam a primeira linha `x y vis`. Na V2, `x,y` são pixels da resolução
  original e são reescalados por `original_width`/`original_height`; a validação considera
  visível apenas `vis == 1`. Não transfira essa convenção silenciosamente para V1.
- Verifique arquivos legíveis, extensões, correspondência frame/GT/label, dimensões,
  visibilidade e continuidade temporal. GTs V2 são carregados sem resize e precisam
  coincidir com a resolução de entrada. A V2 prefere GT de nome exato, com fallback
  único por nome-base em JPG/JPEG/PNG; ausência ou ambiguidade gera erro. Ela usa ordem
  natural numérica com desempate pelo nome, sem exigir IDs consecutivos; V1 usa
  `os.listdir` sem ordenar. Labels ausentes ou inválidos geram erro, não ausência da bola.
  O construtor indexa nomes; leitura das imagens e tamanho dos GTs são verificados no acesso à amostra.
- Separe avaliação por partidas ou clipes independentes e confira sobreposição real.
  Janelas compartilham frames: `random_split` da V1 e caminhos distintos na V2 não
  garantem independência entre treino e validação.
- A validação V2 agrega os três canais de janelas sobrepostas; a inferência usa só o
  atual. Ela conta localização fora de `max_distance` como FP, sem FN adicional.
  Preserve/documente o protocolo ao comparar métricas; não altere sua definição silenciosamente.
- Threshold, raios Hough e tolerância dependem de calibração no goalball. Registre split,
  resolução, seed, versões, checkpoint, limiar e tolerância em experimentos autorizados.
  Só declare ganho de acurácia, convergência, latência ou uso de GPU com medição.

## Execução e validação

Dependências estão em `requirements.txt`; use o ambiente do projeto quando disponível.
O fluxo V1 também importa `scipy`, ainda ausente dessa lista; `torchsummary` é opcional
somente em `show_model_summary()` da V2, não no notebook. Não instale pacotes para uma análise documental.
O treino V1 fixa CUDA; V2 e os scripts de inferência escolhem CPU/CUDA conforme disponibilidade.

Verificação de regressão existente, sem dados reais:

```powershell
python -B -m unittest discover -s tests -v
```

Os testes usam dados sintéticos e não comprovam qualidade em partidas reais ou calibração de Hough.
Para lógica nova não trivial, acrescente um teste pequeno ou uma autoverificação com
asserts; para documentação, confira referências, argumentos e sintaxe das configurações.
Se faltarem dependências, relate a falha de importação: análise de sintaxe não equivale a testes passando.

Consulte as CLIs reais com `python <script>.py --help` quando as dependências estiverem
disponíveis; esses comandos importam bibliotecas antes de exibir ajuda. Exemplos de
argumentos atuais (treino/inferência só quando fizerem parte da tarefa):

```powershell
python trainV1.py --train_path assets/train --val_path assets/val --exp_id run_v1_01
python trainV2.py --train_dataset_path assets/train --val_dataset_path assets/val --exps_path exps/run_v2_01 --logs_path runs/run_v2_01
python inference_video.py --video_paths entrada.mp4 --model_path exps/run_v1_01/model_best.pt --output_path saida_v1.mp4
python predictV2.py --video_path entrada.mp4 --model_path exps/run_v2_01/best_model.pth --video_out saida_v2.mp4
```

No treino V2, valide diretórios e datasets não vazios antes de criar artefatos. A agenda
usa épocas humanas e avalia também a última; salva o primeiro F1 finito, inclusive zero,
e depois apenas melhorias. As losses de época são ponderadas pelo número de amostras.
Na inferência V2, a leitura segue até EOF; a contagem declarada serve apenas ao progresso.
Captura e writer são liberados em caso de erro. Consulte o README e as CLIs para os argumentos atuais.

## Preservação de trabalho e artefatos

- Confira `git status` e preserve alterações locais, relatórios, datasets e pesos.
- Os checkpoints atuais salvam apenas pesos; `--model_path` não retoma otimizador,
  scheduler, época ou melhor métrica. Pesos V1 e V2 não são intercambiáveis.
- A V2 rejeita `--exps_path` com `best_model.pth` ou `last_model.pth` já existente.
  Inicialização por pesos exige um novo diretório de saída; não implemente resume completo sem tarefa específica.
- Use diretórios de experimento e vídeos de saída novos para evitar sobrescritas.
  `.gitignore` não exclui `exps/` nem todos os vídeos; não adicione artefatos gerados ao Git.
- Não inicie treino longo, downloads ou processamento de vídeos só para validar uma
  alteração documental. Preserve calibrações e validações existentes em correções.

## Equipe de três agentes

As definições persistentes estão em `.codex/agents/`, no formato TOML do Codex:

| Agente | Arquivo | Responsabilidade |
| --- | --- | --- |
| `orquestrador` | `.codex/agents/orquestrador.toml` | Delimitar escopo, distribuir trabalho, integrar e validar a entrega |
| `cientista_visao` | `.codex/agents/cientista_visao.toml` | Ciência de dados sênior, visão computacional, dados, treino e inferência |
| `revisor` | `.codex/agents/revisor.toml` | Revisão independente, somente leitura, de correção e regressões |

Em tarefas com frentes independentes, use o orquestrador para envolver o cientista e
pedir revisão do resultado ao revisor. Em tarefas simples, use apenas o papel necessário.
Reutilize agentes existentes; evite delegação recursiva e edição simultânea do mesmo arquivo.
Defina objetivo, arquivos sob responsabilidade, restrições e critério de conclusão em
cada delegação. A revisão ocorre sobre a entrega final; corrija achados relevantes e revalide.
Papéis herdam modelo e permissões da sessão; a instrução de somente leitura do revisor
é comportamental e não cria um sandbox separado.

Para usar: “Use o orquestrador para coordenar esta tarefa, o cientista_visao para os
aspectos de visão computacional e o revisor para revisar a entrega.” Se o cliente não
carregar papéis personalizados nesta sessão, use subagentes com as mesmas instruções
e informe a limitação. Consulte a [documentação oficial](https://learn.chatgpt.com/docs/agent-configuration/subagents).

Ao entregar, informe mudanças, verificações executadas, resultados e limites. Distinga
análise estática, testes sintéticos e experimentos reais; nunca invente resultados.
