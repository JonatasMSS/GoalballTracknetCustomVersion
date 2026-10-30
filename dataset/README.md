# Dados de treino e validação

Esta pasta contém dados, separados dos módulos `tracknet_v1/dataset.py` e `tracknet_v2/dataset.py`.
Os quatro diretórios `tracknet_v{1,2}/{train,val}/` estão vazios, com `.gitkeep` versionado.
Preencha cada conjunto com:

```text
frames_out/<partida>/<clip>/frame1.jpg
gts/<partida>/<clip>/frame1.png
labels/<partida>/<clip>/frame1.txt
```

O TXT usa a primeira linha `x y vis`. Na V2, `x,y` são pixels da resolução original
configurada no YAML; os GTs já precisam estar na resolução de entrada, sem resize.
A V1 mantém seu contrato próprio de coordenadas e visibilidade; consulte o [README](../README.md).
Separe partidas/clips independentes e confira sobreposição entre os conjuntos.

`examples/` preserva um frame e seu TXT; não é um dataset completo para treino.
Dados dos quatro conjuntos e saídas em `runs/` são ignorados pelo Git; exemplos e
placeholders continuam versionados. Nenhum dataset real foi copiado nesta reorganização.
