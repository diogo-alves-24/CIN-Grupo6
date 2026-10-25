# Forrageamento Espacial — Inteligência de Enxame

Simulador 2D em Python e Pygame, com código e identificadores em português.


## Controlos

- **1**: passeio aleatório (referência)
- **2**: comportamento combinado (feromonas + Boids + papéis)
- **R**: reiniciar cenário (mesma semente)
- **Espaço**: pausa
- **S**: mostrar sensores
- **F**: mostrar/ocultar feromonas
- **T**: alternar visualização entre rastos de comida (laranja) e casa (azul)
- **H**: esconder o painel
- **Esc**: sair


## Modelo

- Os agentes movem-se num espaço 2D e só usam observações locais para escolher a direção.
- Feromonas são armazenadas numa grelha, evaporam exponencialmente e sofrem difusão local.
- Agentes em exploração depositam feromona de casa, e agentes carregados depositam feromona de comida.
- O comportamento Boids combina separação, alinhamento e coesão; a separação tem peso superior.
- Os papéis alternam entre **EXPLORADOR** e **RECOLHEDOR** quando há pistas locais de recursos ou ausência delas.

