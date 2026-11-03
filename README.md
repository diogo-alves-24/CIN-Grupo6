# Forrageamento Espacial — Neuroevolução (duas MLP)

Projeto em **Python + Pygame** com agentes autónomos que procuram recursos e os entregam no ninho. Mantém três estratégias para comparação: **Random Walk**, **Combinado** (feromonas + Boids) e **Neuroevolução** (Algoritmo Genético que evolui duas MLP).

## Instalar e executar

```bash
conda activate forrageamento  # ou o teu ambiente Conda com pygame e numpy
python -m pip install -r requirements.txt
python main.py
```

Seleciona o modo no menu. O modo Neuroevolução exige `melhor_genoma.npz` **treinado nesta versão**.

Controlos: `P` (1x → 2x → 4x → 8x), `T` (tipo de feromona), `F` (rastos), `H` (painel), `S` (sensores), `Espaço` (pausa), `R` (reiniciar), `M` (menu), `ESC` (sair).

## Duas redes neuronais especializadas

Em `simulacao/rede_neuronal.py` existem **duas redes separadas**, ambas com topologia **28–10–3**:

- **Procura**: usada enquanto a formiga não transporta recurso.
- **Regresso**: usada depois de recolher um recurso até o entregar.

O mesmo genoma codifica as duas redes, mas cada uma tem **pesos próprios**. O total é **646 parâmetros** (323 por rede). Ambas recebem 24 entradas locais da versão anterior e 4 sinais de memória do próprio agente:

1. Tempo desde a recolha (normalizado).
2. Índice de circularidade do percurso recente (0 = quase retilíneo; 1 = muitas voltas).
3. Rotação assinada acumulada na janela recente.
4. Progresso líquido normalizado na mesma janela.

A janela é de **5 segundos simulados** (`JANELA_MEMORIA`). A memória é formada por **deslocamentos próprios acumulados**, sem acesso às coordenadas globais do ninho/recursos.

As três saídas de cada MLP continuam a ser: correção lateral, correção frontal e **comando de velocidade**. **Não** foi adicionada uma quarta saída. A rede é híbrida: ajusta a direção do Combinado e o genoma todo a zero reproduz exatamente o Combinado.

### Velocidade e energia

A saída de velocidade está em [-1, 1]. A velocidade própria inicial de cada agente é usada no comando neutro. Um comando positivo aumenta-a suavemente até `VELOCIDADE_MAXIMA`; um comando negativo reduz até 20% da velocidade própria, sem atingir valores inferiores a zero.

O gasto energético é artificial e serve apenas para comparar comportamentos:

```
energia += distancia * CUSTO_ENERGIA_DISTANCIA
           * (1 + PENALIZACAO_VELOCIDADE * (velocidade / VELOCIDADE_MAXIMA)**2)
           + CUSTO_ENERGIA_TEMPO * delta_tempo
```

Assim, para uma mesma distância, velocidades mais elevadas consomem mais energia. Também existe um custo basal por tempo, para não beneficiar indefinidamente ficar parado. As constantes ajustáveis encontram-se em `simulacao/configuracao.py`. Não se trata de um modelo físico calibrado de um robô real.

### Fitness do Algoritmo Genético

Em `simulacao/avaliacao.py`:

```
fitness = 1000 * entregues
        + 10 * soma(max(0, 1 - instante_entrega / duracao))
        - 0.015 * energia
        - 2.0 * colisoes_obstaculos
        - 0.05 * contactos_borda
        - 0.5 * bloqueios
```

Prioriza **recursos entregues no ninho**; diferencia controladores que completam a tarefa mais cedo e penaliza energia, colisões e bloqueios. Os pesos são parâmetros iniciais a justificar e testar, não constantes científicas. A fitness é uma **soma ponderada de critérios** (não é otimização de Pareto).

O AG implementa seleção por torneio, cruzamento uniforme, mutação gaussiana, elitismo e imigrantes aleatórios. Começa perto do genoma neutro, para não perder a referência funcional do Combinado. Treino, validação e teste têm sementes separadas.

## Treinar

Teste rápido de funcionamento (não serve para avaliar qualidade):

```bash
python treinar.py --geracoes 2 --populacao 4 --duracao 60 --sementes 42 123 --validacao 9001 9002 --processos 1
```

Treino mais exigente:

```bash
python treinar.py --geracoes 25 --populacao 30 --duracao 300 --sementes 853 1729 4096 7351 --validacao 9001 9002 9003 9004
```

A opção `--sementes-por-geracao` (por omissão, 4) controla quantas sementes de treino são utilizadas em cada geração. As mesmas sementes são usadas para todos os indivíduos dessa geração. O modo Combinado é avaliado apenas uma vez nas sementes de validação.

Ficheiros criados pelo treino: `melhor_genoma.npz` e `historico_treino.json`.

**Os genomas de versões anteriores (incluindo a v5) não são compatíveis com as duas redes 28–10–3.** Não reutilizar o antigo `melhor_genoma.npz`. Guarda uma cópia com outro nome antes de treinar novamente, se precisares dele.

## Avaliar a generalização

```bash
python avaliar.py --duracao 300 --sementes 7001 7002 7003 7004 --processos 1
```

Ou simplesmente `python avaliar.py` para as 20 sementes de teste predefinidas. Comparar entregas, `t50`, `t100`, energia, colisões, bloqueios e intervalos de confiança. Os controladores devem ser avaliados **nas mesmas sementes e com o mesmo modelo energético**. Não considerar melhores genomas apenas porque a fitness aumenta numa validação muito usada para selecionar modelos.

## Requisitos e limitações

- Os exploradores que passaram pelo ninho deixam feromonas de casa. Recolhedores sem recurso (roxos) **não** depositam feromonas; só voltam a depositar feromonas de casa após visitarem o ninho.
- Os agentes carregados depositam feromonas de comida. Continuam a utilizar feromonas e Boids como base; as redes não obtêm a posição global dos objetivos.
- O treino é por **Algoritmo Genético**; os pesos não são ajustados por retropropagação durante uma simulação.
- Ainda falta executar a avaliação experimental completa, incluindo ablação, escalabilidade e robustez, conforme o enunciado do projeto.
- A sintaxe e testes de integração sem interface gráfica foram verificados; a execução gráfica requer Pygame no computador do utilizador. Não há garantia de que a rede evoluída supere o Combinado antes dos testes independentes.
