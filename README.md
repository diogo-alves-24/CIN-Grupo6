# Forrageamento Espacial — Neuroevolução NEAT (sem Boids)

Simulador de formigas em **Python + Pygame + NumPy**. Mantém três modos:

- **Aleatório:** Random Walk, sem coordenação.
- **Combinado:** procura e regresso com feromonas, sensores locais e fuga de ciclos. **Não usa Boids.** O nome é mantido para facilitar a comparação com as versões anteriores.
- **Neuroevolução:** uma rede neuronal evoluída por **NEAT** (topologia e pesos), com 6 saídas — 3 para a procura e 3 para o regresso — que ajustam direção e velocidade do controlador Combinado. Substitui as duas MLP fixas 28–10–3 da versão anterior.

## Instalação

```bash
python -m pip install -r requirements.txt
python main.py
```

Modo Neuroevolução só aparece operacional depois de executar `python treinar.py` nesta versão. Os genomas das versões anteriores (MLP) são **incompatíveis** com o formato NEAT e são recusados com mensagem clara.

**Controlos:** `1/2/3` selecionam o modo; `P` velocidade (1x, 2x, 4x, 8x), `T` tipo de feromona, `F` visibilidade das feromonas, `S` sensores, `H` painel, `Espaço` pausa, `R` reinicia, `C` alterna cenário e reinicia, `M` menu, `ESC` sai.

## Neuroevolução com NEAT

Implementação própria em NumPy (`simulacao/neat.py`), sem dependências novas.

| Componente | Decisão |
| --- | --- |
| Genoma | viéses dos nós + conexões com **número de inovação**; só redes *feedforward* (a memória já entra como sinais locais) |
| Rede | 28 entradas locais, 6 saídas `tanh`: 0–2 usadas sem recurso, 3–5 com recurso (lateral, frontal, velocidade); nós ocultos criados pela evolução |
| Topologia inicial | mínima: entradas ligadas às saídas (168 pesos + 6 viéses). Indivíduo 0 = genoma neutro (tudo a zero = Combinado) |
| Mutação | pesos/viéses (gaussiana, em dois níveis, escalada pela mutação adaptativa), **adicionar conexão** (sem ciclos), **adicionar nó** (divide uma conexão), ativar/desativar |
| Cruzamento | alinhado por inovação; o filho herda a topologia do progenitor mais apto |
| Especiação | distância δ = c₁·E + c₂·D + c₃·W̄; o **limiar adapta-se** para manter `--especies-alvo` espécies; *fitness sharing* |
| Seleção | elitismo global (2), campeão de espécies com ≥ 5 membros, sobrevivência de 30 %, extinção por estagnação (protege as 2 melhores espécies) |

Mantêm-se: fitness multiobjetivo, avaliação robusta (80 % média + 20 % pior quartil), sonda de treino para a mutação adaptativa, validação independente, currículo e paralelismo. Deixou de haver imigrantes aleatórios: a diversidade vem da especiação.

Parâmetros novos de `treinar.py`: `--p-nova-ligacao`, `--p-novo-no`, `--especies-alvo`, `--estagnacao`, `--ligacao-inicial`. `historico_treino.json` regista agora espécies, limiar de compatibilidade e complexidade (nós/conexões) da elite e média da população.

## Alterações da versão v8

### 1. Boids retirados completamente

Removidos os cálculos de **separação, alinhamento e coesão**, tanto do controlador Combinado como dos sensores e da entrada neuronal. A interação descentralizada assenta agora em **feromonas**. Continua a existir desvio local de obstáculos, que não é flocking/Boids.

As entradas da rede continuam a ser 28 sinais locais: as posições usadas para sinais de Boids foram substituídas por **direção de fuga** e **contraste da feromona de comida entre esquerda e direita**. Os sinais de memória da trajetória são locais; não fornecem a posição global do ninho ou recursos. A visualização/avaliador conhecem o mapa, mas a rede não.

### 2. Quatro cenários reproduzíveis

| Cenário | Características |
| --- | --- |
| `proximo` | Comida relativamente próxima do ninho; sem obstáculos |
| `agrupado` | Mapa original, com quatro aglomerados e cinco obstáculos |
| `disperso` | Recursos espalhados pela arena; cinco obstáculos |
| `distante` | Recursos próximos dos quatro cantos; seis obstáculos |

`variado` escolhe o cenário por `semente % 4` na ordem `proximo`, `agrupado`, `disperso`, `distante`. Isto **não** é uma decisão de um agente: é preparação do cenário pelo avaliador, antes de cada simulação. Todos os candidatos são avaliados nas mesmas sementes e condições.

Na interface o cenário inicial é `agrupado`. Usa `C` para alternar durante a demonstração. No treino e no benchmark, o padrão é `variado`.

### 3. Fitness com tempo de conclusão, energia e progresso real

Para cada cenário:

```text
fitness = 1000 * entregas
        + 350 * rapidez_das_entregas
        + 150 * conclusao_rapida
        +  35 * progresso_regresso
        +  80 * cobertura
        - 120 * energia_normalizada
        -  70 * colisoes_normalizadas
        -  50 * tempo_bloqueado_normalizado
```

- **Entregas:** recursos efetivamente entregues (0–36).
- **Rapidez:** soma normalizada de `(1 - instante_entrega/duracao)`. Recompensa entregas antecipadas, mesmo que faltem outras.
- **Conclusão rápida:** só se foram entregues **todos** os recursos, recebe `(1 - t100/duracao)`. Diferencia explicitamente o tempo para terminar tudo.
- **Progresso de regresso:** apenas para formigas que estão **a carregar comida no final**. Compara distância ao ninho no momento da recolha com a distância final, normalizada. Recompensa final, **não** incrementos positivos a cada passo, para não explorar ciclos de aproximação/afastamento. Esta distância global é calculada **só pelo avaliador**; a rede não a recebe.
- **Cobertura:** fração de regiões diferentes visitadas pelo enxame; uma região conta apenas uma vez.
- **Energia:** custo por distância percorrida e velocidade, mais custo basal por tempo.
- **Colisões e bloqueios:** penalizam contacto com obstáculos/bordas e períodos de movimento improdutivo.

Os termos auxiliares estão em `[0,1]` e os seus pesos absolutos somam **855**, menos de 1000. Assim uma entrega adicional domina qualquer combinação de diferenças nos termos auxiliares da mesma avaliação. Os pesos são experimentais, não garantias de maior desempenho.

### 4. Deteção de ciclos

Mantém a memória de 5 segundos de deslocamento/rotação, e deteta bloqueio por deslocamento líquido reduzido **ou** órbita recente persistente. Ativa fuga local durante um intervalo curto. Enquanto uma formiga com comida está em fuga, **não deposita feromona de comida**, para não reforçar um trilho circular. O sistema não usa teletransporte nem coordenação centralizada.

### 5. Evolução (ver secção NEAT acima)

- **Mutação adaptativa:** quando não há melhoria durante `--paciencia-mutacao` gerações numa sonda fixa formada apenas por sementes de treino, aumenta a amplitude até 3x. Não usa a validação para adaptar a mutação.
- **Avaliação robusta:** 80% da fitness média + 20% da média do pior quartil das sementes.
- Suporte opcional a **currículo** de mapas (`--curriculo`): `proximo` → `agrupado` → `disperso` → `distante` → `variado`. A validação continua sempre variada e independente.
- Treino sem Pygame desenhar e possibilidade de vários processos (`--processos`).

## Testes rápidos

```bash
python testar_fitness.py
python testar_projeto.py
python testar_neat.py
python treinar.py --geracoes 2 --populacao 4 --duracao 30 \
  --sementes 100 101 102 103 --validacao 9001 9002 --processos 1
```

Estes testes validam a execução e algumas propriedades do código, **não** demonstram melhoria face ao Combinado.

## Treinar o modelo

Experiência com mapas variados, 180 segundos:

```bash
python treinar.py --geracoes 30 --populacao 60 --duracao 180 \
  --sementes 101 102 103 104 105 106 107 108 \
  --sementes-por-geracao 4 \
  --validacao 9001 9002 9003 9004 \
  --processos 4
```

Para comparar o efeito do currículo, acrescenta `--curriculo`. Se a tarefa com 180 s for demasiado difícil e houver poucas entregas, testa primeiro `--duracao 300`; não assumas que 180 s dá sempre melhor treino. Antes de aumentar para 60 indivíduos, valida o desempenho com 12–30.

São gerados `melhor_genoma.npz` (genoma NEAT, versão `neat_v1`) e `historico_treino.json`.

## Benchmark com sementes nunca usadas

```bash
python avaliar.py --duracao 180 --sementes 7001 7002 7003 7004 7005 7006 7007 7008 --processos 4
```

O benchmark faz comparações emparelhadas entre Random Walk, Combinado e Neuroevolução, com **as mesmas sementes e cenários**, e produz diferenças e intervalos de confiança bootstrap. Também podes forçar `--cenario agrupado` para medir só o mapa original.

**Importante:** compara sempre treino e benchmark com a mesma duração e os mesmos pesos de fitness. Genomas e pontuações de versões anteriores devem ser tratados como experiências separadas. O facto de o código executar corretamente não demonstra que o modelo neuronal já supera o Combinado; são necessários resultados em sementes novas.
