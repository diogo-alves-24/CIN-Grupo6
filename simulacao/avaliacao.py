"""Avaliação reproduzível dos controladores e aptidão usada pelo AG.

A aptidão maximiza o número de recursos realmente ENTREGUES. Entre soluções
com igual número de entregas, favorece entregas mais cedo, menos energia
(simula um custo maior para velocidades elevadas), menos colisões e bloqueios.
Os pesos são hipóteses de modelação, ajustáveis por experimentação.
"""
import os
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import numpy as np
from simulacao.mundo import Simulacao

PESO_ENTREGA = 1000.0
PESO_RAPIDEZ = 10.0    # bónus por entrega: 0 a 10, normalizado pela duração
PESO_ENERGIA = 0.015
PESO_COLISAO = 2.0
PESO_BORDA = 0.05
PESO_BLOQUEIO = 0.5


def calcular_aptidao(tempos_entrega, duracao, energia, colisoes, colisoes_borda, bloqueios):
    """Recompensa entregas e eficiência; zero entregas nunca recebe bónus de rapidez."""
    entregues = len(tempos_entrega)
    rapidez = sum(max(0.0, 1.0 - instante / duracao) for instante in tempos_entrega)
    return (
        PESO_ENTREGA * entregues
        + PESO_RAPIDEZ * rapidez
        - PESO_ENERGIA * energia
        - PESO_COLISAO * colisoes
        - PESO_BORDA * colisoes_borda
        - PESO_BLOQUEIO * bloqueios
    )


def simular(modo, semente, duracao, passo=0.05, genoma=None):
    """Executa a simulação sem janela. Todas as métricas têm unidades explícitas."""
    if duracao <= 0 or passo <= 0:
        raise ValueError('Duração e passo têm de ser positivos.')
    sim = Simulacao(modo, semente=int(semente), genoma=genoma if modo == 'Neuroevolução' else None)
    while sim.tempo_decorrido < duracao - 1e-9 and not sim.terminada:
        sim.atualizar(min(passo, duracao - sim.tempo_decorrido))

    n = len(sim.recursos)
    tempos = sorted(t for agente in sim.agentes for t in agente.tempos_entrega)
    entregues = len(tempos)
    energia = sum(agente.energia for agente in sim.agentes)
    colisoes = sum(agente.colisoes for agente in sim.agentes)
    bordas = sum(agente.colisoes_borda for agente in sim.agentes)
    bloqueios = sum(agente.bloqueios for agente in sim.agentes)
    transportes = [t for agente in sim.agentes for t in agente.tempos_transporte]
    transportes += [agente.relogio - agente.instante_recolha
                   for agente in sim.agentes if agente.transporta_recurso]
    t50 = tempos[n // 2 - 1] if entregues >= n // 2 else duracao
    t100 = tempos[n - 1] if entregues >= n else duracao

    return dict(
        fitness=float(calcular_aptidao(tempos, duracao, energia, colisoes, bordas, bloqueios)),
        entregues=float(entregues),
        t50=float(t50),
        t100=float(t100),
        t_transp=float(np.mean(transportes)) if transportes else 0.0,
        bordas=float(bordas),
        energia=float(energia),
        colisoes=float(colisoes),
        bloqueios=float(bloqueios),
        concluiu=float(sim.terminada),
    )


def avaliar(genoma, sementes, duracao, passo=0.05, modo='Neuroevolução'):
    """Calcula a média sobre as mesmas sementes para comparação justa."""
    if not sementes:
        raise ValueError('É preciso indicar pelo menos uma semente.')
    resultados = [simular(modo, s, duracao, passo, genoma) for s in sementes]
    return {chave: float(np.mean([resultado[chave] for resultado in resultados]))
            for chave in resultados[0]}


def _tarefa(args):
    return avaliar(*args)
