"""Fitness da Neuroevolução e medição reproduzível das simulações.

Entradas/decisões da MLP continuam estritamente LOCAIS. Só o avaliador pode
medir cobertura global do mapa e energia total do enxame.

Os termos secundários estão em [0, 1] e os pesos absolutos somam 855.
Por isso, uma entrega extra (+1000) vale sempre mais do que quaisquer ganhos
nos critérios secundários. Os pesos e escalas são propostas experimentais,
não parâmetros otimizados cientificamente.
"""
import os
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import math
import numpy as np
from simulacao.configuracao import (VELOCIDADE_MAXIMA, CUSTO_ENERGIA_DISTANCIA,
                                    PENALIZACAO_VELOCIDADE, CUSTO_ENERGIA_TEMPO,
                                    POSICAO_NINHO)
from simulacao.mundo import Simulacao

PESO_ENTREGA = 1000.0
PESO_RAPIDEZ = 350.0
PESO_CONCLUSAO = 150.0  # bónus pelo tempo até à última entrega
PESO_PROGRESSO_REGRESSO = 35.0  # progresso FINAL, não acumulado (evita loops)
PESO_COBERTURA = 80.0
PESO_ENERGIA = 120.0
PESO_COLISOES = 70.0
PESO_TEMPO_BLOQUEADO = 50.0

# Uma colisão com obstáculo conta como 1; contacto de borda como 0.25.
PESO_CONTATO_BORDA = 0.25
# Normalização fixa por agente e duração (independente do controlador):
REFERENCIA_COLISOES_POR_AGENTE_30S = 1.0


def limitar01(valor):
    return max(0.0, min(1.0, float(valor)))


def calcular_aptidao(tempos_entrega, duracao, energia, colisoes, colisoes_borda,
                     tempo_bloqueado, cobertura, num_agentes, num_recursos,
                     progresso_regresso=0.0):
    """Devolve (fitness, métricas normalizadas); não consulta dados globais da MLP.

    Escalas constantes para um mesmo cenário e duração, sem normalização por
    candidato. Energia de referência: todos os agentes a VELOCIDADE_MAXIMA até T.
    Contactos: 1 colisão por agente a cada 30 segundos de duração limite.
    Tempo bloqueado: proporção do máximo de agentes * duração limite.
    """
    if duracao <= 0 or num_agentes <= 0 or num_recursos <= 0:
        raise ValueError('Duração, número de agentes e recursos têm de ser positivos.')

    entregues = len(tempos_entrega)
    rapidez = limitar01(sum(max(0.0, 1.0 - t / duracao)
                            for t in tempos_entrega) / num_recursos)
    cobertura = limitar01(cobertura)
    # Só conta se TODAS as entregas foram efetuadas, nunca numa simulação incompleta.
    conclusao = limitar01(1.0 - max(tempos_entrega) / duracao) if entregues == num_recursos else 0.0
    progresso_regresso = limitar01(progresso_regresso)

    # O denominador não depende de quanto tempo um controlador precisou para
    # terminar: todos são comparados com a mesma duração limite.
    energia_limite = num_agentes * duracao * (
        VELOCIDADE_MAXIMA * CUSTO_ENERGIA_DISTANCIA
        * (1.0 + PENALIZACAO_VELOCIDADE) + CUSTO_ENERGIA_TEMPO
    )
    energia_n = limitar01(energia / energia_limite)
    eventos = max(0.0, colisoes) + PESO_CONTATO_BORDA * max(0.0, colisoes_borda)
    escala_colisoes = num_agentes * max(1.0, duracao / 30.0) * REFERENCIA_COLISOES_POR_AGENTE_30S
    colisoes_n = limitar01(eventos / escala_colisoes)
    bloqueado_n = limitar01(tempo_bloqueado / (num_agentes * duracao))

    pontuacao = (
        PESO_ENTREGA * entregues
        + PESO_RAPIDEZ * rapidez
        + PESO_CONCLUSAO * conclusao
        + PESO_PROGRESSO_REGRESSO * progresso_regresso
        + PESO_COBERTURA * cobertura
        - PESO_ENERGIA * energia_n
        - PESO_COLISOES * colisoes_n
        - PESO_TEMPO_BLOQUEADO * bloqueado_n
    )
    componentes = {
        'rapidez': rapidez,
        'conclusao_rapida': conclusao,
        'progresso_regresso': progresso_regresso,
        'cobertura': cobertura,
        'energia_norm': energia_n,
        'colisoes_norm': colisoes_n,
        'bloqueado_norm': bloqueado_n,
    }
    return float(pontuacao), componentes


def simular(modo, semente, duracao, passo=0.05, genoma=None, cenario='variado'):
    """Executa uma avaliação sem janela, devolvendo métricas físicas e fitness."""
    if duracao <= 0 or passo <= 0:
        raise ValueError('Duração e passo têm de ser positivos.')
    sim = Simulacao(modo, semente=int(semente),
                    genoma=genoma if modo == 'Neuroevolução' else None,
                    cenario=cenario)
    while sim.tempo_decorrido < duracao - 1e-9 and not sim.terminada:
        sim.atualizar(min(passo, duracao - sim.tempo_decorrido))

    n = len(sim.recursos)
    tempos = sorted(t for agente in sim.agentes for t in agente.tempos_entrega)
    entregues = len(tempos)
    energia = sum(agente.energia for agente in sim.agentes)
    colisoes = sum(agente.colisoes for agente in sim.agentes)
    bordas = sum(agente.colisoes_borda for agente in sim.agentes)
    bloqueios = sum(agente.bloqueios for agente in sim.agentes)
    tempo_bloqueado = sum(agente.tempo_bloqueado for agente in sim.agentes)
    transportes = [t for agente in sim.agentes for t in agente.tempos_transporte]
    transportes += [agente.relogio - agente.instante_recolha
                   for agente in sim.agentes if agente.transporta_recurso]
    metade = math.ceil(n / 2)
    t50 = tempos[metade - 1] if entregues >= metade else duracao
    t100 = tempos[n - 1] if entregues >= n else duracao

    # O progresso da carga é medido apenas no FINAL de cada simulação.
    # Se a formiga voltar a afastar-se, perde o progresso: não se pode
    # acumular recompensa andando para a frente e para trás.
    progresso_carga = sum(
        limitar01((a.distancia_recolha_ninho - a.posicao.distance_to(
            POSICAO_NINHO)) / max(1.0, a.distancia_recolha_ninho))
        for a in sim.agentes if a.transporta_recurso
    ) / n
    pontuacao, componentes = calcular_aptidao(
        tempos, duracao, energia, colisoes, bordas, tempo_bloqueado,
        sim.cobertura, len(sim.agentes), n,
        progresso_regresso=progresso_carga
    )
    return dict(
        fitness=pontuacao,
        entregues=float(entregues),
        t50=float(t50),
        t100=float(t100),
        t_transp=float(np.mean(transportes)) if transportes else 0.0,
        bordas=float(bordas),
        energia=float(energia),
        colisoes=float(colisoes),
        bloqueios=float(bloqueios),
        tempo_bloqueado=float(tempo_bloqueado),
        concluiu=float(sim.terminada),
        cenario=sim.cenario,
        **componentes,
    )


def agregar_resultados(resultados, peso_piores=0.20, fracao_piores=0.25):
    """Média das métricas e fitness robusta (média + pior quartil).

    Com quatro sementes, o pior quartil corresponde à pior semente.
    Não confundir fitness robusta (seleção durante treino) com a fitness
    individual apresentada no benchmark das sementes de teste.
    """
    if not resultados:
        raise ValueError('É necessária pelo menos uma avaliação.')
    if not 0 <= peso_piores <= 1 or not 0 < fracao_piores <= 1:
        raise ValueError('Peso dos piores e fração dos piores fora de intervalo.')
    medias = {chave: float(np.mean([r[chave] for r in resultados]))
              for chave in resultados[0] if chave not in ('fitness_robusta', 'cenario')}
    valores = sorted(float(r['fitness']) for r in resultados)
    quantos = max(1, math.ceil(fracao_piores * len(valores)))
    media_piores = float(np.mean(valores[:quantos]))
    medias['fitness_piores'] = media_piores
    medias['fitness_robusta'] = (1.0 - peso_piores) * medias['fitness'] + peso_piores * media_piores
    return medias


def avaliar(genoma, sementes, duracao, passo=0.05, modo='Neuroevolução',
           peso_piores=0.20, fracao_piores=0.25, cenario='variado'):
    """Usa sementes idênticas para todos os genomas da mesma geração."""
    if not sementes:
        raise ValueError('É preciso indicar pelo menos uma semente.')
    resultados = [simular(modo, s, duracao, passo, genoma, cenario) for s in sementes]
    return agregar_resultados(resultados, peso_piores, fracao_piores)


def _tarefa(args):
    return avaliar(*args)
