"""Treino evolutivo (AG) da MLP que corrige a política Combinado.

Exemplo completo:  python treinar.py --geracoes 30 --populacao 30 --duracao 300
Experiência rápida: python treinar.py --geracoes 3 --populacao 8 --duracao 60

Principais decisões (ver README):
  * duas MLP independentes (procura e regresso), evoluídas em conjunto;
  * aptidão por entregas, rapidez, energia, colisões e bloqueios;
  * cada MLP controla direção e velocidade (3.ª saída);
  * população inicial concentrada à volta do genoma zero (= Combinado) em vez de
    pesos aleatórios N(0, 0.35), que dão controladores inúteis;
  * várias sementes por avaliação (a simulação é caótica e muito ruidosa);
  * conjunto de validação separado, usado só para escolher o genoma guardado;
  * avaliação dos indivíduos em paralelo (multiprocessing).
"""
import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path
import numpy as np
from simulacao.avaliacao import avaliar, _tarefa
from simulacao.rede_neuronal import TAMANHO_GENOMA, TAMANHO_REDE, ENTRADAS, OCULTOS

# Camadas de saída das DUAS redes: mutação ligeiramente mais forte.
# O genoma contém a rede de procura seguida da rede de regresso.
ESCALA = np.ones(TAMANHO_GENOMA)
for inicio in (0, TAMANHO_REDE):
    comeco_saida = inicio + ENTRADAS * OCULTOS + OCULTOS
    ESCALA[comeco_saida:inicio + TAMANHO_REDE] = 3.0

PASTA = Path(__file__).resolve().parent
ARQUIVO_GENOMA = PASTA / 'melhor_genoma.npz'
ARQUIVO_HISTORICO = PASTA / 'historico_treino.json'


def avaliar_lote(pool, genomas, sementes, duracao, passo, modo='Neuroevolução'):
    tarefas = [(g, sementes, duracao, passo, modo) for g in genomas]
    if pool is None:
        return [_tarefa(t) for t in tarefas]
    return pool.map(_tarefa, tarefas, chunksize=1)


def media_metricas(lista):
    return {k: float(np.mean([m[k] for m in lista])) for k in lista[0]}


def treinar(geracoes, populacao, duracao, sementes_treino, sementes_geracao,
            validacao, taxa_mutacao, desvio_mutacao, sigma_inicial, passo,
            semente_aleatoria, processos):
    if geracoes < 1 or populacao < 4 or duracao <= 0 or not sementes_treino:
        raise ValueError('Verifica gerações, população, duração e sementes.')
    k = min(sementes_geracao, len(sementes_treino))
    rng = np.random.default_rng(semente_aleatoria)
    pool = mp.Pool(processos) if processos > 1 else None
    t0 = time.time()
    try:
        individuos = rng.normal(0.0, sigma_inicial, (populacao, TAMANHO_GENOMA)) * ESCALA
        individuos[0] = 0.0   # genoma neutro == Combinado: ponto de partida garantido

        # Referência (Combinado) nas sementes de validação: uma sim por semente, em paralelo.
        ref = media_metricas([avaliar(None, [s], duracao, passo, 'Combinado') for s in validacao]) \
            if pool is None else media_metricas(
                pool.map(_tarefa, [(None, [s], duracao, passo, 'Combinado') for s in validacao], chunksize=1))
        print(f"Combinado (validação): aptidão {ref['fitness']:.1f} | entregas {ref['entregues']:.1f} "
              f"| t100 {ref['t100']:.1f}s | energia {ref['energia']:.1f} | colisões {ref['colisoes']:.1f}", flush=True)

        melhor_validacao = -float('inf')
        historico = []
        for geracao in range(1, geracoes + 1):
            # Mesmas sementes para TODOS os indivíduos da geração (comparação justa),
            # rodando por um conjunto maior para não memorizar uma arena.
            inicio = ((geracao - 1) * k) % len(sementes_treino)
            sem_ger = [sementes_treino[(inicio + j) % len(sementes_treino)] for j in range(k)]
            metricas = avaliar_lote(pool, individuos, sem_ger, duracao, passo)
            pontuacoes = np.asarray([m['fitness'] for m in metricas])
            ordem = np.argsort(pontuacoes)[::-1]
            elite = individuos[ordem[0]].copy()
            val = media_metricas(pool.map(_tarefa, [(elite, [s], duracao, passo, 'Neuroevolução')
                                                    for s in validacao], chunksize=1)
                                 if pool else [avaliar(elite, [s], duracao, passo) for s in validacao])
            if val['fitness'] > melhor_validacao:
                melhor_validacao = val['fitness']
                np.savez(ARQUIVO_GENOMA, genoma=elite, fitness=melhor_validacao, geracao=geracao,
                         validacao_entregues=val['entregues'], validacao_t100=val['t100'])
            linha = dict(geracao=geracao, melhor_treino=float(pontuacoes[ordem[0]]),
                         media_treino=float(pontuacoes.mean()), validacao=val,
                         melhor_validacao=melhor_validacao, sementes_treino=sem_ger,
                         referencia_combinado=ref, segundos=time.time() - t0)
            historico.append(linha)
            ARQUIVO_HISTORICO.write_text(json.dumps(historico, indent=2), encoding='utf-8')
            print(f"Ger {geracao:>3}/{geracoes} | treino melhor {pontuacoes[ordem[0]]:7.1f} média {pontuacoes.mean():7.1f} "
                  f"| val {val['fitness']:7.1f} ({val['fitness'] - ref['fitness']:+.1f} pts vs Comb.) "
                  f"entregas {val['entregues']:.1f} energia {val['energia']:.0f} "
                  f"t100 {val['t100']:.0f}s | melhor val {melhor_validacao:7.1f} "
                  f"| {time.time() - t0:.0f}s", flush=True)

            def selecionar():
                escolhas = rng.choice(populacao, size=3, replace=False)
                return individuos[escolhas[np.argmax(pontuacoes[escolhas])]]

            novos = [individuos[ordem[0]].copy(), individuos[ordem[1]].copy()]   # elitismo
            while len(novos) < populacao:
                pai, mae = selecionar(), selecionar()
                mascara = rng.random(TAMANHO_GENOMA) < 0.5                       # cruzamento uniforme
                filho = np.where(mascara, pai, mae).copy()
                indices = rng.random(TAMANHO_GENOMA) < taxa_mutacao              # mutação gaussiana
                filho[indices] += rng.normal(0, desvio_mutacao, indices.sum()) * ESCALA[indices]
                np.clip(filho, -3, 3, out=filho)
                novos.append(filho)
            # Imigrantes aleatórios (mesma escala da população inicial, não pesos "grandes").
            for j in range(max(1, populacao // 10)):
                novos[-1 - j] = rng.normal(0, sigma_inicial, TAMANHO_GENOMA) * ESCALA
            individuos = np.asarray(novos)
    finally:
        if pool is not None:
            pool.close(); pool.join()
    print(f'Melhor rede validada: {ARQUIVO_GENOMA}', flush=True)
    print('Confirma o ganho em sementes NUNCA usadas: python avaliar.py', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Evoluir rede MLP através de AG')
    p.add_argument('--geracoes', type=int, default=30)
    p.add_argument('--populacao', type=int, default=30)
    p.add_argument('--duracao', type=float, default=300.0)
    p.add_argument('--passo', type=float, default=0.05)
    p.add_argument('--sementes', type=int, nargs='+', default=[101, 102, 103, 104, 105, 106, 107, 108],
                   help='conjunto de sementes de TREINO (rodado ao longo das gerações)')
    p.add_argument('--sementes-por-geracao', type=int, default=4)
    p.add_argument('--validacao', type=int, nargs='+', default=[9001, 9002, 9003, 9004])
    p.add_argument('--taxa-mutacao', type=float, default=0.10)
    p.add_argument('--desvio-mutacao', type=float, default=0.05)
    p.add_argument('--sigma-inicial', type=float, default=0.05)
    p.add_argument('--semente-aleatoria', type=int, default=2026)
    p.add_argument('--processos', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = p.parse_args()
    treinar(a.geracoes, a.populacao, a.duracao, a.sementes, a.sementes_por_geracao,
            a.validacao, a.taxa_mutacao, a.desvio_mutacao, a.sigma_inicial, a.passo,
            a.semente_aleatoria, a.processos)
