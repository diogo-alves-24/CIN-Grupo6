"""Treino evolutivo por NEAT da rede que corrige a política Combinado.

Exemplo completo:  python treinar.py --geracoes 30 --populacao 30 --duracao 300
Experiência rápida: python treinar.py --geracoes 3 --populacao 8 --duracao 60

Principais decisões (ver README):
  * NEAT: a topologia e os pesos evoluem em conjunto, com números de inovação,
    cruzamento alinhado, especiação e proteção de inovações estruturais;
  * um genoma por indivíduo, com 6 saídas (3 para procura, 3 para regresso);
  * fitness por entregas, rapidez, cobertura, energia, colisões e tempo bloqueado;
  * população inicial perto do genoma zero (= Combinado), com o indivíduo 0 neutro;
  * várias sementes por avaliação e seleção robusta (média + pior quartil);
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
from simulacao.avaliacao import avaliar, _tarefa, agregar_resultados
from simulacao.neat import PopulacaoNEAT, ParametrosNEAT
from simulacao.rede_neuronal import ENTRADAS, SAIDAS, guardar_genoma

PASTA = Path(__file__).resolve().parent
ARQUIVO_GENOMA = PASTA / 'melhor_genoma.npz'
ARQUIVO_HISTORICO = PASTA / 'historico_treino.json'


def avaliar_lote(pool, genomas, sementes, duracao, passo, modo='Neuroevolução',
                 peso_piores=0.20, fracao_piores=0.25, cenario='variado'):
    tarefas = [(g, sementes, duracao, passo, modo, peso_piores, fracao_piores,
                cenario) for g in genomas]
    if pool is None:
        return [_tarefa(t) for t in tarefas]
    return pool.map(_tarefa, tarefas, chunksize=1)




def treinar(geracoes, populacao, duracao, sementes_treino, sementes_geracao,
            validacao, taxa_mutacao, desvio_mutacao, sigma_inicial, passo,
            semente_aleatoria, processos, peso_piores=0.20, fracao_piores=0.25,
            cenario='variado', paciencia_mutacao=5, curriculo=False,
            parametros=None):
    if geracoes < 1 or populacao < 4 or duracao <= 0 or not sementes_treino or not validacao:
        raise ValueError('Verifica gerações, população, duração e sementes.')
    if sementes_geracao < 1 or not 0 <= peso_piores <= 1 or not 0 < fracao_piores <= 1:
        raise ValueError('Verifica sementes por geração, peso e fração dos piores.')
    if set(sementes_treino) & set(validacao):
        raise ValueError('Sementes de treino e validação não devem coincidir.')
    if paciencia_mutacao < 1:
        raise ValueError('A paciência da mutação deve ser positiva.')
    k = min(sementes_geracao, len(sementes_treino))
    rng = np.random.default_rng(semente_aleatoria)
    pool = mp.Pool(processos) if processos > 1 else None
    t0 = time.time()
    try:
        parametros = parametros or ParametrosNEAT()
        parametros.taxa_mutacao = taxa_mutacao
        parametros.desvio_mutacao = desvio_mutacao
        parametros.sigma_inicial = sigma_inicial
        # O genoma 0 é neutro (== Combinado): ponto de partida garantido.
        populacao_neat = PopulacaoNEAT(populacao, ENTRADAS, SAIDAS, rng, parametros)

        # Referência (Combinado) nas sementes de validação: uma sim por semente, em paralelo.
        ref_por_semente = ([avaliar(None, [s], duracao, passo, 'Combinado',
                                    cenario=cenario) for s in validacao]
                            if pool is None else pool.map(
                                _tarefa, [(None, [s], duracao, passo, 'Combinado',
                                           0.20, 0.25, cenario) for s in validacao],
                                chunksize=1))
        ref = agregar_resultados(ref_por_semente, peso_piores, fracao_piores)
        print(f"Combinado (validação): fitness média {ref['fitness']:.1f} "
              f"robusta {ref['fitness_robusta']:.1f} | entregas {ref['entregues']:.1f} "
              f"| t100 {ref['t100']:.1f}s | energia {ref['energia']:.1f} | colisões {ref['colisoes']:.1f}", flush=True)

        melhor_validacao = -float('inf')
        melhor_sonda_treino = -float('inf')
        geracoes_sem_melhoria = 0
        # Mesmas sementes de sondagem em todas as gerações; só TREINO.
        # Não usar validação para ajustar a mutação.
        sementes_sonda = sementes_treino[:min(4, len(sementes_treino))]
        historico = []
        for geracao in range(1, geracoes + 1):
            # Mesmas sementes para TODOS os indivíduos da geração (comparação justa),
            # rodando por um conjunto maior para não memorizar uma arena.
            inicio = ((geracao - 1) * k) % len(sementes_treino)
            sem_ger = [sementes_treino[(inicio + j) % len(sementes_treino)] for j in range(k)]
            # Currículo opcional: começa perto do ninho, avança para mapas
            # mais difíceis, e termina numa mistura de todos os cenários.
            etapas = ('proximo', 'agrupado', 'disperso', 'distante', 'variado')
            cenario_ger = (etapas[min(4, (geracao - 1) * 5 // geracoes)]
                          if curriculo and cenario == 'variado' else cenario)
            individuos = populacao_neat.genomas
            metricas = avaliar_lote(pool, individuos, sem_ger, duracao, passo,
                                    peso_piores=peso_piores, fracao_piores=fracao_piores,
                                    cenario=cenario_ger)
            pontuacoes = np.asarray([m['fitness_robusta'] for m in metricas])
            ordem = np.argsort(pontuacoes)[::-1]
            populacao_neat.registar(pontuacoes)
            elite = individuos[ordem[0]].copiar()
            # Sonda fixa avalia o progresso do treino sem recorrer à validação.
            sonda = avaliar(elite, sementes_sonda, duracao, passo, cenario=cenario)
            if sonda['fitness_robusta'] > melhor_sonda_treino + 1e-6:
                melhor_sonda_treino = sonda['fitness_robusta']
                geracoes_sem_melhoria = 0
            else:
                geracoes_sem_melhoria += 1
            fator_mutacao = min(3.0, 1.0 + geracoes_sem_melhoria // paciencia_mutacao)
            val_por_semente = (pool.map(
                _tarefa, [(elite, [s], duracao, passo, 'Neuroevolução',
                           0.20, 0.25, cenario) for s in validacao],
                chunksize=1) if pool else
                [avaliar(elite, [s], duracao, passo, cenario=cenario) for s in validacao])
            val = agregar_resultados(val_por_semente, peso_piores, fracao_piores)
            if val['fitness_robusta'] > melhor_validacao:
                melhor_validacao = val['fitness_robusta']
                guardar_genoma(ARQUIVO_GENOMA, elite,
                         fitness=melhor_validacao,
                         fitness_media=val['fitness'], geracao=geracao,
                         validacao_entregues=val['entregues'], validacao_t100=val['t100'])
            linha = dict(geracao=geracao, melhor_treino=float(pontuacoes[ordem[0]]),
                         media_treino=float(pontuacoes.mean()), validacao=val,
                         melhor_validacao=melhor_validacao, sementes_treino=sem_ger,
                         referencia_combinado=ref, segundos=time.time() - t0,
                         fator_mutacao=fator_mutacao, sonda_treino=sonda['fitness_robusta'],
                         cenario=cenario, cenario_geracao=cenario_ger,
                         especies=len(populacao_neat.especies),
                         limiar_compatibilidade=populacao_neat.limiar,
                         tamanhos_especies=sorted((len(e.membros) for e in populacao_neat.especies),
                                                  reverse=True),
                         nos_ocultos_elite=len(elite.ids_ocultos),
                         conexoes_ativas_elite=elite.conexoes_ativas,
                         conexoes_totais_elite=len(elite.conexoes),
                         nos_ocultos_media=float(np.mean([len(g.ids_ocultos) for g in individuos])))
            historico.append(linha)
            ARQUIVO_HISTORICO.write_text(json.dumps(historico, indent=2), encoding='utf-8')
            print(f"Ger {geracao:>3}/{geracoes} | treino melhor {pontuacoes[ordem[0]]:7.1f} média {pontuacoes.mean():7.1f} "
                  f"| val robusta {val['fitness_robusta']:7.1f} "
                  f"({val['fitness_robusta'] - ref['fitness_robusta']:+.1f} vs Comb.) "
                  f"entregas {val['entregues']:.1f} cobertura {100*val['cobertura']:.1f}% "
                  f"energia {val['energia']:.0f} bloqueado {val['tempo_bloqueado']:.1f}s "
                  f"t100 {val['t100']:.0f}s | mapa {cenario_ger} | mutação x{fator_mutacao:.1f} "
                  f"| esp {len(populacao_neat.especies)} | rede {elite.resumo()} "
                  f"| melhor val {melhor_validacao:7.1f} "
                  f"| {time.time() - t0:.0f}s", flush=True)

            if geracao < geracoes:
                populacao_neat.reproduzir(fator_mutacao)
    finally:
        if pool is not None:
            pool.close(); pool.join()
    print(f'Melhor genoma NEAT validado: {ARQUIVO_GENOMA}', flush=True)
    print('Confirma o ganho em sementes NUNCA usadas: python avaliar.py', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Evoluir rede neuronal com NEAT')
    p.add_argument('--geracoes', type=int, default=30)
    p.add_argument('--populacao', type=int, default=30)
    p.add_argument('--duracao', type=float, default=300.0)
    p.add_argument('--passo', type=float, default=0.05)
    p.add_argument('--sementes', type=int, nargs='+', default=[101, 102, 103, 104, 105, 106, 107, 108],
                   help='conjunto de sementes de TREINO (rodado ao longo das gerações)')
    p.add_argument('--sementes-por-geracao', type=int, default=4)
    p.add_argument('--validacao', type=int, nargs='+', default=[9001, 9002, 9003, 9004])
    p.add_argument('--taxa-mutacao', type=float, default=0.10,
                   help='fração de pesos/viéses perturbados por mutação')
    p.add_argument('--desvio-mutacao', type=float, default=0.10)
    p.add_argument('--p-nova-ligacao', type=float, default=0.15,
                   help='probabilidade de mutação estrutural: adicionar conexão')
    p.add_argument('--p-novo-no', type=float, default=0.05,
                   help='probabilidade de mutação estrutural: adicionar nó')
    p.add_argument('--especies-alvo', type=int, default=5,
                   help='nº de espécies para o ajuste automático do limiar de compatibilidade')
    p.add_argument('--estagnacao', type=int, default=15,
                   help='gerações sem melhoria até extinguir uma espécie')
    p.add_argument('--ligacao-inicial', type=float, default=1.0,
                   help='fração das ligações entrada→saída na topologia inicial (0-1]')
    p.add_argument('--sigma-inicial', type=float, default=0.05)
    p.add_argument('--semente-aleatoria', type=int, default=2026)
    p.add_argument('--peso-piores', type=float, default=0.20,
                   help='peso dos piores cenários na seleção evolutiva (0 a 1)')
    p.add_argument('--fracao-piores', type=float, default=0.25,
                   help='fração das sementes consideradas difíceis (0 a 1)')
    p.add_argument('--processos', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument('--cenario', choices=('variado', 'proximo', 'agrupado', 'disperso', 'distante'),
                   default='variado', help='variado alterna cenários pela semente')
    p.add_argument('--paciencia-mutacao', type=int, default=5,
                   help='gerações sem melhoria da sonda de treino antes de aumentar mutação')
    p.add_argument('--curriculo', action='store_true',
                   help='evolui de mapas próximos para dispersos e distantes; valida sempre em variados')
    a = p.parse_args()
    if not 0 < a.ligacao_inicial <= 1:
        p.error('--ligacao-inicial tem de estar em (0, 1].')
    parametros = ParametrosNEAT(p_nova_ligacao=a.p_nova_ligacao, p_novo_no=a.p_novo_no,
                                especies_alvo=a.especies_alvo, estagnacao=a.estagnacao,
                                ligacao_inicial=a.ligacao_inicial)
    treinar(a.geracoes, a.populacao, a.duracao, a.sementes, a.sementes_por_geracao,
            a.validacao, a.taxa_mutacao, a.desvio_mutacao, a.sigma_inicial, a.passo,
            a.semente_aleatoria, a.processos, a.peso_piores, a.fracao_piores,
            a.cenario, a.paciencia_mutacao, a.curriculo, parametros)
