"""Benchmark honesto: Random Walk vs Combinado vs Neuroevolução em sementes de TESTE.

    python avaliar.py                       # 20 sementes (7001-7020), 300 s
    python avaliar.py --duracao 120 --sementes 7001 7002 7003

As sementes de teste nunca são usadas no treino nem na validação. A comparação
Neuroevolução vs Combinado é emparelhada (mesma semente) e inclui um intervalo
de confiança a 95 % por bootstrap. Se o intervalo incluir 0, NÃO há evidência de
melhoria (a simulação é caótica: diferenças de 1-2 % são ruído).
"""
import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import numpy as np
from simulacao.avaliacao import simular
from simulacao.rede_neuronal import carregar_genoma

PASTA = Path(__file__).resolve().parent
METRICAS = ('fitness', 'entregues', 't50', 't100', 't_transp', 'conclusao_rapida',
            'progresso_regresso', 'cobertura',
            'energia', 'colisoes', 'bordas', 'tempo_bloqueado', 'bloqueios', 'concluiu')
# Quanto menor melhor:
MENOR_MELHOR = {'t50', 't100', 't_transp', 'bordas', 'energia', 'colisoes', 'tempo_bloqueado', 'bloqueios'}


def _tarefa(args):
    modo, semente, duracao, passo, genoma, cenario = args
    return modo, semente, simular(modo, semente, duracao, passo, genoma, cenario)


def bootstrap_ic(diferencas, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(diferencas, float)
    medias = rng.choice(d, size=(n, d.size), replace=True).mean(axis=1)
    return np.percentile(medias, [2.5, 97.5])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--duracao', type=float, default=300.0)
    p.add_argument('--passo', type=float, default=0.05)
    p.add_argument('--sementes', type=int, nargs='+', default=list(range(7001, 7021)))
    p.add_argument('--genoma', type=Path, default=PASTA / 'melhor_genoma.npz')
    p.add_argument('--processos', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument('--cenario', choices=('variado', 'proximo', 'agrupado', 'disperso', 'distante'),
                   default='variado')
    p.add_argument('--saida', type=Path, default=PASTA / 'resultados_benchmark.json')
    a = p.parse_args()

    genoma = None
    modos = ['Aleatório', 'Combinado']
    if a.genoma.exists():
        try:
            genoma = carregar_genoma(a.genoma)
        except ValueError as erro:
            raise SystemExit(str(erro))
        print(f'Genoma NEAT carregado: {genoma.resumo()}')
        modos.append('Neuroevolução')
    else:
        print('Aviso: sem melhor_genoma.npz — a saltar a Neuroevolução (corre treinar.py).')

    tarefas = [(m, s, a.duracao, a.passo, genoma, a.cenario)
               for m in modos for s in a.sementes]
    if a.processos > 1:
        with mp.Pool(a.processos) as pool:
            saidas = pool.map(_tarefa, tarefas, chunksize=1)
    else:
        saidas = [_tarefa(t) for t in tarefas]
    res = {m: {s: r for mm, s, r in saidas if mm == m} for m in modos}

    print(f"\n{len(a.sementes)} sementes de teste, {a.duracao:.0f} s, cenário {a.cenario} (média ± desvio-padrão)")
    print(f"{'modo':<15}" + ''.join(f'{k:>16}' for k in METRICAS))
    for m in modos:
        linha = f'{m:<15}'
        for k in METRICAS:
            v = np.array([res[m][s][k] for s in a.sementes])
            linha += f'{v.mean():>9.1f}±{v.std(ddof=1) if len(v) > 1 else 0:<5.1f} '
        print(linha)

    if 'Neuroevolução' in res:
        print('\nNeuroevolução − Combinado (emparelhado por semente)')
        for k in METRICAS:
            d = np.array([res['Neuroevolução'][s][k] - res['Combinado'][s][k] for s in a.sementes])
            base = np.mean([res['Combinado'][s][k] for s in a.sementes])
            lo, hi = bootstrap_ic(d)
            melhor = (d < 0) if k in MENOR_MELHOR else (d > 0)
            sinal = 'sem evidência' if lo <= 0 <= hi else ('MELHOR' if (hi < 0) == (k in MENOR_MELHOR) else 'PIOR')
            percentual = f'{100 * d.mean() / base:+6.2f}%' if abs(base) > 1e-9 else 'n/d'
            print(f'  {k:<10} Δ = {d.mean():+9.2f} ({percentual})  IC95% [{lo:+.2f}, {hi:+.2f}]  '
                  f'NE melhor em {int(melhor.sum())}/{len(d)} sementes  -> {sinal}')
    a.saida.write_text(json.dumps({m: {str(s): r for s, r in res[m].items()} for m in modos},
                                  indent=2), encoding='utf-8')
    print(f'\nResultados por semente guardados em {a.saida}')


if __name__ == '__main__':
    main()
