"""Testes do motor NEAT (sem simulações longas).  Executar: python testar_neat.py"""
import tempfile
from pathlib import Path
import numpy as np
from simulacao.neat import (Conexao, Genoma, Inovacoes, ParametrosNEAT, PopulacaoNEAT,
                            RedeCompilada, cruzar, distancia, mutar)
from simulacao.rede_neuronal import (ENTRADAS, SAIDAS, RedeNeuronal, carregar_genoma,
                                     genoma_neutro, guardar_genoma)


def acíclico(g):
    return all(not g.existe_caminho(c.saida, c.entrada) for c in g.conexoes.values())


def main():
    rng = np.random.default_rng(0)
    p = ParametrosNEAT(p_nova_ligacao=0.5, p_novo_no=0.3)

    # 1. Genoma neutro -> saídas exatamente nulas.
    assert not np.any(RedeNeuronal(genoma_neutro()).prever(rng.normal(size=ENTRADAS), True))

    # 2. Rede compilada == cálculo manual numa topologia com um nó oculto.
    inov = Inovacoes(3, 2)
    g = Genoma(3, 2)
    for o, d, w in ((0, 5, .7), (1, 5, -.4), (5, 3, 1.3), (2, 4, .5), (5, 4, -.8)):
        g.conexoes[inov.ligacao(o, d)] = Conexao(inov.ligacao(o, d), o, d, w)
    g.vies[5], g.vies[3], g.vies[4] = .1, -.2, .3
    x = np.array([.3, -.9, .6])
    oculto = np.tanh(.7 * x[0] - .4 * x[1] + .1)
    esperado = [np.tanh(1.3 * oculto - .2), np.tanh(.5 * x[2] - .8 * oculto + .3)]
    assert np.allclose(RedeCompilada(g).prever(x), esperado)

    # 3. Muitas mutações + cruzamentos: sempre acíclico, inovações coerentes.
    inov = Inovacoes(ENTRADAS, SAIDAS)
    populacao = [Genoma.minimo(ENTRADAS, SAIDAS, inov, rng, .05) for _ in range(12)]
    for ronda in range(60):
        for i, g in enumerate(populacao):
            mutar(g, inov, rng, p)
            g.fitness = rng.random()
        filhos = []
        for _ in range(12):
            a, b = rng.choice(populacao, 2, replace=False)
            if b.fitness > a.fitness:
                a, b = b, a
            filhos.append(cruzar(a, b, rng))
        populacao = filhos
        for g in populacao:
            assert acíclico(g)
            for c in g.conexoes.values():           # inovação ↔ (origem, destino)
                assert inov.ligacao(c.entrada, c.saida) == c.inov
            for c in g.conexoes.values():
                assert c.entrada in g.vies or c.entrada < ENTRADAS
                assert c.saida in g.vies
            RedeCompilada(g).prever(rng.normal(size=ENTRADAS))
    assert max(len(g.ids_ocultos) for g in populacao) > 0, 'nunca cresceu'

    # 4. Dividir a mesma conexão em genomas distintos dá o mesmo nó/inovações.
    a, b = (Genoma.minimo(ENTRADAS, SAIDAS, inov, rng) for _ in range(2))
    ca = next(iter(a.conexoes.values()))
    n1, n2 = inov.divisao(ca.inov), inov.divisao(ca.inov)
    assert n1 == n2

    # 5. Distância: 0 para cópias, maior com estrutura diferente.
    g = populacao[0]
    assert distancia(g, g.copiar(), p) == 0
    assert distancia(g, genoma_neutro(), p) > 0

    # 6. Guardar/carregar preserva a função da rede.
    with tempfile.TemporaryDirectory() as pasta:
        ficheiro = Path(pasta) / 'g.npz'
        guardar_genoma(ficheiro, g, fitness=1.0)
        h = carregar_genoma(ficheiro)
    x = rng.normal(size=ENTRADAS)
    assert np.allclose(RedeCompilada(g).prever(x), RedeCompilada(h).prever(x))

    # 7. População: ciclo registar/reproduzir mantém o tamanho e cria espécies.
    pop = PopulacaoNEAT(20, ENTRADAS, SAIDAS, np.random.default_rng(1),
                        ParametrosNEAT(p_nova_ligacao=.5, p_novo_no=.4, limiar_inicial=1.0))
    for _ in range(25):
        pop.registar(np.random.default_rng(2).random(len(pop.genomas)) * 100)
        pop.reproduzir()
        assert len(pop.genomas) == 20
    pop.registar(np.arange(20, dtype=float))
    assert len(pop.especies) >= 2, 'especiação não funcionou'
    print('OK: NEAT (compilação, mutação, cruzamento, inovação, especiação, E/S).')


if __name__ == '__main__':
    main()
