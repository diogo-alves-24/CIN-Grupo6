"""Testes de integração. Requer pygame e numpy do requirements.txt.

Executa: python testar_projeto.py
"""
from simulacao.configuracao import NUM_AGENTES, NUM_RECURSOS, POSICAO_NINHO
from simulacao.avaliacao import simular
from simulacao.mundo import Simulacao, CENARIOS
from simulacao.rede_neuronal import VERSAO_GENOMA, genoma_neutro
from simulacao.sensores import observar


def main():
    for i, cenario in enumerate(CENARIOS):
        sim = Simulacao('Combinado', semente=100 + i, cenario=cenario)
        assert len(sim.recursos) == NUM_RECURSOS
        assert len(sim.agentes) == NUM_AGENTES
        assert sim.cenario == cenario
        assert all(sim._posicao_recurso_valida(r.posicao) for r in sim.recursos)
        assert all(r.posicao.distance_to(POSICAO_NINHO) > 0 for r in sim.recursos)
        obs = observar(sim.agentes[0], sim.recursos, sim.obstaculos, sim.feromonas)
        assert not hasattr(obs, 'separacao')
        assert not hasattr(obs, 'alinhamento')
        assert not hasattr(obs, 'coesao')
        assert -1.0 <= obs.contraste_comida_lateral <= 1.0

    # Em condições e sementes idênticas, o genoma neutro deve coincidir com
    # o controlador Combinado (as duas redes produzem correção exatamente zero).
    for semente in (100, 101, 102, 103):
        referencia = simular('Combinado', semente, duracao=1.0)
        neutra = simular('Neuroevolução', semente, duracao=1.0,
                        genoma=genoma_neutro())
        assert referencia.keys() == neutra.keys()
        for chave in referencia:
            if chave == 'cenario':
                assert referencia[chave] == neutra[chave]
            else:
                assert abs(referencia[chave] - neutra[chave]) < 1e-7, (
                    semente, chave, referencia[chave], neutra[chave])
    assert VERSAO_GENOMA == 'neat_v1'
    print('OK: 4 cenários, sensores sem Boids e genoma neutro equivalente.')


if __name__ == '__main__':
    main()
