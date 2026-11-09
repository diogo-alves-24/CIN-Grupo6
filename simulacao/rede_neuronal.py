"""Rede neuronal do controlador, agora com topologia evoluída por NEAT.

Um único genoma NEAT com 28 entradas locais e 6 saídas:
  * saídas 0-2 -> comandos usados na PROCURA (sem recurso);
  * saídas 3-5 -> comandos usados no REGRESSO (com recurso).
Cada grupo de 3 saídas = correção lateral, correção frontal, ajuste de velocidade.
Os dois comportamentos partilham os nós ocultos que a evolução criar, mas
podem especializar-se por ligações próprias. O genoma neutro (tudo a zero)
produz saídas nulas e reproduz o controlador Combinado.
"""
import numpy as np
from simulacao.neat import Genoma, Inovacoes, RedeCompilada

VERSAO_GENOMA = 'neat_v1'
ENTRADAS = 28
SAIDAS_POR_MODO = 3
SAIDAS = 2 * SAIDAS_POR_MODO


def genoma_neutro():
    """Genoma mínimo com todos os pesos e viéses a zero (== Combinado)."""
    return Genoma.minimo(ENTRADAS, SAIDAS, Inovacoes(ENTRADAS, SAIDAS), None, neutro=True)


def guardar_genoma(caminho, genoma, **extra):
    np.savez(caminho, versao=VERSAO_GENOMA, **genoma.para_dict(), **extra)


def carregar_genoma(caminho):
    """Lê um .npz. Devolve Genoma, ou levanta ValueError se for incompatível."""
    with np.load(caminho, allow_pickle=False) as d:
        versao = str(d['versao']) if 'versao' in d else ''
        if versao != VERSAO_GENOMA or 'neat_nos_id' not in d:
            raise ValueError(
                f'Genoma incompatível (versão "{versao}", esperada "{VERSAO_GENOMA}"). '
                'Os genomas MLP anteriores não servem para NEAT: volta a executar treinar.py.')
        genoma = Genoma.de_dict(d)
    if genoma.entradas != ENTRADAS or genoma.saidas != SAIDAS:
        raise ValueError(
            f'Genoma com {genoma.entradas} entradas e {genoma.saidas} saídas; '
            f'esperadas {ENTRADAS} e {SAIDAS}.')
    return genoma


class RedeNeuronal:
    def __init__(self, genoma):
        if not isinstance(genoma, Genoma):
            raise TypeError('RedeNeuronal espera um Genoma NEAT '
                            '(ver simulacao.rede_neuronal.carregar_genoma).')
        if genoma.entradas != ENTRADAS or genoma.saidas != SAIDAS:
            raise ValueError(f'Genoma incompatível: esperadas {ENTRADAS} entradas e '
                             f'{SAIDAS} saídas, recebidas {genoma.entradas} e {genoma.saidas}.')
        self.rede = RedeCompilada(genoma)

    def prever(self, entradas, transporta_recurso=False):
        dados = np.asarray(entradas, dtype=float)
        if dados.shape != (ENTRADAS,):
            raise ValueError(f'Esperadas {ENTRADAS} entradas, recebidas {dados.shape}.')
        saida = self.rede.prever(dados)
        return saida[SAIDAS_POR_MODO:] if transporta_recurso else saida[:SAIDAS_POR_MODO]
