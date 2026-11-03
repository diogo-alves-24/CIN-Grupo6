"""Duas MLP independentes, evoluídas conjuntamente pelo mesmo Algoritmo Genético.

A primeira controla a procura (sem recurso); a segunda, o regresso (com recurso).
Cada MLP recebe observações locais e produz três comandos: correção lateral,
correção frontal e ajuste de velocidade. O genoma todo a zero é neutro.
"""
import numpy as np

ENTRADAS = 28
OCULTOS = 10
SAIDAS = 3
TAMANHO_REDE = ENTRADAS * OCULTOS + OCULTOS + OCULTOS * SAIDAS + SAIDAS
TAMANHO_GENOMA = 2 * TAMANHO_REDE


class _MLP:
    def __init__(self, genoma):
        indice = 0
        quantidade = ENTRADAS * OCULTOS
        self.pesos_1 = genoma[indice:indice + quantidade].reshape(ENTRADAS, OCULTOS)
        indice += quantidade
        self.vies_1 = genoma[indice:indice + OCULTOS]
        indice += OCULTOS
        quantidade = OCULTOS * SAIDAS
        self.pesos_2 = genoma[indice:indice + quantidade].reshape(OCULTOS, SAIDAS)
        indice += quantidade
        self.vies_2 = genoma[indice:indice + SAIDAS]

    def prever(self, entradas):
        escondida = np.tanh(entradas @ self.pesos_1 + self.vies_1)
        return np.tanh(escondida @ self.pesos_2 + self.vies_2)


class RedeNeuronal:
    def __init__(self, genoma):
        genes = np.asarray(genoma, dtype=float).ravel()
        if genes.size != TAMANHO_GENOMA:
            raise ValueError(
                f'Genoma incompatível: esperado {TAMANHO_GENOMA} parâmetros '
                f'(duas redes 28–10–3), recebido {genes.size}. '
                'É necessário voltar a executar treinar.py.'
            )
        self.procura = _MLP(genes[:TAMANHO_REDE])
        self.regresso = _MLP(genes[TAMANHO_REDE:])

    def prever(self, entradas, transporta_recurso=False):
        dados = np.asarray(entradas, dtype=float)
        if dados.shape != (ENTRADAS,):
            raise ValueError(f'Esperadas {ENTRADAS} entradas, recebidas {dados.shape}.')
        rede = self.regresso if transporta_recurso else self.procura
        return rede.prever(dados)
