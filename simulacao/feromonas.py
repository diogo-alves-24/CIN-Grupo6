"""Duas grelhas de feromonas, com depósito local e evaporação exponencial.

Implementação vetorizada (NumPy). A semântica é a da versão original em listas
Python (evaporação exponencial + difusão com conservação de massa nas
fronteiras), mas ~10x mais rápida, o que torna o treino evolutivo viável.

As grelhas são guardadas com uma margem de zeros (``_MARGEM`` células) para que
as consultas locais por janela não precisem de testes de fronteira.
"""
import math
import numpy as np
import pygame
from simulacao.configuracao import (LARGURA, ALTURA, TAMANHO_CELULA,
    EVAPORACAO_CASA, EVAPORACAO_COMIDA, DIFUSAO_CASA, DIFUSAO_COMIDA, MAX_FEROMONA)

_MARGEM = 6  # >= maior alcance (em células) usado nas consultas


class GrelhaFeromonas:
    def __init__(self):
        self.colunas = math.ceil(LARGURA / TAMANHO_CELULA)
        self.linhas = math.ceil(ALTURA / TAMANHO_CELULA)
        forma = (self.linhas + 2 * _MARGEM, self.colunas + 2 * _MARGEM)
        self._casa = np.zeros(forma)
        self._comida = np.zeros(forma)
        # Em cada célula, um vetor local aponta para onde o explorador vinha.
        # Trata-se de uma pista ambiental, não da localização global do ninho.
        self._vet_casa = np.zeros(forma + (2,))
        self.tempo_atualizacao = 0.0
        viz = np.full((self.linhas, self.colunas), 4.0)
        viz[0, :] -= 1; viz[-1, :] -= 1; viz[:, 0] -= 1; viz[:, -1] -= 1
        self._vizinhos = viz
        self._nucleos = {}

    # --- vistas "interiores" (sem margem), usadas p.ex. pelo desenho ---------
    @property
    def casa(self):
        return self._casa[_MARGEM:-_MARGEM, _MARGEM:-_MARGEM]

    @property
    def comida(self):
        return self._comida[_MARGEM:-_MARGEM, _MARGEM:-_MARGEM]

    def celula(self, posicao):
        return (min(self.colunas - 1, max(0, int(posicao.x // TAMANHO_CELULA))),
                min(self.linhas - 1, max(0, int(posicao.y // TAMANHO_CELULA))))

    def _nucleo(self, alcance):
        """Matrizes pré-calculadas (janela (2a+1)^2) para um dado alcance."""
        if alcance not in self._nucleos:
            assert alcance <= _MARGEM
            idx = np.arange(-alcance, alcance + 1)
            dx, dy = np.meshgrid(idx, idx)            # dx varia nas colunas
            d2 = dx * dx + dy * dy
            dist = np.sqrt(d2)
            dentro = (dist <= alcance) & (d2 > 0)
            seguro = np.where(dist > 0, dist, 1.0)
            w = np.where(dentro, 1.0 / (1.0 + dist * 0.4), 0.0)
            gx = w * dx / seguro
            gy = w * dy / seguro
            disco = (d2 <= alcance * alcance)
            wreg = np.where(disco, 1.0 / (1.0 + d2 * 1.3), 0.0)
            self._nucleos[alcance] = (gx, gy, wreg, dx, dy)
        return self._nucleos[alcance]

    def _janela(self, matriz, posicao, alcance):
        col, lin = self.celula(posicao)
        l0 = lin + _MARGEM - alcance
        c0 = col + _MARGEM - alcance
        return col, lin, (slice(l0, l0 + 2 * alcance + 1), slice(c0, c0 + 2 * alcance + 1))

    def depositar(self, tipo, posicao, quantidade, direcao_regresso=None):
        grelha = self._casa if tipo == 'casa' else self._comida
        coluna, linha = self.celula(posicao)
        i, j = linha + _MARGEM, coluna + _MARGEM
        anterior = grelha[i, j]
        novo = min(MAX_FEROMONA, anterior + quantidade)
        if tipo == 'casa' and direcao_regresso is not None and direcao_regresso.length_squared() > 0.0001:
            vetor = self._vet_casa[i, j]
            # Média ponderada com maior influência dos rastos recentes.
            proporcao = min(0.75, quantidade / max(0.0001, anterior + quantidade))
            d = direcao_regresso.normalize()
            self._vet_casa[i, j, 0] = vetor[0] * (1 - proporcao) + d.x * proporcao
            self._vet_casa[i, j, 1] = vetor[1] * (1 - proporcao) + d.y * proporcao
        grelha[i, j] = novo

    def intensidade(self, tipo, posicao):
        grelha = self._casa if tipo == 'casa' else self._comida
        coluna, linha = self.celula(posicao)
        return float(grelha[linha + _MARGEM, coluna + _MARGEM])

    def direcao(self, tipo, posicao, raio=50):
        """Gradiente estimado a partir de leituras locais (nunca da origem global)."""
        grelha = self._casa if tipo == 'casa' else self._comida
        alcance = max(1, int(raio // TAMANHO_CELULA))
        gx, gy, _, _, _ = self._nucleo(alcance)
        _, _, jan = self._janela(grelha, posicao, alcance)
        valores = grelha[jan]
        valores = np.where(valores > 0.01, valores, 0.0)
        rx = float((valores * gx).sum())
        ry = float((valores * gy).sum())
        n2 = rx * rx + ry * ry
        if n2 <= 0.0001:
            return pygame.Vector2()
        n = math.sqrt(n2)
        return pygame.Vector2(rx / n, ry / n)

    def direcao_regresso(self, posicao, raio=35):
        """Consulta local dos sinais direcionais depositados no ambiente."""
        alcance = max(1, int(raio // TAMANHO_CELULA))
        _, _, wreg, dx, dy = self._nucleo(alcance)
        coluna, linha, jan = self._janela(self._casa, posicao, alcance)
        q = self._casa[jan]
        peso = np.where(q >= 0.06, q, 0.0) * wreg
        peso_total = float(peso.sum())
        if peso_total < 0.01:
            return pygame.Vector2()
        vet = self._vet_casa[jan]
        ox = float((vet[..., 0] * peso).sum()); oy = float((vet[..., 1] * peso).sum())
        lx = float((dx * peso).sum()) / peso_total
        ly = float((dy * peso).sum()) / peso_total
        # Primeiro aproxima-se do trilho; sobre o trilho segue a seta local.
        if self._casa[linha + _MARGEM, coluna + _MARGEM] < 0.05 and lx * lx + ly * ly > .01:
            n = math.hypot(lx, ly)
            return pygame.Vector2(lx / n, ly / n)
        n2 = ox * ox + oy * oy
        if n2 > .0001:
            n = math.sqrt(n2)
            return pygame.Vector2(ox / n, oy / n)
        return pygame.Vector2()

    def atualizar(self, delta_tempo):
        self.tempo_atualizacao += delta_tempo

        if self.tempo_atualizacao < 0.18:
            return

        delta = self.tempo_atualizacao
        self.tempo_atualizacao = 0.0

        for nome in ('casa', 'comida'):

            # Evaporação e difusão específicas de cada feromona
            if nome == 'casa':
                taxa = EVAPORACAO_CASA
                difusao = DIFUSAO_CASA
            else:
                taxa = EVAPORACAO_COMIDA
                difusao = DIFUSAO_COMIDA

            fator = math.exp(-taxa * delta)
            mistura = min(0.2, difusao * delta)

            # Grelha da feromona correspondente
            g = getattr(self, nome)

            # Evaporação
            v = np.where(g >= 0.0005, g * fator, 0.0)

            # Difusão para as células vizinhas
            nova = v * (1 - mistura)
            q = v * mistura / self._vizinhos

            nova[1:, :] += q[:-1, :]
            nova[:-1, :] += q[1:, :]
            nova[:, 1:] += q[:, :-1]
            nova[:, :-1] += q[:, 1:]

            # Atualizar a grelha
            g[...] = nova

            if nome == 'casa':
                # Apagar direções quando as feromonas desaparecem
                vet = self._vet_casa[
                    _MARGEM:-_MARGEM,
                    _MARGEM:-_MARGEM
                ]
                vet[nova < 0.04] = 0.0
