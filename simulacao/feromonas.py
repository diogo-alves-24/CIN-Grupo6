"""Duas grelhas de feromonas, com depósito local e evaporação exponencial."""
import math
import pygame
from simulacao.configuracao import (LARGURA, ALTURA, TAMANHO_CELULA,
    EVAPORACAO, DIFUSAO, MAX_FEROMONA)

class GrelhaFeromonas:
    def __init__(self):
        self.colunas = math.ceil(LARGURA / TAMANHO_CELULA)
        self.linhas = math.ceil(ALTURA / TAMANHO_CELULA)
        self.casa = [[0.0] * self.colunas for _ in range(self.linhas)]
        self.comida = [[0.0] * self.colunas for _ in range(self.linhas)]
        self.tempo_atualizacao = 0.0

    def celula(self, posicao):
        return (min(self.colunas - 1, max(0, int(posicao.x // TAMANHO_CELULA))),
                min(self.linhas - 1, max(0, int(posicao.y // TAMANHO_CELULA))))

    def depositar(self, tipo, posicao, quantidade):
        grelha = self.casa if tipo == 'casa' else self.comida
        coluna, linha = self.celula(posicao)
        grelha[linha][coluna] = min(MAX_FEROMONA, grelha[linha][coluna] + quantidade)

    def intensidade(self, tipo, posicao):
        grelha = self.casa if tipo == 'casa' else self.comida
        coluna, linha = self.celula(posicao)
        return grelha[linha][coluna]

    def direcao(self, tipo, posicao, raio=50):
        """Gradiente estimado a partir de leituras locais (nunca da origem global)."""
        grelha = self.casa if tipo == 'casa' else self.comida
        col, lin = self.celula(posicao)
        alcance = max(1, int(raio // TAMANHO_CELULA))
        resultado = pygame.Vector2()
        for dy in range(-alcance, alcance + 1):
            y = lin + dy
            if not 0 <= y < self.linhas: continue
            for dx in range(-alcance, alcance + 1):
                x = col + dx
                if not 0 <= x < self.colunas or (dx == 0 and dy == 0): continue
                deslocamento = pygame.Vector2(dx, dy)
                distancia = deslocamento.length()
                if distancia > alcance: continue
                valor = grelha[y][x]
                if valor > 0.01:
                    resultado += deslocamento / distancia * (valor / (1 + distancia * .4))
        return resultado.normalize() if resultado.length_squared() > 0.0001 else pygame.Vector2()

    def atualizar(self, delta_tempo):
        self.tempo_atualizacao += delta_tempo
        if self.tempo_atualizacao < 0.18: return
        delta = self.tempo_atualizacao
        self.tempo_atualizacao = 0.0
        fator = math.exp(-EVAPORACAO * delta)
        mistura = min(0.2, DIFUSAO * delta)
        for nome in ('casa', 'comida'):
            antiga = getattr(self, nome)
            nova = [[0.0] * self.colunas for _ in range(self.linhas)]
            for y in range(self.linhas):
                for x in range(self.colunas):
                    valor = antiga[y][x]
                    if valor < 0.0005: continue
                    valor *= fator
                    nova[y][x] += valor * (1-mistura)
                    vizinhos = [(nx,ny) for nx,ny in ((x-1,y),(x+1,y),(x,y-1),(x,y+1))
                                if 0 <= nx < self.colunas and 0 <= ny < self.linhas]
                    quota = valor * mistura / len(vizinhos) if vizinhos else 0
                    for nx,ny in vizinhos: nova[ny][nx] += quota
            setattr(self, nome, nova)
