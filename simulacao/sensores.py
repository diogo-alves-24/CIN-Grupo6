"""Perceção exclusivamente local: recursos, obstáculos, ninho visível e feromonas.

Sem vizinhos, separação, alinhamento ou coesão (Boids removidos).
As amostras de feromonas são feitas em posições próximas do próprio agente.
"""
from dataclasses import dataclass, field
from math import sqrt, hypot, cos, sin
import pygame
from simulacao.configuracao import (ALCANCE_SENSORES, POSICAO_NINHO,
                                    RAIO_NINHO, MAX_FEROMONA)


@dataclass
class Observacao:
    recursos: int = 0
    distancia_obstaculo: float = ALCANCE_SENSORES
    direcao_recurso: pygame.Vector2 = field(default_factory=pygame.Vector2)
    direcao_obstaculo: pygame.Vector2 = field(default_factory=pygame.Vector2)
    feromona_casa: float = 0.0
    feromona_comida: float = 0.0
    direcao_casa: pygame.Vector2 = field(default_factory=pygame.Vector2)
    direcao_comida: pygame.Vector2 = field(default_factory=pygame.Vector2)
    contraste_comida_lateral: float = 0.0
    ninho_visivel: bool = False
    direcao_ninho: pygame.Vector2 = field(default_factory=pygame.Vector2)


def observar(agente, recursos, obstaculos, feromonas=None):
    ax, ay = agente.posicao.x, agente.posicao.y
    alcance2 = ALCANCE_SENSORES ** 2
    obs = Observacao()
    mais_proximo = alcance2 + 1

    for recurso in recursos:
        if not recurso.disponivel:
            continue
        dx = recurso.posicao.x - ax
        dy = recurso.posicao.y - ay
        d2 = dx * dx + dy * dy
        if d2 < alcance2:
            obs.recursos += 1
            if d2 < mais_proximo:
                mais_proximo = d2
                if d2:
                    d = sqrt(d2)
                    obs.direcao_recurso = pygame.Vector2(dx / d, dy / d)

    ox = oy = 0.0
    for obstaculo in obstaculos:
        dx = ax - obstaculo.centro.x
        dy = ay - obstaculo.centro.y
        d = hypot(dx, dy)
        distancia_casca = d - obstaculo.raio
        if distancia_casca < ALCANCE_SENSORES:
            obs.distancia_obstaculo = min(obs.distancia_obstaculo,
                                          max(0.0, distancia_casca))
            if d > 0:
                k = 1.0 / (d * max(6, distancia_casca + 8))
                ox += dx * k
                oy += dy * k
    obs.direcao_obstaculo = pygame.Vector2(ox, oy)

    dnx = POSICAO_NINHO.x - ax
    dny = POSICAO_NINHO.y - ay
    dn = hypot(dnx, dny)
    if dn < ALCANCE_SENSORES + RAIO_NINHO:
        obs.ninho_visivel = True
        if dn > 0:
            obs.direcao_ninho = pygame.Vector2(dnx / dn, dny / dn)

    if feromonas is not None:
        obs.feromona_casa = feromonas.intensidade('casa', agente.posicao)
        obs.feromona_comida = feromonas.intensidade('comida', agente.posicao)
        obs.direcao_casa = feromonas.direcao_regresso(agente.posicao, 40)
        obs.direcao_comida = feromonas.direcao('comida', agente.posicao, 48)
        # Duas leituras laterais de feromonas no referencial próprio da formiga.
        # Não usa a posição de nenhuma outra formiga nem o mapa global.
        lateral = pygame.Vector2(-sin(agente.orientacao), cos(agente.orientacao))
        esquerda = feromonas.intensidade('comida', agente.posicao + lateral * 24)
        direita = feromonas.intensidade('comida', agente.posicao - lateral * 24)
        obs.contraste_comida_lateral = max(-1.0, min(1.0,
                                                   (esquerda - direita) / MAX_FEROMONA))
    return obs
