"""Observações exclusivamente locais, sem coordenadas globais dos objetivos."""
from dataclasses import dataclass, field
import pygame
from simulacao.configuracao import ALCANCE_SENSORES

@dataclass
class Observacao:
    vizinhos: int = 0
    recursos: int = 0
    distancia_obstaculo: float = ALCANCE_SENSORES
    direcao_recurso: pygame.Vector2 = field(default_factory=pygame.Vector2)
    direcao_obstaculo: pygame.Vector2 = field(default_factory=pygame.Vector2)
    separacao: pygame.Vector2 = field(default_factory=pygame.Vector2)
    alinhamento: pygame.Vector2 = field(default_factory=pygame.Vector2)
    coesao: pygame.Vector2 = field(default_factory=pygame.Vector2)
    feromona_casa: float = 0.0
    feromona_comida: float = 0.0
    direcao_casa: pygame.Vector2 = field(default_factory=pygame.Vector2)
    direcao_comida: pygame.Vector2 = field(default_factory=pygame.Vector2)
    ninho_visivel: bool = False
    direcao_ninho: pygame.Vector2 = field(default_factory=pygame.Vector2)


def observar(agente, agentes, recursos, obstaculos, feromonas=None):
    from simulacao.configuracao import POSICAO_NINHO, RAIO_NINHO
    alcance2 = ALCANCE_SENSORES ** 2
    obs = Observacao()
    soma_posicoes = pygame.Vector2()
    soma_direcoes = pygame.Vector2()
    mais_proximo = alcance2 + 1
    for outro in agentes:
        if outro is agente: continue
        delta = outro.posicao - agente.posicao
        d2 = delta.length_squared()
        if d2 <= alcance2:
            obs.vizinhos += 1
            soma_posicoes += delta
            soma_direcoes += pygame.Vector2(__import__('math').cos(outro.orientacao), __import__('math').sin(outro.orientacao))
            if 0 < d2 < 38**2: obs.separacao -= delta / d2
    if obs.vizinhos:
        obs.coesao = soma_posicoes / obs.vizinhos
        obs.alinhamento = soma_direcoes / obs.vizinhos
    for recurso in recursos:
        if not recurso.disponivel: continue
        delta = recurso.posicao - agente.posicao
        d2 = delta.length_squared()
        if d2 < alcance2:
            obs.recursos += 1
            if d2 < mais_proximo:
                mais_proximo = d2
                obs.direcao_recurso = delta.normalize() if d2 else pygame.Vector2()
    for obstaculo in obstaculos:
        delta = agente.posicao - obstaculo.centro
        distancia_casca = delta.length() - obstaculo.raio
        if distancia_casca < ALCANCE_SENSORES:
            obs.distancia_obstaculo = min(obs.distancia_obstaculo, max(0, distancia_casca))
            if delta.length_squared():
                obs.direcao_obstaculo += delta.normalize() / max(6, distancia_casca + 8)
    deslocamento_ninho = POSICAO_NINHO - agente.posicao
    if deslocamento_ninho.length() < ALCANCE_SENSORES + RAIO_NINHO:
        obs.ninho_visivel = True
        if deslocamento_ninho.length_squared(): obs.direcao_ninho = deslocamento_ninho.normalize()
    if feromonas:
        obs.feromona_casa = feromonas.intensidade('casa', agente.posicao)
        obs.feromona_comida = feromonas.intensidade('comida', agente.posicao)
        obs.direcao_casa = feromonas.direcao('casa', agente.posicao, 48)
        obs.direcao_comida = feromonas.direcao('comida', agente.posicao, 48)
    return obs
