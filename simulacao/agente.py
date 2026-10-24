import math
import random
import pygame
from dataclasses import dataclass
from simulacao.configuracao import (
    RAIO_AGENTE, RAIO_RECURSO, RAIO_NINHO, POSICAO_NINHO,
    VELOCIDADE_MAXIMA, VELOCIDADE_ROTACAO, LARGURA, ALTURA,
)
from simulacao.controladores import ControladorAleatorio
from simulacao.sensores import observar

class Agente:
    def __init__(self, numero: int, controlador=None):
        self.numero = numero
        self.posicao = POSICAO_NINHO + pygame.Vector2(random.uniform(-12, 12), random.uniform(-12, 12))
        self.orientacao = random.uniform(-math.pi, math.pi)
        self.velocidade = random.uniform(65., VELOCIDADE_MAXIMA)
        self.controlador = controlador if controlador is not None else ControladorAleatorio()
        self.transporta_recurso = False
        self.energia = 0.
        self.entregas = 0
        self.observacao = None

    def atualizar(self, delta_tempo, agentes, recursos, obstaculos):
        self.observacao = observar(self, agentes, recursos, obstaculos)
        rotacao = self.controlador.calcular_rotacao(delta_tempo, self.observacao)
        self.orientacao += max(-1., min(1., rotacao)) * VELOCIDADE_ROTACAO * delta_tempo
        direcao = pygame.Vector2(math.cos(self.orientacao), math.sin(self.orientacao))
        proxima_posicao = self.posicao + direcao * self.velocidade * delta_tempo

        if proxima_posicao.x < RAIO_AGENTE or proxima_posicao.x > LARGURA - RAIO_AGENTE:
            self.orientacao = math.pi - self.orientacao
            proxima_posicao.x = max(RAIO_AGENTE, min(LARGURA - RAIO_AGENTE, proxima_posicao.x))
        if proxima_posicao.y < RAIO_AGENTE or proxima_posicao.y > ALTURA - RAIO_AGENTE:
            self.orientacao = -self.orientacao
            proxima_posicao.y = max(RAIO_AGENTE, min(ALTURA - RAIO_AGENTE, proxima_posicao.y))
        for obstaculo in obstaculos:
            diferenca = proxima_posicao - obstaculo.centro
            limite = obstaculo.raio + RAIO_AGENTE
            if diferenca.length_squared() < limite * limite:
                normal = diferenca.normalize() if diferenca.length_squared() > 0 else pygame.Vector2(1, 0)
                proxima_posicao = obstaculo.centro + normal * limite
                self.orientacao = math.atan2(normal.y, normal.x) + random.uniform(-.5, .5)

        self.energia += self.posicao.distance_to(proxima_posicao) * .018
        self.posicao = proxima_posicao
        if not self.transporta_recurso:
            for recurso in recursos:
                if recurso.disponivel and self.posicao.distance_squared_to(recurso.posicao) <= (RAIO_AGENTE + RAIO_RECURSO) ** 2:
                    recurso.disponivel = False
                    self.transporta_recurso = True
                    break
        elif self.posicao.distance_squared_to(POSICAO_NINHO) <= (RAIO_NINHO + RAIO_AGENTE) ** 2:
            self.transporta_recurso = False
            self.entregas += 1


@dataclass
class Obstaculo:
    centro: pygame.Vector2
    raio: float

@dataclass
class Recurso:
    posicao: pygame.Vector2
    disponivel: bool = True
