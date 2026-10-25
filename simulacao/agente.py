import math
import random
from dataclasses import dataclass
import pygame
from simulacao.configuracao import (RAIO_AGENTE, RAIO_RECURSO, RAIO_NINHO,
    POSICAO_NINHO, VELOCIDADE_MAXIMA, VELOCIDADE_ROTACAO, LARGURA, ALTURA,
    DEPOSITO_CASA, DEPOSITO_COMIDA)
from simulacao.controladores import ControladorAleatorio, ControladorEnxame
from simulacao.sensores import observar

class Agente:
    def __init__(self, numero, modo='Aleatório'):
        self.numero = numero
        self.posicao = POSICAO_NINHO + pygame.Vector2(random.uniform(-12, 12), random.uniform(-12, 12))
        self.orientacao = random.uniform(-math.pi, math.pi)
        self.velocidade = random.uniform(65., VELOCIDADE_MAXIMA)
        self.controlador = ControladorAleatorio() if modo == 'Aleatório' else ControladorEnxame(modo)
        self.modo = modo
        self.papel = 'EXPLORADOR'
        self.tempo_sem_recursos = 0.0
        self.transporta_recurso = False
        self.energia = 0.0
        self.entregas = 0
        self.observacao = None
        self.tempo_deposito = 0.0
        self.tempo_desde_ninho = 0.0
        self.tempo_desde_recurso = 0.0

    def atualizar(self, delta_tempo, agentes, recursos, obstaculos, feromonas):
        self.observacao = observar(self, agentes, recursos, obstaculos,
                                  feromonas if self.modo != 'Aleatório' else None)
        obs = self.observacao
        if self.modo == 'Combinado' and not self.transporta_recurso:
            if obs.recursos or obs.feromona_comida > .12:
                self.papel = 'RECOLHEDOR'
                self.tempo_sem_recursos = 0
            else:
                self.tempo_sem_recursos += delta_tempo
                if self.tempo_sem_recursos > 8:
                    self.papel = 'EXPLORADOR'
        desejada = self.controlador.direcao(delta_tempo, obs, self)
        alvo = math.atan2(desejada.y, desejada.x)
        diferenca_angular = (alvo - self.orientacao + math.pi) % (2 * math.pi) - math.pi
        self.orientacao += max(-VELOCIDADE_ROTACAO * delta_tempo,
                               min(VELOCIDADE_ROTACAO * delta_tempo, diferenca_angular))
        direcao = pygame.Vector2(math.cos(self.orientacao), math.sin(self.orientacao))
        proxima = self.posicao + direcao * self.velocidade * delta_tempo
        if proxima.x < RAIO_AGENTE or proxima.x > LARGURA - RAIO_AGENTE:
            self.orientacao = math.pi - self.orientacao
            proxima.x = max(RAIO_AGENTE, min(LARGURA - RAIO_AGENTE, proxima.x))
        if proxima.y < RAIO_AGENTE or proxima.y > ALTURA - RAIO_AGENTE:
            self.orientacao = -self.orientacao
            proxima.y = max(RAIO_AGENTE, min(ALTURA - RAIO_AGENTE, proxima.y))
        for obstaculo in obstaculos:
            diferenca = proxima - obstaculo.centro
            limite = obstaculo.raio + RAIO_AGENTE
            if diferenca.length_squared() < limite**2:
                normal = diferenca.normalize() if diferenca.length_squared() else pygame.Vector2(1,0)
                proxima = obstaculo.centro + normal * limite
                self.orientacao = math.atan2(normal.y, normal.x) + random.uniform(-.4, .4)
        proxima.x = max(RAIO_AGENTE, min(LARGURA - RAIO_AGENTE, proxima.x))
        proxima.y = max(RAIO_AGENTE, min(ALTURA - RAIO_AGENTE, proxima.y))
        self.energia += self.posicao.distance_to(proxima) * .018
        self.posicao = proxima
        self.tempo_deposito += delta_tempo
        if self.transporta_recurso:
            self.tempo_desde_recurso += delta_tempo
        else:
            self.tempo_desde_ninho += delta_tempo
        if self.modo == 'Combinado' and self.tempo_deposito >= .15:
            tipo = 'comida' if self.transporta_recurso else 'casa'
            if self.transporta_recurso:
                fator = max(0.18, 1.0 - self.tempo_desde_recurso / 8.0)
                quantidade = DEPOSITO_COMIDA * fator
            else:
                fator = max(0.12, 1.0 - self.tempo_desde_ninho / 8.0)
                quantidade = DEPOSITO_CASA * fator
            feromonas.depositar(tipo, self.posicao, quantidade)
            self.tempo_deposito = 0
        if not self.transporta_recurso:
            for recurso in recursos:
                if recurso.disponivel and self.posicao.distance_squared_to(recurso.posicao) <= (RAIO_AGENTE + RAIO_RECURSO)**2:
                    recurso.disponivel = False
                    self.transporta_recurso = True
                    self.papel = 'RECOLHEDOR'
                    self.tempo_desde_recurso = 0.0
                    break
        elif self.posicao.distance_squared_to(POSICAO_NINHO) <= (RAIO_NINHO + RAIO_AGENTE)**2:
            self.transporta_recurso = False
            self.entregas += 1
            self.tempo_sem_recursos = 0
            self.tempo_desde_ninho = 0.0
            self.papel = 'EXPLORADOR'

@dataclass
class Obstaculo:
    centro: pygame.Vector2
    raio: float

@dataclass
class Recurso:
    posicao: pygame.Vector2
    disponivel: bool = True
