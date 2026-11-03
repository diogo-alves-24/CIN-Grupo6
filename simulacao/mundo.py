import random
from simulacao.rede_neuronal import RedeNeuronal
import pygame
from simulacao.configuracao import (NUM_AGENTES, NUM_RECURSOS, POSICAO_NINHO,
    RAIO_NINHO, LARGURA, ALTURA)
from simulacao.agente import Agente, Obstaculo, Recurso
from simulacao.feromonas import GrelhaFeromonas
from simulacao.controladores import MODOS

class Simulacao:
    def __init__(self, modo='Aleatório', semente=42, genoma=None):
        if modo not in MODOS: raise ValueError('Modo desconhecido')
        random.seed(semente)
        self.modo = modo
        self.genoma = genoma
        if modo == 'Neuroevolução' and genoma is None:
            raise ValueError('É necessário carregar um genoma treinado.')
        rede = RedeNeuronal(genoma) if modo == 'Neuroevolução' else None
        self.obstaculos = [
            Obstaculo(pygame.Vector2(275, 195), 48),
            Obstaculo(pygame.Vector2(820, 185), 58),
            Obstaculo(pygame.Vector2(265, 535), 58),
            Obstaculo(pygame.Vector2(825, 550), 48),
            Obstaculo(pygame.Vector2(550, 170), 29),
        ]
        self.agentes = [Agente(i, modo, rede) for i in range(NUM_AGENTES)]
        self.recursos = self._criar_recursos()
        self.feromonas = GrelhaFeromonas()
        self.tempo_decorrido = 0.0
        self.terminada = False
        self.tempo_final = 0.0
        self.velocidade_simulacao = 1
        self.em_pausa = False
        self.mostrar_sensores = False
        self.mostrar_feromonas = True
        self.tipo_feromona = 'ambas'
        self.mostrar_painel = True

    def _posicao_recurso_valida(self, posicao):
        if posicao.distance_to(POSICAO_NINHO) < RAIO_NINHO + 100: return False
        return all(posicao.distance_to(o.centro) >= o.raio + 20 for o in self.obstaculos)

    def _criar_recursos(self):
        grupos = [pygame.Vector2(125, 105), pygame.Vector2(975, 110),
                  pygame.Vector2(130, 630), pygame.Vector2(970, 625)]
        resultado = []
        for i in range(NUM_RECURSOS):
            for _ in range(200):
                centro = grupos[i % len(grupos)]
                candidata = pygame.Vector2(centro.x + random.gauss(0,42),
                                           centro.y + random.gauss(0,42))
                if (20 < candidata.x < LARGURA-20 and 20 < candidata.y < ALTURA-20
                        and self._posicao_recurso_valida(candidata)):
                    resultado.append(Recurso(candidata))
                    break
            else: raise RuntimeError('Não foi possível posicionar um recurso.')
        return resultado

    def atualizar(self, delta_tempo):
        if self.em_pausa or self.terminada: return
        self.tempo_decorrido += delta_tempo
        self.feromonas.atualizar(delta_tempo)
        for agente in self.agentes:
            agente.atualizar(delta_tempo, self.agentes, self.recursos,
                             self.obstaculos, self.feromonas)

        if sum(a.entregas for a in self.agentes) == len(self.recursos):
            self.terminada = True
            self.tempo_final = self.tempo_decorrido
