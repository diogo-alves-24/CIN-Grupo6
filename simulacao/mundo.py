import random
import pygame
from simulacao.configuracao import NUM_AGENTES, NUM_RECURSOS, POSICAO_NINHO, RAIO_NINHO, LARGURA, ALTURA
from simulacao.agente import Agente
from simulacao.agente import Obstaculo, Recurso

class Simulacao:
    def __init__(self):
        self.obstaculos = [
            Obstaculo(pygame.Vector2(275, 195), 48),
            Obstaculo(pygame.Vector2(820, 185), 58),
            Obstaculo(pygame.Vector2(265, 535), 58),
            Obstaculo(pygame.Vector2(825, 550), 48),
            Obstaculo(pygame.Vector2(550, 170), 29),
        ]
        self.agentes = [Agente(i) for i in range(NUM_AGENTES)]
        self.recursos = self._criar_recursos()
        self.tempo_decorrido = 0.
        self.em_pausa = False
        self.mostrar_sensores = False

    def _posicao_recurso_valida(self, posicao):
        if posicao.distance_to(POSICAO_NINHO) < RAIO_NINHO + 100:
            return False
        return all(posicao.distance_to(obstaculo.centro) >= obstaculo.raio + 20 for obstaculo in self.obstaculos)

    def _criar_recursos(self):
        grupos = [pygame.Vector2(125, 105), pygame.Vector2(975, 110),
                    pygame.Vector2(130, 630), pygame.Vector2(970, 625)]
        resultado = []
        for i in range(NUM_RECURSOS):
            for _ in range(200):
                centro = grupos[i % len(grupos)]
                candidata = pygame.Vector2(centro.x + random.gauss(0, 42),
                                           centro.y + random.gauss(0, 42))
                if (20 < candidata.x < LARGURA - 20 and 20 < candidata.y < ALTURA - 20
                        and self._posicao_recurso_valida(candidata)):
                    resultado.append(Recurso(candidata))
                    break
            else:
                raise RuntimeError('Não foi possível posicionar um recurso.')
        return resultado

    def atualizar(self, delta_tempo):
        if self.em_pausa:
            return
        self.tempo_decorrido += delta_tempo
        for agente in self.agentes:
            agente.atualizar(delta_tempo, self.agentes, self.recursos, self.obstaculos)
