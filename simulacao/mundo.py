import math
import random
from simulacao.rede_neuronal import RedeNeuronal
import pygame
from simulacao.configuracao import (NUM_AGENTES, NUM_RECURSOS, POSICAO_NINHO,
    RAIO_NINHO, LARGURA, ALTURA, TAMANHO_REGIAO_COBERTURA)
from simulacao.agente import Agente, Obstaculo, Recurso
from simulacao.feromonas import GrelhaFeromonas
from simulacao.controladores import MODOS

CENARIOS = ('proximo', 'agrupado', 'disperso', 'distante')


class Simulacao:
    def __init__(self, modo='Aleatório', semente=42, genoma=None, cenario='agrupado'):
        if modo not in MODOS: raise ValueError('Modo desconhecido')
        if cenario not in (*CENARIOS, 'variado'):
            raise ValueError(f'Cenário desconhecido: {cenario}')
        self.cenario = CENARIOS[int(semente) % len(CENARIOS)] if cenario == 'variado' else cenario
        random.seed(semente)
        self.modo = modo
        self.genoma = genoma
        if modo == 'Neuroevolução' and genoma is None:
            raise ValueError('É necessário carregar um genoma treinado.')
        rede = RedeNeuronal(genoma) if modo == 'Neuroevolução' else None
        self.obstaculos = self._criar_obstaculos()
        self.agentes = [Agente(i, modo, rede) for i in range(NUM_AGENTES)]
        self.recursos = self._criar_recursos()
        # Cobertura é métrica GLOBAL do avaliador. Nunca é dada às redes.
        self.regioes_visitadas = set()
        self.colunas_regioes = math.ceil(LARGURA / TAMANHO_REGIAO_COBERTURA)
        self.linhas_regioes = math.ceil(ALTURA / TAMANHO_REGIAO_COBERTURA)
        self.total_regioes = self.colunas_regioes * self.linhas_regioes
        self._registar_cobertura()
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

    def _criar_obstaculos(self):
        if self.cenario == 'proximo':
            return []
        if self.cenario == 'distante':
            dados = [(300, 205, 58), (790, 205, 63), (300, 545, 65),
                     (790, 535, 55), (550, 160, 40), (550, 575, 38)]
        else:
            # Mantém o mapa da versão anterior no cenário agrupado.
            dados = [(275, 195, 48), (820, 185, 58), (265, 535, 58),
                     (825, 550, 48), (550, 170, 29)]
        return [Obstaculo(pygame.Vector2(x, y), raio) for x, y, raio in dados]

    def _posicao_recurso_valida(self, posicao):
        if posicao.distance_to(POSICAO_NINHO) < RAIO_NINHO + 100: return False
        return all(posicao.distance_to(o.centro) >= o.raio + 20 for o in self.obstaculos)

    def _criar_recursos(self):
        grupos = ([pygame.Vector2(125, 105), pygame.Vector2(975, 110),
                   pygame.Vector2(130, 630), pygame.Vector2(970, 625)]
                  if self.cenario != 'distante' else
                  [pygame.Vector2(70, 65), pygame.Vector2(1030, 65),
                   pygame.Vector2(70, 675), pygame.Vector2(1030, 675)])
        resultado = []
        for i in range(NUM_RECURSOS):
            for _ in range(400):
                if self.cenario == 'proximo':
                    angulo = random.uniform(0, math.tau)
                    raio = random.uniform(160, 245)
                    candidata = POSICAO_NINHO + pygame.Vector2(
                        math.cos(angulo) * raio, math.sin(angulo) * raio)
                elif self.cenario == 'disperso':
                    candidata = pygame.Vector2(random.uniform(25, LARGURA - 25),
                                               random.uniform(25, ALTURA - 25))
                else:
                    centro = grupos[i % len(grupos)]
                    candidata = pygame.Vector2(centro.x + random.gauss(0, 42),
                                               centro.y + random.gauss(0, 42))
                if (20 < candidata.x < LARGURA-20 and 20 < candidata.y < ALTURA-20
                        and self._posicao_recurso_valida(candidata)):
                    resultado.append(Recurso(candidata))
                    break
            else: raise RuntimeError('Não foi possível posicionar um recurso.')
        return resultado

    def _registar_cobertura(self):
        """Uma célula só conta uma vez, por mais voltas que o enxame dê."""
        for agente in self.agentes:
            coluna = min(self.colunas_regioes - 1,
                         max(0, int(agente.posicao.x // TAMANHO_REGIAO_COBERTURA)))
            linha = min(self.linhas_regioes - 1,
                        max(0, int(agente.posicao.y // TAMANHO_REGIAO_COBERTURA)))
            self.regioes_visitadas.add((coluna, linha))

    @property
    def cobertura(self):
        """Fração das regiões diferentes visitadas pelo enxame, entre 0 e 1."""
        return len(self.regioes_visitadas) / self.total_regioes

    def atualizar(self, delta_tempo):
        if self.em_pausa or self.terminada: return
        self.tempo_decorrido += delta_tempo
        self.feromonas.atualizar(delta_tempo)
        for agente in self.agentes:
            agente.atualizar(delta_tempo, self.recursos,
                             self.obstaculos, self.feromonas)
        self._registar_cobertura()

        if sum(a.entregas for a in self.agentes) == len(self.recursos):
            self.terminada = True
            self.tempo_final = self.tempo_decorrido
