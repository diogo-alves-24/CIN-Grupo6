import math
import random
from dataclasses import dataclass
from collections import deque
import pygame
from simulacao.configuracao import (RAIO_AGENTE, RAIO_RECURSO, RAIO_NINHO,
    POSICAO_NINHO, VELOCIDADE_MAXIMA, VELOCIDADE_ROTACAO, LARGURA, ALTURA,
    DEPOSITO_CASA, DEPOSITO_COMIDA, CUSTO_ENERGIA_DISTANCIA,
    PENALIZACAO_VELOCIDADE, CUSTO_ENERGIA_TEMPO, JANELA_MEMORIA)
from simulacao.controladores import ControladorAleatorio, ControladorEnxame, ControladorNeuronal
from simulacao.sensores import observar

class Agente:
    def __init__(self, numero, modo='Aleatório', rede=None):
        self.numero = numero
        self.posicao = POSICAO_NINHO + pygame.Vector2(random.uniform(-12, 12), random.uniform(-12, 12))
        self.orientacao = random.uniform(-math.pi, math.pi)
        self.velocidade = random.uniform(65., VELOCIDADE_MAXIMA)
        self.controlador = (ControladorAleatorio() if modo == 'Aleatório' else ControladorNeuronal(rede) if modo == 'Neuroevolução' else ControladorEnxame(modo))
        self.modo = modo
        self.papel = 'EXPLORADOR'
        self.tempo_sem_recursos = 0.0
        self.transporta_recurso = False
        self.energia = 0.0
        self.entregas = 0
        self.observacao = None
        self.tempo_deposito = 0.0
        self.tempo_fuga = 0.0
        self.direcao_fuga = pygame.Vector2()
        self.posicao_referencia = self.posicao.copy()
        self.tempo_verificacao = 0.0
        self.tempo_desde_ninho = 0.0
        self.tempo_desde_recurso = 0.0
        self.pode_depositar_casa = True
        # Indicadores locais para detetar ciclos, sem mapa global.
        self.progresso_local = 1.0
        self.tempo_bloqueado = 0.0
        self.bloqueios = 0
        # Métricas para a função de aptidão e para o relatório.
        self.colisoes = 0          # nº de contactos distintos com obstáculos
        self._em_contacto = False
        self.tempos_entrega = []   # instantes (s de simulação) de cada entrega
        self.relogio = 0.0
        # A saída neuronal 0 mantém a velocidade do controlador Combinado.
        self.comando_velocidade = 0.0
        self.velocidade_atual = self.velocidade
        # Histórico de deslocamentos do PRÓPRIO agente (sem mapa global).
        self.tempo_com_recurso = 0.0
        self.indice_ciclo = 0.0
        self.rotacao_recente = 0.0
        self.progresso_recente = 0.0
        self._deslocamento_relativo = pygame.Vector2()
        self._distancia_total = 0.0
        self._rotacao_total = 0.0
        self._amostras = deque([(0.0, pygame.Vector2(), 0.0, 0.0)])
        self.colisoes_borda = 0       # contactos distintos com os limites da arena
        self._em_borda = False
        self.tempos_transporte = []   # duração de cada transporte (recolha -> entrega)
        self.instante_recolha = 0.0
        self.distancia_recolha_ninho = 0.0  # somente para o avaliador, não para a rede

    def atualizar(self, delta_tempo, recursos, obstaculos, feromonas):
        self.observacao = observar(self, recursos, obstaculos,
                                  feromonas if self.modo != 'Aleatório' else None)
        obs = self.observacao
        if self.modo != 'Aleatório' and not self.transporta_recurso:
            if obs.recursos or obs.feromona_comida > .12:
                self.papel = 'RECOLHEDOR'
                self.pode_depositar_casa = False
                self.tempo_sem_recursos = 0
            else:
                self.tempo_sem_recursos += delta_tempo
                if self.tempo_sem_recursos > 8:
                    self.papel = 'EXPLORADOR'
        if self.modo in ('Combinado', 'Neuroevolução'):
            self.tempo_fuga = max(0.0, self.tempo_fuga - delta_tempo)
            self.tempo_verificacao += delta_tempo
            if self.tempo_verificacao >= 3.0:
                progresso = self.posicao.distance_to(self.posicao_referencia)
                self.progresso_local = min(1.0, progresso / 110.0)
                # Bloqueio: pouco deslocamento líquido ou órbita persistente.
                em_ciclo = self.indice_ciclo > 0.78 and self.progresso_recente < 0.22
                if progresso < 38 or em_ciclo:
                    self.tempo_bloqueado += self.tempo_verificacao
                    self.bloqueios += 1
                if (progresso < 38 or em_ciclo) and self.tempo_fuga <= 0:
                    # Fuga local de uma órbita/engarrafamento, sem teletransporte.
                    angulo = self.orientacao + random.choice((-1, 1)) * random.uniform(.9, 2.2)
                    self.direcao_fuga = pygame.Vector2(math.cos(angulo), math.sin(angulo))
                    self.tempo_fuga = random.uniform(1.4, 2.6)
                self.posicao_referencia = self.posicao.copy()
                self.tempo_verificacao = 0.0
        orientacao_anterior = self.orientacao
        desejada = self.controlador.direcao(delta_tempo, obs, self)
        alvo = math.atan2(desejada.y, desejada.x)
        diferenca_angular = (alvo - self.orientacao + math.pi) % (2 * math.pi) - math.pi
        self.orientacao += max(-VELOCIDADE_ROTACAO * delta_tempo,
                               min(VELOCIDADE_ROTACAO * delta_tempo, diferenca_angular))
        direcao = pygame.Vector2(math.cos(self.orientacao), math.sin(self.orientacao))
        # Comando negativo abranda até 20 % da velocidade própria.
        # Comando positivo acelera progressivamente até ao limite físico.
        # Não há intervalo de saída "achatado" por clipping no máximo.
        comando = max(-1.0, min(1.0, self.comando_velocidade))
        if comando >= 0:
            velocidade = self.velocidade + comando * (VELOCIDADE_MAXIMA - self.velocidade)
        else:
            velocidade = self.velocidade * (1.0 + 0.8 * comando)
        self.velocidade_atual = velocidade
        proxima = self.posicao + direcao * velocidade * delta_tempo
        borda = False
        if proxima.x < RAIO_AGENTE or proxima.x > LARGURA - RAIO_AGENTE:
            borda = True
            self.orientacao = math.pi - self.orientacao
            proxima.x = max(RAIO_AGENTE, min(LARGURA - RAIO_AGENTE, proxima.x))
        if proxima.y < RAIO_AGENTE or proxima.y > ALTURA - RAIO_AGENTE:
            borda = True
            self.orientacao = -self.orientacao
            proxima.y = max(RAIO_AGENTE, min(ALTURA - RAIO_AGENTE, proxima.y))
        if borda and not self._em_borda:
            self.colisoes_borda += 1
        self._em_borda = borda
        self.relogio += delta_tempo
        contacto = False
        for obstaculo in obstaculos:
            diferenca = proxima - obstaculo.centro
            limite = obstaculo.raio + RAIO_AGENTE
            if diferenca.length_squared() < limite**2:
                normal = diferenca.normalize() if diferenca.length_squared() else pygame.Vector2(1,0)
                contacto = True
                proxima = obstaculo.centro + normal * limite
                self.orientacao = math.atan2(normal.y, normal.x) + random.uniform(-.4, .4)
        if contacto and not self._em_contacto:
            self.colisoes += 1
        self._em_contacto = contacto
        proxima.x = max(RAIO_AGENTE, min(LARGURA - RAIO_AGENTE, proxima.x))
        proxima.y = max(RAIO_AGENTE, min(ALTURA - RAIO_AGENTE, proxima.y))
        deslocamento = proxima - self.posicao
        # Energia em unidades simuladas: custo de distância, penalização
        # quadrática da velocidade e custo basal por segundo.
        # Fazer o mesmo percurso mais depressa consome mais energia.
        distancia = deslocamento.length()
        fator_energetico = 1.0 + PENALIZACAO_VELOCIDADE * (velocidade / VELOCIDADE_MAXIMA) ** 2
        self.energia += (distancia * CUSTO_ENERGIA_DISTANCIA * fator_energetico
                         + CUSTO_ENERGIA_TEMPO * delta_tempo)
        self.posicao = proxima

        # Atualizar memória relativa da trajetória: nunca guarda coordenadas
        # do ninho, de fontes ou de outros agentes.
        self._deslocamento_relativo += deslocamento
        self._distancia_total += distancia
        rotacao = (self.orientacao - orientacao_anterior + math.pi) % (2 * math.pi) - math.pi
        self._rotacao_total += rotacao
        self._amostras.append((self.relogio, self._deslocamento_relativo.copy(),
                              self._distancia_total, self._rotacao_total))
        while len(self._amostras) > 2 and self._amostras[1][0] < self.relogio - JANELA_MEMORIA:
            self._amostras.popleft()
        instante, posicao_antiga, distancia_antiga, rotacao_antiga = self._amostras[0]
        linha_reta = (self._deslocamento_relativo - posicao_antiga).length()
        caminho = self._distancia_total - distancia_antiga
        intervalo = max(delta_tempo, self.relogio - instante)
        self.indice_ciclo = max(0.0, min(1.0, 1.0 - linha_reta / caminho)) if caminho > 15 else 0.0
        self.rotacao_recente = max(-1.0, min(1.0, (self._rotacao_total - rotacao_antiga) / (2 * math.pi)))
        self.progresso_recente = min(1.0, linha_reta / (VELOCIDADE_MAXIMA * intervalo))
        self.tempo_deposito += delta_tempo
        if self.transporta_recurso:
            self.tempo_desde_recurso += delta_tempo
            self.tempo_com_recurso += delta_tempo
        else:
            self.tempo_desde_ninho += delta_tempo
        # Apenas exploradores que passaram pelo ninho criam rasto de casa.
        if (not self.transporta_recurso and
            self.posicao.distance_squared_to(POSICAO_NINHO) <= RAIO_NINHO**2):
            self.pode_depositar_casa = True
            self.tempo_desde_ninho = 0.0
            self.papel = 'EXPLORADOR'
        if self.modo != 'Aleatório' and self.tempo_deposito >= .15:
            if self.transporta_recurso and self.tempo_fuga <= 0:
                # Não reforçar trilhos enquanto tenta escapar a um ciclo.
                quantidade = DEPOSITO_COMIDA * max(.18, 1 - self.tempo_desde_recurso / 8)
                feromonas.depositar('comida', self.posicao, quantidade)
            elif self.papel == 'EXPLORADOR' and self.pode_depositar_casa:
                quantidade = DEPOSITO_CASA * max(.12, 1 - self.tempo_desde_ninho / 8)
                feromonas.depositar('casa', self.posicao, quantidade, -deslocamento)
            self.tempo_deposito = 0.0
        if not self.transporta_recurso:
            for recurso in recursos:
                if recurso.disponivel and self.posicao.distance_squared_to(recurso.posicao) <= (RAIO_AGENTE + RAIO_RECURSO)**2:
                    recurso.disponivel = False
                    self.transporta_recurso = True
                    self.papel = 'RECOLHEDOR'
                    self.pode_depositar_casa = False
                    self.tempo_desde_recurso = 0.0
                    self.tempo_com_recurso = 0.0
                    self.instante_recolha = self.relogio
                    self.distancia_recolha_ninho = self.posicao.distance_to(POSICAO_NINHO)
                    break
        elif self.posicao.distance_squared_to(POSICAO_NINHO) <= (RAIO_NINHO + RAIO_AGENTE)**2:
            self.transporta_recurso = False
            self.tempo_com_recurso = 0.0
            self.entregas += 1
            self.tempos_entrega.append(self.relogio)
            self.tempos_transporte.append(self.relogio - self.instante_recolha)
            self.tempo_sem_recursos = 0
            self.tempo_desde_ninho = 0.0
            self.papel = 'EXPLORADOR'
            self.pode_depositar_casa = True

@dataclass
class Obstaculo:
    centro: pygame.Vector2
    raio: float

@dataclass
class Recurso:
    posicao: pygame.Vector2
    disponivel: bool = True
