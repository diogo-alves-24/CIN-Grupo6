"""Dois controladores: passeio aleatório e inteligência de enxame combinada."""
import math
import random
import pygame

MODOS = ('Aleatório', 'Combinado')

def normalizar(vetor):
    return vetor.normalize() if vetor.length_squared() > 0.00001 else pygame.Vector2()

class ControladorAleatorio:
    def __init__(self):
        self.rotacao = random.uniform(-1, 1)
        self.tempo_ate_rotacao = random.uniform(.1, 1)

    def direcao(self, delta_tempo, observacao, agente):
        self.tempo_ate_rotacao -= delta_tempo
        if self.tempo_ate_rotacao <= 0:
            self.tempo_ate_rotacao = random.uniform(.2, 1)
            self.rotacao = random.uniform(-1, 1)
        frente = pygame.Vector2(math.cos(agente.orientacao), math.sin(agente.orientacao))
        return frente.rotate_rad(self.rotacao * .38)

class ControladorEnxame:
    def __init__(self, modo):
        self.modo = modo
        self.aleatorio = ControladorAleatorio()

    def direcao(self, delta_tempo, observacao, agente):
        obs = observacao
        base = self.aleatorio.direcao(delta_tempo, obs, agente)
        vetor = base * 0.8
        vetor += normalizar(obs.direcao_obstaculo) * (3.5 if obs.distancia_obstaculo < 36 else 1.2)
        if self.modo == 'Combinado':
            # Quando o agente já leva um recurso, a prioridade é regressar
            # Mantemos apenas separação para evitar colisões
            vetor += normalizar(obs.separacao) * (3.2 if agente.transporta_recurso else 2.7)
            if not agente.transporta_recurso:
                vetor += normalizar(obs.alinhamento) * .55
                vetor += normalizar(obs.coesao) * .24
        if self.modo == 'Combinado':
            if agente.transporta_recurso:
                vetor += normalizar(obs.direcao_casa) * 4.4
                if obs.ninho_visivel:
                    vetor += obs.direcao_ninho * 6.5
                # Enquanto transporta, ignora a atração por zonas com recurso.
                vetor *= 0.95
            else:
                if obs.recursos: vetor += obs.direcao_recurso * 5
                else: vetor += normalizar(obs.direcao_comida) * (2.6 if agente.papel == 'RECOLHEDOR' else 0.85)
        elif obs.recursos and not agente.transporta_recurso:
            vetor += obs.direcao_recurso * 2.8
        if obs.ninho_visivel and agente.transporta_recurso:
            vetor += obs.direcao_ninho * 5.5
        return normalizar(vetor) if vetor.length_squared() else base
