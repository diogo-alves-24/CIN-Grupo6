"""Dois controladores: passeio aleatório e inteligência de enxame combinada."""
import math
import random
import pygame
from simulacao.configuracao import ALCANCE_SENSORES

MODOS = ('Aleatório', 'Combinado', 'Neuroevolução')

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

        # Escapar de ciclos: um pequeno período de procura de outra trajetória.
        if agente.tempo_fuga > 0:
            vetor = agente.direcao_fuga * 4.0 + base * 0.6
            vetor += normalizar(obs.direcao_obstaculo) * (5.5 if obs.distancia_obstaculo < 45 else 0)
            return normalizar(vetor) if vetor.length_squared() > 0.001 else base

        if agente.transporta_recurso:
            # Regresso por feromonas locais, sem posição global nem odometria.
            regresso = obs.direcao_ninho if obs.ninho_visivel else obs.direcao_casa
            vetor = normalizar(regresso) * (9.0 if obs.ninho_visivel else 5.0)
            if vetor.length_squared() < 0.01:
                # Sem pista: explorar até recuperar um trilho de casa.
                vetor = base * 1.5
            vetor += normalizar(obs.direcao_obstaculo) * (7 if obs.distancia_obstaculo < 45 else .5)
            vetor += normalizar(obs.separacao) * .8
            return normalizar(vetor) if vetor.length_squared() > .001 else base

        vetor = base * 0.9
        vetor += normalizar(obs.direcao_obstaculo) * (5 if obs.distancia_obstaculo < 42 else .8)
        vetor += normalizar(obs.separacao) * 2.7
        vetor += normalizar(obs.alinhamento) * .4
        vetor += normalizar(obs.coesao) * .12
        if obs.recursos:
            vetor += obs.direcao_recurso * 5
        else:
            vetor += normalizar(obs.direcao_comida) * (2.3 if agente.papel == 'RECOLHEDOR' else .65)
        return normalizar(vetor) if vetor.length_squared() > .001 else base


class ControladorNeuronal:
    """Controlador híbrido com duas MLP: procura e regresso.

    Entradas e correções de direção no referencial do próprio agente.
    O genoma neutro (todos zeros) reproduz o Combinado, incluindo a velocidade.
    As duas redes recebem os mesmos sinais locais, mas são treinadas com pesos
    independentes para as duas tarefas.
    """
    GANHO = 1.5

    def __init__(self, rede):
        self.rede = rede
        self.base = ControladorEnxame('Combinado')

    def direcao(self, delta_tempo, observacao, agente):
        o = observacao
        referencia = self.base.direcao(delta_tempo, o, agente)
        fx, fy = math.cos(agente.orientacao), math.sin(agente.orientacao)
        frente = pygame.Vector2(fx, fy)
        lateral = pygame.Vector2(-fy, fx)
        afastamento = normalizar(o.direcao_obstaculo)

        def ego(v):  # (à frente, à esquerda)
            return v.x * fx + v.y * fy, -v.x * fy + v.y * fx

        ninho = o.direcao_ninho if o.ninho_visivel else pygame.Vector2()
        dados = []
        for v in (referencia, o.direcao_casa, o.direcao_comida, o.direcao_recurso,
                  afastamento, normalizar(o.separacao), ninho):
            dados.extend(ego(v))
        dados.extend((
            float(agente.transporta_recurso),
            min(1.0, o.feromona_casa / 8.0),
            min(1.0, o.feromona_comida / 8.0),
            min(1.0, agente.tempo_fuga / 3.0),
            agente.progresso_local,
            min(1.0, agente.tempo_sem_recursos / 12.0),
            min(1.0, o.distancia_obstaculo / ALCANCE_SENSORES),
            min(1.0, o.vizinhos / 6.0),
            float(o.ninho_visivel),
            min(1.0, o.recursos / 4.0),
            # Memória LOCAL: sem localização do ninho nem mapa global.
            min(1.0, agente.tempo_com_recurso / 40.0),
            agente.indice_ciclo,                 # 0 = percurso direto; 1 = quase circular
            agente.rotacao_recente,             # rotação assinada nas últimas ~5 s
            agente.progresso_recente,           # distância líquida / máxima possível
        ))
        saida = self.rede.prever(dados, agente.transporta_recurso)
        # Saída 2 em [-1, 1]: comando contínuo de velocidade. Zero mantém
        # a velocidade do Combinado; não se ultrapassa VELOCIDADE_MAXIMA.
        agente.comando_velocidade = float(saida[2])
        ajuste = lateral * float(saida[0]) + frente * float(saida[1])
        forca = ajuste.length()
        if forca < 1e-9:           # genoma neutro: comportamento idêntico ao Combinado
            return referencia
        vetor = referencia + ajuste * self.GANHO
        # Reflexo de segurança local, proporcional ao desvio introduzido pela rede
        # (não altera o Combinado quando a rede não corrige nada).
        if o.distancia_obstaculo < 28:
            vetor += afastamento * 3.5 * min(1.0, forca)
        return normalizar(vetor) if vetor.length_squared() > 0.0001 else referencia
