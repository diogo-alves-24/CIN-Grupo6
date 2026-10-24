import random
from typing import Protocol
from simulacao.sensores import Observacao

#Temporario como ainda nao temos algoritmos
class ControladorAleatorio:
    def __init__(self):
        self.rotacao = random.uniform(-1., 1.)
        self.tempo_ate_rotacao = random.uniform(.1, 1.)

    def calcular_rotacao(self, delta_tempo: float, observacao: Observacao) -> float:
        self.tempo_ate_rotacao -= delta_tempo
        if self.tempo_ate_rotacao <= 0:
            self.tempo_ate_rotacao = random.uniform(.2, 1.)
            self.rotacao = random.uniform(-1., 1.)
        return self.rotacao


class Controlador(Protocol):
    def calcular_rotacao(self, delta_tempo: float, observacao: Observacao) -> float:
        """Devolve a velocidade angular normalizada no intervalo [-1, 1]."""
        ...
