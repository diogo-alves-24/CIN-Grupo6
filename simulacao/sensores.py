from dataclasses import dataclass
import pygame
from simulacao.configuracao import ALCANCE_SENSORES

@dataclass(frozen=True)
class Observacao:
    vizinhos: int
    recursos: int
    distancia_obstaculo: float
    feromona: float = 0.0  # reservado para implementação posterior


def observar(agente, agentes, recursos, obstaculos) -> Observacao:
    """Observações locais: não disponibiliza posições globais ao controlador."""
    vizinhos = sum(1 for outro in agentes if outro is not agente
                    and agente.posicao.distance_squared_to(outro.posicao) <= ALCANCE_SENSORES ** 2)
    recursos_proximos = sum(1 for recurso in recursos if recurso.disponivel
                           and agente.posicao.distance_squared_to(recurso.posicao) <= ALCANCE_SENSORES ** 2)
    distancia = min((max(0., agente.posicao.distance_to(obstaculo.centro) - obstaculo.raio)
                    for obstaculo in obstaculos
                    if agente.posicao.distance_to(obstaculo.centro) - obstaculo.raio <= ALCANCE_SENSORES),
                   default=ALCANCE_SENSORES)
    return Observacao(vizinhos, recursos_proximos, distancia)
