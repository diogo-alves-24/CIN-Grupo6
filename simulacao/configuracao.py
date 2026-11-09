import pygame
LARGURA, ALTURA = 1100, 740
FPS = 60
NUM_AGENTES = 18
NUM_RECURSOS = 36
ALCANCE_SENSORES = 90.0
RAIO_AGENTE = 7
RAIO_RECURSO = 8
RAIO_NINHO = 48
VELOCIDADE_MAXIMA = 115.0
VELOCIDADE_ROTACAO = 2.8
COR_FUNDO = (18, 24, 34)
BRANCO = (237, 240, 246)
POSICAO_NINHO = pygame.Vector2(LARGURA / 2, ALTURA / 2)
TAMANHO_CELULA = 12
EVAPORACAO_CASA = 0.05  # segundo^-1
EVAPORACAO_COMIDA = 0.13  # segundo^-1
DIFUSAO_CASA = 0.10  # proposta experimental: rasto de regresso mais concentrado
DIFUSAO_COMIDA = 0.18  # proposta experimental: cheiro de comida mais disperso
DEPOSITO_CASA = 0.85
DEPOSITO_COMIDA = 1.3
MAX_FEROMONA = 12.0

# Modelo energético da simulação (unidades artificiais)
# A mesma distância percorrida a maior velocidade custa mais energia.
CUSTO_ENERGIA_DISTANCIA = 0.018
PENALIZACAO_VELOCIDADE = 0.80
CUSTO_ENERGIA_TEMPO = 0.025

# Memória recente de movimento usada apenas pelo controlador neuronal
JANELA_MEMORIA = 5.0  # segundos simulados

# Métricas da avaliação evolutiva (não são entradas das redes)
TAMANHO_REGIAO_COBERTURA = 60  # píxeis: células para medir exploração do enxame
