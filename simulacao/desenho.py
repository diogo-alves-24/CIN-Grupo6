import math
import pygame
from simulacao.configuracao import (LARGURA, ALTURA, COR_FUNDO, BRANCO, ALCANCE_SENSORES,
                              POSICAO_NINHO, RAIO_NINHO, RAIO_RECURSO, RAIO_AGENTE)
from simulacao.metricas import calcular_metricas

def desenhar(ecra, sim, fonte, fonte_pequena):
    ecra.fill(COR_FUNDO)
    for x in range(0, LARGURA, 50):
        pygame.draw.line(ecra, (27, 35, 47), (x, 0), (x, ALTURA))
    for y in range(0, ALTURA, 50):
        pygame.draw.line(ecra, (27, 35, 47), (0, y), (LARGURA, y))
    pygame.draw.circle(ecra, (33, 98, 74), POSICAO_NINHO, RAIO_NINHO)
    pygame.draw.circle(ecra, (108, 228, 164), POSICAO_NINHO, RAIO_NINHO, 3)
    etiqueta = fonte_pequena.render('NINHO', True, BRANCO)
    ecra.blit(etiqueta, etiqueta.get_rect(center=POSICAO_NINHO))
    for obstaculo in sim.obstaculos:
        pygame.draw.circle(ecra, (101, 105, 117), obstaculo.centro, obstaculo.raio)
        pygame.draw.circle(ecra, (160, 167, 180), obstaculo.centro, obstaculo.raio, 2)
    for recurso in sim.recursos:
        if recurso.disponivel:
            pygame.draw.circle(ecra, (245, 217, 87), recurso.posicao, RAIO_RECURSO)
    for agente in sim.agentes:
        posicao = (int(agente.posicao.x), int(agente.posicao.y))
        if sim.mostrar_sensores:
            pygame.draw.circle(ecra, (55, 67, 84), posicao, int(ALCANCE_SENSORES), 1)
        cor = (255, 183, 77) if agente.transporta_recurso else (75, 199, 241)
        pygame.draw.circle(ecra, cor, posicao, RAIO_AGENTE)
        frente = agente.posicao + pygame.Vector2(math.cos(agente.orientacao), math.sin(agente.orientacao)) * 13
        pygame.draw.line(ecra, BRANCO, posicao, (int(frente.x), int(frente.y)), 2)

    m = calcular_metricas(sim)
    linhas = [
        f'Tempo: {sim.tempo_decorrido:6.1f}s   Agentes: {len(sim.agentes)}',
        f"Entregues: {m['entregues']}   Transportados: {m['transportados']}   Disponíveis: {m['disponiveis']}",
        f"Energia acumulada (unid. simuladas): {m['energia']:.0f}",
        'ESPAÇO: pausa   R: reinicia   S: sensores   ESC: sair',
    ]
    painel = pygame.Surface((730, 141), pygame.SRCALPHA)
    painel.fill((7, 12, 20, 224))
    ecra.blit(painel, (12, 12))
    for i, linha in enumerate(linhas):
        imagem = (fonte if i == 0 else fonte_pequena).render(linha, True, BRANCO)
        ecra.blit(imagem, (26, 20 + i * 25))
    if sim.em_pausa:
        pausa = fonte.render('PAUSA', True, (255, 196, 90))
        ecra.blit(pausa, pausa.get_rect(center=(LARGURA // 2, 28)))
