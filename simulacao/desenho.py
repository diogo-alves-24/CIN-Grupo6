import math
import pygame
from simulacao.configuracao import (LARGURA, ALTURA, COR_FUNDO, BRANCO,
    ALCANCE_SENSORES, POSICAO_NINHO, RAIO_NINHO, RAIO_RECURSO, RAIO_AGENTE,
    TAMANHO_CELULA)
from simulacao.metricas import calcular_metricas

def desenhar(ecra, sim, fonte, fonte_pequena):
    ecra.fill(COR_FUNDO)
    for x in range(0, LARGURA, 50):
        pygame.draw.line(ecra, (27,35,47), (x,0), (x,ALTURA))
    for y in range(0, ALTURA, 50):
        pygame.draw.line(ecra, (27,35,47), (0,y), (LARGURA,y))
    if sim.mostrar_feromonas and sim.modo == 'Combinado':
        camada = pygame.Surface((LARGURA, ALTURA), pygame.SRCALPHA)
        tipos = (('casa', (35, 153, 255)), ('comida', (255, 165, 35)))
        for nome, cor in tipos:
            if sim.tipo_feromona not in ('ambas', nome):
                continue
            grelha = getattr(sim.feromonas, nome)
            for y, linha in enumerate(grelha):
                for x, valor in enumerate(linha):
                    if valor < 0.008:
                        continue
                    intensidade = min(1.0, (valor / 2.8) ** 0.48)
                    transparencia = min(215, int(28 + 185 * intensidade))
                    pygame.draw.rect(camada, (*cor, transparencia),
                                     (x * TAMANHO_CELULA, y * TAMANHO_CELULA,
                                      TAMANHO_CELULA, TAMANHO_CELULA))
        ecra.blit(camada, (0, 0))
    pygame.draw.circle(ecra, (33,98,74), POSICAO_NINHO, RAIO_NINHO)
    pygame.draw.circle(ecra, (108,228,164), POSICAO_NINHO, RAIO_NINHO, 3)
    etiqueta = fonte_pequena.render('NINHO', True, BRANCO)
    ecra.blit(etiqueta, etiqueta.get_rect(center=POSICAO_NINHO))
    for obstaculo in sim.obstaculos:
        pygame.draw.circle(ecra, (101,105,117), obstaculo.centro, obstaculo.raio)
        pygame.draw.circle(ecra, (160,167,180), obstaculo.centro, obstaculo.raio, 2)
    for recurso in sim.recursos:
        if recurso.disponivel:
            pygame.draw.circle(ecra, (245,217,87), recurso.posicao, RAIO_RECURSO)
    for agente in sim.agentes:
        posicao = (int(agente.posicao.x), int(agente.posicao.y))
        if sim.mostrar_sensores:
            pygame.draw.circle(ecra, (76,94,118), posicao, int(ALCANCE_SENSORES), 1)
        cor = ((255,183,77) if agente.transporta_recurso else
               (176,117,235) if agente.papel == 'RECOLHEDOR' else (75,199,241))
        pygame.draw.circle(ecra, cor, posicao, RAIO_AGENTE)
        frente = agente.posicao + pygame.Vector2(math.cos(agente.orientacao), math.sin(agente.orientacao))*13
        pygame.draw.line(ecra, BRANCO, posicao, (int(frente.x),int(frente.y)), 2)
    m = calcular_metricas(sim)
    linhas = [
        f'FORRAGEAMENTO | {sim.modo.upper()}',
        f'Tempo: {sim.tempo_decorrido:6.1f}s  Agentes: {len(sim.agentes)}  Exploradores: {m["exploradores"]}  Recolhedores: {m["recolhedores"]}',
        f'Entregues: {m["entregues"]}  Transportados: {m["transportados"]}  Disponíveis: {m["disponiveis"]}',
        f'Energia acumulada (unid. simuladas): {m["energia"]:.0f}',
        '1: Random Walk  2: Combinado  |  R: reiniciar  H: esconder painel',
        'ESPAÇO: pausa  S: sensores  F: rastos  T: ambos/comida/casa  ESC: sair',
    ]
    if sim.mostrar_painel:
        painel = pygame.Surface((1045, 169), pygame.SRCALPHA)
        painel.fill((7, 12, 20, 224))
        ecra.blit(painel, (12, 12))
        for i, linha in enumerate(linhas):
            img = (fonte if i == 0 else fonte_pequena).render(linha, True, BRANCO)
            ecra.blit(img, (26, 20 + i * 25))
    if sim.em_pausa:
        pausa = fonte.render('PAUSA', True, (255,196,90))
        ecra.blit(pausa, pausa.get_rect(center=(LARGURA//2,195)))
