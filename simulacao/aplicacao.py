import pygame
from simulacao.configuracao import LARGURA, ALTURA, FPS
from simulacao.mundo import Simulacao
from simulacao.desenho import desenhar

def executar():
    pygame.init()
    ecra = pygame.display.set_mode((LARGURA, ALTURA))
    pygame.display.set_caption('Forrageamento Espacial - Simulação')
    relogio = pygame.time.Clock()
    fonte = pygame.font.SysFont('arial', 21, bold=True)
    fonte_pequena = pygame.font.SysFont('arial', 18)
    simulacao = Simulacao()
    em_execucao = True
    while em_execucao:
        delta_tempo = min(relogio.tick(FPS) / 1000., .05)
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                em_execucao = False
            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_ESCAPE:
                    em_execucao = False
                elif evento.key == pygame.K_SPACE:
                    simulacao.em_pausa = not simulacao.em_pausa
                elif evento.key == pygame.K_r:
                    simulacao = Simulacao()
                elif evento.key == pygame.K_s:
                    simulacao.mostrar_sensores = not simulacao.mostrar_sensores
        simulacao.atualizar(delta_tempo)
        desenhar(ecra, simulacao, fonte, fonte_pequena)
        pygame.display.flip()
    pygame.quit()
