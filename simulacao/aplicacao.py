import pygame
from simulacao.configuracao import LARGURA, ALTURA, FPS
from simulacao.mundo import Simulacao
from simulacao.desenho import desenhar
from simulacao.controladores import MODOS

def executar():
    pygame.init()
    ecra = pygame.display.set_mode((LARGURA, ALTURA))
    pygame.display.set_caption('Forrageamento Espacial - Inteligência de Enxame')
    relogio = pygame.time.Clock()
    fonte = pygame.font.SysFont('arial', 21, bold=True)
    fonte_pequena = pygame.font.SysFont('arial', 18)
    simulacao = Simulacao()
    em_execucao = True
    while em_execucao:
        delta_tempo = min(relogio.tick(FPS) / 1000., .05)
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT: em_execucao = False
            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_ESCAPE: em_execucao = False
                elif evento.key == pygame.K_SPACE: simulacao.em_pausa = not simulacao.em_pausa
                elif evento.key == pygame.K_r:
                    novo = Simulacao(simulacao.modo)
                    novo.mostrar_painel = simulacao.mostrar_painel
                    novo.mostrar_feromonas = simulacao.mostrar_feromonas
                    novo.tipo_feromona = simulacao.tipo_feromona
                    novo.mostrar_sensores = simulacao.mostrar_sensores
                    simulacao = novo
                elif evento.key == pygame.K_s: simulacao.mostrar_sensores = not simulacao.mostrar_sensores
                elif evento.key == pygame.K_f: simulacao.mostrar_feromonas = not simulacao.mostrar_feromonas
                elif evento.key == pygame.K_h:
                    simulacao.mostrar_painel = not simulacao.mostrar_painel
                elif evento.key == pygame.K_t:
                    tipos = ('ambas', 'comida', 'casa')
                    atual = tipos.index(simulacao.tipo_feromona)
                    simulacao.tipo_feromona = tipos[(atual + 1) % len(tipos)]
                elif evento.key in (pygame.K_1, pygame.K_2):
                    novo = Simulacao(MODOS[evento.key - pygame.K_1])
                    novo.mostrar_painel = simulacao.mostrar_painel
                    novo.mostrar_feromonas = simulacao.mostrar_feromonas
                    novo.tipo_feromona = simulacao.tipo_feromona
                    novo.mostrar_sensores = simulacao.mostrar_sensores
                    simulacao = novo
        simulacao.atualizar(delta_tempo)
        desenhar(ecra, simulacao, fonte, fonte_pequena)
        pygame.display.flip()
    pygame.quit()
