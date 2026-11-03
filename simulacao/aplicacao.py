"""Menu simples e execução visual de três controladores."""
from pathlib import Path
import numpy as np
import pygame
from simulacao.configuracao import LARGURA, ALTURA, FPS
from simulacao.mundo import Simulacao
from simulacao.desenho import desenhar, desenhar_resultado
from simulacao.controladores import MODOS
from simulacao.rede_neuronal import TAMANHO_GENOMA

# Passo FIXO igual ao do treino/benchmark (0,05 s): o que se vê no vídeo é
# exatamente a dinâmica em que a rede foi treinada e avaliada.
PASSO_SIMULACAO = 0.05

FICHEIRO_GENOMA = Path(__file__).resolve().parent.parent / 'melhor_genoma.npz'


def criar_simulacao(indice):
    modo = MODOS[indice]
    if modo == 'Neuroevolução':
        if not FICHEIRO_GENOMA.exists():
            return None
        with np.load(FICHEIRO_GENOMA, allow_pickle=False) as dados:
            genoma = dados['genoma'].copy()
        if genoma.size != TAMANHO_GENOMA:   # pesos de uma arquitetura antiga
            return None
        return Simulacao(modo, genoma=genoma)
    return Simulacao(modo)


def executar():
    pygame.init()
    ecra = pygame.display.set_mode((LARGURA, ALTURA))
    pygame.display.set_caption('Forrageamento Espacial - Inteligência de Enxame')
    relogio = pygame.time.Clock()
    fonte = pygame.font.SysFont('arial', 21, bold=True)
    pequena = pygame.font.SysFont('arial', 18)
    estado = 'MENU'
    simulacao = None
    mensagem = ''
    executando = True
    acumulador = 0.0
    botoes = [pygame.Rect(340, 230+i*100, 420, 70) for i in range(len(MODOS))]

    while executando:
        delta = min(relogio.tick(FPS) / 1000.0, .05)
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                executando = False
            elif estado == 'MENU':
                escolha = None
                if evento.type == pygame.KEYDOWN:
                    if evento.key == pygame.K_ESCAPE:
                        executando = False
                    elif pygame.K_1 <= evento.key <= pygame.K_3:
                        escolha = evento.key - pygame.K_1
                elif evento.type == pygame.MOUSEBUTTONDOWN and evento.button == 1:
                    for i, botao in enumerate(botoes):
                        if botao.collidepoint(evento.pos):
                            escolha = i
                            break
                if escolha is not None:
                    simulacao = criar_simulacao(escolha)
                    if simulacao is None:
                        mensagem = 'Sem genoma válido: corre python treinar.py'
                    else:
                        estado = 'SIMULACAO'
                        mensagem = ''
            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_ESCAPE:
                    executando = False
                elif evento.key == pygame.K_m:
                    estado = 'MENU'
                    simulacao = None
                elif evento.key == pygame.K_r:
                    novo = criar_simulacao(MODOS.index(simulacao.modo))
                    for nome in ('mostrar_painel', 'mostrar_feromonas', 'tipo_feromona',
                                 'mostrar_sensores', 'velocidade_simulacao'):
                        setattr(novo, nome, getattr(simulacao, nome))
                    simulacao = novo
                elif evento.key == pygame.K_SPACE:
                    simulacao.em_pausa = not simulacao.em_pausa
                elif evento.key == pygame.K_h:
                    simulacao.mostrar_painel = not simulacao.mostrar_painel
                elif evento.key == pygame.K_f:
                    simulacao.mostrar_feromonas = not simulacao.mostrar_feromonas
                elif evento.key == pygame.K_s:
                    simulacao.mostrar_sensores = not simulacao.mostrar_sensores
                elif evento.key == pygame.K_t:
                    tipos = ('ambas', 'comida', 'casa')
                    simulacao.tipo_feromona = tipos[(tipos.index(simulacao.tipo_feromona) + 1) % 3]
                elif evento.key == pygame.K_p:
                    velocidades = (1, 2, 4, 8)
                    simulacao.velocidade_simulacao = velocidades[(velocidades.index(simulacao.velocidade_simulacao) + 1) % 4]
        if estado == 'MENU':
            ecra.fill((18, 24, 34))
            for i, botao in enumerate(botoes):
                ativo = (i != 2 or FICHEIRO_GENOMA.exists())
                pygame.draw.rect(ecra, (35, 110, 100) if ativo else (55, 58, 65), botao, border_radius=9)
                nome = ('RANDOM WALK', 'COMBINADO', 'NEUROEVOLUÇÃO')[i]
                img = fonte.render(nome, True, (237, 240, 246))
                ecra.blit(img, img.get_rect(center=botao.center))
            if mensagem:
                img = pequena.render(mensagem, True, (255, 190, 110))
                ecra.blit(img, img.get_rect(center=(LARGURA//2, 590)))
        else:
            if not (simulacao.em_pausa or simulacao.terminada):
                acumulador += delta * simulacao.velocidade_simulacao
                while acumulador >= PASSO_SIMULACAO and not simulacao.terminada:
                    simulacao.atualizar(PASSO_SIMULACAO)
                    acumulador -= PASSO_SIMULACAO
            desenhar(ecra, simulacao, fonte, pequena)
            if simulacao.terminada:
                desenhar_resultado(ecra, simulacao)
        pygame.display.flip()
    pygame.quit()
