"""Verificações rápidas da função de fitness (sem executar simulações longas).

Executar: python testar_fitness.py
"""
from simulacao.avaliacao import calcular_aptidao, agregar_resultados


def fitness(tempos, energia=0, colisoes=0, bordas=0, bloqueado=0, cobertura=0):
    valor, componentes = calcular_aptidao(
        tempos, 300.0, energia, colisoes, bordas, bloqueado,
        cobertura, num_agentes=18, num_recursos=36,
    )
    assert all(0 <= v <= 1 for v in componentes.values()), componentes
    return valor


def main():
    # Uma entrega extra deve valer mais que TODAS as métricas secundárias.
    pior_uma = fitness([299.9], energia=1e9, colisoes=1e9,
                       bordas=1e9, bloqueado=1e9, cobertura=0.0)
    melhor_zero = fitness([], energia=0.0, colisoes=0.0,
                          bordas=0.0, bloqueado=0.0, cobertura=1.0)
    assert pior_uma > melhor_zero, (pior_uma, melhor_zero)

    # A entrega antecipada deve ser mais recompensada.
    assert fitness([30.0]) > fitness([250.0])

    # Percorrer menos energia e evitar bloqueios deve melhorar fitness.
    assert fitness([120.0], energia=100.0) > fitness([120.0], energia=10000.0)
    assert fitness([120.0], bloqueado=0.0) > fitness([120.0], bloqueado=1000.0)

    # Maior proximidade FINAL ao ninho com comida no fim ajuda.
    def pontuacao_progresso(valor):
        return calcular_aptidao([120.0], 300.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 18, 36, progresso_regresso=valor)[0]
    assert pontuacao_progresso(0.8) > pontuacao_progresso(0.2)

    # A conclusão antecipada vale mais que completar tudo tarde.
    cedo = fitness([50.0] * 36)
    tarde = fitness([250.0] * 36)
    assert cedo > tarde

    # Mesmo os termos secundários no seu melhor não superam uma entrega extra.
    mais_entrega_pior = fitness([300.0], energia=1e9, colisoes=1e9,
                               bordas=1e9, bloqueado=1e9, cobertura=0)
    zero_entrega_melhor = fitness([], energia=0, colisoes=0,
                                 bordas=0, bloqueado=0, cobertura=1)
    assert mais_entrega_pior > zero_entrega_melhor

    # Cobertura mais ampla deve ajudar sem repetir bónus por zonas já visitadas.
    assert fitness([], cobertura=0.5) > fitness([], cobertura=0.2)

    # Fitness robusta: 80 % da média + 20 % da pior de 4 sementes.
    valores = [dict(fitness=v, entregues=0.0) for v in (10.0, 20.0, 30.0, 40.0)]
    resultado = agregar_resultados(valores)
    assert abs(resultado['fitness'] - 25.0) < 1e-9
    assert abs(resultado['fitness_robusta'] - 22.0) < 1e-9
    print('OK: entregas, rapidez, conclusão, progresso, energia, bloqueios, cobertura e robustez.')


if __name__ == '__main__':
    main()
