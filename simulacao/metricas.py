def calcular_metricas(simulacao):
    return {
        'entregues': sum(a.entregas for a in simulacao.agentes),
        'transportados': sum(a.transporta_recurso for a in simulacao.agentes),
        'disponiveis': sum(r.disponivel for r in simulacao.recursos),
        'energia': sum(a.energia for a in simulacao.agentes),
    }
