"""NEAT (NeuroEvolution of Augmenting Topologies) em NumPy puro.

Implementação própria, sem dependências extra, com os ingredientes do artigo
original (Stanley & Miikkulainen, 2002):

  * genomas com genes de nós e de conexões, identificados por **números de
    inovação** históricos (permitem alinhar genomas diferentes no cruzamento);
  * **topologia mínima** inicial (entradas ligadas diretamente às saídas) que
    cresce por mutação estrutural: *adicionar conexão* e *adicionar nó*;
  * **cruzamento** alinhado por inovação (genes disjuntos/excedentes vêm do
    progenitor mais apto);
  * **especiação** por distância de compatibilidade e *fitness sharing*, para
    proteger inovações estruturais recentes;
  * elitismo, estagnação de espécies e sobrevivência por fração.

Só são geradas redes *feedforward* (sem ciclos): a memória do agente já entra
como sinais locais de entrada. O conjunto de nós é: entradas [0, E), saídas
[E, E+S) e ocultos (ids ≥ E+S, criados por mutação). O viés é um parâmetro de
cada nó não-entrada. Com todos os pesos e viéses a zero as saídas são
exatamente zero (genoma neutro).
"""
from collections import defaultdict
from dataclasses import dataclass
import math
import numpy as np


# --------------------------------------------------------------------------
# Genoma
# --------------------------------------------------------------------------
class Conexao:
    __slots__ = ('inov', 'entrada', 'saida', 'peso', 'ativa')

    def __init__(self, inov, entrada, saida, peso, ativa=True):
        self.inov = int(inov)
        self.entrada = int(entrada)
        self.saida = int(saida)
        self.peso = float(peso)
        self.ativa = bool(ativa)

    def copiar(self):
        return Conexao(self.inov, self.entrada, self.saida, self.peso, self.ativa)


class Genoma:
    """Genoma NEAT: viéses dos nós (saídas e ocultos) + conexões por inovação."""

    def __init__(self, entradas, saidas):
        self.entradas = int(entradas)
        self.saidas = int(saidas)
        self.vies = {n: 0.0 for n in self.ids_saida}   # id -> viés
        self.conexoes = {}                              # inovação -> Conexao
        self.fitness = 0.0

    # --- nós -------------------------------------------------------------
    @property
    def ids_entrada(self):
        return range(self.entradas)

    @property
    def ids_saida(self):
        return range(self.entradas, self.entradas + self.saidas)

    @property
    def ids_ocultos(self):
        base = self.entradas + self.saidas
        return sorted(n for n in self.vies if n >= base)

    def copiar(self):
        g = Genoma(self.entradas, self.saidas)
        g.vies = dict(self.vies)
        g.conexoes = {k: c.copiar() for k, c in self.conexoes.items()}
        g.fitness = self.fitness
        return g

    # --- fábricas --------------------------------------------------------
    @classmethod
    def minimo(cls, entradas, saidas, inovacoes, rng, sigma=0.05,
               fracao_ligacoes=1.0, neutro=False):
        """Topologia mínima: entradas → saídas. `neutro` põe tudo a zero."""
        g = cls(entradas, saidas)
        for origem in range(entradas):
            for destino in g.ids_saida:
                if neutro or fracao_ligacoes >= 1.0 or rng.random() < fracao_ligacoes:
                    peso = 0.0 if neutro else rng.normal(0.0, sigma)
                    g.conexoes[inovacoes.ligacao(origem, destino)] = Conexao(
                        inovacoes.ligacao(origem, destino), origem, destino, peso)
        if not neutro:
            for n in g.ids_saida:
                g.vies[n] = float(rng.normal(0.0, sigma))
        return g

    # --- informação ------------------------------------------------------
    @property
    def conexoes_ativas(self):
        return sum(1 for c in self.conexoes.values() if c.ativa)

    def resumo(self):
        return (f'{len(self.ids_ocultos)} nós ocultos, '
                f'{self.conexoes_ativas}/{len(self.conexoes)} conexões ativas')

    def existe_caminho(self, origem, alvo):
        """Há caminho origem → alvo (conexões ativas ou não)? Evita ciclos."""
        seguintes = defaultdict(list)
        for c in self.conexoes.values():
            seguintes[c.entrada].append(c.saida)
        pilha, visto = [origem], {origem}
        while pilha:
            n = pilha.pop()
            if n == alvo:
                return True
            for m in seguintes[n]:
                if m not in visto:
                    visto.add(m)
                    pilha.append(m)
        return False

    # --- (de)serialização para .npz ------------------------------------
    def para_dict(self):
        nos = sorted(self.vies)
        cons = sorted(self.conexoes.values(), key=lambda c: c.inov)
        return dict(
            neat_entradas=np.int64(self.entradas),
            neat_saidas=np.int64(self.saidas),
            neat_nos_id=np.asarray(nos, dtype=np.int64),
            neat_nos_vies=np.asarray([self.vies[n] for n in nos], dtype=float),
            neat_con_inov=np.asarray([c.inov for c in cons], dtype=np.int64),
            neat_con_entrada=np.asarray([c.entrada for c in cons], dtype=np.int64),
            neat_con_saida=np.asarray([c.saida for c in cons], dtype=np.int64),
            neat_con_peso=np.asarray([c.peso for c in cons], dtype=float),
            neat_con_ativa=np.asarray([c.ativa for c in cons], dtype=bool),
        )

    @classmethod
    def de_dict(cls, d):
        g = cls(int(d['neat_entradas']), int(d['neat_saidas']))
        g.vies = {int(n): float(v) for n, v in zip(d['neat_nos_id'], d['neat_nos_vies'])}
        for i, e, s, p, a in zip(d['neat_con_inov'], d['neat_con_entrada'],
                                 d['neat_con_saida'], d['neat_con_peso'],
                                 d['neat_con_ativa']):
            g.conexoes[int(i)] = Conexao(i, e, s, p, a)
        return g


# --------------------------------------------------------------------------
# Rede compilada (avaliação rápida)
# --------------------------------------------------------------------------
class RedeCompilada:
    """Converte um genoma numa lista de camadas densas para avaliação em NumPy.

    Só se calculam os nós que influenciam alguma saída. A camada de um nó é o
    caminho mais longo desde as entradas, logo todas as suas fontes já estão
    calculadas quando ele é avaliado.
    """

    def __init__(self, genoma):
        E, S = genoma.entradas, genoma.saidas
        self.entradas, self.saidas = E, S
        recebe = defaultdict(list)
        for c in genoma.conexoes.values():
            if c.ativa:
                recebe[c.saida].append(c)

        # nós que contribuem para as saídas (pesquisa para trás)
        necessarios = set(genoma.ids_saida)
        pilha = list(necessarios)
        while pilha:
            n = pilha.pop()
            for c in recebe[n]:
                if c.entrada not in necessarios:
                    necessarios.add(c.entrada)
                    pilha.append(c.entrada)

        camada = {n: 0 for n in range(E)}

        def profundidade(n):
            if n not in camada:
                camada[n] = 1 + max((profundidade(c.entrada) for c in recebe[n]), default=0)
            return camada[n]

        ocultos = [n for n in genoma.ids_ocultos if n in necessarios]
        for n in list(genoma.ids_saida) + ocultos:
            profundidade(n)

        indice = {n: n for n in range(E)}
        for n in genoma.ids_saida:
            indice[n] = n                       # saídas ficam em [E, E+S)
        for k, n in enumerate(ocultos):
            indice[n] = E + S + k
        total = E + S + len(ocultos)

        W = np.zeros((total, total))
        vies = np.zeros(total)
        for n in list(genoma.ids_saida) + ocultos:
            vies[indice[n]] = genoma.vies.get(n, 0.0)
            for c in recebe[n]:
                W[indice[c.entrada], indice[n]] += c.peso

        por_camada = defaultdict(list)
        for n in list(genoma.ids_saida) + ocultos:
            por_camada[camada[n]].append(indice[n])
        self.camadas = []
        for k in sorted(por_camada):
            cols = np.asarray(sorted(por_camada[k]), dtype=np.intp)
            self.camadas.append((cols, np.ascontiguousarray(W[:, cols]), vies[cols].copy()))
        self.total = total

    def prever(self, entradas):
        a = np.zeros(self.total)
        a[:self.entradas] = entradas
        for cols, pesos, vies in self.camadas:
            a[cols] = np.tanh(a @ pesos + vies)
        return a[self.entradas:self.entradas + self.saidas].copy()


# --------------------------------------------------------------------------
# Inovações históricas
# --------------------------------------------------------------------------
class Inovacoes:
    """Regista números de inovação para toda a execução.

    A mesma ligação (origem→destino) tem sempre o mesmo número; dividir a mesma
    conexão cria sempre o mesmo nó e as mesmas duas conexões novas.
    """

    def __init__(self, entradas, saidas):
        self.proxima_inov = 0
        self.proximo_no = entradas + saidas
        self.por_ligacao = {}
        self.por_divisao = {}

    def ligacao(self, origem, destino):
        chave = (int(origem), int(destino))
        if chave not in self.por_ligacao:
            self.por_ligacao[chave] = self.proxima_inov
            self.proxima_inov += 1
        return self.por_ligacao[chave]

    def divisao(self, inov):
        """Devolve (id_do_no_novo) para dividir a conexão `inov`."""
        if inov not in self.por_divisao:
            self.por_divisao[inov] = self.proximo_no
            self.proximo_no += 1
        return self.por_divisao[inov]


# --------------------------------------------------------------------------
# Parâmetros, operadores genéticos
# --------------------------------------------------------------------------
@dataclass
class ParametrosNEAT:
    # mutação de pesos / viéses
    p_mutar_pesos: float = 0.80       # probabilidade de um genoma sofrer mutação de pesos
    taxa_mutacao: float = 0.10        # fração de genes perturbados nesse caso
    desvio_mutacao: float = 0.10      # desvio da perturbação gaussiana
    p_substituir_peso: float = 0.05   # um gene perturbado é antes substituído por valor novo
    sigma_novo: float = 0.10          # desvio dos valores novos (pesos de novas conexões)
    limite_peso: float = 3.0
    # mutação estrutural
    p_nova_ligacao: float = 0.15
    p_novo_no: float = 0.05
    p_alternar: float = 0.01          # ativar/desativar uma conexão
    # compatibilidade e especiação
    c_excedentes: float = 1.0
    c_disjuntos: float = 1.0
    c_pesos: float = 0.4
    limiar_inicial: float = 3.0
    limiar_passo: float = 0.3
    limiar_minimo: float = 0.3
    especies_alvo: int = 5
    # reprodução
    elite_global: int = 2
    tamanho_campeao: int = 5          # espécies com ≥ N membros preservam o campeão
    sobrevivencia: float = 0.30
    p_so_mutacao: float = 0.25        # filho sem cruzamento
    estagnacao: int = 15              # gerações sem melhoria antes de extinguir espécie
    # inicialização
    sigma_inicial: float = 0.05
    ligacao_inicial: float = 1.0      # fração das ligações entrada→saída presentes


def distancia(a, b, p):
    """δ = c1·E + c2·D + c3·W̄ (contagens sem normalização; o limiar adapta-se)."""
    if not a.conexoes or not b.conexoes:
        return float(len(a.conexoes) + len(b.conexoes))
    comuns = a.conexoes.keys() & b.conexoes.keys()
    limite = min(max(a.conexoes), max(b.conexoes))
    diferentes = a.conexoes.keys() ^ b.conexoes.keys()
    excedentes = sum(1 for i in diferentes if i > limite)
    disjuntos = len(diferentes) - excedentes
    dif_peso = (sum(abs(a.conexoes[i].peso - b.conexoes[i].peso) for i in comuns)
                / len(comuns)) if comuns else 0.0
    return p.c_excedentes * excedentes + p.c_disjuntos * disjuntos + p.c_pesos * dif_peso


def cruzar(mais_apto, outro, rng):
    """Cruzamento alinhado por inovação.

    O filho herda TODOS os genes do progenitor mais apto (logo a topologia dele,
    que é acíclica); nos genes comuns, peso e viés vêm de qualquer um dos pais.
    Gene desativado num dos pais: 75 % de ficar desativado.
    """
    filho = Genoma(mais_apto.entradas, mais_apto.saidas)
    for inov, ca in mais_apto.conexoes.items():
        cb = outro.conexoes.get(inov)
        if cb is None:
            filho.conexoes[inov] = ca.copiar()
            continue
        c = ca.copiar()
        c.peso = ca.peso if rng.random() < 0.5 else cb.peso
        if not (ca.ativa and cb.ativa):
            c.ativa = rng.random() >= 0.75
        filho.conexoes[inov] = c
    for n, v in mais_apto.vies.items():
        filho.vies[n] = v if (n not in outro.vies or rng.random() < 0.5) else outro.vies[n]
    return filho


def _adicionar_conexao(g, inovacoes, rng, p):
    fontes = list(g.ids_entrada) + g.ids_ocultos
    destinos = g.ids_ocultos + list(g.ids_saida)
    for _ in range(20):
        origem = int(rng.choice(fontes))
        destino = int(rng.choice(destinos))
        if origem == destino:
            continue
        inov = inovacoes.ligacao(origem, destino)
        if inov in g.conexoes or g.existe_caminho(destino, origem):
            continue
        g.conexoes[inov] = Conexao(inov, origem, destino, rng.normal(0.0, p.sigma_novo))
        return True
    return False


def _adicionar_no(g, inovacoes, rng, p):
    candidatas = [c for c in g.conexoes.values() if c.ativa]
    if not candidatas:
        return False
    c = candidatas[int(rng.integers(len(candidatas)))]
    novo = inovacoes.divisao(c.inov)
    if novo in g.vies:       # essa divisão já existe neste genoma
        return False
    c.ativa = False
    g.vies[novo] = 0.0
    i1 = inovacoes.ligacao(c.entrada, novo)
    i2 = inovacoes.ligacao(novo, c.saida)
    g.conexoes[i1] = Conexao(i1, c.entrada, novo, 1.0)
    g.conexoes[i2] = Conexao(i2, novo, c.saida, c.peso)
    return True


def mutar(g, inovacoes, rng, p, fator=1.0):
    """Mutações de pesos/viéses (escaladas por `fator`) e estruturais."""
    if rng.random() < p.p_mutar_pesos:
        # Dois níveis: a maioria dos filhos recebe perturbações pequenas, 20 % maiores.
        escala = p.desvio_mutacao * fator * (2.0 if rng.random() < 0.20 else 1.0)
        for c in g.conexoes.values():
            if rng.random() < p.taxa_mutacao:
                if rng.random() < p.p_substituir_peso:
                    c.peso = rng.normal(0.0, p.sigma_novo)
                else:
                    c.peso += rng.normal(0.0, escala)
                c.peso = float(np.clip(c.peso, -p.limite_peso, p.limite_peso))
        for n in g.vies:
            if rng.random() < p.taxa_mutacao:
                g.vies[n] = float(np.clip(g.vies[n] + rng.normal(0.0, escala),
                                          -p.limite_peso, p.limite_peso))
    if rng.random() < p.p_nova_ligacao:
        _adicionar_conexao(g, inovacoes, rng, p)
    if rng.random() < p.p_novo_no:
        _adicionar_no(g, inovacoes, rng, p)
    if g.conexoes and rng.random() < p.p_alternar:
        c = g.conexoes[int(rng.choice(list(g.conexoes)))]
        c.ativa = not c.ativa
    return g


# --------------------------------------------------------------------------
# População com especiação
# --------------------------------------------------------------------------
class Especie:
    def __init__(self, id_, representante):
        self.id = id_
        self.representante = representante
        self.membros = []
        self.melhor = -math.inf
        self.sem_melhoria = 0


class PopulacaoNEAT:
    """Ciclo:  avaliar `genomas` → `registar(fitness)` → `reproduzir()`."""

    def __init__(self, tamanho, entradas, saidas, rng, parametros=None):
        if tamanho < 4:
            raise ValueError('A população NEAT precisa de pelo menos 4 genomas.')
        self.p = parametros or ParametrosNEAT()
        self.rng = rng
        self.entradas, self.saidas = entradas, saidas
        self.inovacoes = Inovacoes(entradas, saidas)
        self.limiar = self.p.limiar_inicial
        self.especies = []
        self._proxima_especie = 1
        # Indivíduo 0 = genoma neutro (reproduz o controlador Combinado).
        self.genomas = [Genoma.minimo(entradas, saidas, self.inovacoes, rng, neutro=True)]
        for _ in range(tamanho - 1):
            self.genomas.append(Genoma.minimo(
                entradas, saidas, self.inovacoes, rng, self.p.sigma_inicial,
                self.p.ligacao_inicial))

    # --- especiação ------------------------------------------------------
    def registar(self, pontuacoes):
        for g, f in zip(self.genomas, pontuacoes):
            g.fitness = float(f)
        for e in self.especies:
            e.membros = []
        for g in self.genomas:
            for e in self.especies:
                if distancia(g, e.representante, self.p) < self.limiar:
                    e.membros.append(g)
                    break
            else:
                e = Especie(self._proxima_especie, g)
                self._proxima_especie += 1
                e.membros.append(g)
                self.especies.append(e)
        self.especies = [e for e in self.especies if e.membros]
        for e in self.especies:
            melhor = max(m.fitness for m in e.membros)
            if melhor > e.melhor + 1e-9:
                e.melhor, e.sem_melhoria = melhor, 0
            else:
                e.sem_melhoria += 1
        # Ajuste do limiar rumo ao número-alvo de espécies (usado na próxima geração).
        if len(self.especies) < self.p.especies_alvo:
            self.limiar = max(self.p.limiar_minimo, self.limiar - self.p.limiar_passo)
        elif len(self.especies) > self.p.especies_alvo:
            self.limiar += self.p.limiar_passo

    # --- reprodução ------------------------------------------------------
    def reproduzir(self, fator_mutacao=1.0):
        p, rng = self.p, self.rng
        tamanho = len(self.genomas)
        ordenados = sorted(self.genomas, key=lambda g: g.fitness, reverse=True)
        novos = [g.copiar() for g in ordenados[:p.elite_global]]

        # Extinção por estagnação, protegendo as 2 melhores espécies.
        ranking = sorted(self.especies, key=lambda e: e.melhor, reverse=True)
        protegidas = {id(e) for e in ranking[:2]}
        vivas = [e for e in self.especies
                 if e.sem_melhoria < p.estagnacao or id(e) in protegidas]

        # Fitness sharing: aptidão normalizada média de cada espécie.
        fmin = min(g.fitness for g in self.genomas)
        fmax = max(g.fitness for g in self.genomas)
        amplitude = max(fmax - fmin, 1e-9)
        ajustada = np.array([np.mean([(m.fitness - fmin) / amplitude + 0.05
                                      for m in e.membros]) for e in vivas])
        restante = tamanho - len(novos)
        quota = ajustada / ajustada.sum() * restante
        aloc = np.floor(quota).astype(int)
        resto = restante - aloc.sum()
        for i in np.argsort(quota - aloc)[::-1][:resto]:
            aloc[i] += 1

        for e, n in zip(vivas, aloc):
            n = int(n)
            if n <= 0:
                continue
            membros = sorted(e.membros, key=lambda g: g.fitness, reverse=True)
            if len(membros) >= p.tamanho_campeao:
                novos.append(membros[0].copiar())
                n -= 1
            pool = membros[:max(2, math.ceil(p.sobrevivencia * len(membros)))]
            for _ in range(n):
                if len(pool) == 1 or rng.random() < p.p_so_mutacao:
                    filho = pool[int(rng.integers(len(pool)))].copiar()
                else:
                    i, j = rng.choice(len(pool), size=2, replace=False)
                    a, b = pool[i], pool[j]
                    if b.fitness > a.fitness:
                        a, b = b, a
                    filho = cruzar(a, b, rng)
                novos.append(mutar(filho, self.inovacoes, rng, p, fator_mutacao))

        while len(novos) < tamanho:       # salvaguarda contra arredondamentos
            novos.append(mutar(ordenados[0].copiar(), self.inovacoes, rng, p, fator_mutacao))

        for e in vivas:
            e.representante = e.membros[int(rng.integers(len(e.membros)))]
        self.especies = vivas
        self.genomas = novos[:tamanho]
        for g in self.genomas:
            g.fitness = 0.0
