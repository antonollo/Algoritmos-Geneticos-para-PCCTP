from random import random
import matplotlib.pyplot as plt
import numpy as np

"""
Estrutura do Cromossomo:
[[Posições Iniciais][Posições Finais]]
"""

class Individuo:
    def __init__(self, qntConteineres, matrizTerminal, matrizBalsa, maxPerDeck, geracao=0):
        self.qntConteineres = qntConteineres
        self.matrizTerminal = matrizTerminal
        self.matrizBalsa = matrizBalsa
        self.posicoesFinaisPossiveis = []
        self.qntPerDeck = [0, 0, 0]
        self.maxPerDeck = maxPerDeck
        self.cromossomo = self.construirCromossomo()
        self.quant_movimentos = self.calcularMovimentos()
        self.notaFitness = self.fitness()
        self.geracao = geracao

    def construirCromossomo(self):
        cromossomo = []
        posicoesIniciaisPossiveis = []
        for lines in range(self.matrizTerminal[0]):
            for columns in range(self.matrizTerminal[1]):
                for decks in range(self.matrizTerminal[2]):
                    posicoesIniciaisPossiveis.append([lines + 1, columns + 1, decks + 1])
        for lines in range(self.matrizBalsa[0]):
            for columns in range(self.matrizBalsa[1]):
                for decks in range(self.matrizBalsa[2]):
                    self.posicoesFinaisPossiveis.append([lines + 1, columns + 1, decks + 1])
        for i in range(len(posicoesIniciaisPossiveis)):
            selected = self.selecionarPosicao()
            cromossomo.append([posicoesIniciaisPossiveis[i], selected])
        return cromossomo

    def fitness(self, alfa=0.7, beta=0.3):
        """
        Função de fitness com forte ênfase em formação em cruz.
        - alfa: peso para o equilíbrio (estabilidade)
        - beta: peso para a quantidade de movimentos
        """
        movimentos = self.quant_movimentos
        fitness = 0
        penalidade = 0
        centro_geo = [self.matrizBalsa[0] / 2 + 0.5, self.matrizBalsa[1] / 2 + 0.5, self.matrizBalsa[2] / 2 + 0.5]
        centro_massa = np.mean([[gene[1][0], gene[1][1], gene[1][2]] for gene in self.cromossomo], axis=0)
        balanco_x = (centro_massa[0] - centro_geo[0]) ** 2
        balanco_y = (centro_massa[1] - centro_geo[1]) ** 2
        balanco = np.sqrt(balanco_x + balanco_y)
        penalidade_cruz = self.calcular_penalidade_cruz(centro_geo) * 3.0
        penalidade_deck = self.calcular_penalidade_deck(8, 10)
        if movimentos > 190:
            penalidade += 10
        fitness = 1 / (1 + (alfa * (balanco + penalidade_cruz)) + (beta * movimentos) + penalidade + penalidade_deck)
        return fitness

    def calcular_penalidade_deck(self, min_deck, max_deck):
        """
        Penaliza decks que têm menos que min_deck ou mais que max_deck contêineres. Restrições do problema.
        """
        penalidade = 0
        contagem_deck = [0] * self.matrizBalsa[2]
        for gene in self.cromossomo:
            deck = gene[1][2] - 1
            contagem_deck[deck] += 1
        for count in contagem_deck:
            if count < min_deck:
                penalidade += (min_deck - count) ** 2
            elif count > max_deck:
                penalidade += (count - max_deck) ** 2
        return penalidade * 0.5

    def calcular_penalidade_cruz(self, centro_geo):
        """
        Como se busca favorecer contêineres nas posições mais ao centro do barco para buscar mais estabilidade, o formato de cruz acaba 
        sendo a disposição de carregamento mais óbvia para ser seguida, então porque não penalizar o algoritmo por desrespeitar uma estrutura
        que foca em deixar os contêineres bem centralizados e organizados?
        """
        penalidade = 0
        linhas_centrais = [int(centro_geo[0]), int(centro_geo[0]) - 1]
        colunas_centrais = [int(centro_geo[1]), int(centro_geo[1]) - 1]
        for gene in self.cromossomo:
            pos = gene[1]
            linha, coluna = pos[0], pos[1]
            if linha not in linhas_centrais and coluna not in colunas_centrais:
                dist_linha = min([abs(linha - l) for l in linhas_centrais])
                dist_coluna = min([abs(coluna - c) for c in colunas_centrais])
                dist_min = min(dist_linha, dist_coluna)
                penalidade += dist_min ** 2.5
        return penalidade / len(self.cromossomo)

    def selecionarPosicao(self):
        centro_geo = [self.matrizBalsa[0] / 2 + 0.5, self.matrizBalsa[1] / 2 + 0.5, self.matrizBalsa[2] / 2 + 0.5]
        linhas_centrais = [int(centro_geo[0]), int(centro_geo[0]) - 1]
        colunas_centrais = [int(centro_geo[1]), int(centro_geo[1]) - 1]
        if not self.posicoesFinaisPossiveis:
            raise Exception("Sem posições disponíveis restantes")
        posicoes_centro = []
        posicoes_cruz = []
        posicoes_outras = []
        for pos in self.posicoesFinaisPossiveis:
            linha, coluna = pos[0], pos[1]
            if linha in linhas_centrais and coluna in colunas_centrais:
                posicoes_centro.append(pos)
            elif linha in linhas_centrais or coluna in colunas_centrais:
                posicoes_cruz.append(pos)
            else:
                posicoes_outras.append(pos)
        for lista in [posicoes_centro, posicoes_cruz, posicoes_outras]:
            lista.sort(key=lambda pos: np.sqrt((pos[0] - centro_geo[0]) ** 2 + (pos[1] - centro_geo[1]) ** 2))
        posicoes_ordenadas = posicoes_centro + posicoes_cruz + posicoes_outras
        for i, posicaoSelecionada in enumerate(posicoes_ordenadas):
            z = posicaoSelecionada[2]
            abaixo = [posicaoSelecionada[0], posicaoSelecionada[1], z - 1]
            if abaixo in self.posicoesFinaisPossiveis:
                continue
            deck = z
            if self.qntPerDeck[deck - 1] < self.maxPerDeck:
                self.qntPerDeck[deck - 1] += 1
                self.posicoesFinaisPossiveis.remove(posicaoSelecionada)
                if self.qntPerDeck[deck - 1] == self.maxPerDeck:
                    self.posicoesFinaisPossiveis = [
                        pos for pos in self.posicoesFinaisPossiveis if pos[2] != deck
                    ]
                return posicaoSelecionada
        while True:
            if not self.posicoesFinaisPossiveis:
                raise Exception("Sem posições disponíveis restantes")
            posicaoSelecionada = self.posicoesFinaisPossiveis[round(random() * (len(self.posicoesFinaisPossiveis) - 1))]
            z = posicaoSelecionada[2]
            abaixo = [posicaoSelecionada[0], posicaoSelecionada[1], z - 1]
            if abaixo in self.posicoesFinaisPossiveis:
                posicaoSelecionada = abaixo
                continue
            else:
                deck = z
                if self.qntPerDeck[deck - 1] < self.maxPerDeck:
                    self.qntPerDeck[deck - 1] += 1
                    self.posicoesFinaisPossiveis.remove(posicaoSelecionada)
                    if self.qntPerDeck[deck - 1] == self.maxPerDeck:
                        self.posicoesFinaisPossiveis = [
                            pos for pos in self.posicoesFinaisPossiveis if pos[2] != deck
                        ]
                    break
                else:
                    self.posicoesFinaisPossiveis.remove(posicaoSelecionada)
                    continue
        return posicaoSelecionada

    def calcularMovimentos(self):
        movimentos = 0
        for conteineres in self.cromossomo:
            if conteineres[0][0] == conteineres[1][0]:
                movimentos += 6
                continue
            else:
                movimentos += 8
        return movimentos


class AlgoritmoGeneticoAdpt:
    def __init__(self, NUM_POP, taxaMutacao, taxaCrossover):
        self.NUM_POP = NUM_POP
        self.taxaMutacao = taxaMutacao
        self.taxaMutacaoMin = 0.01
        self.taxaMutacaoMax = 0.1
        self.taxaCrossover = taxaCrossover
        self.taxaCrossoverMin = 0.7
        self.taxaCrossoverMax = 0.95
        self.populacao = []
        self.geracao = 0
        self.soma_avaliacao = 0
        self.melhorIndividuoEncontrado = 0
        self.listaMelhorIndividuo = []
        self.historicoFitness = []
        self.historicoTaxas = []
        self.geracoesSemMelhoria = 0
        self.melhorFitnessAnterior = 0

    def inicializarPopulacao(self, qntConteineres, matrizTerminal, matrizBalsa, maxPerDeck):
        for i in range(self.NUM_POP):
            self.populacao.append(Individuo(qntConteineres, matrizTerminal, matrizBalsa, maxPerDeck))
        self.melhorIndividuoEncontrado = self.populacao[0]

    def totalAvaliacao(self):
        totalAvaliacao = 0
        for individuo in self.populacao:
            totalAvaliacao += individuo.notaFitness
        self.soma_avaliacao = totalAvaliacao
        return totalAvaliacao

    def melhorIndividuo(self, individuo):
        if self.melhorIndividuoEncontrado.notaFitness < individuo.notaFitness:
            self.melhorIndividuoEncontrado = individuo
            self.geracoesSemMelhoria = 0
        else:
            self.geracoesSemMelhoria += 1

    def roulette_selection(self, soma_avaliacao):
        pai = -1
        valor_sorteado = random() * soma_avaliacao
        soma = 0
        i = 0
        while i < len(self.populacao) and soma < valor_sorteado:
            soma += self.populacao[i].notaFitness
            pai += 1
            i += 1
        return self.populacao[pai]

    def ordenar_populacao(self):
        for individuo in self.populacao:
            individuo.fitness()
        self.populacao = sorted(self.populacao, key=lambda populacao: populacao.notaFitness,
                                reverse=True)

    def onePointCrossover(self, pai1, pai2, gen):
        if random() > self.taxaCrossover:
            return [pai1, pai2]
        cutPoint = round(random() * (len(pai1.cromossomo) - 1))
        filho1 = Individuo(pai1.qntConteineres, pai1.matrizTerminal, pai1.matrizBalsa, pai1.maxPerDeck, gen + 1)
        filho2 = Individuo(pai2.qntConteineres, pai2.matrizTerminal, pai2.matrizBalsa, pai2.maxPerDeck, gen + 1)
        filho1.cromossomo = [gene[:] for gene in pai1.cromossomo]
        filho2.cromossomo = [gene[:] for gene in pai2.cromossomo]
        posicoes_finais_filho1 = [gene[1] for gene in filho1.cromossomo]
        posicoes_finais_filho2 = [gene[1] for gene in filho2.cromossomo]
        for i in range(cutPoint, len(filho1.cromossomo)):
            posicao_inicial_1 = filho1.cromossomo[i][0]
            posicao_inicial_2 = filho2.cromossomo[i][0]
            filho1.cromossomo[i] = [posicao_inicial_1, pai2.cromossomo[i][1]]
            filho2.cromossomo[i] = [posicao_inicial_2, pai1.cromossomo[i][1]]
        filho1 = self.corrigirRestricoes(filho1)
        filho2 = self.corrigirRestricoes(filho2)
        filho1.quant_movimentos = filho1.calcularMovimentos()
        filho2.quant_movimentos = filho2.calcularMovimentos()
        filho1.notaFitness = filho1.fitness()
        filho2.notaFitness = filho2.fitness()
        return [filho1, filho2]

    def corrigirRestricoes(self, individuo):
        individuo.qntPerDeck = [0, 0, 0]
        posicoes_finais_usadas = set()
        for i, gene in enumerate(individuo.cromossomo):
            posicao_final = tuple(gene[1])
            if posicao_final in posicoes_finais_usadas:
                individuo.cromossomo[i][1] = None
            else:
                deck = gene[1][2]
                if individuo.qntPerDeck[deck - 1] < individuo.maxPerDeck:
                    if deck > 1:
                        abaixo = tuple([gene[1][0], gene[1][1], deck - 1])
                        if abaixo not in posicoes_finais_usadas:
                            individuo.cromossomo[i][1] = None
                            continue
                    posicoes_finais_usadas.add(posicao_final)
                    individuo.qntPerDeck[deck - 1] += 1
                else:
                    individuo.cromossomo[i][1] = None
        posicoes_finais_disponiveis = []
        for lines in range(individuo.matrizBalsa[0]):
            for columns in range(individuo.matrizBalsa[1]):
                for decks in range(individuo.matrizBalsa[2]):
                    pos = [lines + 1, columns + 1, decks + 1]
                    if tuple(pos) not in posicoes_finais_usadas:
                        posicoes_finais_disponiveis.append(pos)
        tentativas_maximas = 100
        for i, gene in enumerate(individuo.cromossomo):
            if gene[1] is None:
                encontrou_posicao = False
                tentativas = 0
                while not encontrou_posicao and tentativas < tentativas_maximas:
                    tentativas += 1
                    if not posicoes_finais_disponiveis:
                        for lines in range(individuo.matrizBalsa[0]):
                            for columns in range(individuo.matrizBalsa[1]):
                                for decks in range(individuo.matrizBalsa[2]):
                                    pos = [lines + 1, columns + 1, decks + 1]
                                    if tuple(pos) not in posicoes_finais_usadas:
                                        posicoes_finais_disponiveis.append(pos)
                        if not posicoes_finais_disponiveis:
                            for deck_idx in range(len(individuo.qntPerDeck)):
                                if individuo.qntPerDeck[deck_idx] < individuo.maxPerDeck:
                                    for lines in range(individuo.matrizBalsa[0]):
                                        for columns in range(individuo.matrizBalsa[1]):
                                            pos = [lines + 1, columns + 1, deck_idx + 1]
                                            if tuple(pos) not in posicoes_finais_usadas:
                                                posicoes_finais_disponiveis.append(pos)
                                                break
                                        if posicoes_finais_disponiveis:
                                            break
                                if posicoes_finais_disponiveis:
                                    break
                    if not posicoes_finais_disponiveis:
                        deck_com_espaco = -1
                        for deck_idx in range(len(individuo.qntPerDeck)):
                            if individuo.qntPerDeck[deck_idx] < individuo.maxPerDeck + 1:
                                deck_com_espaco = deck_idx
                                break
                        if deck_com_espaco != -1:
                            for lines in range(individuo.matrizBalsa[0]):
                                for columns in range(individuo.matrizBalsa[1]):
                                    pos = [lines + 1, columns + 1, deck_com_espaco + 1]
                                    if tuple(pos) not in posicoes_finais_usadas:
                                        posicoes_finais_disponiveis.append(pos)
                                        break
                                if posicoes_finais_disponiveis:
                                    break
                    if not posicoes_finais_disponiveis:
                        posicao = [1, 1, 1]
                        posicoes_finais_disponiveis.append(posicao)
                    if posicoes_finais_disponiveis:
                        idx = round(random() * (len(posicoes_finais_disponiveis) - 1))
                        posicao = posicoes_finais_disponiveis[idx]
                        deck = posicao[2]
                        valida = True
                        if deck > 1:
                            abaixo = tuple([posicao[0], posicao[1], deck - 1])
                            if abaixo not in posicoes_finais_usadas:
                                posicoes_finais_disponiveis.pop(idx)
                                valida = False
                                continue
                        if individuo.qntPerDeck[deck - 1] < individuo.maxPerDeck and valida:
                            individuo.cromossomo[i][1] = posicao
                            posicoes_finais_usadas.add(tuple(posicao))
                            individuo.qntPerDeck[deck - 1] += 1
                            posicoes_finais_disponiveis.pop(idx)
                            encontrou_posicao = True
                            break
                        else:
                            posicoes_finais_disponiveis.pop(idx)
                if not encontrou_posicao:
                    for deck in range(individuo.matrizBalsa[2]):
                        for linha in range(individuo.matrizBalsa[0]):
                            for coluna in range(individuo.matrizBalsa[1]):
                                posicao = [linha + 1, coluna + 1, deck + 1]
                                if tuple(posicao) not in posicoes_finais_usadas:
                                    individuo.cromossomo[i][1] = posicao
                                    posicoes_finais_usadas.add(tuple(posicao))
                                    individuo.qntPerDeck[deck] += 1
                                    encontrou_posicao = True
                                    break
                            if encontrou_posicao:
                                break
                        if encontrou_posicao:
                            break
                    if not encontrou_posicao:
                        posicao = [1, 1, 1]
                        individuo.cromossomo[i][1] = posicao
        return individuo

    def swapValido(self, individuo, idx1, idx2):
        """Verifica se o Swap ocorrido por causa da mutação não vai ocasionar numa solução inválida. Se estiver invalidado, a correção é devidamente feita."""
        pos1 = individuo.cromossomo[idx1][1]
        pos2 = individuo.cromossomo[idx2][1]
        individuo.cromossomo[idx1][1], individuo.cromossomo[idx2][1] = pos2, pos1
        valido = True
        posicoes_ocupadas = set(tuple(gene[1]) for gene in individuo.cromossomo)
        for gene in individuo.cromossomo:
            deck = gene[1][2]
            if deck > 1:
                abaixo = (gene[1][0], gene[1][1], deck - 1)
                if abaixo not in posicoes_ocupadas:
                    valido = False
                    break
        individuo.cromossomo[idx1][1], individuo.cromossomo[idx2][1] = pos1, pos2
        return valido

    def mutateSwap(self, individuo):
        """
        Fenômeno da mutação de genes de um indivíduo. É um formato simples, mas que não modifica bruscamente a estrutura do problema, que é basicamente 
        trocar a posição final de um contêiner com o de outro contêiner.
        """
        if random() < self.taxaMutacao:
            idx1 = round(random() * (len(individuo.cromossomo) - 1))
            idx2 = round(random() * (len(individuo.cromossomo) - 1))
            while idx1 == idx2:
                idx2 = round(random() * (len(individuo.cromossomo) - 1))
            pos1 = individuo.cromossomo[idx1][1]
            pos2 = individuo.cromossomo[idx2][1]
            if self.swapValido(individuo, idx1, idx2):
                individuo.cromossomo[idx1][1], individuo.cromossomo[idx2][1] = individuo.cromossomo[idx2][1], \
                    individuo.cromossomo[idx1][1]
                individuo.quant_movimentos = individuo.calcularMovimentos()
                individuo.notaFitness = individuo.fitness()
        return individuo

    def calcularDiversidade(self):
        """Calcula a diversidade da população com base nos cromossomos. A partir da diversidade, se pode determinar se as taxas de mutação e crossing-over precisam ser modificadas."""
        if len(self.populacao) <= 1:
            return 0
        fitness_values = [ind.notaFitness for ind in self.populacao]
        mean_fitness = sum(fitness_values) / len(fitness_values)
        variance = sum((x - mean_fitness) ** 2 for x in fitness_values) / len(fitness_values)
        max_fitness = max(fitness_values)
        min_fitness = min(fitness_values)
        range_fitness = max_fitness - min_fitness if max_fitness != min_fitness else 1
        normalized_variance = variance / (range_fitness ** 2)
        return normalized_variance

    def ajustarTaxas(self, fitness_medio, fitness_max, diversidade):
        """Ajusta as taxas de crossover e mutação com base no desempenho (diversidade) atual."""
        estagnacao = (fitness_max - fitness_medio) / fitness_max if fitness_max > 0 else 1
        melhoria = 0
        if self.melhorFitnessAnterior > 0:
            melhoria = (fitness_max - self.melhorFitnessAnterior) / self.melhorFitnessAnterior
        self.melhorFitnessAnterior = fitness_max
        if estagnacao < 0.05 or diversidade < 0.1:
            self.taxaCrossover = min(self.taxaCrossover * 1.05, self.taxaCrossoverMax)
        elif diversidade > 0.3 and melhoria > 0.01:
            self.taxaCrossover = max(self.taxaCrossover * 0.98, self.taxaCrossoverMin)
        if self.geracoesSemMelhoria > 20:
            self.taxaMutacao = min(self.taxaMutacao * 1.1, self.taxaMutacaoMax)
        elif melhoria > 0.01:
            self.taxaMutacao = max(self.taxaMutacao * 0.95, self.taxaMutacaoMin)
        self.historicoTaxas.append((self.taxaCrossover, self.taxaMutacao))

    def run_aga(self, NUM_GER, qntConteineres, matrizTerminal, matrizBalsa, maxPerDeck):
        try:
            self.inicializarPopulacao(qntConteineres, matrizTerminal, matrizBalsa, maxPerDeck)
            self.ordenar_populacao()
            self.melhorFitnessAnterior = self.populacao[0].notaFitness
            self.listaMelhorIndividuo.append(self.populacao[0].notaFitness)
            for geracao in range(NUM_GER):
                soma_avaliacao = self.totalAvaliacao()
                fitness_values = [ind.notaFitness for ind in self.populacao]
                fitness_max = max(fitness_values)
                fitness_medio = sum(fitness_values) / len(fitness_values)
                diversidade = self.calcularDiversidade()
                self.historicoFitness.append((fitness_medio, fitness_max))
                self.ajustarTaxas(fitness_medio, fitness_max, diversidade)
                nova_Pop = []
                melhor = self.populacao[0]
                nova_Pop.append(melhor)
                for gerados in range(0, self.NUM_POP - 1, 2):
                    try:
                        pai1 = self.roulette_selection(soma_avaliacao)
                        pai2 = self.roulette_selection(soma_avaliacao)
                        filhos = self.onePointCrossover(pai1, pai2, geracao)
                        for filho in filhos:
                            if len(nova_Pop) < self.NUM_POP:
                                nova_Pop.append(self.mutateSwap(filho))
                    except Exception as e:
                        print(f"Erro ao gerar filhos na geração {geracao}: {e}")
                        if len(nova_Pop) < self.NUM_POP:
                            nova_Pop.append(pai1)
                        if len(nova_Pop) < self.NUM_POP:
                            nova_Pop.append(pai2)
                while len(nova_Pop) < self.NUM_POP:
                    idx = round(random() * (len(self.populacao) - 1))
                    nova_Pop.append(self.populacao[idx])
                self.populacao = list(nova_Pop)
                self.ordenar_populacao()
                melhorIndividuoGeracao = self.populacao[0]
                self.melhorIndividuo(melhorIndividuoGeracao)
                self.listaMelhorIndividuo.append(self.melhorIndividuoEncontrado.notaFitness)
                if not hasattr(self, 'historicoMovimentos'):
                    self.historicoMovimentos = []
                self.historicoMovimentos.append(melhorIndividuoGeracao.quant_movimentos)
                if geracao % 100 == 0 or geracao == NUM_GER - 1:
                    print(f"Geração {geracao}: Melhor fitness = {melhorIndividuoGeracao.notaFitness}")
                    print(f"Taxa de Crossover: {self.taxaCrossover:.4f}, Taxa de Mutação: {self.taxaMutacao:.4f}")
                    print(f"Diversidade: {diversidade:.4f}, Gerações sem melhoria: {self.geracoesSemMelhoria}")
            print(f"A melhor solução encontrada foi no indivíduo da geração {self.melhorIndividuoEncontrado.geracao}")
            print(f"O fitness desse indivíduo foi: {self.melhorIndividuoEncontrado.notaFitness}")
            for i in self.melhorIndividuoEncontrado.cromossomo:
                print(f"Entrada: {i[0]}\nSaída: {i[1]}")
            print(
                f"--------------------------------------- Quantidade de Movimentos: {self.melhorIndividuoEncontrado.quant_movimentos}")
            return self.listaMelhorIndividuo
        except Exception as e:
            print(f"Erro geral no algoritmo: {e}")
            import traceback
            traceback.print_exc()
            return self.listaMelhorIndividuo


def plot_fitness_evolution(ag):
    """
    Plota a evolução do fitness, quantidade de movimentos e características do melhor indivíduo ao longo das gerações usando três eixos Y,
    justamente para compreender melhor o comportamento do Algoritmo Genético Adaptativo.
    """
    fig, ax1 = plt.subplots(figsize=(14, 8))
    geracoes = range(len(ag.historicoFitness))
    fitness_medio = [f[0] for f in ag.historicoFitness]
    fitness_max = [f[1] for f in ag.historicoFitness]
    movimentos_por_geracao = []
    for i in range(len(ag.historicoFitness)):
        if i < len(ag.populacao):
            movimentos_por_geracao.append(ag.populacao[0].quant_movimentos)
        else:
            movimentos_por_geracao.append(movimentos_por_geracao[-1] if movimentos_por_geracao else 0)
    fitness_global = np.array(ag.listaMelhorIndividuo)
    fitness_max = np.array(fitness_max)
    fitness_medio = np.array(fitness_medio)
    cor1 = 'darkgreen'
    ax1.set_xlabel('Geração', fontsize=12)
    ax1.set_ylabel('Fitness', color=cor1, fontsize=12)
    ax1.plot(fitness_global, '-', color=cor1, label='Melhor Fitness Global', linewidth=2)
    ax1.plot(fitness_max, '--', color='forestgreen', label='Melhor Fitness da Geração', linewidth=1.5)
    ax1.plot(fitness_medio, ':', color='mediumseagreen', label='Fitness Médio', linewidth=1)
    ax1.tick_params(axis='y', labelcolor=cor1)
    ax1.grid(True, alpha=0.3)
    cor2 = 'darkblue'
    ax2 = ax1.twinx()
    ax2.set_ylabel('Quantidade de Movimentos', color=cor2, fontsize=12)
    ax2.plot(movimentos_por_geracao, '-', color=cor2, label='Movimentos do Melhor Indivíduo', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=cor2)
    cor3 = 'darkred'
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel('Taxas de Evolução', color=cor3, fontsize=12)
    if ag.historicoTaxas:
        crossover_rates = [t[0] for t in ag.historicoTaxas]
        mutation_rates = [t[1] for t in ag.historicoTaxas]
        ax3.plot(crossover_rates, '-.', color='firebrick', label='Taxa de Crossover', linewidth=1)
        ax3.plot(mutation_rates, '-.', color='indianred', label='Taxa de Mutação', linewidth=1)
    ax3.tick_params(axis='y', labelcolor=cor3)
    plt.title('Evolução do AG: Fitness, Movimentos e Parâmetros por Geração', fontsize=14)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3,
               loc='upper center', bbox_to_anchor=(0.5, -0.12),
               ncol=3, frameon=True, fontsize=10)
    melhor_individuo = ag.melhorIndividuoEncontrado
    plt.figtext(0.02, 0.02,
                f"Melhor solução: Geração {melhor_individuo.geracao} | "
                f"Fitness: {melhor_individuo.notaFitness:.4f} | "
                f"Movimentos: {melhor_individuo.quant_movimentos}",
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.show()


def plot_solution_2d(individuo, matrizBalsa):
    """
    Plota a disposição dos contêineres na balsa em 2D (por deck)
    """
    altura_balsa = matrizBalsa[2]
    linhas_balsa = matrizBalsa[0]
    colunas_balsa = matrizBalsa[1]
    conteiner_por_deck = [0] * altura_balsa
    for gene in individuo.cromossomo:
        deck = gene[1][2] - 1
        conteiner_por_deck[deck] += 1
    print(f"\nTotal de contêineres: {len(individuo.cromossomo)}")
    print(f"Distribuição por deck: {conteiner_por_deck}")
    print(f"Total de movimentos: {individuo.quant_movimentos}")
    print(f"Fitness: {individuo.notaFitness}")
    fig, axs = plt.subplots(1, altura_balsa, figsize=(15, 6))
    if altura_balsa == 1:
        axs = [axs]
    for z in range(altura_balsa):
        deck = np.zeros((linhas_balsa, colunas_balsa))
        for idx, gene in enumerate(individuo.cromossomo):
            posicao_final = gene[1]
            if posicao_final[2] - 1 == z:
                linha = posicao_final[0] - 1
                coluna = posicao_final[1] - 1
                deck[linha, coluna] = idx + 1
        im = axs[z].imshow(deck, cmap='Blues', vmin=0, vmax=len(individuo.cromossomo))
        axs[z].set_title(f"Deck {z + 1} ({conteiner_por_deck[z]} contêineres)")
        for i in range(linhas_balsa):
            for j in range(colunas_balsa):
                if deck[i, j] > 0:
                    axs[z].text(j, i, f"{int(deck[i, j])}", ha="center", va="center",
                                color="white", fontweight='bold')
    plt.suptitle(f"Distribuição dos Contêineres por Deck (Total: {len(individuo.cromossomo)})")
    plt.tight_layout()
    plt.show()


def plot_solution_3d(individuo, matrizBalsa):
    """
    Plota a disposição dos contêineres na balsa em 3D
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    linhas_balsa = matrizBalsa[0]
    colunas_balsa = matrizBalsa[1]
    altura_balsa = matrizBalsa[2]
    cores = plt.cm.jet(np.linspace(0, 1, len(individuo.cromossomo)))
    centro_massa = np.mean([[gene[1][0], gene[1][1], gene[1][2]] for gene in individuo.cromossomo], axis=0)
    centro_geo = [matrizBalsa[0] / 2 - 0.5, matrizBalsa[1] / 2 - 0.5, matrizBalsa[2] / 2 - 0.5]
    for idx, gene in enumerate(individuo.cromossomo):
        posicao_final = gene[1]
        x = posicao_final[0] - 1
        y = posicao_final[1] - 1
        z = posicao_final[2] - 1
        ax.bar3d(y, x, z, 0.9, 0.9, 0.9, color=cores[idx], alpha=0.7)
        ax.text(y + 0.45, x + 0.45, z + 0.45, str(idx + 1),
                horizontalalignment='center', verticalalignment='center')
    ax.scatter(centro_massa[1] - 1, centro_massa[0] - 1, centro_massa[2] - 1,
               color='red', s=100, marker='*', label='Centro de Massa')
    ax.scatter(centro_geo[1], centro_geo[0], centro_geo[2],
               color='green', s=100, marker='o', label='Centro Geométrico')
    for z in range(altura_balsa):
        for x in range(linhas_balsa + 1):
            ax.plot([0, colunas_balsa], [x, x], [z, z], 'k-', alpha=0.2)
        for y in range(colunas_balsa + 1):
            ax.plot([y, y], [0, linhas_balsa], [z, z], 'k-', alpha=0.2)
    ax.set_xlabel('Coluna')
    ax.set_ylabel('Linha')
    ax.set_zlabel('Deck')
    ax.set_title(
        f"Visualização 3D da Balsa (Contêineres: {len(individuo.cromossomo)}, Movimentos: {individuo.quant_movimentos})")
    ax.set_xlim(0, colunas_balsa)
    ax.set_ylim(0, linhas_balsa)
    ax.set_zlim(0, altura_balsa)
    ax.view_init(elev=30, azim=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualizar_resultados(ag, matrizBalsa):
    """
    Visualiza todos os resultados do algoritmo genético
    """
    plot_fitness_evolution(ag)
    melhor_individuo = ag.melhorIndividuoEncontrado
    plot_solution_2d(melhor_individuo, matrizBalsa)
    plot_solution_3d(melhor_individuo, matrizBalsa)
    centro_massa = np.mean([[gene[1][0], gene[1][1], gene[1][2]] for gene in melhor_individuo.cromossomo], axis=0)
    centro_geo = [matrizBalsa[0] / 2, matrizBalsa[1] / 2, matrizBalsa[2] / 2]
    balanco = abs(centro_massa[0] - centro_geo[0]) + abs(centro_massa[1] - centro_geo[1])


if __name__ == "__main__":
    matrizTerminal = [6, 2, 2]
    matrizBalsa = [4, 4, 3]
    NUM_GEN = 1000
    NUM_POP = 100
    MUTATION_TAXE = 0.05
    CROSSOVER_TAXE = 0.9
    QNT_CONTEINERES = 24
    MAX_PER_DECK = 10
    aga = AlgoritmoGeneticoAdpt(NUM_POP, MUTATION_TAXE, CROSSOVER_TAXE)
    aga.run_aga(NUM_GEN, QNT_CONTEINERES, matrizTerminal, matrizBalsa, MAX_PER_DECK)
    visualizar_resultados(aga, matrizBalsa)