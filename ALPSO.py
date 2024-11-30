import numpy as np
import random
import copy
import math

from scipy.stats import norm
''' Population initialization function '''
def initial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = round((random.random() * (ub[j] - lb[j]) + lb[j][0])[0])

    return X, lb, ub


'''Boundary checking function'''
def BorderCheck(X, ub, lb, pop, dim):
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            elif X[i, j] < lb[j]:
                X[i, j] = lb[j]
    return X


'''Calculate the fitness function'''


def CaculateFitness(X, fun):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness


'''Ordering of fitness'''


def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index


'''Sorting positions according to fitness'''


def SortPosition(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew


def update_fun(pop,X,V,w,c1,c2,Pbest,GbestPositon,dim,Vmin,Vmax,lb,ub,fitness,fun,fitnessPbest):
    # Speed Update
    for j in range(pop):
        V[j, :] = w * V[j, :] + c1 * np.random.random() * (Pbest[j, :] - X[j, :]) + c2 * np.random.random() * (
                GbestPositon - X[j, :])
        # Velocity boundary check
        for ii in range(dim):
            if V[j, ii] < Vmin[ii]:
                V[j, ii] = Vmin[ii]
            if V[j, ii] > Vmax[ii]:
                V[j, ii] = Vmax[ii]
        # Location Updates
        X[j, :] = X[j, :] + V[j, :]
        # Position boundary check
        for ii in range(dim):
            if X[j, ii] < 0:
                V[j, ii] = lb[ii]
            if X[j, ii] > 0:
                V[j, ii] = ub[ii]
        fitness[j] = fun(X[j, :])
    return fitness.copy()

'''ALPSO'''


def PSO(pop, dim, lb, ub, MaxIter, fun, Vmin, Vmax):
    w = 0.9  # inertial factor
    max_w = 0.9
    min_w = 0.4
    c1_1,c1_2 = 4,0
    c2_1,c2_2 = 0,4
    X, lb, ub = initial(pop, dim, ub, lb)
    V, Vmin, Vmax = initial(pop, dim, Vmax, Vmin)
    fitness_list = []
    fitness = CaculateFitness(X, fun)
    fitness_list.append(fitness.copy())
    fitness, sortIndex = SortFitness(fitness)
    X = SortPosition(X, sortIndex)
    GbestScore = copy.copy(fitness[0])
    GbestPositon = copy.copy(X[0, :])
    Curve = np.zeros([MaxIter, 1])
    Pbest = copy.copy(X)
    fitnessPbest = copy.copy(fitness)
    T = 0
    for i in range(MaxIter):
        w = max_w - (max_w - min_w) * (i / MaxIter)
        c1 = c1_1 - (c1_1-c1_2)*i/MaxIter
        c2 = c2_1 + (c2_2-c2_1)*i/MaxIter

        for j in range(pop):
            V[j, :] = w * V[j, :] + c1 * np.random.random() * (Pbest[j, :] - X[j, :]) + c2 * np.random.random() * (
                    GbestPositon - X[j, :])

            for ii in range(dim):
                if V[j, ii] < Vmin[ii]:
                    V[j, ii] = Vmin[ii]
                if V[j, ii] > Vmax[ii]:
                    V[j, ii] = Vmax[ii]

            X[j, :] = X[j, :] + V[j, :]

            for ii in range(dim):
                if X[j, ii] < 0:
                    V[j, ii] = 0
                if X[j, ii] > 0:
                    V[j, ii] = 1
            fitness[j] = fun(X[j, :])

            flag = 0
            if fitness[j] < fitnessPbest[j]:
                Pbest[j, :] = copy.copy(X[j, :])
                fitnessPbest[j] = copy.copy(fitness[j])
                flag = 1

        fitness_list.append(fitness.copy())
        if flag != 1:  # No Individual Optimal Solution Update
            T = T+1
        rand = random.random()  # Generating Random Numbers

        if (math.exp(T)-1)/(math.exp(10)-1) > rand:
            candicate = np.zeros([len(GbestPositon), 1])
            for d in range(len(GbestPositon)):
                rand2 = random.random()  # Generating Random Numbers
                Prob_candicate = random.random()
                if Prob_candicate > rand2:
                    candicate[d] = GbestPositon[d]
                else:
                    random_values = random.sample(range(pop), 2)  # Randomly generate two particles
                    if fitnessPbest[random_values[0]] < fitnessPbest[random_values[1]]:
                        average = np.mean(Pbest, axis=0).tolist()[d]
                        squared_array = (Pbest[:,d] - average) ** 2
                        sigama = np.sqrt(np.mean(squared_array))
                        pdf_value = norm.pdf(0, loc=average, scale=sigama)
                        candicate[d] = Pbest[random_values[0]][d] + pdf_value

                    else:
                        average = np.mean(Pbest, axis=0).tolist()[d]
                        squared_array = (Pbest[:, d] - average) ** 2
                        sigama = np.sqrt(np.mean(squared_array))
                        pdf_value = norm.pdf(0, loc=average, scale=sigama)
                        candicate[d] = Pbest[random_values[1]][d] + pdf_value

            fitness2 = update_fun(pop, X, V, w, c1, c2, Pbest, GbestPositon, dim, Vmin,
                                                               Vmax, lb, ub, fitness, fun, fitnessPbest)

            candicate = candicate.reshape(-1)

            fitness3 = update_fun(pop, X, V, w, c1, c2, Pbest, candicate, dim, Vmin,
                                                                    Vmax, lb, ub, fitness, fun, fitnessPbest)
            result1 = np.subtract(fitness2, fitness_list[-1])

            result2 = np.subtract(fitness3, fitness_list[-1])

            sum_result1 = np.sum(result1)
            sum_result2 = np.sum(result2)

            if sum_result1 >= sum_result2:
                pass
                T = 0

            else:
                GbestPositon = candicate
                T = T-1
        else:
            if fitness.min() < GbestScore[0]:
                GbestScore[0] = copy.copy(fitness.min())
                GbestPositon = copy.copy(X[np.argmin(fitness), :])

        Curve[i] = GbestScore

    return GbestScore, GbestPositon, Curve