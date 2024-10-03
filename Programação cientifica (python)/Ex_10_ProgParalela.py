from mpi4py import MPI
import numpy as np
import sympy as sp
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def monte_carlo(f, a, b, n, comm):

    start_time = time.time()

    local_n = n // size
    remainder = n % size

    if rank < remainder:
        local_n += 1

    local_x = np.random.uniform(a, b, local_n)
    local_y = np.array([f.subs(sp.Symbol('x'), xi) for xi in local_x], dtype=float)

    local_sum = np.sum(local_y)

    global_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

    end_time = time.time()
    exe_time = end_time - start_time

    if rank == 0:
        integral = (b - a) * global_sum / n
        return integral, exe_time
    else:
        return None, exe_time



a, b = 0, 1
n_monte_carlo = 100000

x = sp.Symbol('x')
f_2a = 4 / (1 + x**2)

valor_exato_2a = float(sp.integrate(f_2a, (x, a, b)))

monte_carlo_2a, monte_carlo_time_2a = monte_carlo(f_2a, a, b, n_monte_carlo, comm)

monte_carlo_times = comm.gather(monte_carlo_time_2a, root=0)

if rank == 0:
    print(f"Valor Exato: {valor_exato_2a:.6f}")
    print(f"Monte Carlo: {monte_carlo_2a:.6f}")
    print("Tempos de execução por processo:")
    for i, time in enumerate(monte_carlo_times):
        print(f"Processo {i}: {time:.6f} segundos")


"""
# executanco com 1 processo:
Valor Exato: 3.141593
Monte Carlo: 3.137081
Tempos de execuþÒo por processo:
Processo 0: 77.610039 segundos

# executanco com 3 processo:
Valor Exato: 3.141593
Monte Carlo: 3.142745
Tempos de execuþÒo por processo:
Processo 0: 42.518595 segundos
Processo 1: 42.456088 segundos
Processo 2: 42.220828 segundos

# executanco com 4 processo:
Valor Exato: 3.141593
Monte Carlo: 3.142912
Tempos de execuþÒo por processo:
Processo 0: 36.620098 segundos
Processo 1: 36.604473 segundos
Processo 2: 36.635721 segundos
Processo 3: 36.651346 segundos

# executanco com 6 processo:
Valor Exato: 3.141593
Monte Carlo: 3.140785
Tempos de execuþÒo por processo:
Processo 0: 36.547867 segundos
Processo 1: 37.183975 segundos
Processo 2: 36.682182 segundos
Processo 3: 36.650934 segundos
Processo 4: 37.121475 segundos
Processo 5: 36.337866 segundos

# executanco com 8 processo:
Valor Exato: 3.141593
Monte Carlo: 3.144139
Tempos de execuþÒo por processo:
Processo 0: 37.685586 segundos
Processo 1: 36.385313 segundos
Processo 2: 36.885302 segundos
Processo 3: 37.137006 segundos
Processo 4: 37.623087 segundos
Processo 5: 35.694441 segundos
Processo 6: 37.060604 segundos
Processo 7: 35.270859 segundos

# executanco com 16 processo:
Valor Exato: 3.141593
Monte Carlo: 3.141524
Tempos de execuþÒo por processo:
Processo 0: 35.020881 segundos
Processo 1: 39.581412 segundos
Processo 2: 33.407603 segundos
Processo 3: 39.924605 segundos
Processo 4: 38.092461 segundos
Processo 5: 33.734579 segundos
Processo 6: 33.640809 segundos
Processo 7: 38.233083 segundos
Processo 8: 37.559088 segundos
Processo 9: 33.375178 segundos
Processo 10: 33.124536 segundos
Processo 11: 36.743652 segundos
Processo 12: 37.167198 segundos
Processo 13: 39.313952 segundos
Processo 14: 38.969055 segundos
Processo 15: 33.328294 segundos

"""