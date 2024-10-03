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
f_2b = sp.sqrt(x + sp.sqrt(x))

valor_exato_2b = float(sp.integrate(f_2b, (x, a, b)))

monte_carlo_2b, monte_carlo_time_2b = monte_carlo(f_2b, a, b, n_monte_carlo, comm)

monte_carlo_times = comm.gather(monte_carlo_time_2b, root=0)

if rank == 0:
    print(f"Valor Exato: {valor_exato_2b:.6f}")
    print(f"Monte Carlo: {monte_carlo_2b:.6f}")
    print("Tempos de execução por processo:")
    for i, time in enumerate(monte_carlo_times):
        print(f"Processo {i}: {time:.6f} segundos")



"""
# executanco com 1 processo:
Valor Exato: 1.045301
Monte Carlo: 1.044738
Tempos de execuþÒo por processo:
Processo 0: 33.470233 segundos

# executanco com 4 processo:
Valor Exato: 1.045301
Monte Carlo: 1.045337
Tempos de execuþÒo por processo:
Processo 0: 15.482623 segundos
Processo 1: 15.561172 segundos
Processo 2: 15.482632 segundos
Processo 3: 15.451383 segundos

# executanco com 8 processo:
alor Exato: 1.045301
Monte Carlo: 1.046412
Tempos de execuþÒo por processo:
Processo 0: 14.883817 segundos
Processo 1: 15.450114 segundos
Processo 2: 16.248695 segundos
Processo 3: 15.668928 segundos
Processo 4: 15.778305 segundos
Processo 5: 15.481424 segundos
Processo 6: 15.104703 segundos
Processo 7: 14.478036 segundos

# executanco com 16 processo:
Valor Exato: 1.045301
Monte Carlo: 1.047067
Tempos de execuþÒo por processo:
Processo 0: 13.664580 segundos
Processo 1: 11.512378 segundos
Processo 2: 16.974042 segundos
Processo 3: 19.789744 segundos
Processo 4: 14.682906 segundos
Processo 5: 16.334302 segundos
Processo 6: 12.222299 segundos
Processo 7: 18.648513 segundos
Processo 8: 17.482194 segundos
Processo 9: 13.663800 segundos
Processo 10: 19.257810 segundos
Processo 11: 11.436521 segundos
Processo 12: 19.838270 segundos
Processo 13: 16.061236 segundos
Processo 14: 15.043771 segundos
Processo 15: 17.707468 segundos
"""