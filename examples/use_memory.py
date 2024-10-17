import time
from set_memory import calculate_sum_squares, memory


start = time.time()
res = calculate_sum_squares(1, 10_000_001)
print(res)
end = time.time()
print(f'Time taken: {end - start}')
