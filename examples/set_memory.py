from joblib import Memory
import time

# Set up memory caching
memory = Memory('./cachedir', verbose=0)

@memory.cache
def calculate_sum_squares(start=1,end=101):
    return sum([i ** 2 for i in range(start, end)])

start = time.time()
res = calculate_sum_squares(1, 10_000_001)
print(res)
end = time.time()
print(f'Time taken: {end - start}')


