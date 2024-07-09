import json
import random

def generate_and_sort_numbers(count=1):
    numbers = [random.randint(1, 1000) for _ in range(count)]
    sorted_numbers = sorted(numbers)
    return numbers, sorted_numbers

def main():
    original, sorted_list = generate_and_sort_numbers()
    
    results = {
        'original': original,
        'sorted': sorted_list
    }
    
    with open('sorting_results.json', 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
