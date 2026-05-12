import time
from ex2_part1 import run_part1
from ex2_part2 import run_part2
from ex2_part3 import run_part3
from ex2_part4 import run_part4

def main():
    print("="*60)
    print("STARTING DEEP LEARNING EXERCISE 2")
    print("NOTE: Close the Matplotlib windows to advance to the next step.")
    print("="*60)
    time.sleep(2)

    # Execute Part 1
    print("\n>>> EXECUTING PART 1: Autoencoder <<<")
    run_part1()

    # Execute Part 2
    print("\n>>> EXECUTING PART 2: Classifier <<<")
    run_part2()

    # Execute Part 3
    print("\n>>> EXECUTING PART 3: Pre-trained Representation <<<")
    run_part3()

    # Execute Part 4
    print("\n>>> EXECUTING PART 4: Task Specific Encoding <<<")
    run_part4()

    print("\n" + "="*60)
    print("ALL PARTS COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()