import csv
import json

csv_columns = [
    "task_id",
    "prompt",
    "declaration",
    "canonical_solution",
    "test",
    "example_test",
]

# Open the JSONL file
with open("humaneval_rust.jsonl", "r") as json_file:
    # Open the CSV file for writing
    with open("rust.csv", "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_columns)

        # Write the header row
        writer.writeheader()

        # Iterate over each line in the JSONL file
        for line in json_file:
            # Parse each line as JSON
            data = json.loads(line)

            # Write the data to the CSV file
            writer.writerow(data)
