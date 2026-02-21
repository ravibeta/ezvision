def find_missing_intervals(file_path):
    # Read file and parse numbers
    with open(file_path, 'r') as f:
        content = f.read().strip()
    
    if len(content) % 6 != 0:
        raise ValueError("Invalid file format: Length not divisible by 6")
    
    existing = set()
    for i in range(0, len(content), 6):
        num_str = content[i:i+6]
        existing.add(int(num_str))

    # Find missing intervals
    missing = []
    current_start = None
    max_number = 17854  # Upper bound of search range

    for num in range(1, max_number + 1):
        if num not in existing:
            if current_start is None:
                current_start = num
            current_end = num
        else:
            if current_start is not None:
                missing.append((current_start, current_end))
                current_start = None
                
    # Add final interval if needed
    if current_start is not None:
        missing.append((current_start, max_number))

    return missing

def format_interval(interval):
    start, end = interval
    if start == end:
        return f"{start:06d}"
    return f"{start:06d}-{end:06d}"

# Example usage
if __name__ == "__main__":
    missing_intervals = find_missing_intervals("ids.txt")
    
    print("Missing number intervals:")
    for interval in missing_intervals:
        print(format_interval(interval))
